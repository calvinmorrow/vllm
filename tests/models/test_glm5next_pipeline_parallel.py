# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.glm5next.nvidia import model as glm5next_mod
from vllm.model_executor.models.interfaces import supports_pp
from vllm.sequence import IntermediateTensors


class _BoundaryLayer(nn.Module):
    def __init__(
        self,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.output_residual = residual
        self.output_post = post
        self.output_comb = comb
        self.received: tuple[torch.Tensor | None, ...] | None = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del positions
        self.received = (residual, post, comb)
        return (
            hidden_states,
            residual if self.output_residual is None else self.output_residual,
            post if self.output_post is None else self.output_post,
            comb if self.output_comb is None else self.output_comb,
        )


def _model(mhc: bool, num_streams: int = 3) -> glm5next_mod.Glm5NextModel:
    model = glm5next_mod.Glm5NextModel.__new__(glm5next_mod.Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=5,
        mhc=mhc,
        mhc_num_residual_streams=num_streams,
    )
    model.is_sequence_parallel = False
    model.norm = nn.Identity()
    return model


@pytest.mark.cpu_test
@pytest.mark.parametrize("mhc", [False, True])
def test_glm5next_pp_intermediate_tensor_allocator(mhc: bool):
    model = _model(mhc)

    tensors = model.make_empty_intermediate_tensors(
        batch_size=7,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    expected_shapes = {
        "hidden_states": (7, 5),
        "residual": (7, 3, 5) if mhc else (7, 5),
    }
    expected_dtypes = {
        "hidden_states": torch.bfloat16,
        "residual": torch.bfloat16,
    }
    if mhc:
        expected_shapes.update(post=(7, 3, 1), comb=(7, 3, 3))
        expected_dtypes.update(post=torch.float32, comb=torch.float32)

    assert tensors.tensors.keys() == expected_shapes.keys()
    for name, tensor in tensors.items():
        assert tensor.shape == expected_shapes[name]
        assert tensor.dtype == expected_dtypes[name]
        assert tensor.device.type == "cpu"


@pytest.mark.cpu_test
def test_glm5next_mhc_pp_boundary_preserves_continuation_state(monkeypatch):
    hidden_states = torch.randn(2, 5, dtype=torch.bfloat16)
    residual = torch.randn(2, 3, 5, dtype=torch.bfloat16)
    post = torch.randn(2, 3, 1)
    comb = torch.randn(2, 3, 3)

    first_stage = _model(mhc=True)
    first_layer = _BoundaryLayer(residual, post, comb)
    first_stage._active_layers = nn.ModuleList([first_layer])
    monkeypatch.setattr(
        glm5next_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    intermediate_tensors = first_stage(
        input_ids=None,
        positions=torch.tensor([0, 1]),
        intermediate_tensors=None,
        inputs_embeds=hidden_states,
    )

    assert isinstance(intermediate_tensors, IntermediateTensors)
    assert intermediate_tensors.tensors.keys() == {
        "hidden_states",
        "residual",
        "post",
        "comb",
    }
    assert intermediate_tensors["post"] is post
    assert intermediate_tensors["comb"] is comb

    second_stage = _model(mhc=True)
    second_layer = _BoundaryLayer()
    second_stage._active_layers = nn.ModuleList([second_layer])
    monkeypatch.setattr(
        glm5next_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    second_stage(
        input_ids=None,
        positions=torch.tensor([0, 1]),
        intermediate_tensors=intermediate_tensors,
    )

    assert second_layer.received is not None
    received_residual, received_post, received_comb = second_layer.received
    assert received_residual is residual
    assert received_post is post
    assert received_comb is comb


@pytest.mark.cpu_test
def test_glm5next_non_mhc_pp_boundary_uses_two_tensor_contract(monkeypatch):
    hidden_states = torch.randn(2, 5, dtype=torch.bfloat16)
    residual = torch.randn(2, 5, dtype=torch.bfloat16)

    first_stage = _model(mhc=False)
    first_stage._active_layers = nn.ModuleList([_BoundaryLayer(residual)])
    monkeypatch.setattr(
        glm5next_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    intermediate_tensors = first_stage(
        input_ids=None,
        positions=torch.tensor([0, 1]),
        intermediate_tensors=None,
        inputs_embeds=hidden_states,
    )

    assert isinstance(intermediate_tensors, IntermediateTensors)
    assert intermediate_tensors.tensors.keys() == {"hidden_states", "residual"}

    second_stage = _model(mhc=False)
    second_layer = _BoundaryLayer()
    second_stage._active_layers = nn.ModuleList([second_layer])
    monkeypatch.setattr(
        glm5next_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    second_stage(
        input_ids=None,
        positions=torch.tensor([0, 1]),
        intermediate_tensors=intermediate_tensors,
    )

    assert second_layer.received is not None
    received_residual, received_post, received_comb = second_layer.received
    assert received_residual is residual
    assert received_post is None
    assert received_comb is None


@pytest.mark.cpu_test
def test_glm5next_causal_lm_delegates_pp_allocator(monkeypatch):
    sentinel = IntermediateTensors({"hidden_states": torch.zeros(1, 1)})

    class _InnerModel(nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            del vllm_config, prefix

        def make_empty_intermediate_tensors(self, batch_size, dtype, device):
            del batch_size, dtype, device
            return sentinel

    monkeypatch.setattr(glm5next_mod, "Glm5NextModel", _InnerModel)
    monkeypatch.setattr(
        glm5next_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    config = SimpleNamespace(vocab_size=11, hidden_size=5, logit_scale=1.0)
    target = glm5next_mod.Glm5NextForCausalLM(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            quant_config=None,
        )
    )

    assert supports_pp(target)
    assert (
        target.make_empty_intermediate_tensors(
            batch_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        is sentinel
    )
