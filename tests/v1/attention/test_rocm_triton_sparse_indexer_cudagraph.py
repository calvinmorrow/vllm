# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph coverage for the gfx1151 ROCm Triton sparse indexer."""

from __future__ import annotations

import pytest
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1151
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops import rocm_triton_sparse_indexer  # noqa: F401
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    indexer_k_quant_and_cache_triton,
)
from vllm.v1.worker.workspace import current_workspace_manager, init_workspace_manager


_requires_gfx1151 = pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx1151()),
    reason="requires ROCm gfx1151",
)


@_requires_gfx1151
@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE])
def test_rocm_triton_sparse_indexer_decode_cudagraph_replay(mode: CUDAGraphMode):
    """The registered fixed-shape decode op replays without an eager break."""
    device = torch.device("cuda")
    batch_size, block_size, max_model_len, topk_tokens = 2, 64, 128, 16
    prefix = "model.layers.0.self_attn.indexer.k_cache"

    def make_state(seed: int):
        generator = torch.Generator(device=device).manual_seed(seed)
        hidden_states = torch.empty((batch_size, 1), device=device)
        q_fp8 = (
            torch.randn(
                (batch_size, 64, 128), device=device, generator=generator
            )
            * 0.125
        ).to(current_platform.fp8_dtype())
        k = torch.randn(
            (batch_size, 128), device=device, dtype=torch.bfloat16, generator=generator
        )
        weights = torch.rand(
            (batch_size, 64), device=device, dtype=torch.float32, generator=generator
        )
        kv_cache = torch.zeros(
            (4, block_size, 132), dtype=torch.uint8, device=device
        )
        output = torch.empty(
            (batch_size, topk_tokens), dtype=torch.int32, device=device
        )
        return hidden_states, q_fp8, k, weights, kv_cache, output

    block_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([[70], [33]], dtype=torch.int32, device=device)
    decode_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens.view(-1),
        max_seq_len=max_model_len,
        slot_mapping=torch.tensor([128, 192], dtype=torch.int64, device=device),
        num_decodes=batch_size,
        num_decode_tokens=batch_size,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table,
            seq_lens=seq_lens,
            decode_lens=decode_lens,
            requires_padding=False,
            schedule_metadata=torch.empty(0, dtype=torch.int32, device=device),
        ),
    )
    context = ForwardContext({}, {prefix: metadata}, {})

    def run(state):
        hidden_states, q_fp8, k, weights, kv_cache, output = state
        with override_forward_context(context):
            result = torch.ops.vllm.rocm_triton_sparse_attn_indexer(
                hidden_states,
                prefix,
                kv_cache,
                q_fp8,
                k,
                weights,
                128,
                "dynamic",
                topk_tokens,
                128,
                max_model_len,
                max_model_len,
                output,
                False,
            )
        assert result.data_ptr() == output.data_ptr()
        return output.clone(), kv_cache.clone()

    eager_state = make_state(0)
    init_workspace_manager(device)
    eager_output, eager_cache = run(eager_state)

    replay_reference = make_state(1)
    init_workspace_manager(device)
    expected_output, expected_cache = run(replay_reference)

    captured_state = make_state(0)
    init_workspace_manager(device)
    workspace = current_workspace_manager()
    workspace.get_simultaneous(
        ((batch_size, max_model_len), torch.float32),
        ((2, batch_size, 1, 1024), torch.uint64),
    )
    workspace_ptr = workspace._current_workspaces[0].data_ptr()
    captured_pointers = tuple(t.data_ptr() for t in captured_state[1:]) + (
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        workspace_ptr,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), override_forward_context(
        ForwardContext({}, {prefix: metadata}, {}, cudagraph_runtime_mode=mode)
    ):
        capture = BreakableCUDAGraphCapture()
        with capture:
            result = torch.ops.vllm.rocm_triton_sparse_attn_indexer(
                captured_state[0],
                prefix,
                captured_state[4],
                captured_state[1],
                captured_state[2],
                captured_state[3],
                128,
                "dynamic",
                topk_tokens,
                128,
                max_model_len,
                max_model_len,
                captured_state[5],
                False,
            )
            assert result.data_ptr() == captured_state[5].data_ptr()

        capture.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()

    assert capture.num_eager_breaks == 0
    assert tuple(t.data_ptr() for t in captured_state[1:]) + (
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        workspace._current_workspaces[0].data_ptr(),
    ) == captured_pointers
    torch.testing.assert_close(captured_state[5], eager_output)
    torch.testing.assert_close(captured_state[4], eager_cache)

    replay_state = make_state(1)
    captured_state[1].copy_(replay_state[1])
    captured_state[2].copy_(replay_state[2])
    captured_state[3].copy_(replay_state[3])
    captured_state[4].zero_()
    captured_state[5].fill_(-1)

    with torch.cuda.stream(stream):
        capture.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()

    torch.testing.assert_close(captured_state[5], expected_output)
    torch.testing.assert_close(captured_state[4], expected_cache)


@_requires_gfx1151
def test_rocm_triton_sparse_indexer_eager_reference_mutates_kv_cache():
    """Keep the cache-write portion of the integration fixture nontrivial."""
    kv_cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device="cuda")
    k = torch.ones((1, 128), dtype=torch.bfloat16, device="cuda")
    slot_mapping = torch.zeros(1, dtype=torch.int64, device="cuda")
    indexer_k_quant_and_cache_triton(k, kv_cache, slot_mapping, 128, "dynamic")
    assert torch.count_nonzero(kv_cache).item() > 0
