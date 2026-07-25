# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import _warmup_layer_mhc


def test_warmup_layer_dispatches_fused_post_pre_for_each_mhc_function() -> None:
    hidden_size = 8
    hc_mult = 2
    layer = SimpleNamespace(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        hc_attn_fn=torch.empty(1),
        hc_attn_scale=object(),
        hc_attn_base=object(),
        hc_ffn_fn=object(),
        hc_ffn_scale=object(),
        hc_ffn_base=object(),
        rms_norm_eps=1e-6,
        hc_eps=1e-5,
        hc_post_alpha=2.0,
        hc_sinkhorn_iters=4,
    )
    layer.hc_pre = MagicMock(
        side_effect=lambda residual, *_: (
            torch.empty(residual.shape[0], hidden_size),
            torch.empty(residual.shape[0], hc_mult),
            torch.empty(residual.shape[0], hc_mult, hc_mult),
        )
    )
    layer.mhc_fused_post_pre = MagicMock()

    _warmup_layer_mhc(layer, [1, 16])

    assert [call.args[0].shape[0] for call in layer.hc_pre.call_args_list] == [
        1,
        1,
        16,
        16,
    ]
    assert [
        call.args[0].shape for call in layer.mhc_fused_post_pre.call_args_list
    ] == [
        (1, hidden_size),
        (1, hidden_size),
        (16, hidden_size),
        (16, hidden_size),
    ]
    assert [
        call.args[1].shape for call in layer.mhc_fused_post_pre.call_args_list
    ] == [
        (1, hc_mult, hidden_size),
        (1, hc_mult, hidden_size),
        (16, hc_mult, hidden_size),
        (16, hc_mult, hidden_size),
    ]
    assert [call.args[4] for call in layer.mhc_fused_post_pre.call_args_list] == [
        layer.hc_attn_fn,
        layer.hc_ffn_fn,
        layer.hc_attn_fn,
        layer.hc_ffn_fn,
    ]
