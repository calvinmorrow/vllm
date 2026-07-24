# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1151 Triton sparse indexer for DeepSeek V4.

Provides a Triton-based sparse indexer path for gfx1151 that replaces
the AITER-only top-k selection with native Triton bitonic sort kernels.

Reuses existing Triton score computation (rocm_fp8_mqa_logits,
rocm_fp8_paged_mqa_logits) and K-cache management from
rocm_aiter_mla_sparse.py. Only the top-k selection step is replaced.
"""

from __future__ import annotations

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.utils.torch_utils import LayerNameType, _resolve_layer_name
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_rocm():
    from vllm.platforms.rocm import (
        on_gfx942 as _on_gfx942,
    )
    from vllm.platforms.rocm import (
        on_gfx950 as _on_gfx950,
    )
    from vllm.platforms.rocm import (
        on_gfx1151 as _on_gfx1151,
    )
else:
    _on_gfx942 = lambda: False  # type: ignore[assignment]
    _on_gfx950 = lambda: False  # type: ignore[assignment]
    _on_gfx1151 = lambda: False  # type: ignore[assignment]


def is_gfx1151_triton_sparse_indexer_available() -> bool:
    """Return True when the gfx1151 Triton sparse-indexer path is available.

    Narrow, operation-specific capability check. Does not affect AITER
    eligibility or broad ROCm capability gates.
    """
    return _on_gfx1151()


@eager_break_during_capture
def rocm_triton_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    """Sparse attention indexer for gfx1151 using Triton top-k selection.

    Mirrors rocm_aiter_sparse_attn_indexer but replaces the
    torch.ops._C.top_k_per_row_prefill/decode calls with native
    Triton bitonic sort kernels.
    """
    from vllm import envs
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_k_quant_cache_triton,
        indexer_k_quant_and_cache_triton,
        rocm_fp8_mqa_logits,
        rocm_fp8_paged_mqa_logits,
    )
    from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
        triton_sparse_indexer_topk_decode,
        triton_sparse_indexer_topk_prefill,
    )

    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()

    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    if not isinstance(attn_metadata, dict):
        # Profiling early-exit: reserve workspace memory
        workspace_manager = current_workspace_manager()
        workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )
        # Decode logits buffer: gfx1151 Triton path uses
        # fp8_paged_mqa_logits_torch which returns 2D logits
        # [batch_size * next_n, max_model_len], not 3D.
        workspace_manager.get_simultaneous(
            ((hidden_states.shape[0], max_model_len), torch.float32),
        )
        max_logits_elems = (
            envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        )
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )
        return topk_indices_buffer

    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None
    assert scale_fmt is not None

    slot_mapping = layer_attn_metadata.slot_mapping
    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens

    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    elif not skip_k_cache_insert:
        raise ValueError("k must be provided when skip_k_cache_insert is False")

    if not skip_k_cache_insert:
        indexer_k_quant_and_cache_triton(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1

    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None

        workspace_manager = current_workspace_manager()
        k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )

        for chunk in prefill_metadata.chunks:
            k_fp8 = k_fp8_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            cp_gather_indexer_k_quant_cache_triton(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
                token_to_seq=chunk.token_to_seq,
            )

            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            topk_indices.fill_(-1)

            # Triton top-k selection (replaces torch.ops._C.top_k_per_row_prefill)
            result = triton_sparse_indexer_topk_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_tokens,
            )
            topk_indices.copy_(result)

    if has_decode:
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None

        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens

        if decode_metadata.requires_padding:
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )

        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        topk_indices.fill_(-1)

        # Triton top-k selection (replaces torch.ops._C.top_k_per_row_decode)
        result = triton_sparse_indexer_topk_decode(
            logits,
            decode_metadata.seq_lens.repeat_interleave(next_n),
            topk_tokens,
        )
        topk_indices.copy_(result)

        if decode_metadata.requires_padding:
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, next_n, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


# ---------------------------------------------------------------------------
# Torch custom op registration for proper cudagraph handling
# ---------------------------------------------------------------------------

def _rocm_triton_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


if _on_gfx1151():
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="rocm_triton_sparse_attn_indexer",
        op_func=rocm_triton_sparse_attn_indexer,
        mutates_args=["topk_indices_buffer"],
        fake_impl=_rocm_triton_sparse_attn_indexer_fake,
        dispatch_key=current_platform.dispatch_key,
    )
