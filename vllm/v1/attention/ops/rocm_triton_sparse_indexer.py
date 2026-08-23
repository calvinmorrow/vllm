# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Triton sparse indexer for DeepSeek V4.

Provides a Triton-based sparse indexer path on supported ROCm devices that
replaces the AITER-only top-k selection with a device-only stable-sort fallback.

Reuses existing Triton score computation (rocm_fp8_mqa_logits,
rocm_fp8_paged_mqa_logits) and K-cache management from
rocm_aiter_mla_sparse.py. Only the top-k selection step is replaced.
"""

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1151
from vllm.utils.torch_utils import LayerNameType, _resolve_layer_name
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager


def is_rocm_triton_sparse_indexer_available() -> bool:
    """Return whether the ROCm sparse-indexer fallback is available.

    The Triton score-producing and cache dependencies have only been validated
    on gfx1151. AITER is selected by the caller when enabled.
    """
    return current_platform.is_rocm() and on_gfx1151()


def rocm_triton_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """Dispatch the capture-safe decode path or a narrow eager fallback."""
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        k_cache_prefix = _resolve_layer_name(k_cache_prefix)
        layer_attn_metadata = attn_metadata[k_cache_prefix]
        assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
        decode_metadata = layer_attn_metadata.decode
        if (
            decode_metadata is not None
            and (
                decode_metadata.requires_padding
                or decode_metadata.seq_lens.shape[-1] != 1
            )
        ):
            raise NotImplementedError(
                "The gfx1151 Triton sparse indexer supports only unpadded "
                "single-token decode"
            )
        if layer_attn_metadata.num_prefills:
            return _rocm_triton_sparse_attn_indexer_eager_fallback(
                hidden_states,
                k_cache_prefix,
                kv_cache,
                q_fp8,
                k,
                weights,
                quant_block_size,
                scale_fmt,
                topk_tokens,
                head_dim,
                max_model_len,
                total_seq_lens,
                topk_indices_buffer,
                skip_k_cache_insert,
            )
    return _rocm_triton_sparse_attn_indexer_impl(
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_fp8,
        k,
        weights,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        skip_k_cache_insert,
    )


@eager_break_during_capture
def _rocm_triton_sparse_attn_indexer_eager_fallback(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor | None,
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
    return _rocm_triton_sparse_attn_indexer_impl(
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_fp8,
        k,
        weights,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        skip_k_cache_insert,
    )


def _rocm_triton_sparse_attn_indexer_impl(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor | None,
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
    """Sparse attention indexer using the ROCm Triton top-k fallback.

    Mirrors rocm_aiter_sparse_attn_indexer but replaces the
    torch.ops._C.top_k_per_row_prefill/decode calls with native
    Triton bitonic sort kernels.
    """
    from vllm import envs
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_k_quant_cache_triton,
        indexer_k_quant_and_cache_triton,
    )
    from vllm.v1.attention.ops.triton_fp8_mqa_logits import triton_fp8_mqa_logits
    from vllm.v1.attention.ops.triton_fp8_paged_mqa_logits import (
        triton_fp8_paged_mqa_logits_gfx1151,
    )
    from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
        sparse_indexer_topk_scratch_shape,
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
        # Decode logits buffer uses
        # fp8_paged_mqa_logits_torch which returns 2D logits
        # [batch_size * next_n, max_model_len], not 3D.
        workspace_manager.get_simultaneous(
            ((hidden_states.shape[0], max_model_len), torch.float32),
            (
                sparse_indexer_topk_scratch_shape(
                    hidden_states.shape[0], max_model_len, topk_tokens
                ),
                torch.uint64,
            ),
        )
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
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

        for chunk in prefill_metadata.chunks:
            num_chunk_tokens = chunk.token_end - chunk.token_start
            k_fp8, k_scale, logits, topk_scratch = (
                workspace_manager.get_simultaneous(
                    ((chunk.total_seq_lens, head_dim), fp8_dtype),
                    ((chunk.total_seq_lens, 4), torch.uint8),
                    ((num_chunk_tokens, chunk.total_seq_lens), torch.float32),
                    (
                        sparse_indexer_topk_scratch_shape(
                            num_chunk_tokens, chunk.total_seq_lens, topk_tokens
                        ),
                        torch.uint64,
                    ),
                )
            )

            cp_gather_indexer_k_quant_cache_triton(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
                token_to_seq=chunk.token_to_seq,
            )

            triton_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                k_fp8,
                k_scale.view(torch.float32),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                out=logits,
            )

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            triton_sparse_indexer_topk_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_tokens,
                topk_indices,
                topk_scratch,
            )

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

        # Reserve the logits and selector planes together. The logits producer
        # takes the first view; retaining the second prevents scratch from
        # overlapping logits and fixes the workspace capacity before capture.
        logits, topk_scratch = current_workspace_manager().get_simultaneous(
            ((num_padded_tokens, max_model_len), torch.float32),
            (
                sparse_indexer_topk_scratch_shape(
                    num_padded_tokens, max_model_len, topk_tokens
                ),
                torch.uint64,
            ),
        )
        logits = triton_fp8_paged_mqa_logits_gfx1151(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            max_model_len=max_model_len,
            out=logits,
        )

        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        # Decode metadata stores one length per padded token as a contiguous
        # [batch_size, next_n] tensor. The selector accepts a rank-1 view;
        # view() preserves its capture-stable storage and never allocates.
        topk_seq_lens = decode_metadata.seq_lens.view(-1)
        triton_sparse_indexer_topk_decode(
            logits,
            topk_seq_lens,
            topk_tokens,
            topk_indices,
            topk_scratch,
        )

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
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
    compress_ratio: int = 1,
) -> torch.Tensor:
    return topk_indices_buffer


if is_rocm_triton_sparse_indexer_available():
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="rocm_triton_sparse_attn_indexer",
        op_func=rocm_triton_sparse_attn_indexer,
        mutates_args=["kv_cache", "topk_indices_buffer"],
        fake_impl=_rocm_triton_sparse_attn_indexer_fake,
        dispatch_key=current_platform.dispatch_key,
    )
