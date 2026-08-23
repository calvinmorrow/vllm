# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton paged FP8 MQA logits for the gfx1151 sparse indexer decode path."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import current_workspace_manager


@triton.jit
def _paged_fp8_mqa_logits_kernel(
    q_ptr,
    kv_cache_ptr,
    kv_scales_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    next_n,
    max_model_len,
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_kv_block,
    stride_kv_tile,
    stride_kv_token,
    stride_kv_head_tile,
    stride_kv_head,
    stride_scale_block,
    stride_scale_token,
    stride_weights_row,
    stride_weights_head,
    stride_context_b,
    stride_context_n,
    stride_block_table_b,
    stride_block_table_page,
    stride_logits_row,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    batch = row // next_n
    token = row % next_n

    context_len = tl.load(
        context_lens_ptr + batch * stride_context_b + token * stride_context_n
    )
    query_offset = context_len - 1
    logical_k = tile * BLOCK_KV + tl.arange(0, BLOCK_KV)
    logical_block = logical_k // BLOCK_SIZE
    block_offset = logical_k % BLOCK_SIZE
    physical_block = tl.load(
        block_tables_ptr
        + batch * stride_block_table_b
        + logical_block * stride_block_table_page,
        mask=logical_k < max_model_len,
        other=0,
    ).to(tl.int64)

    head = tl.arange(0, HEADS)
    dim = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + batch * stride_q_b
        + token * stride_q_n
        + head[:, None] * stride_q_h
        + dim[None, :] * stride_q_d
    )
    weights = tl.load(
        weights_ptr + row * stride_weights_row + head * stride_weights_head
    ).to(tl.float32)

    key_dim = tl.arange(0, HEAD_DIM)[:, None]
    key_offset = block_offset[None, :]
    key_ptrs = (
        kv_cache_ptr
        + physical_block[None, :] * stride_kv_block
        + (key_offset // 16) * stride_kv_tile
        + (key_offset % 16) * stride_kv_token
        + (key_dim // 16) * stride_kv_head_tile
        + (key_dim % 16) * stride_kv_head
    )
    valid = (logical_k < context_len) & (logical_k <= query_offset)
    keys = tl.load(key_ptrs, mask=valid[None, :], other=0.0)
    scales = tl.load(
        kv_scales_ptr
        + physical_block * stride_scale_block
        + block_offset * stride_scale_token,
        mask=valid,
        other=0.0,
    )

    scores = tl.dot(q, keys, input_precision="ieee")
    scores = tl.maximum(scores, 0.0) * weights[:, None]
    logits = tl.sum(scores, axis=0) * scales
    tl.store(
        logits_ptr + row * stride_logits_row + logical_k,
        logits,
        mask=valid,
    )


def triton_fp8_paged_mqa_logits_gfx1151(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute fixed-shape paged indexer logits without host scalar reads.

    This initial gfx1151 implementation supports the deployed non-speculative
    DeepSeek V4 indexer geometry: ``next_n=1``, 64 heads, and head size 128.
    """
    if q_fp8.ndim != 4 or q_fp8.shape[1:] != (1, 64, 128):
        raise RuntimeError(
            "gfx1151 Triton paged MQA logits requires q_fp8 shaped [B, 1, 64, 128]"
        )
    if kv_cache.ndim != 4 or kv_cache.shape[2:] != (1, 132):
        raise RuntimeError(
            "gfx1151 Triton paged MQA logits requires packed FP8 KV cache"
        )
    if kv_cache.dtype != torch.uint8 or weights.dtype != torch.float32:
        raise RuntimeError(
            "gfx1151 Triton paged MQA logits received unsupported dtypes"
        )
    if context_lens.ndim not in (1, 2):
        raise RuntimeError(
            "gfx1151 Triton paged MQA logits requires rank-1 or rank-2 lengths"
        )

    batch_size, next_n, _, head_dim = q_fp8.shape
    block_size = kv_cache.shape[1]
    if block_size % 16 or block_size % 64:
        raise RuntimeError(
            "gfx1151 Triton paged MQA logits requires block size divisible by 64"
        )
    if weights.shape != (batch_size * next_n, 64):
        raise RuntimeError("gfx1151 Triton paged MQA logits received invalid weights")

    context_lens = context_lens.view(batch_size, next_n)
    num_blocks = kv_cache.shape[0]
    cache_flat = kv_cache.view(num_blocks, -1)
    cache_values = cache_flat[:, : block_size * head_dim].view(q_fp8.dtype)
    cache_scales = cache_flat[:, block_size * head_dim :].view(torch.float32)
    if out is None:
        (logits,) = current_workspace_manager().get_simultaneous(
            ((batch_size * next_n, max_model_len), torch.float32),
        )
    else:
        if (
            out.device != q_fp8.device
            or out.dtype != torch.float32
            or out.shape != (batch_size * next_n, max_model_len)
            or not out.is_contiguous()
        ):
            raise RuntimeError(
                "out must be a contiguous FP32 tensor on q_fp8.device with shape "
                "[batch_size * next_n, max_model_len]"
            )
        logits = out
    logits.fill_(float("-inf"))

    grid = (batch_size * next_n, triton.cdiv(max_model_len, 64))
    _paged_fp8_mqa_logits_kernel[grid](
        q_fp8,
        cache_values,
        cache_scales,
        weights,
        context_lens,
        block_tables,
        logits,
        next_n,
        max_model_len,
        *q_fp8.stride(),
        cache_values.stride(0),
        16 * head_dim,
        16,
        16 * 16,
        1,
        cache_scales.stride(0),
        cache_scales.stride(1),
        *weights.stride(),
        *context_lens.stride(),
        *block_tables.stride(),
        logits.stride(0),
        HEADS=64,
        HEAD_DIM=128,
        BLOCK_SIZE=block_size,
        BLOCK_KV=64,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return logits
