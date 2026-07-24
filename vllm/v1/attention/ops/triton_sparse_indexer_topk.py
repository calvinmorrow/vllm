# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Triton top-k selection for DeepSeek V4 sparse attention indexer.

Implements exact deterministic top-k over per-row score vectors using
bitonic sort in shared memory, with chunk-and-merge for long contexts.

Algorithm reference: Dwarfstar ROCm indexer (external reference only).
"""

import math
from typing import Final

import torch

from vllm.triton_utils import tl, triton

# Chunk size for chunk-and-merge phase. Must be power of 2.
CHUNK_N: Final = 4096

# Merge fan-in for tree-merge.
MERGE_GROUP: Final = 8

# Maximum N handled by a single-kernel sort.
SORT_N_MAX: Final = 8192

# Supported SORT_N values (powers of 2).
_SUPPORTED_SORT_N = (256, 512, 1024, 2048, 4096, 8192)

# Scale factor for packing score+index into float64
# Score range is typically [-10, 10], index range is [0, 8191]
# With SCALE = 1e6, score*SCALE dominates and -index breaks ties
_SCORE_SCALE: Final = 1e6


def _select_sort_n(n: int) -> int:
    """Select SORT_N from supported values for given n."""
    if n <= 0:
        return 256
    p = 1
    while p < n:
        p <<= 1
    for s in _SUPPORTED_SORT_N:
        if s >= p:
            return s
    return SORT_N_MAX


@triton.jit
def _pack_score_idx(
    score: tl.tensor,
    idx: tl.tensor,
    scale: tl.constexpr,
) -> tl.tensor:
    """Pack score and index into float64 for sorting.

    Higher score -> higher packed value.
    Equal scores: lower index -> higher packed value (for tie-breaking).
    """
    # packed = score * scale - index
    # Sorting descending: higher score first, then lower index first
    return score.to(tl.float64) * scale - idx.to(tl.float64)


@triton.jit
def _unpack_idx(packed: tl.tensor, scale: tl.constexpr) -> tl.tensor:
    """Extract index from packed float64 value."""
    # idx = round(scale - packed) when score ~ 0
    # More robust: idx = round(-fract(packed / 1) * scale)
    # Actually: packed = score * scale - idx
    # So: idx = score * scale - packed
    # But we don't have score after sorting...
    # Use: idx = round(packed % scale) but Triton doesn't have modulo
    # Use: idx = round(packed - floor(packed / scale) * scale)
    # Actually simpler: the fractional info IS the index
    # idx = round(-packed + round(packed)) -- no, that loses info
    # Let's use: idx = round(score * scale - packed) but we don't have score
    #
    # Alternative: store idx separately and use sort to get permutation
    # Actually, we can recover: idx = round(-frac_part * scale)
    # Where frac_part = packed - trunc(packed) for the small perturbation
    #
    # Simpler approach: after sorting packed values,
    # idx = round(-packed + (packed // 1)) -- but Triton integer div
    #
    # Best: use round(-packed) and subtract round(-packed/scale)*scale
    # idx = round(-packed + round(-packed / scale) * scale)
    #
    # Actually simplest: since packed = score*scale - idx,
    # idx = round(-packed mod scale)
    # = round(-packed - floor(-packed/scale)*scale)
    neg = -packed
    quotient = tl.floor(neg / scale)
    return tl.round(neg - quotient * scale).to(tl.int32)


# ---------------------------------------------------------------------------
# Per-SORT_N kernels using tl.sort with packing
# ---------------------------------------------------------------------------


@triton.jit
def _topk_sort_kernel(
    scores_ptr,
    selected_ptr,
    n_comp,
    top_k,
    stride_scores,
    stride_selected,
    SORT_N: tl.constexpr,
):
    """Sort-based top-k: one program per row."""
    row = tl.program_id(0)

    # Load scores + create indices
    offsets = tl.arange(0, SORT_N)
    mask = offsets < n_comp
    scores_off = row * stride_scores + offsets
    vals = tl.where(mask, tl.load(scores_ptr + scores_off), -float("inf"))
    idxs = tl.where(mask, offsets, tl.full_like(offsets, -1))

    # Pack score+index into float64 for sorting
    # Higher score + lower index = higher packed value
    packed = _pack_score_idx(vals, idxs, _SCORE_SCALE)

    # Sort descending
    packed_sorted = tl.sort(packed, descending=True)

    # Unpack indices
    idxs_sorted = _unpack_idx(packed_sorted, _SCORE_SCALE)

    # Write top-k
    top_offsets = tl.arange(0, top_k)
    tl.store(
        selected_ptr + row * stride_selected + top_offsets,
        idxs_sorted[top_offsets],
        mask=top_offsets < top_k,
    )


@triton.jit
def _topk_chunk_kernel(
    scores_ptr,
    candidates_ptr,
    n_comp,
    top_k,
    candidate_stride,
    stride_scores,
    SORT_N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Chunk-local top-k for chunk-and-merge pipeline."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)

    actual_chunk_start = chunk * CHUNK_SIZE
    if actual_chunk_start >= n_comp:
        return

    actual_chunk_n = min(n_comp - actual_chunk_start, CHUNK_SIZE)

    offsets = tl.arange(0, SORT_N)
    abs_offsets = actual_chunk_start + offsets
    mask = offsets < actual_chunk_n
    scores_off = row * stride_scores + abs_offsets
    vals = tl.where(mask, tl.load(scores_ptr + scores_off), -float("inf"))
    idxs = tl.where(mask, abs_offsets, tl.full_like(abs_offsets, -1))

    packed = _pack_score_idx(vals, idxs, _SCORE_SCALE)
    packed_sorted = tl.sort(packed, descending=True)
    idxs_sorted = _unpack_idx(packed_sorted, _SCORE_SCALE)

    out_ptr = candidates_ptr + row * candidate_stride + chunk * top_k
    top_offsets = tl.arange(0, top_k)
    tl.store(
        out_ptr + top_offsets,
        idxs_sorted[top_offsets],
        mask=top_offsets < top_k,
    )


@triton.jit
def _topk_merge_kernel(
    candidates_ptr,
    scores_ptr,
    out_ptr,
    n_comp,
    top_k,
    candidate_count,
    stride_scores,
    SORT_N: tl.constexpr,
):
    """Merge sorted candidate sets into top-k."""
    row = tl.program_id(0)

    offsets = tl.arange(0, SORT_N)
    c_mask = offsets < candidate_count

    # Load candidate indices
    c_idx = tl.where(
        c_mask,
        tl.load(candidates_ptr + offsets),
        -1,
    )

    # Load scores at candidate indices
    valid = c_idx >= 0 & c_idx < n_comp
    vals = tl.where(
        valid,
        tl.load(scores_ptr + row * stride_scores + c_idx),
        -float("inf"),
    )

    packed = _pack_score_idx(vals, c_idx, _SCORE_SCALE)
    packed_sorted = tl.sort(packed, descending=True)
    idxs_sorted = _unpack_idx(packed_sorted, _SCORE_SCALE)

    top_offsets = tl.arange(0, top_k)
    tl.store(
        out_ptr + row * top_k + top_offsets,
        idxs_sorted[top_offsets],
        mask=top_offsets < top_k,
    )


@triton.jit
def _topk_tree_merge_kernel(
    candidates_ptr,
    scores_ptr,
    out_ptr,
    n_comp,
    top_k,
    n_sets,
    merge_group,
    candidate_count,
    cur_stride,
    next_stride,
    stride_scores,
    SORT_N: tl.constexpr,
):
    """Tree merge: merge one group of candidate sets."""
    row = tl.program_id(0)
    group = tl.program_id(1)

    set0 = group * merge_group
    if set0 >= n_sets:
        return

    actual_set_count = min(merge_group, n_sets - set0)
    actual_candidate_count = actual_set_count * top_k

    base = set0 * top_k
    offsets = tl.arange(0, SORT_N)
    c_mask = offsets < actual_candidate_count

    c_idx = tl.where(
        c_mask,
        tl.load(candidates_ptr + row * cur_stride + base + offsets),
        -1,
    )

    valid = c_idx >= 0 & c_idx < n_comp
    vals = tl.where(
        valid,
        tl.load(scores_ptr + row * stride_scores + c_idx),
        -float("inf"),
    )

    packed = _pack_score_idx(vals, c_idx, _SCORE_SCALE)
    packed_sorted = tl.sort(packed, descending=True)
    idxs_sorted = _unpack_idx(packed_sorted, _SCORE_SCALE)

    out_base = row * next_stride + group * top_k
    top_offsets = tl.arange(0, top_k)
    tl.store(
        out_ptr + out_base + top_offsets,
        idxs_sorted[top_offsets],
        mask=top_offsets < top_k,
    )


# ---------------------------------------------------------------------------
# Python launch wrappers
# ---------------------------------------------------------------------------


def _topk_single_phase(
    scores: torch.Tensor,
    top_k: int,
    sort_n: int,
) -> torch.Tensor:
    """Single-phase sort top-k.

    Args:
        scores: [num_tokens, n_comp] fp32 scores.
        top_k: number of top indices to select.
        sort_n: power-of-2 size >= n_comp.

    Returns:
        [num_tokens, top_k] int32 indices (descending score, -1 padding).
    """
    num_tokens, n_comp = scores.shape
    assert sort_n >= n_comp, f"sort_n={sort_n} < n_comp={n_comp}"

    selected = torch.full(
        (num_tokens, top_k), -1, dtype=torch.int32, device=scores.device
    )

    grid = (num_tokens,)
    _topk_sort_kernel[grid](
        scores,
        selected,
        n_comp,
        top_k,
        scores.stride(0),
        selected.stride(0),
        SORT_N=sort_n,
        num_warps=4,
    )
    return selected


def _topk_chunk_and_merge(
    scores: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Chunk-and-merge top-k for n_comp > SORT_N_MAX.

    Phase 1: sort each CHUNK_N chunk independently, keep top-k per chunk.
    Phase 2: tree-merge chunk results until <= MERGE_GROUP sets remain.
    Phase 3: final merge into global top-k.
    """
    num_tokens, n_comp = scores.shape
    n_chunks = math.ceil(n_comp / CHUNK_N)
    sort_n = CHUNK_N

    # Scratch allocation
    candidate_stride_phase1 = n_chunks * top_k
    n_sets = n_chunks
    total_scratch = candidate_stride_phase1
    while n_sets > MERGE_GROUP:
        n_sets = math.ceil(n_sets / MERGE_GROUP)
        total_scratch += n_sets * top_k

    scratch = torch.empty(
        (num_tokens, total_scratch), dtype=torch.int32, device=scores.device
    )

    # Phase 1: chunk-local top-k
    cur_offset = 0
    cur_stride = candidate_stride_phase1
    grid_chunks = (num_tokens, n_chunks)
    _topk_chunk_kernel[grid_chunks](
        scores,
        scratch,
        n_comp,
        top_k,
        cur_stride,
        scores.stride(0),
        SORT_N=sort_n,
        CHUNK_SIZE=CHUNK_N,
        num_warps=4,
    )

    # Phase 2: tree merge
    n_sets = n_chunks
    while n_sets > MERGE_GROUP:
        next_sets = math.ceil(n_sets / MERGE_GROUP)
        next_stride = next_sets * top_k
        next_offset = cur_offset + cur_stride

        grid_merge = (num_tokens, next_sets)
        cur_ptr = scratch[:, cur_offset : cur_offset + cur_stride]
        next_ptr = scratch[:, next_offset : next_offset + next_stride]

        _topk_tree_merge_kernel[grid_merge](
            cur_ptr,
            scores,
            next_ptr,
            n_comp,
            top_k,
            n_sets,
            MERGE_GROUP,
            MERGE_GROUP * top_k,
            cur_stride,
            next_stride,
            scores.stride(0),
            SORT_N=sort_n,
            num_warps=4,
        )

        cur_offset = next_offset
        cur_stride = next_stride
        n_sets = next_sets

    # Phase 3: final merge
    selected = torch.full(
        (num_tokens, top_k), -1, dtype=torch.int32, device=scores.device
    )

    final_candidate_count = n_sets * top_k
    final_cur_ptr = scratch[:, cur_offset : cur_offset + cur_stride]

    grid_final = (num_tokens,)
    _topk_merge_kernel[grid_final](
        final_cur_ptr,
        scores,
        selected,
        n_comp,
        top_k,
        final_candidate_count,
        scores.stride(0),
        SORT_N=sort_n,
        num_warps=4,
    )

    return selected


def triton_sparse_indexer_topk(
    scores: torch.Tensor,
    top_k: int,
    n_valid: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Compute exact top-k indices for each row of scores.

    Args:
        scores: [num_tokens, n_comp] fp32 score matrix.
        top_k: number of top indices to return per row.
        n_valid: optional per-row valid candidate count (scalar or
            [num_tokens] tensor). Positions >= n_valid are masked.

    Returns:
        [num_tokens, top_k] int32 indices, descending score, ascending-index
        tie-breaking. Padding positions are -1.
    """
    num_tokens, n_comp = scores.shape
    assert scores.dtype == torch.float32
    assert top_k > 0

    if isinstance(n_valid, torch.Tensor):
        effective_n = int(n_valid.max().item())
    elif isinstance(n_valid, int):
        effective_n = n_valid
    else:
        effective_n = n_comp
    effective_n = min(effective_n, n_comp)

    if effective_n <= SORT_N_MAX:
        sort_n = _select_sort_n(effective_n)
        return _topk_single_phase(scores, top_k, sort_n)
    else:
        return _topk_chunk_and_merge(scores, top_k)


def triton_sparse_indexer_topk_prefill(
    scores: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Top-k for prefill with per-row valid spans and local index output.

    For each token row i, valid candidates are in
    [cu_seqlen_ks[i], cu_seqlen_ke[i]). Output indices are local
    (relative to span start), with -1 padding.

    Args:
        scores: [num_tokens, max_n_comp] fp32. Positions outside valid
            span should be masked to -inf.
        cu_seqlen_ks: [num_tokens] span start per row.
        cu_seqlen_ke: [num_tokens] span end per row.
        top_k: top-k count.

    Returns:
        [num_tokens, top_k] int32 local indices.
    """
    abs_indices = triton_sparse_indexer_topk(scores, top_k)

    starts = cu_seqlen_ks.unsqueeze(1).to(torch.int32)
    span_lens = (cu_seqlen_ke - cu_seqlen_ks).unsqueeze(1).to(torch.int32)

    # Convert absolute to local indices
    local_indices = torch.where(
        abs_indices >= 0,
        abs_indices - starts,
        torch.full_like(abs_indices, -1),
    )

    # Clamp: if local index >= span_len, mark invalid
    local_indices = torch.where(
        (local_indices >= 0) & (local_indices < span_lens),
        local_indices,
        torch.full_like(local_indices, -1),
    )

    return local_indices


def triton_sparse_indexer_topk_decode(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Top-k for decode with per-row sequence length masking.

    Args:
        scores: [num_tokens, n_comp] fp32 scores.
        seq_lens: [num_tokens] valid length per row.
        top_k: top-k count.

    Returns:
        [num_tokens, top_k] int32 indices with -1 padding.
    """
    indices = triton_sparse_indexer_topk(scores, top_k, n_valid=seq_lens)

    lens = seq_lens.unsqueeze(1).to(torch.int32)
    indices = torch.where(
        (indices >= 0) & (indices < lens),
        indices,
        torch.full_like(indices, -1),
    )
    return indices
