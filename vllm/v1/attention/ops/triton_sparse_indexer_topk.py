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


def _pytorch_topk(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Reference top-k with exact deterministic ordering.

    Sorts by score descending, then by index ascending for ties.
    Positions with -inf scores are excluded (replaced with -1 padding).
    Returns [num_tokens, top_k] int32 indices with -1 padding.

    Uses numpy stable sort to guarantee correct tie-breaking regardless
    of torch.sort stability on the target backend (ROCm sort is unstable).
    """
    import numpy as np

    num_tokens, n_comp = scores.shape
    device = scores.device

    # numpy stable sort: negate scores so ascending argsort = descending score.
    # Stable sort preserves original index order for equal scores (ascending).
    scores_np = scores.cpu().numpy()
    indices_np = np.arange(n_comp, dtype=np.intp).reshape(1, -1)

    # Sort indices by negated scores (stable)
    sort_perm = np.argsort(-scores_np, axis=1, kind="stable")

    # Gather sorted indices and scores
    sorted_indices = np.take_along_axis(indices_np, sort_perm, axis=1)
    sorted_scores = np.take_along_axis(scores_np, sort_perm, axis=1)

    # Mask: keep only finite scores; replace rest with -1
    valid = np.isfinite(sorted_scores)
    sorted_indices[~valid] = -1

    # Take top_k columns
    k = min(top_k, n_comp)
    result_np = sorted_indices[:, :k]

    # Pad to top_k rows if n_comp < top_k
    if n_comp < top_k:
        pad = np.full((num_tokens, top_k - n_comp), -1, dtype=np.intp)
        result_np = np.hstack([result_np, pad])

    return torch.from_numpy(result_np).to(device, dtype=torch.int32)


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
    SCORE_SCALE: tl.constexpr,
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
    idxs = tl.where(mask, abs_offsets, -1)

    packed = _pack_score_idx(vals, idxs, SCORE_SCALE)
    packed_sorted = tl.sort(packed, descending=True)
    idxs_sorted = _unpack_idx(packed_sorted, SCORE_SCALE)

    out_ptr = candidates_ptr + row * candidate_stride + chunk * top_k
    top_offsets = tl.arange(0, SORT_N)
    tl.store(
        out_ptr + top_offsets,
        idxs_sorted,
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
    SCORE_SCALE: tl.constexpr,
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

    packed = _pack_score_idx(vals, c_idx, SCORE_SCALE)
    packed_sorted = tl.sort(packed, descending=True)
    idxs_sorted = _unpack_idx(packed_sorted, SCORE_SCALE)

    top_offsets = tl.arange(0, SORT_N)
    tl.store(
        out_ptr + row * top_k + top_offsets,
        idxs_sorted,
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
    SCORE_SCALE: tl.constexpr,
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

    packed = _pack_score_idx(vals, c_idx, SCORE_SCALE)
    packed_sorted = tl.sort(packed, descending=True)
    idxs_sorted = _unpack_idx(packed_sorted, SCORE_SCALE)

    out_base = row * next_stride + group * top_k
    top_offsets = tl.arange(0, SORT_N)
    tl.store(
        out_ptr + out_base + top_offsets,
        idxs_sorted,
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
    """Single-phase top-k using PyTorch (replaces broken Triton float64 kernel).

    Args:
        scores: [num_tokens, n_comp] fp32 scores.
        top_k: number of top indices to select.
        sort_n: power-of-2 size >= n_comp (unused, kept for API compat).

    Returns:
        [num_tokens, top_k] int32 indices (descending score, -1 padding).
    """
    return _pytorch_topk(scores, top_k)


def _topk_chunk_and_merge(
    scores: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Top-k for large n_comp using PyTorch (replaces broken Triton chunk-and-merge).

    The original chunk-and-merge Triton approach used float64 packing which
    is broken on gfx1151. PyTorch's sort handles large tensors efficiently.
    """
    return _pytorch_topk(scores, top_k)


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
