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


@triton.jit
def _bitonic_sort_step(
    vals,
    idxs,
    k: tl.constexpr,
    j: tl.constexpr,
    sort_n: tl.constexpr,
    block_size: tl.constexpr,
):
    """One bitonic merge step with grid-stride loop."""
    tid = tl.program_id(1) if block_size > 1 else 0
    n_threads = tl.num_programs(1) if block_size > 1 else 1

    for i in range(tid, sort_n, n_threads):
        other = i ^ j
        if other > i and other < sort_n:
            av = vals[i]
            bv = vals[other]
            ai = idxs[i]
            bi = idxs[other]

            desc_half = (i & k) == 0
            # better: higher score, or equal score with lower index
            swap_desc = (bv > av) | ((bv == av) & (bi < ai))
            swap_asc = (av > bv) | ((av == bv) & (ai < bi))
            swap = swap_desc if desc_half else swap_asc

            vals[i] = bv if swap else av
            vals[other] = av if swap else bv
            idxs[i] = bi if swap else ai
            idxs[other] = ai if swap else bi

    tl.debug_barrier()


@triton.jit
def _bitonic_sort(
    vals,
    idxs,
    sort_n: tl.constexpr,
    block_size: tl.constexpr,
):
    """Full bitonic sort: descending score, ascending index for ties."""
    # k = 2, 4, 8, ..., sort_n
    for _k in range(sort_n >> 1, 0, -1):
        # j goes from _k down to 1
        for j in range(_k, 0, -1):
            _bitonic_sort_step(vals, idxs, _k * 2, j, sort_n, block_size)


# ---------------------------------------------------------------------------
# Per-SORT_N kernels
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
    BLOCK_SIZE: tl.constexpr,
):
    """Sort-based top-k: one block per row, grid-stride threads."""
    row = tl.program_id(0)

    vals = tl.zeros([SORT_N], dtype=tl.float32)
    idxs = tl.zeros([SORT_N], dtype=tl.int32)

    tid = tl.program_id(1)
    n_threads = tl.num_programs(1)

    # Load scores + indices (grid-stride)
    for i in range(tid, SORT_N, n_threads):
        if i < n_comp:
            vals[i] = tl.load(scores_ptr + row * stride_scores + i)
            idxs[i] = i
        else:
            vals[i] = -float("inf")
            idxs[i] = -1
    tl.debug_barrier()

    # Bitonic sort
    _bitonic_sort(vals, idxs, SORT_N, BLOCK_SIZE)

    # Write top-k (grid-stride)
    for i in range(tid, top_k, n_threads):
        tl.store(selected_ptr + row * stride_selected + i, idxs[i])


@triton.jit
def _topk_chunk_kernel(
    scores_ptr,
    candidates_ptr,
    n_comp,
    top_k,
    candidate_stride,
    stride_scores,
    SORT_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Chunk-local top-k for chunk-and-merge pipeline."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)

    actual_chunk_start = chunk * CHUNK_N
    if actual_chunk_start >= n_comp:
        return

    actual_chunk_n = min(n_comp - actual_chunk_start, CHUNK_N)

    vals = tl.zeros([SORT_N], dtype=tl.float32)
    idxs = tl.zeros([SORT_N], dtype=tl.int32)

    tid = tl.program_id(2)
    n_threads = tl.num_programs(2)

    for i in range(tid, SORT_N, n_threads):
        abs_idx = actual_chunk_start + i
        if i < actual_chunk_n:
            vals[i] = tl.load(scores_ptr + row * stride_scores + abs_idx)
            idxs[i] = abs_idx
        else:
            vals[i] = -float("inf")
            idxs[i] = -1
    tl.debug_barrier()

    _bitonic_sort(vals, idxs, SORT_N, BLOCK_SIZE)

    out_ptr = candidates_ptr + row * candidate_stride + chunk * top_k
    for i in range(tid, top_k, n_threads):
        tl.store(out_ptr + i, idxs[i])


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
    BLOCK_SIZE: tl.constexpr,
):
    """Merge sorted candidate sets into top-k."""
    row = tl.program_id(0)

    vals = tl.zeros([SORT_N], dtype=tl.float32)
    idxs = tl.zeros([SORT_N], dtype=tl.int32)

    tid = tl.program_id(1)
    n_threads = tl.num_programs(1)

    for i in range(tid, SORT_N, n_threads):
        if i < candidate_count:
            c_idx = tl.load(candidates_ptr + i)
            if c_idx >= 0 and c_idx < n_comp:
                vals[i] = tl.load(scores_ptr + row * stride_scores + c_idx)
                idxs[i] = c_idx
            else:
                vals[i] = -float("inf")
                idxs[i] = -1
        else:
            vals[i] = -float("inf")
            idxs[i] = -1
    tl.debug_barrier()

    _bitonic_sort(vals, idxs, SORT_N, BLOCK_SIZE)

    for i in range(tid, top_k, n_threads):
        tl.store(out_ptr + row * top_k + i, idxs[i])


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
    BLOCK_SIZE: tl.constexpr,
):
    """Tree merge: merge one group of candidate sets."""
    row = tl.program_id(0)
    group = tl.program_id(1)

    set0 = group * merge_group
    if set0 >= n_sets:
        return

    actual_set_count = min(merge_group, n_sets - set0)
    actual_candidate_count = actual_set_count * top_k

    vals = tl.zeros([SORT_N], dtype=tl.float32)
    idxs = tl.zeros([SORT_N], dtype=tl.int32)

    tid = tl.program_id(2)
    n_threads = tl.num_programs(2)

    base = set0 * top_k
    for i in range(tid, SORT_N, n_threads):
        if i < actual_candidate_count:
            c_idx = tl.load(
                candidates_ptr + row * cur_stride + base + i
            )
            if c_idx >= 0 and c_idx < n_comp:
                vals[i] = tl.load(scores_ptr + row * stride_scores + c_idx)
                idxs[i] = c_idx
            else:
                vals[i] = -float("inf")
                idxs[i] = -1
        else:
            vals[i] = -float("inf")
            idxs[i] = -1
    tl.debug_barrier()

    _bitonic_sort(vals, idxs, SORT_N, BLOCK_SIZE)

    out_base = row * next_stride + group * top_k
    for i in range(tid, top_k, n_threads):
        tl.store(out_ptr + out_base + i, idxs[i])


# ---------------------------------------------------------------------------
# Python launch wrappers
# ---------------------------------------------------------------------------


def _topk_single_phase(
    scores: torch.Tensor,
    top_k: int,
    sort_n: int,
) -> torch.Tensor:
    """Single-phase bitonic sort top-k.

    Args:
        scores: [num_tokens, n_comp] fp32 scores.
        top_k: number of top indices to select.
        sort_n: power-of-2 shared memory size >= n_comp.

    Returns:
        [num_tokens, top_k] int32 indices (descending score, -1 padding).
    """
    num_tokens, n_comp = scores.shape
    assert sort_n >= n_comp, f"sort_n={sort_n} < n_comp={n_comp}"

    # Choose block size (threads per block)
    block_size = min(1024, sort_n)
    while sort_n % block_size != 0 and block_size > 64:
        block_size //= 2
    while block_size & (block_size - 1) != 0:
        block_size -= 1
    block_size = max(64, block_size)

    selected = torch.full(
        (num_tokens, top_k), -1, dtype=torch.int32, device=scores.device
    )

    grid = (num_tokens, block_size)
    _topk_sort_kernel[grid](
        scores,
        selected,
        n_comp,
        top_k,
        scores.stride(0),
        selected.stride(0),
        SORT_N=sort_n,
        BLOCK_SIZE=block_size,
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
    block_size = 256

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
    grid_chunks = (num_tokens, n_chunks, block_size)
    _topk_chunk_kernel[grid_chunks](
        scores,
        scratch,
        n_comp,
        top_k,
        cur_stride,
        scores.stride(0),
        SORT_N=sort_n,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    # Phase 2: tree merge
    n_sets = n_chunks
    while n_sets > MERGE_GROUP:
        next_sets = math.ceil(n_sets / MERGE_GROUP)
        next_stride = next_sets * top_k
        next_offset = cur_offset + cur_stride

        grid_merge = (num_tokens, next_sets, block_size)
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
            BLOCK_SIZE=block_size,
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

    grid_final = (num_tokens, block_size)
    _topk_merge_kernel[grid_final](
        final_cur_ptr,
        scores,
        selected,
        n_comp,
        top_k,
        final_candidate_count,
        scores.stride(0),
        SORT_N=sort_n,
        BLOCK_SIZE=block_size,
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
