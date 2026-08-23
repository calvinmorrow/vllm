# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-only exact top-k selection for the sparse attention indexer.

Scores are selected in 1024-column chunks, then reduced by pairwise merges.
The caller provides two uint64 scratch planes; each stores packed sortable keys
with shape ``[rows, chunks, candidate_k]``. Thus scratch is bounded by
``2 * rows * ceil(width / 1024) * next_power_of_two(top_k)`` uint64 elements.
The implementation supports any positive score width whose caller-provided
scratch has sufficient capacity, and top-k values through ``MAX_TOP_K``. It
never allocates or aliases scores on the selected path.
"""

import torch

from vllm.triton_utils import tl, triton

CHUNK_WIDTH = 1024
MAX_TOP_K = 2048
_SENTINEL = -9223372036854775808


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _tensors_overlap(first: torch.Tensor, second: torch.Tensor) -> bool:
    first_start = first.data_ptr()
    first_end = first_start + first.numel() * first.element_size()
    second_start = second.data_ptr()
    second_end = second_start + second.numel() * second.element_size()
    return first_start < second_end and second_start < first_end


def sparse_indexer_topk_scratch_shape(
    rows: int, width: int, top_k: int
) -> tuple[int, int, int, int]:
    """Return the caller-owned uint64 ping/pong scratch shape.

    The shape is ``[2, rows, ceil(width / 1024), next_power_of_two(top_k)]``.
    It is a pure-Python capacity helper: callers must reserve it before graph
    capture, and insufficient scratch is rejected before any kernel launch.
    """
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise RuntimeError("rows must be a non-negative integer")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise RuntimeError("score width must be a positive integer")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 0 < top_k <= MAX_TOP_K
    ):
        raise RuntimeError(f"top_k must be in [1, {MAX_TOP_K}]")
    return 2, rows, (width + CHUNK_WIDTH - 1) // CHUNK_WIDTH, _next_power_of_two(
        top_k
    )


def sparse_indexer_topk_scratch_numel(rows: int, width: int, top_k: int) -> int:
    """Return the exact uint64 element capacity required by the selector."""
    planes, rows, chunks, candidate_k = sparse_indexer_topk_scratch_shape(
        rows, width, top_k
    )
    return planes * rows * chunks * candidate_k


def _validate_inputs(
    scores: torch.Tensor,
    top_k: int,
    out: torch.Tensor,
    scratch: torch.Tensor,
) -> tuple[int, int]:
    if (
        scores.ndim != 2
        or scores.dtype != torch.float32
        or not scores.is_cuda
        or scores.stride(-1) != 1
    ):
        raise RuntimeError(
            "scores must be a CUDA/ROCm [rows, width] FP32 tensor with a "
            "contiguous last dimension"
        )
    rows, width = scores.shape
    _, _, chunks, candidate_k = sparse_indexer_topk_scratch_shape(rows, width, top_k)
    if (
        out.dtype != torch.int32
        or not out.is_cuda
        or out.device != scores.device
        or out.ndim != 2
        or out.shape[0] != rows
        or out.shape[1] < top_k
        or out.stride(-1) != 1
    ):
        raise RuntimeError(
            "out must be a CUDA/ROCm contiguous-last-dimension int32 tensor "
            "with shape [rows, at least top_k]"
        )
    if (
        scratch.dtype != torch.uint64
        or not scratch.is_cuda
        or scratch.device != scores.device
        or not scratch.is_contiguous()
        or scratch.numel() < sparse_indexer_topk_scratch_numel(rows, width, top_k)
    ):
        raise RuntimeError(
            "scratch must be a contiguous CUDA/ROCm uint64 tensor with capacity "
            "for two selector scratch planes"
        )
    if (
        _tensors_overlap(scores, out)
        or _tensors_overlap(scores, scratch)
        or _tensors_overlap(out, scratch)
    ):
        raise RuntimeError("scores, out, and scratch must not overlap")
    return chunks, candidate_k


@triton.jit
def _sparse_indexer_chunk_select_kernel(
    scores_ptr,
    starts_ptr,
    ends_ptr,
    scratch_ptr,
    width,
    score_stride,
    scratch_row_stride,
    scratch_chunk_stride,
    chunk_width: tl.constexpr,
    sentinel: tl.constexpr,
    repeats: tl.constexpr,
    candidate_k: tl.constexpr,
    local_k: tl.constexpr,
    validity_mode: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * chunk_width + tl.arange(0, chunk_width)
    scores = tl.load(
        scores_ptr + row * score_stride + cols,
        mask=cols < width,
        other=float("-inf"),
    )
    source_row = row // repeats
    start = 0
    end = width
    if validity_mode == 1:
        end = tl.minimum(tl.maximum(tl.load(ends_ptr + source_row), 0), width)
    elif validity_mode == 2:
        start = tl.minimum(tl.maximum(tl.load(starts_ptr + source_row), 0), width)
        end = tl.minimum(tl.maximum(tl.load(ends_ptr + source_row), start), width)
    finite = (scores == scores) & (scores != float("inf")) & (scores != -float("inf"))
    valid = (cols < width) & (cols >= start) & (cols < end) & finite

    bits = scores.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    bits = tl.where(scores == 0.0, 0, bits)
    ordered = tl.where(
        (bits & 0x80000000) != 0,
        (~bits) & 0xFFFFFFFF,
        bits ^ 0x80000000,
    )
    key = ((ordered << 32) | (0xFFFFFFFF - cols.to(tl.int64))) ^ sentinel
    selected = tl.topk(tl.where(valid, key, sentinel), local_k)
    selected = tl.bitonic_merge(selected, descending=True)
    if candidate_k > local_k:
        padding = tl.full((candidate_k - local_k,), sentinel, tl.int64)
        selected = tl.cat(selected, padding, can_reorder=True)

    ranks = tl.arange(0, candidate_k)
    offsets = (
        row * scratch_row_stride + chunk * scratch_chunk_stride + ranks
    )
    tl.store(scratch_ptr + offsets, selected.to(tl.uint64, bitcast=True))


@triton.jit
def _sparse_indexer_merge_kernel(
    scratch_ptr,
    source_base,
    destination_base,
    source_row_stride,
    source_chunk_stride,
    destination_row_stride,
    destination_chunk_stride,
    source_chunks,
    sentinel: tl.constexpr,
    candidate_k: tl.constexpr,
):
    row = tl.program_id(0)
    destination_chunk = tl.program_id(1)
    source_chunk = destination_chunk * 2
    ranks = tl.arange(0, candidate_k)
    left_offsets = (
        source_base
        + row * source_row_stride
        + source_chunk * source_chunk_stride
        + ranks
    )
    right_offsets = left_offsets + source_chunk_stride
    has_left = source_chunk < source_chunks
    has_right = source_chunk + 1 < source_chunks
    left = tl.load(
        scratch_ptr + left_offsets,
        mask=has_left,
        other=0,
    ).to(tl.int64, bitcast=True)
    right = tl.load(
        scratch_ptr + right_offsets,
        mask=has_right,
        other=0,
    ).to(tl.int64, bitcast=True)
    left = tl.where(has_left, left, sentinel)
    right = tl.where(has_right, right, sentinel)
    selected = tl.topk(
        tl.cat(left, right, can_reorder=True), candidate_k
    )
    selected = tl.bitonic_merge(selected, descending=True)
    offsets = (
        row * destination_row_stride
        + destination_chunk * destination_chunk_stride
        + ranks
    )
    tl.store(
        scratch_ptr + destination_base + offsets,
        selected.to(tl.uint64, bitcast=True),
    )


@triton.jit
def _sparse_indexer_write_kernel(
    scratch_ptr,
    scratch_base,
    out_ptr,
    out_stride,
    scratch_row_stride,
    width,
    top_k: tl.constexpr,
    candidate_k: tl.constexpr,
    sentinel: tl.constexpr,
    validity_mode: tl.constexpr,
    starts_ptr,
):
    row = tl.program_id(0)
    ranks = tl.arange(0, candidate_k)
    keys = tl.load(
        scratch_ptr + scratch_base + row * scratch_row_stride + ranks
    ).to(
        tl.int64, bitcast=True
    )
    columns = (0xFFFFFFFF - (keys & 0xFFFFFFFF)).to(tl.int32)
    values = tl.where(keys != sentinel, columns, -1)
    if validity_mode == 2:
        start = tl.minimum(tl.maximum(tl.load(starts_ptr + row), 0), width)
        values = tl.where(values >= 0, values - start.to(tl.int32), values)
    tl.store(out_ptr + row * out_stride + ranks, values, mask=ranks < top_k)


def _device_topk(
    scores: torch.Tensor,
    starts: torch.Tensor | None,
    ends: torch.Tensor | None,
    top_k: int,
    out: torch.Tensor,
    scratch: torch.Tensor,
    repeats: int,
    validity_mode: int,
) -> torch.Tensor:
    chunks, candidate_k = _validate_inputs(scores, top_k, out, scratch)
    rows, width = scores.shape
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise RuntimeError("repeats must be a positive integer")
    if rows % repeats:
        raise RuntimeError("repeats must divide the number of score rows")
    if validity_mode == 2 and starts is None:
        raise RuntimeError("prefill validity starts must be provided")
    if validity_mode:
        if starts is not None and (
            starts.ndim != 1
            or not starts.is_cuda
            or starts.device != scores.device
            or starts.dtype not in (torch.int32, torch.int64)
            or starts.stride(0) != 1
        ):
            raise RuntimeError(
                "validity starts must be a same-device contiguous rank-1 int32 "
                "or int64 CUDA/ROCm tensor"
            )
        if (
            ends is None
            or ends.ndim != 1
            or not ends.is_cuda
            or ends.device != scores.device
            or ends.dtype not in (torch.int32, torch.int64)
            or ends.stride(0) != 1
        ):
            raise RuntimeError(
                "validity ends must be a same-device contiguous rank-1 int32 "
                "or int64 CUDA/ROCm tensor"
            )
        if ends.numel() * repeats != rows:
            raise RuntimeError("validity bounds must have one element per source row")
        if validity_mode == 2 and starts is not None and starts.shape != ends.shape:
            raise RuntimeError("prefill validity bounds must have matching shapes")

    plane_stride = rows * chunks * candidate_k
    row_stride = chunks * candidate_k
    local_k = min(candidate_k, CHUNK_WIDTH)
    _sparse_indexer_chunk_select_kernel[(rows, chunks)](
        scores,
        starts if starts is not None else scratch,
        ends if ends is not None else scratch,
        scratch,
        width,
        scores.stride(0),
        row_stride,
        candidate_k,
        chunk_width=CHUNK_WIDTH,
        sentinel=_SENTINEL,
        repeats=repeats,
        candidate_k=candidate_k,
        local_k=local_k,
        validity_mode=validity_mode,
        num_warps=4,
    )

    source_plane = 0
    source_chunks = chunks
    while source_chunks > 1:
        destination_plane = 1 - source_plane
        destination_chunks = (source_chunks + 1) // 2
        _sparse_indexer_merge_kernel[(rows, destination_chunks)](
            scratch,
            source_plane * plane_stride,
            destination_plane * plane_stride,
            row_stride,
            candidate_k,
            row_stride,
            candidate_k,
            source_chunks,
            sentinel=_SENTINEL,
            candidate_k=candidate_k,
            num_warps=4,
        )
        source_plane = destination_plane
        source_chunks = destination_chunks

    _sparse_indexer_write_kernel[(rows,)](
        scratch,
        source_plane * plane_stride,
        out,
        out.stride(0),
        row_stride,
        width,
        top_k=top_k,
        candidate_k=candidate_k,
        sentinel=_SENTINEL,
        validity_mode=validity_mode,
        starts_ptr=starts if starts is not None else scratch,
        num_warps=4,
    )
    return out


def _stable_topk_reference(
    scores: torch.Tensor,
    top_k: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the stable-sort reference used exclusively by focused tests."""
    scores = scores.masked_fill(~torch.isfinite(scores), -torch.inf)
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, -torch.inf)
    sorted_scores, sorted_indices = torch.sort(
        scores, dim=-1, descending=True, stable=True
    )
    result = torch.full(
        (scores.shape[0], top_k), -1, dtype=torch.int32, device=scores.device
    )
    count = min(top_k, scores.shape[1])
    result[:, :count] = torch.where(
        torch.isfinite(sorted_scores[:, :count]),
        sorted_indices[:, :count].to(torch.int32),
        -1,
    )
    return result


def triton_sparse_indexer_topk(
    scores: torch.Tensor,
    top_k: int,
    out: torch.Tensor,
    scratch: torch.Tensor,
) -> torch.Tensor:
    """Write exact score-descending, index-ascending selections into ``out``."""
    return _device_topk(scores, None, None, top_k, out, scratch, 1, 0)


def triton_sparse_indexer_topk_prefill(
    scores: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    top_k: int,
    out: torch.Tensor,
    scratch: torch.Tensor,
) -> torch.Tensor:
    """Write top-k row-local indices for each clamped ``[start, end)`` span."""
    return _device_topk(
        scores, cu_seqlen_ks, cu_seqlen_ke, top_k, out, scratch, 1, 2
    )


def triton_sparse_indexer_topk_decode(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    out: torch.Tensor,
    scratch: torch.Tensor,
    repeats: int = 1,
) -> torch.Tensor:
    """Write top-k indices for each clamped decode prefix into ``out``."""
    return _device_topk(scores, None, seq_lens, top_k, out, scratch, repeats, 1)
