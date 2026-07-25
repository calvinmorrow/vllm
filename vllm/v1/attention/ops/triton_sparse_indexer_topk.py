# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-only interim top-k for the sparse attention indexer."""

import torch


# Kept for callers and focused coverage while this implementation uses sort.
SORT_N_MAX = 8192


def _select_sort_n(n: int) -> int:
    """Return the historical sort bucket for ``n``."""
    for sort_n in (256, 512, 1024, 2048, 4096, SORT_N_MAX):
        if n <= sort_n:
            return sort_n
    return SORT_N_MAX


def _stable_topk(
    scores: torch.Tensor,
    top_k: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select stable descending indices, padding masked scores with ``-1``."""
    num_rows, width = scores.shape
    scores = scores.masked_fill(~torch.isfinite(scores), -torch.inf)
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, -torch.inf)

    sorted_scores, sorted_indices = torch.sort(
        scores, dim=-1, descending=True, stable=True
    )
    k = min(top_k, width)
    result = torch.full(
        (num_rows, top_k), -1, dtype=torch.int32, device=scores.device
    )
    selected_indices = sorted_indices[:, :k].to(torch.int32)
    result[:, :k] = torch.where(
        torch.isfinite(sorted_scores[:, :k]),
        selected_indices,
        torch.full_like(selected_indices, -1),
    )
    return result


def _prefix_mask(
    scores: torch.Tensor, n_valid: torch.Tensor | int
) -> torch.Tensor:
    """Build a GPU mask for each row's valid prefix."""
    num_rows, width = scores.shape
    if isinstance(n_valid, int):
        lengths = torch.full(
            (num_rows,), n_valid, device=scores.device, dtype=torch.long
        )
    else:
        lengths = n_valid.to(device=scores.device, dtype=torch.long)
    lengths = lengths.clamp(0, width)
    columns = torch.arange(width, device=scores.device)
    return columns.unsqueeze(0) < lengths.unsqueeze(1)


def triton_sparse_indexer_topk(
    scores: torch.Tensor,
    top_k: int,
    n_valid: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Return stable score-descending top-k indices for each GPU score row.

    Equal scores retain their original ascending column order. Per-row prefixes
    outside ``n_valid`` are masked before selection and ``-inf`` is ``-1``.
    """
    assert scores.ndim == 2
    assert scores.dtype == torch.float32
    assert scores.is_cuda
    assert top_k > 0
    valid_mask = None if n_valid is None else _prefix_mask(scores, n_valid)
    return _stable_topk(scores, top_k, valid_mask)


def triton_sparse_indexer_topk_prefill(
    scores: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Select each row's ``[start, end)`` span and return local indices."""
    assert scores.ndim == 2
    assert scores.dtype == torch.float32
    assert scores.is_cuda
    assert top_k > 0

    width = scores.shape[1]
    starts = cu_seqlen_ks.to(device=scores.device, dtype=torch.long).clamp(0, width)
    ends = cu_seqlen_ke.to(device=scores.device, dtype=torch.long).clamp(0, width)
    ends = torch.maximum(ends, starts)
    columns = torch.arange(width, device=scores.device)
    valid_mask = (columns.unsqueeze(0) >= starts.unsqueeze(1)) & (
        columns.unsqueeze(0) < ends.unsqueeze(1)
    )
    absolute = _stable_topk(scores, top_k, valid_mask)
    return torch.where(
        absolute >= 0,
        absolute - starts.unsqueeze(1).to(torch.int32),
        absolute,
    )


def triton_sparse_indexer_topk_decode(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Select each row's valid decode prefix before applying top-k."""
    return triton_sparse_indexer_topk(scores, top_k, n_valid=seq_lens)
