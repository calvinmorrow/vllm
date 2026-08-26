# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure helpers for W8A8 Block-FP8 benchmark evidence collection."""

from collections.abc import Sequence
from typing import TypeVar

import torch

T = TypeVar("T")


def rotate_timing_order(items: Sequence[T], pass_index: int) -> list[T]:
    """Return a cyclic timing order for one confirmation pass.

    Args:
        items: Canonical candidate order.
        pass_index: Zero-based confirmation pass index.

    Returns:
        The candidate order rotated so every candidate occupies every ordinal
        position over a complete cycle.
    """
    if not items:
        return []
    offset = pass_index % len(items)
    return list(items[offset:]) + list(items[:offset])


def tensors_within_tolerance(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    """Return whether finite tensors satisfy PyTorch's declared tolerance."""
    return bool(
        torch.isfinite(actual).all()
        and torch.isfinite(expected).all()
        and torch.allclose(actual, expected, rtol=rel_tol, atol=abs_tol)
    )
