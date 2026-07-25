# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark stable GPU sort for sparse-indexer top-k target shapes.

Run from the repository root:
    .venv/bin/python benchmarks/kernels/benchmark_sparse_indexer_stable_sort_rocm.py
"""

import argparse
import statistics

import torch

WIDTHS = (4096, 8192, 16384)
TOP_K = 512


def stable_topk(scores: torch.Tensor) -> torch.Tensor:
    """Return stable, descending top-k column indices for every score row."""
    return torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :TOP_K]


def verify_stable_ties(device: torch.device) -> None:
    """Check that equal scores retain ascending input-column order."""
    scores = torch.tensor(
        [[4.0, 9.0, 9.0, 9.0, 1.0], [7.0, 7.0, 6.0, 7.0, 6.0]],
        device=device,
    )
    actual = stable_topk(scores)[:, :5].cpu()
    expected = torch.tensor([[1, 2, 3, 0, 4], [0, 1, 3, 2, 4]])
    if not torch.equal(actual, expected):
        raise RuntimeError(f"stable tie verification failed: {actual.tolist()}")


def benchmark(scores: torch.Tensor, warmup: int, repeats: int) -> float:
    """Return median GPU-event latency in milliseconds."""
    for _ in range(warmup):
        stable_topk(scores)

    elapsed_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        stable_topk(scores)
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))
    return statistics.median(elapsed_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.rows <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("rows and repeats must be positive; warmup must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/CUDA GPU is required")

    device = torch.device("cuda")
    verify_stable_ties(device)
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"rows={args.rows}, top_k={TOP_K}, warmup={args.warmup}, repeats={args.repeats}")
    print("width  median_ms  rows/s        candidates/s")
    for width in WIDTHS:
        scores = torch.rand((args.rows, width), device=device, dtype=torch.float32)
        median_ms = benchmark(scores, args.warmup, args.repeats)
        rows_per_second = args.rows * 1_000 / median_ms
        candidates_per_second = rows_per_second * width
        print(
            f"{width:5d}  {median_ms:9.3f}  {rows_per_second:12.0f}  "
            f"{candidates_per_second:14.0f}"
        )


if __name__ == "__main__":
    main()
