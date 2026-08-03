# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark sparse-indexer stable-sort reference and device-only top-k.

Run from the repository root, for example:
    .venv/bin/python benchmarks/kernels/benchmark_sparse_indexer_stable_sort_rocm.py \
        --width 8192 --rows 16 --top-k 512 --mode decode
"""

import argparse
import statistics

import torch

from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
    _stable_topk_reference,
    sparse_indexer_topk_scratch_shape,
    triton_sparse_indexer_topk_decode,
    triton_sparse_indexer_topk_prefill,
)


def _reference(
    scores: torch.Tensor,
    top_k: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    columns = torch.arange(scores.shape[1], device=scores.device)
    starts = starts.clamp(0, scores.shape[1])
    ends = torch.maximum(ends.clamp(0, scores.shape[1]), starts)
    valid = (columns[None, :] >= starts[:, None]) & (
        columns[None, :] < ends[:, None]
    )
    result = _stable_topk_reference(scores, top_k, valid).to(torch.int32)
    if mode == "prefill":
        result = torch.where(result >= 0, result - starts[:, None], result)
    return result


def _latencies(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))
    return elapsed_ms


def _report(name: str, elapsed_ms: list[float]) -> None:
    quantiles = statistics.quantiles(elapsed_ms, n=10, method="inclusive")
    print(
        f"{name:12s} median={statistics.median(elapsed_ms):8.3f} ms "
        f"p10={quantiles[0]:8.3f} ms p90={quantiles[8]:8.3f} ms "
        f"stdev={statistics.pstdev(elapsed_ms):8.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--valid-fraction", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.width <= 0 or args.rows <= 0 or args.top_k <= 0:
        raise ValueError("width, rows, and top-k must be positive")
    if not 0 < args.valid_fraction <= 1:
        raise ValueError("valid-fraction must be in (0, 1]")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/CUDA GPU is required")

    device = torch.device("cuda")
    scores = torch.rand((args.rows, args.width), device=device, dtype=torch.float32)
    span = max(1, int(args.width * args.valid_fraction))
    starts = torch.randint(
        0, args.width - span + 1, (args.rows,), device=device, dtype=torch.int32
    )
    ends = starts + span
    if args.mode == "decode":
        starts.zero_()

    out = torch.empty((args.rows, args.top_k), device=device, dtype=torch.int32)
    scratch = torch.empty(
        sparse_indexer_topk_scratch_shape(args.rows, args.width, args.top_k),
        device=device,
        dtype=torch.uint64,
    )
    expected = _reference(scores, args.top_k, starts, ends, args.mode)
    if args.mode == "decode":
        actual = triton_sparse_indexer_topk_decode(
            scores, ends, args.top_k, out, scratch
        )
    else:
        actual = triton_sparse_indexer_topk_prefill(
            scores, starts, ends, args.top_k, out, scratch
        )
    torch.testing.assert_close(actual, expected)

    reference = lambda: _reference(scores, args.top_k, starts, ends, args.mode)
    if args.mode == "decode":
        replacement = lambda: triton_sparse_indexer_topk_decode(
            scores, ends, args.top_k, out, scratch
        )
    else:
        replacement = lambda: triton_sparse_indexer_topk_prefill(
            scores, starts, ends, args.top_k, out, scratch
        )
    print(
        f"device={torch.cuda.get_device_name(device)} mode={args.mode} "
        f"rows={args.rows} width={args.width} top_k={args.top_k} "
        f"span={span}"
    )
    _report("reference", _latencies(reference, args.warmup, args.repeats))
    _report("replacement", _latencies(replacement, args.warmup, args.repeats))


if __name__ == "__main__":
    main()
