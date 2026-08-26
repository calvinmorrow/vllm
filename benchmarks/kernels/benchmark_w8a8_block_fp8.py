# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from sglang quantization/tuning_block_wise_kernel.py

"""Benchmark the existing W8A8 block-FP8 Triton kernel.

Two modes:

1. Legacy tuning mode (no ``--targets`` / ``--manifest``): sweep the fixed
   DeepSeek-V3 weight-shape list and write the best config per (N, K), exactly
   as before.
2. Evidence mode (``--targets`` and/or ``--manifest``): benchmark an explicit
   (M, N, K) target matrix with a production-equivalent layout (padded B row
   pitch, E8M0->fp32 scale route, BF16 output), collect raw timing samples,
   full statistics, and a numerical correctness check against
   ``native_w8a8_block_matmul``, and emit a machine-readable JSON record.

Evidence mode is the path used to produce static configuration evidence for a
specific target device (gfx1151 / AMD Radeon 8060S).
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import time
from datetime import datetime
from typing import Any

import torch
from tqdm import tqdm

from benchmarks.kernels._block_fp8_benchmark_utils import (
    rotate_timing_order,
    tensors_within_tolerance,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
    _w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.platform_utils import get_device_name_as_file_name

mp.set_start_method("spawn", force=True)

assert current_platform.is_cuda() or current_platform.is_rocm(), (
    "Only support tune w8a8 block fp8 kernel on CUDA/ROCm device."
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
}

# Generic fallback config selected by the serving route when no static JSON
# file exists. It is the default baseline for evidence mode.
GENERIC_FALLBACK: dict[str, int] = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}

# Production FP8 row padding applied by ``_maybe_pad_fp8_weight`` on ROCm when
# ``VLLM_ROCM_FP8_PADDING`` is set (default True): each one-byte FP8 weight row
# is padded by 256 bytes when the current row pitch is already 512-byte
# aligned. All six confirmed target K values satisfy that, so their
# production B row pitch is K + 256.
_FP8_PAD_BYTES = 256


def _b_row_pitch(K: int, pad_mode: str) -> int:
    """Return the logical B row pitch (stride along N) for a given K.

    Args:
        K: Reduction dimension of the logical B tensor [N, K].
        pad_mode: ``"off"`` (contiguous, pitch K), ``"prod"`` (reproduce
            ``_maybe_pad_fp8_weight``: pad to 512-byte rows when already
            512-byte aligned), or an integer number of extra one-byte padding
            elements.

    Returns:
        int: Row pitch in elements. Equals K for the contiguous layout.
    """
    if pad_mode == "off":
        return K
    if pad_mode == "prod":
        if K % 512 == 0:
            return K + _FP8_PAD_BYTES
        return K
    return K + int(pad_mode)


def make_operands(
    M: int,
    N: int,
    K: int,
    block_n: int,
    block_k: int,
    scale_dtype: str,
    b_row_pitch: int,
    seed: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a production-shaped operand set for the block-FP8 kernel.

    B is backed by a buffer of row pitch ``b_row_pitch`` (>= K) so the logical
    [N, K] view may be non-contiguous, exactly as a padded production weight.
    The logical [N, K] values are identical to a contiguous [N, K] allocation
    of the same seed; ``B.contiguous()`` is the value-identical reference.

    Args:
        M: Activation rows (flattened token count).
        N: Output features (weight rows).
        K: Reduction dimension (weight columns).
        block_n: Weight block N size (128).
        block_k: Weight block K size (128).
        scale_dtype: ``"fp32"`` or ``"e8m0"``. e8m0 generates exponent-only
            scales then upcasts them with the wrapper's exact function.
        b_row_pitch: B row pitch in elements (K contiguous, K+256 padded).
        seed: Deterministic RNG seed.
        device: Torch device string.

    Returns:
        (A, B, As, Bs). A is contiguous [M, K] fp8. B is a [N, K] view with the
        given row pitch. As is [M, ceil(K/block_k)]. Bs is
        [ceil(N/block_n), ceil(K/block_k)].
    """
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    torch.manual_seed(seed)
    A = (
        (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5)
        * 2
        * fp8_max
    ).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    pad = b_row_pitch - K
    if pad == 0:
        B = (
            (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5)
            * 2
            * fp8_max
        ).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    else:
        B_full = (
            (torch.rand(N, b_row_pitch, dtype=torch.float32, device=device)
             - 0.5)
            * 2
            * fp8_max
        ).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        B_full[:, K:] = 0
        B = B_full[:, :K]

    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    if scale_dtype == "e8m0":
        As = torch.randint(
            120, 130, (M, k_tiles), device=device, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        As = _upcast_e8m0_to_fp32(As).contiguous()
        Bs = torch.randint(
            120, 130, (n_tiles, k_tiles), device=device, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        Bs = _upcast_e8m0_to_fp32(Bs).contiguous()
    else:
        As = (
            torch.rand(M, k_tiles, dtype=torch.float32, device=device)
            * factor_for_scale
        )
        Bs = (
            torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=device)
            * factor_for_scale
        )

    return A, B, As, Bs


def launch_candidate(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    config: dict[str, Any],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Launch the existing Triton kernel once with an explicit config.

    Mirrors the serving wrapper's launch exactly (single-axis grid, strides,
    meta-params). The E8M0->fp32 upcast and config lookup the wrapper performs
    are config-independent pre-launch steps and are excluded from the timed
    region; the caller supplies already-upcast fp32 scales.

    Args:
        A: Contiguous [M, K] fp8 activation.
        B: [N, K] weight view (strides honored).
        As: [M, ceil(K/block_k)] fp32 activation scales.
        Bs: [ceil(N/block_n), ceil(K/block_k)] fp32 weight scales.
        block_size: [block_n, block_k], e.g. [128, 128].
        config: Meta-parameter dict for the kernel.
        output_dtype: Output tensor dtype.

    Returns:
        C: [M, N] tensor in output_dtype.
    """
    block_n, block_k = block_size[0], block_size[1]
    M, K = A.shape[-2], A.shape[-1]
    N = B.shape[0]
    C = A.new_empty((M, N), dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )
    return C


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linearly-interpolated percentile of an ascending-sorted sample list."""
    if not sorted_vals:
        raise ValueError("empty sample list")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _one_sample(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    config: dict[str, Any],
    output_dtype: torch.dtype,
) -> float:
    """Time a single launch, returning microseconds (one sync to read events)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    launch_candidate(A, B, As, Bs, block_size, config, output_dtype)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def summarize(samples: list[float]) -> dict[str, float]:
    """Compute the required descriptive statistics for a sample list."""
    s = sorted(samples)
    mean = statistics.fmean(s)
    stdev = statistics.stdev(s) if len(s) > 1 else 0.0
    return {
        "n": len(s),
        "mean_us": mean,
        "median_us": _percentile(s, 50),
        "p10_us": _percentile(s, 10),
        "p90_us": _percentile(s, 90),
        "stdev_us": stdev,
        "cv": (stdev / mean) if mean else 0.0,
    }


def correctness_check(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    config: dict[str, Any],
    output_dtype: torch.dtype,
    ref: torch.Tensor,
    rel_tol: float,
    abs_tol: float,
) -> dict[str, Any]:
    """Compare a candidate launch against a reference output.

    Args:
        A, B, As, Bs: Operands (B may be non-contiguous).
        block_size: [block_n, block_k].
        config: Candidate meta-params.
        output_dtype: Output dtype.
        ref: Reference [M, N] tensor (native block matmul on value-identical
            contiguous operands).
        rel_tol: Relative-error pass threshold (fraction of ref magnitude).
        abs_tol: Absolute-error pass threshold.

    Returns:
        Dict with absolute/relative error summaries, declared tolerances, pass
        status, and output dtype.
    """
    out = launch_candidate(A, B, As, Bs, block_size, config, output_dtype)
    out_f = out.to(torch.float32)
    ref_f = ref.to(torch.float32)
    diff = (out_f - ref_f).abs()
    ref_abs = ref_f.abs()
    max_abs = float(diff.max().item())
    denom = float(ref_abs.mean().item())
    mean_rel = (float(diff.mean().item()) / denom) if denom else 0.0
    relative_denom = ref_abs.clamp_min(abs_tol)
    max_rel = float((diff / relative_denom).max().item())
    finite = bool(torch.isfinite(out_f).all() and torch.isfinite(ref_f).all())
    passed = tensors_within_tolerance(out_f, ref_f, rel_tol, abs_tol)
    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "mean_rel_err": mean_rel,
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
        "finite": finite,
        "zero_reference_count": int((ref_abs == 0).sum().item()),
        "passed": passed,
        "out_dtype": str(out.dtype),
    }


def _cfg_key(c: dict[str, int]) -> tuple:
    return tuple(sorted(c.items()))


def _load_candidate_file(path: str) -> list[dict[str, int]]:
    """Load a JSON list of candidate config dicts (the recorded procedure)."""
    with open(path) as f:
        fam = json.load(f)
    dedup: list[dict[str, int]] = []
    keyset = set()
    for c in fam:
        k = _cfg_key(c)
        if k not in keyset:
            keyset.add(k)
            dedup.append(dict(c))
    return dedup


def _resolve_targets(args) -> list[tuple[int, int, int]]:
    """Resolve the (M, N, K) target list from flags and/or manifest."""
    targets: list[tuple[int, int, int]] = []
    for s in args.targets or []:
        parts = [p for p in s.replace("(", "").replace(")", "").split(",")
                 if p.strip()]
        if len(parts) == 3:
            targets.append(tuple(int(p) for p in parts))
        elif len(parts) == 2:
            n, k = (int(p) for p in parts)
            for mm in args.m_values:
                targets.append((mm, n, k))
        else:
            raise ValueError(f"bad --targets entry {s!r}; use M,N,K")
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        for row in manifest["targets"]:
            targets.append(
                (int(row["M"]), int(row["N"]), int(row["K"]))
            )
    if not targets:
        raise ValueError("no targets given; use --targets or --manifest")
    seen = set()
    uniq = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _software_identity() -> dict[str, Any]:
    """Record the software/hardware identity for the evidence record."""
    try:
        import vllm

        vllm_version = getattr(vllm, "__version__", "unknown")
        vllm_file = getattr(vllm, "__file__", "unknown")
    except Exception:
        vllm_version, vllm_file = "unknown", "unknown"
    props = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "vllm_version": vllm_version,
        "vllm_file": vllm_file,
        "triton": getattr(__import__("triton"), "__version__", "unknown"),
        "gcn_arch": getattr(props, "gcnArchName", None),
        "num_cu": props.multi_processor_count,
    }


def run_evidence(args) -> int:
    """Run the explicit-target evidence-collection path.

    For each target (M, N, K): build production-layout operands, gate every
    candidate against the native block-matmul reference, then measure the
    surviving candidates with cyclically rotated single-sample passes for
    anti-drift, and emit full statistics plus raw samples. A winner
    is a correct candidate whose median beats the baseline by at least
    ``accept_pct`` and by more than the combined observed variability, with a
    stable CV. The record is written incrementally after every target.

    Args:
        args: Parsed CLI arguments with targets/manifest and timing options.

    Returns:
        0 on success, non-zero on a fatal (non-candidate) error.
    """
    block_n, block_k = args.block_n, args.block_k
    out_dtype = DTYPE_MAP[args.out_dtype]
    device = f"cuda:{args.device}"
    torch.cuda.set_device(device)
    device_name = get_device_name_as_file_name()
    targets = _resolve_targets(args)
    family = (
        _load_candidate_file(args.candidate_file)
        if args.candidate_file
        else []
    )

    from tests.kernels.quant_utils import native_w8a8_block_matmul

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_path = os.path.join(args.out_dir, f"block_fp8_evidence_{stamp}.json")

    scale_note = (
        "e8m0 scales generated then upcast with the wrapper's exact function "
        "_upcast_e8m0_to_fp32 before launch; the upcast is a config-"
        "independent pre-launch step excluded from the timed region, so the "
        "kernel is timed kernel-only and sees fp32 scales."
        if args.scale_dtype == "e8m0" else
        "fp32 scales generated directly; kernel timed as kernel-only; "
        "the production E8M0 upcast is a config-independent pre-launch step."
    )
    record: dict[str, Any] = {
        "meta": {
            "generated": datetime.now().isoformat(),
            "device_name": torch.cuda.get_device_name(device),
            "device_token": device_name,
            "software": _software_identity(),
            "block_size": [block_n, block_k],
            "scale_dtype": args.scale_dtype,
            "scale_note": scale_note,
            "output_dtype": args.out_dtype,
            "b_row_pitch_mode": args.b_pad,
            "seed": args.seed,
            "warmup": args.warmup,
            "confirm_iters": args.confirm_iters,
            "accept_pct": args.accept_pct,
            "max_cv": args.max_cv,
            "correctness": {"rel_tol": args.rel_tol, "abs_tol": args.abs_tol},
            "baseline": GENERIC_FALLBACK,
            "n_candidates": len(family) + 1,
            "candidate_source": (args.candidate_file or "none"),
            "timing_order": {
                "policy": "cyclic_rotation",
                "description": (
                    "The canonical baseline-first order is rotated once per "
                    "pass so every candidate occupies every ordinal position "
                    "over a complete cycle."
                ),
            },
        },
        "results": [],
    }

    for (M, N, K) in targets:
        pitch = _b_row_pitch(K, args.b_pad)
        A, B, As, Bs = make_operands(
            M, N, K, block_n, block_k, args.scale_dtype, pitch, args.seed
        )
        # Value-identical contiguous reference (native matmul requires B
        # contiguous); this copy is NOT timed.
        ref_out = native_w8a8_block_matmul(
            A, B.contiguous(), As, Bs, [block_n, block_k], out_dtype
        )

        target_rec: dict[str, Any] = {
            "M": M,
            "N": N,
            "K": K,
            "b_stride": list(B.stride()),
            "b_logical_shape": list(B.shape),
            "as_stride": list(As.stride()),
            "bs_stride": list(Bs.stride()),
            "padded": pitch != K,
            "row_pitch": pitch,
            "candidates": [],
            "rejected": [],
            "winner": None,
        }

        # Baseline first, then the family (de-duplicated against baseline).
        all_cfgs = [dict(GENERIC_FALLBACK)] + [
            c for c in family if _cfg_key(c) != _cfg_key(GENERIC_FALLBACK)
        ]

        # Correctness gate: every candidate must match the native reference.
        survivors: list[dict[str, int]] = []
        cc_by_key: dict[tuple, dict[str, Any]] = {}
        for c in all_cfgs:
            try:
                cc = correctness_check(
                    A, B, As, Bs, [block_n, block_k], c, out_dtype, ref_out,
                    args.rel_tol, args.abs_tol,
                )
            except Exception as e:
                target_rec["rejected"].append(
                    {"config": c, "reason": "launch_error", "detail": repr(e)}
                )
                continue
            cc_by_key[_cfg_key(c)] = cc
            if not cc["passed"]:
                target_rec["rejected"].append(
                    {"config": c, "reason": "correctness", "detail": cc}
                )
                continue
            survivors.append(c)

        # Warm/compile every survivor once (untimed) so JIT is excluded.
        for c in survivors:
            for _ in range(args.warmup):
                launch_candidate(A, B, As, Bs, [block_n, block_k], c,
                                 out_dtype)
        torch.cuda.synchronize()

        # One sample per config per pass, with a cyclic order to avoid a fixed
        # baseline-first or candidate-position bias.
        order = list(survivors)
        base_key = _cfg_key(GENERIC_FALLBACK)
        if base_key in {_cfg_key(c) for c in survivors}:
            order = [GENERIC_FALLBACK] + [
                c for c in survivors if _cfg_key(c) != base_key
            ]
        samples: dict[tuple, list[float]] = {
            _cfg_key(c): [] for c in order
        }
        target_rec["timing_order"] = {
            "policy": "cyclic_rotation",
            "canonical_order": order,
            "rotation_offsets": (
                [
                    pass_index % len(order)
                    for pass_index in range(args.confirm_iters)
                ]
                if order
                else []
            ),
        }
        for pass_index in range(args.confirm_iters):
            for c in rotate_timing_order(order, pass_index):
                key = _cfg_key(c)
                samples[key].append(
                    _one_sample(A, B, As, Bs, [block_n, block_k], c, out_dtype)
                )

        base_med = None
        for c in order:
            key = _cfg_key(c)
            stats = summarize(samples[key])
            rec: dict[str, Any] = {
                "config": c,
                "stats": stats,
                "raw_samples_us": samples[key],
                "correctness": cc_by_key[key],
            }
            if key == base_key:
                base_med = stats["median_us"]
                rec["role"] = "baseline_fallback"
            else:
                rec["role"] = "candidate"
                rec["pct_change_vs_baseline_median"] = (
                    (stats["median_us"] - base_med) / base_med * 100.0
                    if base_med else None
                )
            target_rec["candidates"].append(rec)

        # Winner: a correct candidate whose median beats the baseline by at
        # least accept_pct AND by more than the combined observed variability,
        # with a stable CV.
        winner = None
        baseline_rec = next(
            (r for r in target_rec["candidates"]
             if r["role"] == "baseline_fallback"),
            None,
        )
        if baseline_rec is not None and base_med is not None:
            bm = baseline_rec["stats"]["median_us"]
            bcv = baseline_rec["stats"]["cv"]
            for r in target_rec["candidates"]:
                if r["role"] != "candidate":
                    continue
                if not r["correctness"]["passed"]:
                    continue
                if r["stats"]["cv"] > args.max_cv:
                    continue
                improvement = (bm - r["stats"]["median_us"]) / bm * 100.0
                combined_var = max(bcv, r["stats"]["cv"]) * 100.0
                if improvement >= args.accept_pct and improvement > combined_var:
                    if winner is None or r["stats"]["median_us"] < winner[
                        "stats"]["median_us"]:
                        winner = r
        target_rec["winner"] = winner["config"] if winner else None
        record["results"].append(target_rec)

        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)
        print(
            f"[M={M} N={N} K={K}] winner={target_rec['winner']} "
            f"base_med={base_med and round(base_med, 2)}us "
            f"correct={len(survivors)}/{len(all_cfgs)}",
            flush=True,
        )

    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"EVIDENCE_DONE {record_path}", flush=True)
    return 0


def get_configs_compute_bound():
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [64, 128]:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 16, 32, 64]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                }
                            )
    return configs


def get_weight_shapes(tp_size):
    # NOTE(HandH1998): The weight shapes only works for DeepSeek-V3.
    # Modify them, if you tune for another different model.
    # cannot TP
    total = [
        (512 + 64, 7168),
        (2112, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (7168, 16384),
        (7168, 18432),
    ]
    # N can TP
    n_tp = [
        (18432 * 2, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (24576, 1536),
        (12288, 7168),
        (4096, 7168),
    ]
    # K can TP
    k_tp = [(7168, 18432), (7168, 16384), (7168, 2048)]

    weight_shapes = []
    for t in total:
        weight_shapes.append(t)
    for n_t in n_tp:
        new_t = (n_t[0] // tp_size, n_t[1])
        weight_shapes.append(new_t)
    for k_t in k_tp:
        new_t = (k_t[0], k_t[1] // tp_size)
        weight_shapes.append(new_t)
    return weight_shapes


def w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    config: dict[str, Any],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Legacy DeepSeek-V3 tuner launcher (contiguous B). Preserved unchanged.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight (contiguous in this legacy path).
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: 2-dim block size, e.g. [128, 128].
        config: Meta-parameter dict for the kernel.
        output_dtype: The dtype of the returned tensor.

    Returns:
        torch.Tensor: The result of the block-scaled matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    if A.dtype == torch.float8_e4m3fn:
        kernel = _w8a8_triton_block_scaled_mm
    else:
        raise RuntimeError("Currently, only support tune w8a8 block fp8 kernel.")

    kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


def benchmark_config(
    A, B, As, Bs, block_size, config, out_dtype=torch.float16, num_iters=10
):
    def run():
        w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype)

    torch.accelerator.synchronize()
    for _ in range(5):
        run()
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        torch.accelerator.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def tune(M, N, K, block_size, out_dtype, search_space, input_type):
    factor_for_scale = 1e-2

    if input_type == "fp8":
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        A_fp32 = (
            (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5)
            * 2
            * fp8_max
        )
        A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        B_fp32 = (
            (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5)
            * 2
            * fp8_max
        )
        B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    else:
        raise RuntimeError("Currently, only support tune w8a8 block fp8 kernel.")

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * factor_for_scale
    Bs = (
        torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda")
        * factor_for_scale
    )

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space):
        try:
            kernel_time = benchmark_config(
                A,
                B,
                As,
                Bs,
                block_size,
                config,
                out_dtype,
                num_iters=10,
            )
        except triton.runtime.autotuner.OutOfResources:
            continue

        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    now = datetime.now()
    print(f"{now.ctime()}] Completed tuning for batch_size={M}")
    assert best_config is not None
    return best_config


def save_configs(
    N,
    K,
    block_n,
    block_k,
    configs,
    save_path,
    input_type="fp8",
) -> None:
    os.makedirs(save_path, exist_ok=True)
    device_name = get_device_name_as_file_name()
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype={input_type}_w8a8,"
        f"block_shape=[{block_n},{block_k}].json"
    )

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing best config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def tune_on_gpu(args_dict):
    """Run tuning on a specific GPU."""
    gpu_id = args_dict["gpu_id"]
    batch_sizes = args_dict["batch_sizes"]
    weight_shapes = args_dict["weight_shapes"]
    args = args_dict["args"]

    torch.accelerator.set_device_index(gpu_id)
    print(f"Starting tuning on GPU {gpu_id} with batch sizes {batch_sizes}")

    block_n = args.block_n
    block_k = args.block_k
    out_dtype = DTYPE_MAP[args.out_dtype]
    save_path = args.save_path
    input_type = args.input_type

    search_space = get_configs_compute_bound()
    search_space = [
        config for config in search_space if block_k % config["BLOCK_SIZE_K"] == 0
    ]

    start = time.time()
    for shape in tqdm(weight_shapes, desc=f"GPU {gpu_id} - Shapes"):
        N, K = shape[0], shape[1]
        print(f"[GPU {gpu_id}] Tune for weight shape of `N: {N}, K: {K}`")
        benchmark_results = [
            tune(
                batch_size,
                N,
                K,
                [block_n, block_k],
                out_dtype,
                search_space,
                input_type,
            )
            for batch_size in tqdm(batch_sizes, desc=f"GPU {gpu_id} - Batch sizes")
        ]
        best_configs = {M: config for M, config in zip(batch_sizes, benchmark_results)}
        save_configs(N, K, block_n, block_k, best_configs, save_path, input_type)

    end = time.time()
    print(f"Tuning on GPU {gpu_id} took {end - start:.2f} seconds")


def distribute_batch_sizes(batch_sizes, num_gpus):
    batches_per_gpu = []
    for i in range(num_gpus):
        start_idx = i * len(batch_sizes) // num_gpus
        end_idx = (i + 1) * len(batch_sizes) // num_gpus
        batches_per_gpu.append(batch_sizes[start_idx:end_idx])
    return batches_per_gpu


def main(args):
    print(args)
    if args.targets or args.manifest:
        raise SystemExit(run_evidence(args))

    num_gpus = torch.accelerator.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available for tuning")
    print(f"Found {num_gpus} GPUs for parallel tuning")

    torch.cuda.init()

    if args.batch_size is None:
        batch_sizes = [
            1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256,
            512, 1024, 1536, 2048, 3072, 4096,
        ]
    else:
        batch_sizes = [args.batch_size]
        num_gpus = 1

    weight_shapes = get_weight_shapes(args.tp_size)

    batches_per_gpu = distribute_batch_sizes(batch_sizes, num_gpus)

    process_args = []
    for gpu_id in range(num_gpus):
        process_args.append(
            {
                "gpu_id": gpu_id,
                "batch_sizes": batches_per_gpu[gpu_id],
                "weight_shapes": weight_shapes,
                "args": args,
            }
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_gpus) as pool:
        pool.map(tune_on_gpu, process_args)

    print("Multi-GPU tuning completed")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="""Tune triton w8a8 block fp8 for DeepSeek-V3/DeepSeek-R1:
    python benchmark_w8a8_block_fp8.py --tp-size 8 --input-type fp8
Or run explicit-target evidence collection:
    python benchmark_w8a8_block_fp8.py --targets 1,4096,8192 \\
        --targets 8,4096,8192 --b-pad prod --scale-dtype e8m0 \\
        --out-dtype bfloat16 --out-dir tmp/bench
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--tp-size", "-tp", type=int, default=8)
    parser.add_argument("--input-type", type=str, choices=["fp8"], default="fp8")
    parser.add_argument(
        "--out-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16", "half"],
        default="float16",
    )
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--save-path", type=str, default="./")

    # Evidence-mode arguments.
    parser.add_argument(
        "--targets",
        type=str,
        action="append",
        default=None,
        help="Repeated M,N,K target (e.g. 1,4096,8192) or N,K with --m-values.",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        action="append",
        default=[1, 2, 4, 8, 16, 69, 256],
        help="M values to expand for N,K-form targets.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="JSON manifest with a 'targets' list of {M,N,K}.",
    )
    parser.add_argument(
        "--candidate-file",
        type=str,
        default=None,
        help="JSON list of candidate config dicts (identical procedure).",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="./")
    parser.add_argument(
        "--scale-dtype",
        type=str,
        choices=["fp32", "e8m0"],
        default="e8m0",
    )
    parser.add_argument(
        "--b-pad",
        type=str,
        default="prod",
        help="off | prod | <int extra padding elements>",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--confirm-iters", type=int, default=30)
    parser.add_argument("--rel-tol", type=float, default=0.001)
    parser.add_argument("--abs-tol", type=float, default=1.0)
    parser.add_argument(
        "--accept-pct",
        type=float,
        default=3.0,
        help="Min percent median improvement vs baseline to accept.",
    )
    parser.add_argument(
        "--max-cv", type=float, default=0.15, help="Max CV to accept."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
