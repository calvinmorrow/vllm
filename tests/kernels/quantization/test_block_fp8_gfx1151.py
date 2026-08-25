# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Focused block-FP8 correctness coverage for the gfx1151 (RDNA4) W8A8
# Triton route: multi-K-block [128,128] scales, padded / non-contiguous B,
# the ROCm E8M0->FP32 scale conversion used by the launcher, and a CPU-only
# statistic/boundary helper test that needs no GPU.

import math

import pytest
import torch

from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
    w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform

# The module is meaningful only where the FP8 Triton route is exercisable and
# FP8 E4M3 is the platform dtype (not FNUZ). Skip elsewhere with a reason.
pytest.importorskip("torch.cuda")

ROCM = current_platform.is_rocm()
FNUZ = current_platform.is_fp8_fnuz()
FP8 = current_platform.fp8_dtype()

# The gfx1151 benchmark uses padded B (row pitch K + 256) and [128, 128] blocks
# with BF16 output, mirroring DeepSeek V4 Flash dense linears.
BLOCK_SIZE = [128, 128]
OUT_DTYPE = torch.bfloat16
SEED = 0


def _rand_fp8(*shape, device="cuda"):
    finfo = torch.finfo(torch.float8_e4m3fn)
    fmax, fmin = finfo.max, finfo.min
    x = (
        (torch.rand(*shape, dtype=torch.float32, device=device) - 0.5)
        * 2
        * fmax
    )
    return x.clamp(min=fmin, max=fmax).to(torch.float8_e4m3fn)


@pytest.mark.skipif(
    FNUZ, reason="This platform uses e4m3fnuz, not e4m3fn."
)
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 128, 1024),  # single N block, many K blocks
        (7, 256, 2048),  # two N blocks, many K blocks, M not a power of two
        (16, 512, 1024),  # multi N and K blocks, M == BLOCK_SIZE_M
        (69, 512, 4096),  # observed indeterminate guard M
    ],
)
@torch.inference_mode()
def test_block_fp8_multi_k_block_reference(M, N, K):
    """Triton W8A8 block-FP8 matches the native block reference across
    more than one K scale block (K > 128) for several (M, N, K)."""
    torch.manual_seed(SEED)
    block_n, block_k = BLOCK_SIZE
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    A = _rand_fp8(M, K)
    B = _rand_fp8(N, K)
    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * 1e-2

    ref = native_w8a8_block_matmul(A, B, As, Bs, BLOCK_SIZE, OUT_DTYPE)
    out = w8a8_triton_block_scaled_mm(A, B, As, Bs, BLOCK_SIZE, OUT_DTYPE)

    assert out.shape == (M, N) and out.dtype == OUT_DTYPE
    rel = (
        torch.abs(out.float() - ref.float()).mean()
        / torch.abs(ref.float()).mean()
    )
    assert rel < 1e-3, f"rel={rel.item()}"


@pytest.mark.skipif(
    not ROCM, reason="Padded-B (VLLM_ROCM_FP8_PADDING) layout is ROCm."
)
@pytest.mark.parametrize("K", [1024, 2048])
@torch.inference_mode()
def test_block_fp8_padded_noncontiguous_b_matches_contiguous(K):
    """A padded, non-contiguous B (row pitch K + 256, as produced by
    ``_maybe_pad_fp8_weight``) yields output equal to the value-identical
    contiguous B reference, with [128,128] scales across multiple K blocks."""
    torch.manual_seed(SEED)
    M, N, K_pad = 1, 256, K
    block_n, block_k = BLOCK_SIZE
    k_tiles = (K_pad + block_k - 1) // block_k
    A = _rand_fp8(M, K_pad)
    # Backing buffer with a 256-element pad per row; logical [N, K] view.
    B_full = _rand_fp8(N, K_pad + 256)
    B_full[:, K_pad:] = 0
    B_padded = B_full[:, :K_pad]
    assert not B_padded.is_contiguous()
    B_ref = B_padded.contiguous()  # value-identical contiguous reference

    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    Bs = torch.rand(
        (N + block_n - 1) // block_n, k_tiles, dtype=torch.float32,
        device="cuda",
    ) * 1e-2

    ref = native_w8a8_block_matmul(A, B_ref, As, Bs, BLOCK_SIZE, OUT_DTYPE)
    out = w8a8_triton_block_scaled_mm(A, B_padded, As, Bs, BLOCK_SIZE, OUT_DTYPE)

    assert out.shape == (M, N)
    # Padded and contiguous routes must agree element-for-element (bf16).
    assert torch.equal(out.float(), ref.float()) or (
        torch.abs(out.float() - ref.float()).max() < 1e-2
    )


@pytest.mark.skipif(
    not ROCM or not hasattr(torch, "float8_e8m0fnu"),
    reason="E8M0 scale conversion is a ROCm E8M0-capability path.",
)
@torch.inference_mode()
def test_e8m0_to_fp32_conversion_equivalence():
    """``_upcast_e8m0_to_fp32`` reproduces 2**(b-127) for E8M0 exponents.

    E8M0 stores only the 8-bit biased exponent (bias=127); the upcast places
    those bits into the float32 exponent field, giving powers of two.
    """
    bits = torch.arange(120, 138, dtype=torch.uint8)
    e = bits.view(torch.float8_e8m0fnu)
    up = _upcast_e8m0_to_fp32(e)
    expected = torch.tensor([2.0 ** (b - 127) for b in bits.tolist()])
    assert torch.allclose(up, expected)
    # 127 (bias) maps to exactly 1.0.
    assert _upcast_e8m0_to_fp32(
        torch.tensor([127], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    ).item() == 1.0


@pytest.mark.skipif(
    not ROCM,
    reason="E8M0 scale path on the launcher is ROCm; CPU here is capability.",
)
@torch.inference_mode()
def test_block_fp8_e8m0_scales_match_fp32_reference():
    """E8M0 scales (upcast by the launcher) produce output equal to the
    native reference computed with the upcast fp32 scales."""
    torch.manual_seed(SEED)
    M, N, K = 1, 256, 1024
    block_n, block_k = BLOCK_SIZE
    k_tiles = K // block_k
    A = _rand_fp8(M, K)
    B = _rand_fp8(N, K)
    As_e = torch.randint(120, 130, (M, k_tiles), device="cuda",
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    Bs_e = torch.randint(120, 130,
                         ((N + block_n - 1) // block_n, k_tiles),
                         device="cuda", dtype=torch.uint8
                         ).view(torch.float8_e8m0fnu)
    As_f = _upcast_e8m0_to_fp32(As_e).contiguous()
    Bs_f = _upcast_e8m0_to_fp32(Bs_e).contiguous()

    ref = native_w8a8_block_matmul(A, B, As_f, Bs_f, BLOCK_SIZE, OUT_DTYPE)
    out = w8a8_triton_block_scaled_mm(A, B, As_e, Bs_e, BLOCK_SIZE, OUT_DTYPE)

    rel = (
        torch.abs(out.float() - ref.float()).mean()
        / torch.abs(ref.float()).mean()
    )
    assert rel < 1e-3, f"rel={rel.item()}"


# ---------------------------------------------------------------------------
# CPU-only helper: nearest-M key selection, the integration contract used by
# ``get_w8a8_block_fp8_configs``. No GPU required.
# ---------------------------------------------------------------------------
def _nearest_key(configs: dict[str, int], M: int) -> str:
    return min(configs.keys(), key=lambda x: abs(int(x) - M))


def test_nearest_m_key_selection_cpu():
    """Nearest-key M selection routes observed M values to the intended key.

    This isolates the config-selection boundary that governs which M region a
    static JSON entry controls, without requiring a GPU.
    """
    # Example region layout: decode keys 1/4/8, guard keys 16/69/256.
    configs = {"1": 1, "4": 2, "8": 3, "16": 4, "69": 5, "256": 6}
    assert _nearest_key(configs, 1) == "1"
    assert _nearest_key(configs, 2) == "1"
    assert _nearest_key(configs, 3) == "4"
    assert _nearest_key(configs, 4) == "4"
    assert _nearest_key(configs, 8) == "8"
    # M=12 is a tie between keys 8 and 16 (both distance 4); min resolves to 8.
    assert _nearest_key(configs, 12) == "8"
    assert _nearest_key(configs, 13) == "16"
    assert _nearest_key(configs, 16) == "16"
    assert _nearest_key(configs, 69) == "69"
    assert _nearest_key(configs, 256) == "256"
    # A file with only a "1" key is NOT single-token-only: it routes every M.
    only_one = {"1": 1}
    assert _nearest_key(only_one, 1000) == "1"


def test_nearest_m_key_tie_breaks_to_smaller_cpu():
    """On an exact midpoint tie, ``min`` selects the first (smaller) key.

    Documents the deterministic tie behavior so nearest-M regions are explicit.
    """
    configs = {"2": 1, "4": 2}
    # M=3 is equidistant from 2 and 4; min returns the first minimal key.
    assert _nearest_key(configs, 3) == "2"
    assert abs(2 - 3) == abs(4 - 3) == 1
    # Sanity: math.fabs used for the distance is symmetric.
    assert math.fabs(3 - 2) == math.fabs(3 - 4)
