# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only tests for W8A8 Block-FP8 static configuration selection."""

import json
import logging
from pathlib import Path

import pytest

from vllm.model_executor.layers.quantization.utils import fp8_utils

DEVICE_TOKEN = "AMD_Radeon_8060S"
GENERIC_FALLBACK = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}
SIX_SHAPES = [
    (4096, 8192),
    (4096, 4096),
    (4096, 2048),
    (32768, 1024),
    (8192, 1024),
    (1536, 4096),
]


def _config_filename(N: int, K: int) -> str:
    return (
        f"N={N},K={K},device_name={DEVICE_TOKEN},"
        "dtype=fp8_w8a8,block_shape=[128,128].json"
    )


def _nearest_key(configs: dict[int, dict], M: int) -> int:
    return min(configs, key=lambda key: abs(key - M))


@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear the cached loader around every filesystem-isolated test."""
    fp8_utils.get_w8a8_block_fp8_configs.cache_clear()
    yield
    fp8_utils.get_w8a8_block_fp8_configs.cache_clear()


@pytest.fixture
def isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the real loader's derived config directory to temporary files."""
    monkeypatch.setattr(fp8_utils, "__file__", str(tmp_path / "fp8_utils.py"))
    monkeypatch.setattr(
        fp8_utils,
        "get_device_name_as_file_name",
        lambda: DEVICE_TOKEN,
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir


def _write_config(
    config_dir: Path, N: int, K: int, m_keys: dict[str, dict]
) -> Path:
    path = config_dir / _config_filename(N, K)
    path.write_text(json.dumps(m_keys, indent=2) + "\n")
    fp8_utils.get_w8a8_block_fp8_configs.cache_clear()
    return path


def test_missing_file_returns_none_and_warns(isolated_config_dir, caplog):
    """An absent file returns None and emits the fallback warning."""
    with caplog.at_level(logging.WARNING):
        result = fp8_utils.get_w8a8_block_fp8_configs(4096, 8192, 128, 128)

    assert result is None
    assert "Config file not found" in caplog.text


def test_present_file_returns_config_and_info_log(isolated_config_dir, caplog):
    """The loader reads an isolated JSON file through its production path."""
    m_keys = {
        "1": {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        }
    }
    path = _write_config(isolated_config_dir, 4096, 8192, m_keys)

    with caplog.at_level(logging.INFO):
        result = fp8_utils.get_w8a8_block_fp8_configs(4096, 8192, 128, 128)

    assert result == {1: m_keys["1"]}
    assert "Using configuration from" in caplog.text
    assert str(path) in caplog.text
    assert "Config file not found" not in caplog.text


def test_uncovered_shape_returns_none_with_isolated_config(isolated_config_dir):
    """A config for one shape does not cover a sibling shape."""
    _write_config(isolated_config_dir, 4096, 8192, {"1": GENERIC_FALLBACK})

    assert fp8_utils.get_w8a8_block_fp8_configs(4096, 2048, 128, 128) is None


@pytest.mark.parametrize(
    ("M", "expected_key"),
    [(1, 1), (2, 1), (3, 4), (6, 4), (7, 8), (12, 8), (13, 16), (69, 69),
     (256, 256)],
)
def test_nearest_m_selection_uses_documented_boundaries(M, expected_key):
    """Nearest-M selection retains its deterministic midpoint behavior."""
    configs = {key: {} for key in (1, 4, 8, 16, 69, 256)}

    assert _nearest_key(configs, M) == expected_key


@pytest.mark.parametrize("N,K", [(32768, 1024), (8192, 1024)])
def test_m256_no_winner_configs_preserve_generic_fallback(N, K):
    """Explicit M=256 keys preserve fallback where no tuned winner was found."""
    config_dir = Path(fp8_utils.__file__).resolve().parent / "configs"
    config = json.loads((config_dir / _config_filename(N, K)).read_text())

    assert config["256"] == GENERIC_FALLBACK
    int_configs = {int(key): value for key, value in config.items()}
    assert _nearest_key(int_configs, 256) == 256


def test_installed_configs_cover_only_the_six_confirmed_shapes():
    """Each committed Radeon config file maps to an approved local shape."""
    config_dir = Path(fp8_utils.__file__).resolve().parent / "configs"
    installed = {
        path.name
        for path in config_dir.glob("*AMD_Radeon_8060S*dtype=fp8_w8a8*")
    }
    expected = {_config_filename(N, K) for N, K in SIX_SHAPES}

    assert installed == expected
