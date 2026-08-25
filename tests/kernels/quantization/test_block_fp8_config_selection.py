# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# CPU-only integration harness for the W8A8 block-FP8 static-config selection
# path (Req 9.2-9.3). No GPU required: exercises get_w8a8_block_fp8_configs
# file lookup, log messages, nearest-M key selection, and fallback behavior.

import json
import logging
import os
from pathlib import Path
from unittest import mock

import pytest

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_w8a8_block_fp8_configs,
)

# The config directory that get_w8a8_block_fp8_configs reads from.
CONFIGS_DIR = Path(
    os.path.dirname(
        os.path.realpath(
            "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
        )
    )
) / "configs"

# The exact six confirmed runtime (N, K) shapes.
SIX_SHAPES = [
    (4096, 8192),
    (4096, 4096),
    (4096, 2048),
    (32768, 1024),
    (8192, 1024),
    (1536, 4096),
]

DEVICE_TOKEN = "AMD_Radeon_8060S"


def _config_filename(N: int, K: int) -> str:
    return (
        f"N={N},K={K},device_name={DEVICE_TOKEN},"
        f"dtype=fp8_w8a8,block_shape=[128,128].json"
    )


def _make_config_file(N: int, K: int, m_keys: dict[str, dict]) -> str:
    """Write a config file into the real configs dir; return its path."""
    path = CONFIGS_DIR / _config_filename(N, K)
    path.write_text(json.dumps(m_keys, indent=2) + "\n")
    get_w8a8_block_fp8_configs.cache_clear()
    return str(path)


def _remove_config_file(N: int, K: int) -> None:
    path = CONFIGS_DIR / _config_filename(N, K)
    if path.exists():
        path.unlink()
    get_w8a8_block_fp8_configs.cache_clear()


class TestConfigFileSelection:
    """Verify get_w8a8_block_fp8_configs file lookup and log messages."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Ensure a clean slate before and after each test."""
        for N, K in SIX_SHAPES:
            _remove_config_file(N, K)
        yield
        for N, K in SIX_SHAPES:
            _remove_config_file(N, K)

    def test_missing_file_returns_none_and_warns(self, caplog):
        """No config file -> None + the missing-file warning."""
        with caplog.at_level(logging.WARNING):
            result = get_w8a8_block_fp8_configs(4096, 8192, 128, 128)
        assert result is None
        assert "Config file not found" in caplog.text

    def test_present_file_returns_config_and_info_log(self, caplog):
        """Config file present -> dict + the 'Using configuration from' log."""
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
        path = _make_config_file(4096, 8192, m_keys)
        with caplog.at_level(logging.INFO):
            result = get_w8a8_block_fp8_configs(4096, 8192, 128, 128)
        assert result is not None
        assert 1 in result
        assert result[1]["BLOCK_SIZE_M"] == 16
        assert "Using configuration from" in caplog.text
        assert path in caplog.text
        assert "Config file not found" not in caplog.text


class TestNearestMSelection:
    """Verify the nearest-M key selection used by the launcher."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        for N, K in SIX_SHAPES:
            _remove_config_file(N, K)

    def _nearest_key(self, configs: dict[int, dict], M: int) -> int:
        """Replicate the launcher's nearest-M selection exactly."""
        return min(configs.keys(), key=lambda x: abs(x - M))

    def test_single_key_routes_all_m(self):
        """A file with only key '1' routes every M to it (not single-token)."""
        configs = {1: {"BLOCK_SIZE_M": 16}}
        for M in (1, 2, 4, 8, 16, 69, 256, 1000):
            assert self._nearest_key(configs, M) == 1

    def test_multi_key_routes_to_intended_region(self):
        """With keys 1/4/8/16/69/256, each M routes to the intended key."""
        configs = {1: {}, 4: {}, 8: {}, 16: {}, 69: {}, 256: {}}
        assert self._nearest_key(configs, 1) == 1
        assert self._nearest_key(configs, 2) == 1
        assert self._nearest_key(configs, 3) == 4
        assert self._nearest_key(configs, 4) == 4
        assert self._nearest_key(configs, 5) == 4
        # M=6 is a tie between keys 4 and 8 (both distance 2); min resolves to
        # the first-encountered (smaller) key 4.
        assert self._nearest_key(configs, 6) == 4
        assert self._nearest_key(configs, 7) == 8
        assert self._nearest_key(configs, 8) == 8
        # M=12 is a tie between keys 8 and 16 (both distance 4); min resolves
        # to 8. M=13 crosses the 8/16 boundary to 16.
        assert self._nearest_key(configs, 12) == 8
        assert self._nearest_key(configs, 13) == 16
        assert self._nearest_key(configs, 16) == 16
        assert self._nearest_key(configs, 20) == 16
        assert self._nearest_key(configs, 69) == 69
        assert self._nearest_key(configs, 100) == 69
        assert self._nearest_key(configs, 256) == 256

    def test_tie_breaks_to_first_minimal_key(self):
        """On an exact midpoint tie, min returns the first (smaller) key."""
        configs = {2: {}, 4: {}}
        assert self._nearest_key(configs, 3) == 2

    def test_guard_m_69_routes_to_guard_key(self):
        """M=69 (indeterminate guard) routes to the 69 key when present."""
        configs = {1: {}, 4: {}, 8: {}, 16: {}, 69: {}, 256: {}}
        assert self._nearest_key(configs, 69) == 69
        # M=69 is closer to 69 than to 16 or 256.
        assert abs(69 - 69) < abs(69 - 16)
        assert abs(69 - 69) < abs(69 - 256)


class TestUncoveredShapesPreserveFallback:
    """Verify uncovered shapes still fall back to the default config."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        for N, K in SIX_SHAPES:
            _remove_config_file(N, K)

    def test_uncovered_shape_returns_none(self):
        """An (N,K) with no config file returns None (fallback path)."""
        # Install a config for one shape only.
        _make_config_file(4096, 8192, {"1": {"BLOCK_SIZE_M": 16}})
        # A different shape must still return None.
        result = get_w8a8_block_fp8_configs(4096, 2048, 128, 128)
        assert result is None

    def test_all_six_shapes_start_uncovered(self):
        """With no config files installed, all six shapes return None."""
        for N, K in SIX_SHAPES:
            assert get_w8a8_block_fp8_configs(N, K, 128, 128) is None

    def test_installed_shape_is_covered_uncovered_siblings_are_not(self):
        """Installing one shape covers only that shape; siblings stay None."""
        _make_config_file(32768, 1024, {"1": {"BLOCK_SIZE_M": 16}})
        covered = get_w8a8_block_fp8_configs(32768, 1024, 128, 128)
        assert covered is not None and 1 in covered
        for N, K in SIX_SHAPES:
            if (N, K) != (32768, 1024):
                assert (
                    get_w8a8_block_fp8_configs(N, K, 128, 128) is None
                ), f"{(N, K)} should be uncovered"
