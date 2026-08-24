# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils


def _reset_telemetry_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fp8_utils, "_LOG_BLOCK_FP8_SHAPES", True)
    monkeypatch.setattr(fp8_utils, "_MAX_LOGGED_BLOCK_FP8_SHAPES", 128)
    fp8_utils._logged_block_fp8_shapes.clear()


def _log_shape() -> None:
    fp8_utils.log_block_fp8_shape(
        "test",
        torch.empty((2, 4)),
        torch.empty((8, 4)),
        torch.empty((2, 1)),
        torch.empty((1, 1)),
        (128, 128),
        torch.bfloat16,
        "triton",
    )


class _BlockFp8Module(torch.nn.Module):
    def __init__(self, weight_shape: tuple[int, int] = (128, 256)) -> None:
        super().__init__()
        self.weight = torch.empty(weight_shape, dtype=torch.float8_e4m3fn)
        self.weight_scale_inv = torch.empty(
            (weight_shape[0] // 128, weight_shape[1] // 128)
        )


def _block_fp8_module(
    weight_shape: tuple[int, int] = (128, 256),
) -> _BlockFp8Module:
    return _BlockFp8Module(weight_shape)


def test_block_fp8_shape_telemetry_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(fp8_utils, "logger", logger)
    monkeypatch.setattr(fp8_utils, "_LOG_BLOCK_FP8_SHAPES", False)

    _log_shape()

    logger.info.assert_not_called()


def test_block_fp8_shape_telemetry_is_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_telemetry_state(monkeypatch)
    logger = Mock()
    monkeypatch.setattr(fp8_utils, "logger", logger)

    _log_shape()
    _log_shape()

    logger.info.assert_called_once()
    assert logger.info.call_args.args[2:6] == (2, 8, 4, (128, 128))


def test_block_fp8_shape_telemetry_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_telemetry_state(monkeypatch)
    monkeypatch.setattr(fp8_utils, "_MAX_LOGGED_BLOCK_FP8_SHAPES", 1)
    logger = Mock()
    monkeypatch.setattr(fp8_utils, "logger", logger)

    _log_shape()
    fp8_utils.log_block_fp8_shape(
        "test",
        torch.empty((3, 4)),
        torch.empty((8, 4)),
        torch.empty((3, 1)),
        torch.empty((1, 1)),
        (128, 128),
        torch.bfloat16,
        "triton",
    )

    logger.info.assert_called_once()


def test_block_fp8_weight_inventory_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(fp8_utils, "logger", logger)
    monkeypatch.setattr(fp8_utils, "_LOG_BLOCK_FP8_SHAPES", False)

    fp8_utils.log_block_fp8_weight_inventory(
        "test", (("linear", _block_fp8_module()),)
    )

    logger.info.assert_not_called()


def test_block_fp8_weight_inventory_aggregates_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_telemetry_state(monkeypatch)
    logger = Mock()
    monkeypatch.setattr(fp8_utils, "logger", logger)
    modules = (
        ("first", _block_fp8_module()),
        ("second", _block_fp8_module()),
        ("not_block_scaled", torch.nn.Module()),
    )

    fp8_utils.log_block_fp8_weight_inventory("test", modules)
    fp8_utils.log_block_fp8_weight_inventory("test", modules)

    logger.info.assert_called_once()
    entries = logger.info.call_args.args[2]
    assert entries[0][0:2] == ((128, 256), (1, 2))
    assert entries[0][4:] == (2, "first")
