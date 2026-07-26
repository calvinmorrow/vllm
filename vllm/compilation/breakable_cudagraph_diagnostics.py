# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in aggregate diagnostics for breakable CUDA graph execution."""

from __future__ import annotations

import time
from collections import Counter
from typing import Final

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_LOG_INTERVAL_S: Final = 10.0
_LOG_CHECK_INTERVAL_EVENTS: Final = 8
_MAX_COUNTER_KEYS: Final = 32


class BreakableCUDAGraphDiagnostics:
    """Collect bounded, worker-local counters without synchronizing the device."""

    def __init__(self) -> None:
        self.enabled = envs.VLLM_BREAKABLE_CUDAGRAPH_DIAGNOSTICS
        self._last_log_time = time.monotonic() if self.enabled else 0.0
        self._counters: Counter[str] = Counter()
        self._durations_ms: dict[str, float] = {}
        self._dispatches: Counter[str] = Counter()
        self._events_since_log = 0

    def record(self, name: str, count: int = 1) -> None:
        if not self.enabled:
            return
        self._counters[name] += count
        self._maybe_log()

    def record_duration_ms(self, name: str, elapsed_s: float) -> None:
        if not self.enabled:
            return
        self._durations_ms[name] = (
            self._durations_ms.get(name, 0.0) + elapsed_s * 1_000
        )
        self._maybe_log()

    def record_dispatch(
        self, num_tokens: int, padded_tokens: int, mode: str
    ) -> None:
        if not self.enabled:
            return
        self._counters["dispatches"] += 1
        self._counters["padding_tokens"] += padded_tokens - num_tokens
        key = f"{num_tokens}->{padded_tokens}:{mode}"
        if key in self._dispatches or len(self._dispatches) < _MAX_COUNTER_KEYS:
            self._dispatches[key] += 1
        else:
            self._dispatches["other"] += 1
        self._maybe_log()

    def _maybe_log(self) -> None:
        self._events_since_log += 1
        if self._events_since_log < _LOG_CHECK_INTERVAL_EVENTS:
            return
        self._events_since_log = 0
        now = time.monotonic()
        if now - self._last_log_time < _LOG_INTERVAL_S:
            return
        self._last_log_time = now
        if not self._counters:
            return
        counters = " ".join(
            f"{name}={count}" for name, count in sorted(self._counters.items())
        )
        durations = " ".join(
            f"{name}={elapsed_ms:.1f}"
            for name, elapsed_ms in sorted(self._durations_ms.items())
        )
        dispatches = ",".join(
            f"{key}:{count}" for key, count in sorted(self._dispatches.items())
        )
        logger.info(
            "BREAKABLE_CUDAGRAPH_DIAG interval_s=%.0f %s %s "
            "dispatches_by_shape=%s",
            _LOG_INTERVAL_S,
            counters,
            durations,
            dispatches or "none",
        )
        self._counters.clear()
        self._durations_ms.clear()
        self._dispatches.clear()

    def snapshot(self) -> tuple[dict[str, int], dict[str, float], dict[str, int]]:
        """Return pending data for unit tests and debug inspection."""
        return dict(self._counters), dict(self._durations_ms), dict(self._dispatches)


breakable_cudagraph_diagnostics = BreakableCUDAGraphDiagnostics()
