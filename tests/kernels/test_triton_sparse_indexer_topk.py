# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused GPU tests for sparse-indexer stable top-k fallback."""

import pytest
import torch

from vllm.platforms import current_platform


pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.on_gfx1151()),
    reason="requires ROCm gfx1151",
)


def _device_tensor(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=torch.float32)


class TestSparseIndexerStableTopk:
    def test_stable_ties_choose_lower_indices_first(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        result = triton_sparse_indexer_topk(_device_tensor([[2, 3, 3, 3, 1]]), 5)
        expected = torch.tensor([[1, 2, 3, 0, 4]], device="cuda", dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_decode_masks_high_scores_outside_each_row_bound(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk_decode,
        )

        scores = _device_tensor([[1, 4, 99, 98], [5, 2, 8, 99]])
        lengths = torch.tensor([2, 3], device="cuda")
        result = triton_sparse_indexer_topk_decode(scores, lengths, top_k=4)
        expected = torch.tensor(
            [[1, 0, -1, -1], [2, 0, 1, -1]],
            device="cuda",
            dtype=torch.int32,
        )
        torch.testing.assert_close(result, expected)

    def test_prefill_masks_high_scores_outside_span_before_selection(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk_prefill,
        )

        scores = _device_tensor([[99, 98, 3, 1, 4, 2, 97], [1, 3, 2, 96, 95, 94, 93]])
        starts = torch.tensor([2, 0], device="cuda")
        ends = torch.tensor([6, 3], device="cuda")
        result = triton_sparse_indexer_topk_prefill(scores, starts, ends, top_k=5)
        expected = torch.tensor(
            [[2, 0, 3, 1, -1], [1, 2, 0, -1, -1]],
            device="cuda",
            dtype=torch.int32,
        )
        torch.testing.assert_close(result, expected)

    def test_nonfinite_scores_are_padding(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        scores = _device_tensor(
            [[3, float("nan"), 1, float("inf"), float("-inf"), 3]]
        )
        result = triton_sparse_indexer_topk(scores, top_k=7)
        expected = torch.tensor(
            [[0, 5, 2, -1, -1, -1, -1]],
            device="cuda",
            dtype=torch.int32,
        )
        torch.testing.assert_close(result, expected)

    def test_stable_sort_capture_replay_matches_eager(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        width, top_k = 8192, 512
        captured_scores = (
            (torch.arange(width, device="cuda") % 8).flip(0).float()[None, :]
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_result = triton_sparse_indexer_topk(captured_scores, top_k)
        graph.replay()
        torch.cuda.synchronize()

        expected = triton_sparse_indexer_topk(captured_scores, top_k)
        torch.testing.assert_close(captured_result, expected)

        captured_scores.copy_(captured_scores.roll(1, dims=1))
        graph.replay()
        torch.cuda.synchronize()

        expected = triton_sparse_indexer_topk(captured_scores, top_k)
        torch.testing.assert_close(captured_result, expected)

    def test_stable_sort_microbenchmark_correctness(self):
        """Exercise stable sort at target width without a flaky time threshold."""
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        width, top_k = 8192, 512
        scores = (torch.arange(width, device="cuda") % 8).flip(0).float()[None, :]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = triton_sparse_indexer_topk(scores, top_k)
        end.record()
        end.synchronize()
        assert start.elapsed_time(end) >= 0

        _, expected = torch.sort(scores, dim=-1, descending=True, stable=True)
        torch.testing.assert_close(result, expected[:, :top_k].to(torch.int32))
