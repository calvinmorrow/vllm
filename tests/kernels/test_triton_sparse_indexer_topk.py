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


def _make_paged_mqa_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        indexer_k_quant_and_cache_triton,
    )

    torch.manual_seed(0)
    batch_size, block_size, max_model_len = 2, 64, 128
    num_blocks = 4
    kv_cache = torch.zeros(
        (num_blocks, block_size, 132), dtype=torch.uint8, device="cuda"
    )
    block_tables = torch.tensor(
        [[2, 0], [3, 1]], dtype=torch.int32, device="cuda"
    )
    context_lens = torch.tensor([70, 33], dtype=torch.int32, device="cuda")
    slot_mapping = torch.cat(
        (
            torch.cat(
                (
                    torch.arange(64, device="cuda") + 2 * block_size,
                    torch.arange(6, device="cuda"),
                )
            ),
            torch.arange(33, device="cuda") + 3 * block_size,
        )
    ).to(torch.int64)
    k = torch.randn(103, 128, dtype=torch.bfloat16, device="cuda") * 0.125
    indexer_k_quant_and_cache_triton(
        k, kv_cache, slot_mapping, quant_block_size=128, scale_fmt="dynamic"
    )
    q = (torch.randn(batch_size, 1, 64, 128, device="cuda") * 0.125).to(
        torch.float8_e4m3fnuz
    )
    weights = torch.rand(batch_size, 64, dtype=torch.float32, device="cuda")
    return q, kv_cache.unsqueeze(-2), weights, context_lens, block_tables


def _paged_mqa_logits_reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    block_size, head_dim = cache.shape[1], q.shape[-1]
    cache_flat = cache.squeeze(-2).view(cache.shape[0], -1)
    values = cache_flat[:, : block_size * head_dim].view(q.dtype)
    scales = cache_flat[:, block_size * head_dim :].view(torch.float32)
    logits = torch.full(
        (q.shape[0], max_model_len), float("-inf"), dtype=torch.float32, device=q.device
    )
    dims = torch.arange(head_dim, device=q.device)
    for batch in range(q.shape[0]):
        context_len = int(context_lens[batch])
        positions = torch.arange(context_len, device=q.device)
        blocks = block_tables[batch, positions // block_size].long()
        offsets = (
            (positions[:, None] % block_size // 16) * (16 * head_dim)
            + (positions[:, None] % 16) * 16
            + (dims[None, :] // 16) * (16 * 16)
            + dims[None, :] % 16
        )
        keys = values[blocks[:, None], offsets].float()
        scores = torch.relu(keys @ q[batch, 0].float().T) * weights[batch]
        logits[batch, :context_len] = scores.sum(dim=1) * scales[blocks, positions % block_size]
    return logits


class TestTritonPagedMqaLogits:
    def test_matches_torch_reference_with_paged_shuffled_cache(self):
        from vllm.v1.attention.ops.triton_fp8_paged_mqa_logits import (
            triton_fp8_paged_mqa_logits_gfx1151,
        )
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            reset_workspace_manager,
        )

        q, cache, weights, context_lens, block_tables = _make_paged_mqa_inputs()
        reset_workspace_manager()
        init_workspace_manager(torch.device("cuda"))
        actual = triton_fp8_paged_mqa_logits_gfx1151(
            q, cache, weights, context_lens, block_tables, max_model_len=128
        ).clone()
        expected = _paged_mqa_logits_reference(
            q, cache, weights, context_lens, block_tables, max_model_len=128
        )

        valid = torch.arange(128, device="cuda")[None, :] < context_lens[:, None]
        torch.testing.assert_close(actual[valid], expected[valid], atol=2e-2, rtol=2e-2)
        assert torch.isneginf(actual[~valid]).all()
        reset_workspace_manager()

    def test_cuda_graph_replay_matches_eager(self):
        from vllm.v1.attention.ops.triton_fp8_paged_mqa_logits import (
            triton_fp8_paged_mqa_logits_gfx1151,
        )
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            init_workspace_manager,
            reset_workspace_manager,
        )

        q, cache, weights, context_lens, block_tables = _make_paged_mqa_inputs()
        reset_workspace_manager()
        init_workspace_manager(torch.device("cuda"))
        eager = triton_fp8_paged_mqa_logits_gfx1151(
            q, cache, weights, context_lens, block_tables, max_model_len=128
        ).clone()
        current_workspace_manager().lock()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = triton_fp8_paged_mqa_logits_gfx1151(
                q, cache, weights, context_lens, block_tables, max_model_len=128
            )
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(captured, eager)
        captured_ptr = captured.data_ptr()

        q_bytes = q.view(torch.uint8)
        q_bytes.copy_(q_bytes.roll(1, dims=2))
        expected = _paged_mqa_logits_reference(
            q, cache, weights, context_lens, block_tables, max_model_len=128
        )
        graph.replay()
        torch.cuda.synchronize()

        assert captured.data_ptr() == captured_ptr
        torch.testing.assert_close(captured, expected, atol=2e-2, rtol=2e-2)
        reset_workspace_manager()
