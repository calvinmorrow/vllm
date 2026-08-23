# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused GPU tests for the sparse-indexer device-only top-k selector."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1151

_requires_gfx1151 = pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx1151()),
    reason="requires ROCm gfx1151",
)


def _device_tensor(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=torch.float32)


def _output(rows: int, top_k: int) -> torch.Tensor:
    return torch.empty((rows, top_k), device="cuda", dtype=torch.int32)


def _scratch(rows: int, width: int, top_k: int) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
        sparse_indexer_topk_scratch_shape,
    )

    return torch.empty(
        sparse_indexer_topk_scratch_shape(rows, width, top_k),
        device="cuda",
        dtype=torch.uint64,
    )


def _reference(
    scores: torch.Tensor,
    top_k: int,
    starts: torch.Tensor | None = None,
    ends: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
        _stable_topk_reference,
    )

    columns = torch.arange(scores.shape[1], device=scores.device)
    valid = None
    if starts is not None:
        assert ends is not None
        starts = starts.clamp(0, scores.shape[1])
        ends = ends.clamp(0, scores.shape[1])
        ends = torch.maximum(ends, starts)
        valid = (columns[None, :] >= starts[:, None]) & (
            columns[None, :] < ends[:, None]
        )
    return _stable_topk_reference(scores, top_k, valid)


def test_scratch_shape_validates_static_arguments() -> None:
    from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
        sparse_indexer_topk_scratch_numel,
        sparse_indexer_topk_scratch_shape,
    )

    assert sparse_indexer_topk_scratch_shape(0, 1, 1) == (2, 0, 1, 1)
    assert sparse_indexer_topk_scratch_shape(3, 2049, 513) == (2, 3, 3, 1024)
    assert sparse_indexer_topk_scratch_numel(3, 2049, 513) == 18_432
    assert sparse_indexer_topk_scratch_shape(1, 16_385, 1) == (2, 1, 17, 1)
    with pytest.raises(RuntimeError, match="score width"):
        sparse_indexer_topk_scratch_shape(1, 0, 1)
    with pytest.raises(RuntimeError, match="top_k"):
        sparse_indexer_topk_scratch_shape(1, 1, 0)


@_requires_gfx1151
class TestSparseIndexerDeviceTopk:
    def test_ties_across_selector_width_and_signed_zero_match_reference(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        width, top_k = 2048, 512
        scores = (torch.arange(width, device="cuda") % 7).flip(0).float()[None, :]
        scores[0, 600] = 4.0
        scores[0, 1700] = 4.0
        scores[0, 100] = 0.0
        scores[0, 1100] = -0.0
        out = _output(1, top_k)
        result = triton_sparse_indexer_topk(
            scores, top_k, out, _scratch(1, width, top_k)
        )
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(result, _reference(scores, top_k))

    def test_ties_at_global_512_boundary_across_1024_chunks(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        width, top_k = 2048, 512
        scores = torch.zeros((1, width), device="cuda", dtype=torch.float32)
        tied_columns = torch.cat(
            (
                torch.arange(256, device="cuda"),
                torch.arange(1024, 1281, device="cuda"),
            )
        )
        scores[0, tied_columns] = 1.0
        result = triton_sparse_indexer_topk(
            scores, top_k, _output(1, top_k), _scratch(1, width, top_k)
        )
        torch.testing.assert_close(result, _reference(scores, top_k))
        assert result[0, -1].item() == 1279

    def test_width_above_16384_matches_reference(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        width, top_k = 16_385, 4
        scores = torch.zeros((1, width), device="cuda", dtype=torch.float32)
        scores[0, torch.tensor([1, 1024, 16_384], device="cuda")] = torch.tensor(
            [1.0, 2.0, 3.0], device="cuda"
        )
        result = triton_sparse_indexer_topk(
            scores, top_k, _output(1, top_k), _scratch(1, width, top_k)
        )
        torch.testing.assert_close(result, _reference(scores, top_k))

    def test_decode_clamps_prefixes_excludes_nonfinite_and_pads(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk_decode,
        )

        scores = _device_tensor(
            [[1, 4, 99, float("inf")], [5, float("nan"), 8, float("-inf")]]
        )
        lengths = torch.tensor([-2, 99], device="cuda", dtype=torch.int32)
        out = _output(2, 6)
        result = triton_sparse_indexer_topk_decode(
            scores, lengths, 6, out, _scratch(2, 4, 6)
        )
        expected = _reference(
            scores,
            6,
            torch.zeros_like(lengths),
            lengths,
        )
        torch.testing.assert_close(result, expected)

    def test_decode_repeated_rows_uses_source_sequence_bounds(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk_decode,
        )

        scores = _device_tensor(
            [[1, 5, 9, 8], [2, 4, 7, 6], [3, 8, 1, 0], [9, 2, 1, 0]]
        )
        lengths = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
        out = _output(4, 4)
        result = triton_sparse_indexer_topk_decode(
            scores, lengths, 4, out, _scratch(4, 4, 4), repeats=2
        )
        expected = _reference(
            scores,
            4,
            torch.zeros(4, device="cuda", dtype=torch.int32),
            lengths.repeat_interleave(2),
        )
        torch.testing.assert_close(result, expected)

    def test_prefill_clamps_spans_and_returns_row_local_indices(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk_prefill,
        )

        scores = _device_tensor([[99, 98, 3, 1, 4, 2, 97], [1, 3, 2, 96, 95, 94, 93]])
        starts = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
        ends = torch.tensor([6, -3], device="cuda", dtype=torch.int32)
        out = _output(2, 8)
        result = triton_sparse_indexer_topk_prefill(
            scores, starts, ends, 8, out, _scratch(2, 7, 8)
        )
        expected = _reference(scores, 8, starts, ends)
        clamped_starts = starts.clamp(0, scores.shape[1])
        expected = torch.where(
            expected >= 0, expected - clamped_starts[:, None], expected
        )
        torch.testing.assert_close(result, expected)

    def test_output_capacity_is_validated(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        with pytest.raises(RuntimeError, match="out must"):
            triton_sparse_indexer_topk(
                _device_tensor([[1, 2]]), 2, _output(1, 1), _scratch(1, 2, 2)
            )

    def test_scratch_capacity_helper_uses_ping_pong_chunk_capacity(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            sparse_indexer_topk_scratch_numel,
            sparse_indexer_topk_scratch_shape,
        )

        assert sparse_indexer_topk_scratch_shape(3, 2049, 513) == (2, 3, 3, 1024)
        assert sparse_indexer_topk_scratch_numel(3, 2049, 513) == 18_432

    def test_scratch_capacity_is_validated(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        scores = _device_tensor([[1, 2]])
        with pytest.raises(RuntimeError, match="scratch must"):
            triton_sparse_indexer_topk(
                scores,
                2,
                _output(1, 2),
                torch.empty(1, device="cuda", dtype=torch.uint64),
            )

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
        (q.shape[0], max_model_len),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
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
        logits[batch, :context_len] = scores.sum(dim=1) * scales[
            blocks, positions % block_size
        ]
    return logits


@_requires_gfx1151
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
            q,
            cache,
            weights,
            context_lens,
            block_tables,
            max_model_len=128,
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
