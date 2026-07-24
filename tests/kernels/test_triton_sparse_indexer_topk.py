# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Triton sparse indexer top-k selection.

Validates exact deterministic top-k semantics:
- Descending score ordering
- Ascending index tie-breaking
- -inf masked positions excluded
- -1 padding when candidates < top-k
- Local index conversion for prefill spans
"""

import math
from typing import Final

import pytest
import torch

from vllm.platforms import current_platform

# Mirror constants from the implementation for parametrize tests
CHUNK_N: Final = 4096
MERGE_GROUP: Final = 8
SORT_N_MAX: Final = 8192


def _ref_topk(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """PyTorch reference: descending score, ascending index for ties."""
    num_tokens, n_comp = scores.shape
    # torch.topk is descending by default
    _, indices = torch.topk(scores, k=min(top_k, n_comp), dim=-1, sorted=True)
    # torch.topk doesn't guarantee stable tie-breaking by index,
    # so we do a stable sort manually
    result = torch.full(
        (num_tokens, top_k), -1, dtype=torch.int32, device=scores.device
    )
    for t in range(num_tokens):
        # Create (score, -index) pairs for stable sort
        # Higher score first; for equal scores, lower index first
        paired = list(enumerate(scores[t].tolist()))
        paired.sort(key=lambda x: (-x[1], x[0]))
        for pos, (idx, _score) in enumerate(paired[:top_k]):
            result[t, pos] = idx
    return result


def _ref_topk_prefill(
    scores: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Reference prefill top-k with local index output."""
    num_tokens = scores.shape[0]
    abs_indices = _ref_topk(scores, top_k)
    result = torch.full_like(abs_indices, -1)
    for t in range(num_tokens):
        start = cu_seqlen_ks[t].item()
        end = cu_seqlen_ke[t].item()
        for k in range(top_k):
            idx = abs_indices[t, k].item()
            if idx >= 0 and start <= idx < end:
                result[t, k] = idx - start
    return result


def _ref_topk_decode(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Reference decode top-k with seq_len masking."""
    num_tokens = scores.shape[0]
    indices = _ref_topk(scores, top_k)
    for t in range(num_tokens):
        slen = seq_lens[t].item()
        for k in range(top_k):
            if indices[t, k].item() >= slen:
                indices[t, k] = -1
    return indices


class TestTopkHelpers:
    """Test Python helper functions (no GPU needed)."""

    def test_select_sort_n_small(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            _select_sort_n,
        )

        assert _select_sort_n(1) == 256
        assert _select_sort_n(128) == 256
        assert _select_sort_n(256) == 256
        assert _select_sort_n(257) == 512
        assert _select_sort_n(512) == 512
        assert _select_sort_n(1024) == 1024
        assert _select_sort_n(4096) == 4096
        assert _select_sort_n(8192) == 8192

    def test_select_sort_n_large(self):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            _select_sort_n,
        )

        # > 8192 caps at 8192 (chunk-and-merge will be used)
        assert _select_sort_n(16384) == 8192
        assert _select_sort_n(65536) == 8192

    def test_chunk_count(self):
        """Verify chunk count calculation."""
        assert math.ceil(100 / CHUNK_N) == 1
        assert math.ceil(4096 / CHUNK_N) == 1
        assert math.ceil(4097 / CHUNK_N) == 2
        assert math.ceil(8192 / CHUNK_N) == 2
        assert math.ceil(16384 / CHUNK_N) == 4

    def test_merge_tree_depth(self):
        """Verify merge tree reduces correctly."""
        n_sets = 16
        depth = 0
        while n_sets > MERGE_GROUP:
            n_sets = math.ceil(n_sets / MERGE_GROUP)
            depth += 1
        # 16 -> 2 (1 merge level)
        assert depth == 1
        assert n_sets == 2

        n_sets = 64
        depth = 0
        while n_sets > MERGE_GROUP:
            n_sets = math.ceil(n_sets / MERGE_GROUP)
            depth += 1
        # 64 -> 8 (1 level), stops
        assert depth == 1
        assert n_sets == 8


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Triton GPU tests require ROCm hardware",
)
class TestTopkGpu:
    """GPU tests: compare Triton top-k against PyTorch reference."""

    @pytest.mark.parametrize(
        "num_tokens,n_comp,top_k",
        [
            (1, 64, 8),
            (4, 128, 16),
            (8, 256, 32),
            (2, 512, 64),
            (1, 1024, 128),
            (4, 2048, 256),
            (2, 4096, 512),
            (1, 4096, 1024),
            (1, 8192, 512),
        ],
    )
    def test_single_phase(self, num_tokens, n_comp, top_k):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        torch.manual_seed(42)
        scores = torch.randn(num_tokens, n_comp, device="cuda")

        result = triton_sparse_indexer_topk(scores, top_k)
        expected = _ref_topk(scores, top_k).to("cuda")

        assert result.shape == (num_tokens, top_k)
        assert result.dtype == torch.int32
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "num_tokens,n_comp,top_k",
        [
            (1, 16384, 512),
            (2, 16384, 512),
            (1, 32768, 512),
        ],
    )
    def test_chunk_and_merge(self, num_tokens, n_comp, top_k):
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        torch.manual_seed(42)
        scores = torch.randn(num_tokens, n_comp, device="cuda")

        result = triton_sparse_indexer_topk(scores, top_k)
        expected = _ref_topk(scores, top_k).to("cuda")

        assert result.shape == (num_tokens, top_k)
        torch.testing.assert_close(result, expected)

    def test_tie_breaking(self):
        """Verify ascending-index tie-breaking for equal scores."""
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        # All equal scores: indices should be 0, 1, 2, ..., top_k-1
        scores = torch.ones(1, 10, device="cuda")
        result = triton_sparse_indexer_topk(scores, 5)
        expected = torch.tensor([[0, 1, 2, 3, 4]], device="cuda", dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_minus_inf_masking(self):
        """Positions with -inf scores should not appear in top-k."""
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        scores = torch.tensor(
            [[3.0, float("-inf"), 1.0, float("-inf"), 2.0]],
            device="cuda",
            dtype=torch.float32,
        )
        result = triton_sparse_indexer_topk(scores, 4)
        # Top 3 valid: indices 0, 4, 2 (scores 3, 2, 1)
        # 4th position padded with -1
        expected = torch.tensor(
            [[0, 4, 2, -1]], device="cuda", dtype=torch.int32
        )
        torch.testing.assert_close(result, expected)

    def test_fewer_candidates_than_topk(self):
        """When n_comp < top_k, remaining positions are -1."""
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        scores = torch.tensor(
            [[3.0, 1.0, 2.0]], device="cuda", dtype=torch.float32
        )
        result = triton_sparse_indexer_topk(scores, 5)
        expected = torch.tensor(
            [[0, 2, 1, -1, -1]], device="cuda", dtype=torch.int32
        )
        torch.testing.assert_close(result, expected)

    def test_prefill_local_indices(self):
        """Prefill top-k returns local indices relative to span start."""
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk_prefill,
        )

        # 2 tokens, max 8 candidates each
        # Token 0: valid range [2, 6) -> 4 candidates at abs indices 2,3,4,5
        # Token 1: valid range [0, 3) -> 3 candidates at abs indices 0,1,2
        scores = torch.tensor(
            [
                [
                    float("-inf"), float("-inf"), 3.0, 1.0,
                    4.0, 2.0, float("-inf"), float("-inf"),
                ],
                [
                    1.0, 3.0, 2.0, float("-inf"),
                    float("-inf"), float("-inf"), float("-inf"),
                    float("-inf"),
                ],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        cu_seqlen_ks = torch.tensor([2, 0], device="cuda")
        cu_seqlen_ke = torch.tensor([6, 3], device="cuda")

        result = triton_sparse_indexer_topk_prefill(
            scores, cu_seqlen_ks, cu_seqlen_ke, top_k=5
        )
        expected = _ref_topk_prefill(scores, cu_seqlen_ks, cu_seqlen_ke, 5).to(
            "cuda"
        )
        torch.testing.assert_close(result, expected)

    def test_decode_seq_len_masking(self):
        """Decode top-k respects per-row sequence lengths."""
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk_decode,
        )

        scores = torch.tensor(
            [
                [3.0, 1.0, 4.0, 2.0, 5.0],
                [1.0, 2.0, float("-inf"), float("-inf"), float("-inf")],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        # Token 0: seq_len=3 (only indices 0,1,2 valid)
        # Token 1: seq_len=2 (only indices 0,1 valid)
        seq_lens = torch.tensor([3, 2], device="cuda")

        result = triton_sparse_indexer_topk_decode(scores, seq_lens, top_k=4)
        expected = _ref_topk_decode(scores, seq_lens, 4).to("cuda")
        torch.testing.assert_close(result, expected)

    def test_deterministic_ordering(self):
        """Verify output is sorted: descending score, ascending index."""
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        torch.manual_seed(123)
        scores = torch.randn(4, 256, device="cuda")

        result = triton_sparse_indexer_topk(scores, 64)

        # Check ordering for each row
        for t in range(4):
            row_indices = result[t]
            for k in range(63):
                if row_indices[k] == -1 or row_indices[k + 1] == -1:
                    break
                idx_a = row_indices[k].item()
                idx_b = row_indices[k + 1].item()
                score_a = scores[t, idx_a].item()
                score_b = scores[t, idx_b].item()
                # Earlier position must have >= score
                assert score_a >= score_b, (
                    f"Row {t}: position {k} score {score_a} < "
                    f"position {k+1} score {score_b}"
                )
                if math.isclose(score_a, score_b):
                    # Equal scores: lower index first
                    assert idx_a < idx_b, (
                        f"Row {t}: tie at position {k}, "
                        f"idx {idx_a} >= {idx_b}"
                    )


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.on_gfx1151()),
    reason="Hardware-gated test requires gfx1151",
)
class TestGfx1151Hardware:
    """gfx1151-specific hardware tests."""

    def test_deepseek_v4_geometry(self):
        """Test with DeepSeek V4 target-like shapes."""
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        # Target: top_k=512, various context lengths
        for n_comp in [1024, 2048, 4096, 8192]:
            scores = torch.randn(8, n_comp, device="cuda", dtype=torch.float32)
            result = triton_sparse_indexer_topk(scores, 512)
            expected = _ref_topk(scores, 512).to("cuda")
            torch.testing.assert_close(
                result,
                expected,
                err_msg=f"Failed for n_comp={n_comp}",
            )

    def test_long_context_chunk_and_merge(self):
        """Test chunk-and-merge with long context."""
        from vllm.v1.attention.ops.triton_sparse_indexer_topk import (
            triton_sparse_indexer_topk,
        )

        # Long context: 16384 candidates -> 4 chunks of 4096
        scores = torch.randn(2, 16384, device="cuda", dtype=torch.float32)
        result = triton_sparse_indexer_topk(scores, 512)
        expected = _ref_topk(scores, 512).to("cuda")
        torch.testing.assert_close(result, expected)
