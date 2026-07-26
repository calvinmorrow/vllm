# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer
from vllm.model_executor.layers import sparse_attn_indexer
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.attention.ops import rocm_triton_sparse_indexer
from vllm.v1.kv_cache_interface import MLAAttentionSpec


@pytest.mark.parametrize(
    ("is_rocm", "is_gfx1151", "expected"),
    [
        (True, True, AttentionCGSupport.NEVER),
        (True, False, AttentionCGSupport.UNIFORM_BATCH),
        (False, False, AttentionCGSupport.UNIFORM_BATCH),
    ],
)
def test_indexer_cudagraph_support_excludes_rocm_fallback(
    monkeypatch, is_rocm, is_gfx1151, expected
):
    monkeypatch.setattr(indexer.current_platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setattr(
        indexer.current_platform, "on_gfx1151", lambda: is_gfx1151
    )

    support = DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(None, None)

    assert support is expected


@pytest.mark.parametrize(
    ("is_rocm", "is_gfx1151", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_rocm_triton_sparse_indexer_availability(
    monkeypatch, is_rocm, is_gfx1151, expected
):
    monkeypatch.setattr(
        rocm_triton_sparse_indexer.current_platform, "is_rocm", lambda: is_rocm
    )
    monkeypatch.setattr(
        rocm_triton_sparse_indexer.current_platform,
        "on_gfx1151",
        lambda: is_gfx1151,
    )

    assert (
        rocm_triton_sparse_indexer.is_rocm_triton_sparse_indexer_available()
        is expected
    )


@pytest.mark.parametrize(
    ("fallback_available", "aiter_enabled", "expected"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_rocm_sparse_indexer_dispatch_preserves_aiter_priority(
    monkeypatch, fallback_available, aiter_enabled, expected
):
    monkeypatch.setattr(
        sparse_attn_indexer,
        "is_rocm_triton_sparse_indexer_available",
        lambda: fallback_available,
    )
    monkeypatch.setattr(
        sparse_attn_indexer.rocm_aiter_ops, "is_enabled", lambda: aiter_enabled
    )

    assert sparse_attn_indexer.use_rocm_triton_sparse_indexer() is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_storage_block_size():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    # storage_block_size = block_size // compress_ratio = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    vllm_config = create_vllm_config(max_model_len=1024)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
    )

    # Construct a single request where:
    # - num_computed = 240 (=> compressed_pos_start = 60)
    # - query_len = 40 (=> num_groups = 10)
    # => compressed positions are 60..69 which cross the storage block boundary at 64.
    query_start_loc = torch.tensor([0, 40], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([280], dtype=torch.int32, device=device)  # 240 + 40

    # Two blocks: compressed positions 0..63 map to block 5, 64..127 map to block 7.
    block_table_tensor = torch.tensor([[5, 7]], dtype=torch.int32, device=device)

    # Dummy uncompressed slot mapping (length == uncompressed num_actual_tokens).
    slot_mapping = torch.full((40,), -123, dtype=torch.int64, device=device)

    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=40,
        max_query_len=40,
        max_seq_len=280,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    # The compressed slot_mapping retains the original uncompressed size (40).
    # Only every compress_ratio-th position gets a valid slot; the rest are -1.
    assert md.slot_mapping.numel() == 40
    valid_slots = md.slot_mapping[md.slot_mapping >= 0]
    assert valid_slots.numel() == 10  # 40 tokens / compress_ratio 4

    storage_bs = kv_cache_spec.storage_block_size  # 64
    # Compressed positions 60..63 land in block 5, positions 64..69 in block 7.
    expected = torch.tensor(
        [
            5 * storage_bs + 60,
            5 * storage_bs + 61,
            5 * storage_bs + 62,
            5 * storage_bs + 63,
        ]
        + [
            7 * storage_bs + 0,
            7 * storage_bs + 1,
            7 * storage_bs + 2,
            7 * storage_bs + 3,
            7 * storage_bs + 4,
            7 * storage_bs + 5,
        ],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(valid_slots, expected)
