# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.model_executor.layers import sparse_attn_indexer
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    BuildPrefillChunkMetadataKernel,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.block_table import get_block_table_width


def test_rocm_triton_sparse_indexer_availability(monkeypatch):
    from vllm.v1.attention.ops import rocm_triton_sparse_indexer

    monkeypatch.setattr(
        rocm_triton_sparse_indexer.current_platform, "is_rocm", lambda: True
    )
    monkeypatch.setattr(rocm_triton_sparse_indexer, "on_gfx1151", lambda: True)
    assert rocm_triton_sparse_indexer.is_rocm_triton_sparse_indexer_available()

    monkeypatch.setattr(rocm_triton_sparse_indexer, "on_gfx1151", lambda: False)
    assert not rocm_triton_sparse_indexer.is_rocm_triton_sparse_indexer_available()


def test_rocm_triton_sparse_indexer_dispatch_requires_no_aiter(monkeypatch):
    monkeypatch.setattr(
        sparse_attn_indexer,
        "is_rocm_triton_sparse_indexer_available",
        lambda: True,
    )
    monkeypatch.setattr(sparse_attn_indexer.rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(
        sparse_attn_indexer.rocm_aiter_ops,
        "is_rdna_aiter_enabled",
        lambda: False,
    )
    assert sparse_attn_indexer.use_rocm_triton_sparse_indexer()

    monkeypatch.setattr(sparse_attn_indexer.rocm_aiter_ops, "is_enabled", lambda: True)
    assert not sparse_attn_indexer.use_rocm_triton_sparse_indexer()


def test_rocm_triton_sparse_indexer_accepts_compressed_kv(monkeypatch):
    from vllm.v1.attention.ops import rocm_triton_sparse_indexer

    sentinel = object()
    monkeypatch.setattr(
        rocm_triton_sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    monkeypatch.setattr(
        rocm_triton_sparse_indexer,
        "_rocm_triton_sparse_attn_indexer_impl",
        lambda *_args: sentinel,
    )

    result = rocm_triton_sparse_indexer.rocm_triton_sparse_attn_indexer(
        torch.empty(0),
        "indexer.k_cache",
        torch.empty(0),
        torch.empty(0),
        None,
        torch.empty(0),
        128,
        "dynamic",
        1,
        128,
        1,
        1,
        torch.empty(0, dtype=torch.int32),
        compress_ratio=4,
    )

    assert result is sentinel


def test_indexer_warmup_normalizes_zero_compress_ratios():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[0, 0, 4, 128, 0])
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = BuildPrefillChunkMetadataKernel().get_warmup_keys(config)

    assert {key.COMPRESS_RATIO for key in keys} == {1, 4, 128}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_num_states():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    # num_states = block_size // tokens_per_state = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    vllm_config = create_vllm_config(max_model_len=1024)
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024)
    block_table_width = get_block_table_width(max_num_blocks, kv_cache_spec.block_size)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
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

    storage_bs = kv_cache_spec.num_states  # 64
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
