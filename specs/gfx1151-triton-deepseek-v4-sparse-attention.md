# Spec: gfx1151 Triton DeepSeek V4 sparse-attention support

## Tracking

- Branch: `rocm-autoround-deepseek-v4-flash-moe`
- Primary serving target: `Intel/DeepSeek-V4-Flash-W4A16-AutoRound`
- Related existing spec: `specs/rocm-autoround-deepseek-v4-flash-moe.md`
- Algorithmic reference only: `/a0/usr/projects/dwarfstar/rocm/ds4_rocm_indexer.cuh`

## Goal

Enable DeepSeek V4 Flash sparse attention to run on ROCm `gfx1151` without AITER by adding a native vLLM Triton/Python sparse-indexer path. This removes the current AITER-only failure that occurs after the Intel AutoRound checkpoint loads and its routed MoE layers have already selected `TritonWNA16Experts`.

Success means that the exact Intel AutoRound target generates a non-empty completion on `gfx1151` with `VLLM_ROCM_USE_AITER=0`, using the existing AutoRound INC and Triton WNA16 MoE code unchanged and a new Triton sparse-indexer implementation.

## Scope

### In scope

- An operation-specific, AITER-independent ROCm sparse-indexer capability and dispatch path for `gfx1151`.
- Reuse and validate vLLM's existing Triton score and downstream sparse-attention kernels where they work on `gfx1151`.
- Native Triton/Python exact top-k selection for DeepSeek V4 indexer scores, including the target-relevant long-context path.
- Correct prefill and decode sparse-indexer operation, including cache layout, valid-range semantics, padded speculative decode, and output index conventions.
- Unit and ROCm hardware correctness tests for the new kernels and selector integration.
- End-to-end single-GPU and two-node pipeline-parallel smoke validation with `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` on `gfx1151` hardware.
- Narrow user-facing documentation of the validated `gfx1151` requirement and AITER-independent behavior, after successful hardware validation.

### Out of scope

- Any AITER source, package, build, capability-gate, or dependency change.
- Changes to Intel AutoRound/INC checkpoint detection, format handling, linear kernels, or `MoeWNA16Method` / `TritonWNA16Experts`, unless baseline evidence proves a separate regression there.
- A HIP/C++ extension, CUDA/HIP kernel port, CUB/rocPRIM dependency, or copying Dwarfstar code into vLLM.
- Generic RDNA support: `gfx1100`, `gfx1150`, `gfx12xx`, and every other architecture remain unsupported unless independently validated later.
- A performance target, benchmarking gate, autotuning project, or fusion of score generation and top-k selection in the first implementation.
- General DeepSeek-family support, arbitrary sparse-attention models, arbitrary top-k workloads, or a broad rewrite of ROCm attention backend selection.
- Tensor parallel or expert parallel correctness/support expansion.
- Multi-node communication changes, NCCL/RDMA tuning, or Ray transport changes. Pipeline-parallel validation consumes the completed local backend but must not alter distributed infrastructure.

## Context

### Observed failure

The target checkpoint successfully reaches the intended quantized MoE route on ROCm:

```text
Using MoEPrepareAndFinalizeNoDPEPModular
Using TritonWNA16Experts
```

It then fails during profile-run attention before MoE inference:

```text
RuntimeError: Sparse attention indexer ROCm path is only supported on AITER.
Please enable aiter with VLLM_ROCM_USE_AITER=1
```

The relevant vLLM call chain is:

```text
DeepseekV4 attention
  -> SparseAttnIndexer.forward_hip()
  -> rocm_aiter_ops.is_enabled()
  -> exception when false
```

This failure is independent of AutoRound weight loading, WNA16 MoE selection, pipeline parallelism, and RDMA.

### Current vLLM owners

- `vllm/model_executor/layers/sparse_attn_indexer.py`
  - `SparseAttnIndexer.forward_hip()` currently invokes the AITER-named custom op only when `rocm_aiter_ops.is_enabled()` is true, otherwise raises the fatal error.
- `vllm/_aiter_ops.py`
  - AITER availability is intentionally broad and currently tied to package discovery, ROCm, and an MI300-class gate. Do not weaken or repurpose it.
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`
  - Despite its historical name, it already contains substantial generic ROCm Triton implementation: indexer K quantization/cache insertion, context-parallel K-cache gather, prefill indexer flow, decode indexer flow, and downstream sparse-attention kernels.
- `vllm/v1/attention/ops/triton_fp8_mqa_logits.py`
  - Contains a vendored Triton FP8 MQA score kernel and explicitly documents DeepSeek V4 indexer geometry of 64 heads and head size 128.
- `vllm/config/compilation.py`
  - Registers `vllm::rocm_aiter_sparse_attn_indexer`; inspect custom-op registration before renaming or adding a new operation.
- `vllm/platforms/rocm.py`
  - Enables `+sparse_attn_indexer` as a compilation custom operation. Preserve existing platform behavior outside the narrow new branch.

The current orchestration obtains score tensors then calls these compiled selector ops:

```python
torch.ops._C.top_k_per_row_prefill(...)
torch.ops._C.top_k_per_row_decode(...)
```

A gfx1151 path must not assume those compiled selector ops are usable; provide native Triton selection instead.

### Dwarfstar reference

Dwarfstar is algorithmic evidence and must remain external reference material. Its `rocm/ds4_rocm_indexer.cuh` provides a known-working ROCm `gfx1151` design:

- score computation with the DeepSeek V4-relevant 64-head × 128-dimension geometry;
- causal/valid candidate masking by writing negative infinity;
- deterministic exact top-k ordering: larger score first, then smaller index for equal score;
- specialized exact top-k for candidate counts up to 8192;
- exact long-context selection by top-k per 4096-wide candidate chunk followed by tree merges.

Port the semantics and algorithm into idiomatic vLLM Triton/Python. Do not copy Dwarfstar HIP/CUDA source, use `rocwmma`, use CUB, add a HIP extension, or inherit its flat FP32 cache assumptions.

### vLLM data contracts

The implementation must preserve vLLM's existing data and metadata contracts rather than adopt Dwarfstar's internal layouts:

- Input indexer query is FP8 and has target-relevant score geometry `[M, 64, 128]`.
- Prefill scores use gathered indexer K FP8 values/scales, `DeepseekV32IndexerMetadata.prefill`, per-chunk `cu_seqlen_ks`, and `cu_seqlen_ke`.
- Decode scores use paged indexer KV cache, `DeepseekV32IndexerMetadata.decode`, sequence lengths, and block tables.
- Indexer K cache uses vLLM's existing quantization/cache-insertion representation and scale format.
- Top-k output is `int32`, contains local logical indices for each query row, and uses `-1` for invalid/unavailable entries.
- Prefill top-k indices must be local to each row's legal `[start, end)` candidate span, rather than absolute columns in a concatenated logits matrix.
- Existing packed/unpacked decode handling using `pack_seq_triton()` and `unpack_seq_triton()` must continue to work.

## Requirements

1. **Record a `gfx1151` baseline before changing behavior.** On the target ROCm environment, run focused invocations of the existing `rocm_fp8_mqa_logits()` and `rocm_fp8_paged_mqa_logits()` for target-like shapes and compare with an explicit reference. Record whether each compiles and is numerically correct on `gfx1151`.

2. **Do not change AITER eligibility.** Leave `is_aiter_found_and_supported()`, `rocm_aiter_ops.is_enabled()`, all AITER environment variables, and existing MI300 AITER dispatch semantics unchanged.

3. **Add narrow operation-specific capability selection.** Implement a sparse-indexer-specific ROCm Triton support predicate/selector. It must select the new path only for a physically verified `gfx1151` ROCm environment and must not make general AITER operations, generic RDNA devices, or unsupported architectures appear supported.

4. **Make the sparse indexer runnable without AITER.** Update `SparseAttnIndexer.forward_hip()` so the verified `gfx1151` selector calls a generic ROCm Triton sparse-indexer custom op/path when `VLLM_ROCM_USE_AITER=0`. Retain the present AITER path for its current supported environment. Unsupported ROCm architectures must fail with an actionable message that identifies the available supported routes.

5. **Keep ownership and naming accurate.** If an AITER-named custom op is being used as a generic Triton path, refactor/rename or add an explicitly named generic ROCm Triton operation. Update its fake implementation and compilation registration together. Do not leave a misleading AITER requirement in the `gfx1151` runtime route.

6. **Reuse existing Triton scoring only when proven.** If the baseline validates both existing score functions, retain them. If either fails to compile or produces incorrect values on `gfx1151`, add the smallest native Triton implementation necessary, with the same vLLM FP8 cache and scale representation. Do not port the Dwarfstar FP32 score kernel as an incompatible replacement.

7. **Implement exact native Triton indexer top-k.** Add a dedicated module close to the ROCm sparse attention operations, for example `vllm/v1/attention/ops/triton_sparse_indexer_topk.py`. It must provide prefill and decode-capable selection over the score rows with the vLLM contracts above.

8. **Use the Dwarfstar-derived scalable exact selection algorithm.** For target-relevant `topk` and context lengths, use tile/chunk-local exact top-k and exact hierarchical merge. The initial wide-candidate chunk size may be 4096 or another measured viable Triton tile size, but it must preserve exact global top-k membership and ordering. Do not use a serial one-thread-per-row fallback as the normal long-context implementation.

9. **Preserve deterministic ordering.** For finite equal scores, the lower logical index must rank first. Masked candidates must behave as negative infinity. Rows with fewer valid candidates than requested top-k must return `-1` in remaining positions and must not expose out-of-range indices.

10. **Integrate native selection in both flows.** Replace the `top_k_per_row_prefill` and `top_k_per_row_decode` usage only on the new `gfx1151` path. Preserve existing selector behavior elsewhere. Handle prefill chunks, causal/valid spans, paged decode, decode lengths, and speculative-decode padding/unpacking.

11. **Do not regress current sparse-attention consumers.** The output from the new selector must feed the existing DeepSeek V4 sparse attention path without a model-name exception, cache-layout conversion, CPU synchronization, or host-side top-k.

12. **Add unit/reference tests.** Add narrowly scoped tests that compare score and selected indices to explicit PyTorch reference calculations. Tests must cover score masking, local-vs-global index conversion, deterministic ties, fewer-than-k candidates, prefill chunks, decode sequence lengths, and padded speculative decode behavior.

13. **Add hardware-gated `gfx1151` kernel correctness coverage.** Add tests that run only when the active ROCm GPU is `gfx1151` and otherwise skip with a precise reason. Cover target-like score geometry, prefill and decode top-k output, and downstream sparse-attention output against reference tensors with stated tolerances.

14. **Validate the full target checkpoint without AITER.** On a `gfx1151` runner with sufficient unified/GPU memory, use automatic quantization detection to load `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` with `VLLM_ROCM_USE_AITER=0`. Generate from a fixed short prompt and assert a non-empty completion. Capture logs proving `TritonWNA16Experts` remains selected and the new Triton sparse-indexer route is selected.

15. **Validate requested pipeline-parallel deployment.** After the single-GPU target smoke test passes, run the corresponding two-node `gfx1151` pipeline-parallel deployment over the intended RDMA fabric. It must load and generate one non-empty fixed-prompt completion without AITER. This test validates only compatibility of the completed local sparse-indexer implementation with pipeline parallelism; it does not create an RDMA performance claim.

16. **Document only verified support.** After hardware validation, update the applicable DeepSeek V4 ROCm documentation to state: `gfx1151` is the initial verified Triton sparse-indexer target; AITER is not required for that route; AutoRound W4A16 target support still relies on its existing Triton WNA16 MoE route; and no performance promise or generic RDNA claim is made.

## Constraints

- All implementation code must be native Python/Triton in the vLLM repository.
- Dwarfstar may be read to reproduce algorithmic behavior only. It is not a source dependency, vendored component, or license/copying shortcut.
- Preserve the current CUDA, XPU, CPU, existing ROCm/AITER, and MI300 behaviors.
- Preserve the existing AutoRound format boundary and MoE backend selection. This work must not make an unsupported quantization layout appear valid.
- Do not use `torch.topk` or other host-side/PyTorch fallback in the production gfx1151 sparse-indexer route.
- Avoid a global device-name exception in kernel code. Architecture capability dispatch belongs at the operation/backend boundary.
- Python commands must use `uv` and `.venv/bin/python`, never system Python or bare pip.
- Do not install dependencies without explicit user approval.
- Maintain 88-character Python line length and repository style.
- Do not modify generated files, lock files, or unrelated documentation.

## Key decisions (confirmed)

- **Goal:** serve the Intel AutoRound DeepSeek V4 Flash target on `gfx1151` without AITER through a Triton sparse-indexer implementation.
- **Initial hardware scope:** `gfx1151` only.
- **Implementation style:** native vLLM Triton/Python; no AITER and no HIP/C++ extension.
- **Reference use:** Dwarfstar is an algorithmic reference only.
- **Validation shape:** one integrated spec with both unit/kernel testing and real end-to-end validation, rather than isolated sibling specs that cannot independently establish success.
- **Performance:** correctness and successful serving are the initial release bar. No throughput or latency threshold is required.
- **Existing quantization route:** preserve INC AutoRound and Triton WNA16 MoE; do not reopen that already-selected path unless evidence shows a distinct issue.

## Edge cases and failure modes

- **Existing score kernel fails on gfx1151:** retain the same API and vLLM FP8 cache semantics, then add a minimal Triton score fallback. Do not switch to Dwarfstar's FP32 cache representation.
- **Top-k score ties:** order by ascending logical index after descending score.
- **Scores outside valid range:** treat as negative infinity and never return their index as a valid selection.
- **Candidate count less than top-k:** fill excess output with `-1`.
- **Long candidate sequences:** use chunk-and-merge rather than an unbounded per-row sort or serial selector.
- **Prefill chunk boundaries:** convert output to query-local indices exactly once; do not leak concatenated workspace offsets.
- **Speculative decode padding:** padded rows must not alter valid rows after unpacking; padded output must not become model-visible.
- **AITER is installed/enabled on gfx1151:** the selected `gfx1151` route must remain independently testable with `VLLM_ROCM_USE_AITER=0`; no accidental AITER import or operation dispatch may be required.
- **No qualifying hardware:** do not claim implementation validation. Unit tests can run, but the hardware-gated and end-to-end criteria remain blocked rather than passing by skip.
- **Two-node PP failure after local pass:** diagnose the pipeline configuration separately; do not alter sparse-indexer numerical semantics or silently disable PP to call the requirement fulfilled.

## Execution guidance

- **Recommended model:** a lower-cost code model such as Sonnet is suitable after baseline facts are captured; escalate only for a real Triton/RDNA compiler ambiguity.
- **Agent shape:** one ordered implementation session. The score baseline must precede code changes because the existing score kernels may already be sufficient.
- **Order:** baseline score validation → exact top-k tests/module → `gfx1151` dispatch/integration → hardware kernel correctness → single-GPU target smoke → two-node PP smoke → documentation.
- **Parallelism:** static unit/reference test design and Dwarfstar semantic comparison can run in parallel. Kernel implementation, integration, and hardware validation are ordered.
- **Hardware:** do not attempt the full 284B model load until focused kernel tests pass. Do not run concurrent huge model processes.

## Definition of done

- A verified `gfx1151` environment runs DeepSeek V4 sparse indexer operations without AITER.
- The new Triton selector returns exact, deterministic, contract-compatible top-k indices for prefill and decode.
- Existing AutoRound target routed experts still select `TritonWNA16Experts`.
- The Intel target produces a non-empty fixed-prompt completion with `VLLM_ROCM_USE_AITER=0` on one `gfx1151` GPU.
- The target produces a non-empty fixed-prompt completion in the intended two-node pipeline-parallel deployment without AITER.
- Existing supported AITER and non-ROCm routes remain unchanged.

## Evaluation criteria

- [ ] `git diff --check` exits 0.
- [ ] `.venv/bin/python -m pytest tests/kernels/attention/test_rocm_triton_attn_dsv4.py -v` exits 0 on its applicable environment.
- [ ] The added reference tests for native sparse-indexer top-k exit 0 and assert descending score order, ascending-index tie break, correct local indices, `-1` padding, and masked-index exclusion.
- [ ] The added prefill test covers at least one multi-chunk query range and verifies indices against an explicit PyTorch reference.
- [ ] The added decode test covers at least one paged sequence-length/block-table case and one padded speculative-decode case against reference output.
- [ ] On `gfx1151`, hardware-gated score tests for target-like `[M, 64, 128]` geometry pass against a stated-tolerance reference.
- [ ] On `gfx1151`, hardware-gated top-k tests pass for both a candidate count at or below the one-chunk limit and a count above it, demonstrating exact hierarchical merge behavior.
- [ ] The `gfx1151` route succeeds when launched with `VLLM_ROCM_USE_AITER=0`; a test or observable log proves it did not require `aiter`.
- [ ] Existing AITER eligibility predicates are unchanged except for additive, operation-specific dispatch code; no broad architecture expansion of `rocm_aiter_ops` is present in the diff.
- [ ] `.venv/bin/python -m pytest tests/quantization/test_auto_round.py tests/quantization/test_moe_wna16.py -v` exits 0.
- [ ] On a sufficiently provisioned `gfx1151` runner, a fixed short prompt passed to `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` produces a non-empty completion with automatic INC/AutoRound detection and `VLLM_ROCM_USE_AITER=0`.
- [ ] The single-GPU smoke log includes `TritonWNA16Experts` and the new generic ROCm Triton sparse-indexer selection, and contains neither the current AITER-only exception nor Marlin MoE selection.
- [ ] On the intended dual-`gfx1151` RDMA environment with pipeline parallelism, the same target produces a non-empty fixed-prompt completion with `VLLM_ROCM_USE_AITER=0`.
- [ ] `pre-commit run ruff-check --files <each changed Python file>` exits 0.
- [ ] Documentation states only the validated `gfx1151` initial support boundary and does not imply generic RDNA or performance support.
- [ ] No modified tracked file is outside the sparse-indexer implementation, directly relevant tests, narrowly relevant dispatch/configuration, and associated documentation unless the baseline evidence establishes necessity.

Hardware checks cannot be automated in this container because its Python environment is CUDA-oriented and does not expose the available ROCm device to Torch. A human must review the target runner logs, actual architecture, completion output, and two-node PP run.

## Handoff

1. **Execute** — A clean implementation session on branch `rocm-autoround-deepseek-v4-flash-moe` follows this specification alone. It records the `gfx1151` score baseline before changing dispatch, then implements the narrow Triton top-k and integration path.
2. **Verify** — An independent fresh session runs every applicable evaluation command and the required `gfx1151` kernel, single-GPU, and two-node PP smoke tests. Hardware absence is reported as blocked, never as passing.
3. **Review** — An adversarial fresh review compares the diff to the constraints, emphasizing: no AITER expansion, no HIP/C++ dependency, correct top-k semantics, retained AutoRound/MoE boundaries, and no broad RDNA claim.
4. **Gate** — Walk every evaluation checkbox. Any failed numerical, dispatch, or end-to-end criterion returns to Execute with the exact observed failure.
5. **Ship** — After human review of every changed line and all applicable criteria pass, run the repository duplicate-work checks before creating a PR. The PR description must disclose AI assistance, explain duplicate-work findings, and list test plus hardware validation results.

This is a single integrated execution specification. It has no parallel sibling specifications because unit tests alone cannot establish the desired real-world `gfx1151` serving outcome.

## Implementation status

- [x] Dwarfstar algorithm analysis: top-k semantics, bitonic sort, chunk-and-merge
- [x] Triton top-k module: `vllm/v1/attention/ops/triton_sparse_indexer_topk.py`
  - Single-phase bitonic sort (N ≤ 8192)
  - Chunk-and-merge for N > 8192 (CHUNK_N=4096, MERGE_GROUP=8)
  - Prefill local-index conversion
  - Decode seq-len masking
- [x] gfx1151 dispatch: `vllm/v1/attention/ops/rocm_triton_sparse_indexer.py`
  - Capability predicate (`is_gfx1151_triton_sparse_indexer_available`)
  - Full indexer flow mirroring `rocm_aiter_sparse_attn_indexer`
- [x] Integration: `SparseAttnIndexer.forward_hip()` dispatches to Triton path on gfx1151
- [x] Unit tests: `tests/kernels/test_triton_sparse_indexer_topk.py`
  - Helper function tests (CPU, no GPU needed)
  - GPU-gated correctness tests (ROCm hardware)
  - gfx1151-specific hardware tests
- [ ] gfx1151 hardware kernel correctness validation (blocked: no gfx1151 hardware)
- [ ] Single-GPU end-to-end smoke test (blocked: no gfx1151 hardware)
- [ ] Two-node pipeline-parallel smoke test (blocked: no gfx1151 hardware)
- [ ] Documentation update (post-validation)
