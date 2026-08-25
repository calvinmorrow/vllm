# Spec: gfx1151 W8A8 Block-FP8 Triton Benchmarking

## Tracking

- Related evidence: `tmp/rocm_deepseek_v4_flash_w8a8_block_fp8_tuning_report.md`
- Related evidence: `tmp/gfx1151_block_fp8_config_discovery_report.md`
- Related handoff: `ROCM_GFX1151_BLOCK_FP8_CONFIG_HANDOFF.md`
- Separate, out-of-scope work: PP/RCCL/RDMA transport tuning; W4A16/MXFP4 MoE; sparse-indexer tuning.

## Goal

Produce reproducible, target-hardware evidence for static configurations of the existing W8A8 Block-FP8 Triton kernel on AMD Radeon 8060S / gfx1151, then install only configurations that are measurably faster, numerically correct, and selected by the real vLLM route.

This matters because the DeepSeek V4 Flash deployment uses the generic fallback for confirmed dense Block-FP8 shapes on gfx1151. The GPU has no native FP8 matrix path for this workload, making the fallback's tile, warp, stage, and grouping choices a material local decode cost.

## Scope

### In scope

- Make the existing Block-FP8 benchmark reproduce the relevant production geometry and layout.
- Benchmark the six runtime-confirmed local `(N,K)` shape families on the gfx1151 node.
- Compare static candidates with the existing generic fallback at defined `M` values.
- Validate candidate output against the existing block-FP8 reference.
- Add only benchmark-backed `AMD_Radeon_8060S` JSON configurations.
- Verify configuration selection, eager execution, graph replay, and a warmed serving workload.
- Keep reproducibility records, runtime inventory, and raw timing results in `tmp/`.

### Out of scope

- New Triton kernels, a bespoke GEMV path, dispatch changes, autotuning in the serving path, or changes to `fp8_utils.py` dispatch/config-selection semantics.
- AITER backend support or priority changes. AITER is not supported for this gfx1151 path.
- W4A16 / MXFP4 routed-MoE tuning, FP8 KV/cache tuning, attention/indexer redesign, or RCCL/NCCL/RDMA work.
- Configurations for inventory-only, unobserved shape `(8192,4096)`.
- Copying MI300, NVIDIA, or third-party configuration values without target-hardware benchmark evidence.
- Modifying model weights, quantization formats, or loading behavior.

### Related specs

The existing sparse-indexer specs may proceed independently. This spec must not wait for their implementation or for the transport work, but its end-to-end outcome must be reprofiled after transport changes land.

## Context

### Repository and environment

- Repository root: `/a0/usr/projects/vllm`
- Python policy: use `uv` and `.venv/bin/python`; never use bare `python3`, `pip`, or `pip install`.
- Target GPU: `AMD Radeon 8060S Graphics`, `gfx1151`.
- Verified stack at discovery time: PyTorch `2.12.0+rocm7.14.0`; HIP `7.14.60850`; `torch.cuda.is_available() == True`; `torch.float8_e4m3fn` available.
- Required library path for this local environment:

```bash
cd /a0/usr/projects/vllm
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

- Current device filename token: `AMD_Radeon_8060S`. Reconfirm it through `get_device_name_as_file_name()` at the beginning of the execution; it controls the config filename.
- Do not install dependencies unless separately approved.

### Existing kernel and static-config contract

The serving route is:

```text
TritonFp8BlockScaledMMKernel.apply_block_scaled_mm
  -> torch.ops.vllm.w8a8_triton_block_scaled_mm_func
  -> w8a8_triton_block_scaled_mm
  -> get_w8a8_block_fp8_configs(N, K, 128, 128)
  -> _w8a8_triton_block_scaled_mm
```

Relevant files:

| Path | Role |
|---|---|
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | Existing W8A8 Block-FP8 wrapper, ROCm scale upcast, fallback config, and filename/nearest-M selection. Do not alter its serving behavior. |
| `benchmarks/kernels/benchmark_w8a8_block_fp8.py` | Existing sweep benchmark; requires focused extension before it can provide acceptable evidence. |
| `tests/kernels/quantization/test_block_fp8.py` | Existing numerical-reference test location. |
| `tests/kernels/quantization/test_block_fp8_shape_telemetry.py` | Existing CPU-oriented telemetry tests; not a performance benchmark. |
| `vllm/model_executor/layers/quantization/utils/configs/` | Destination for only validated static JSON files. |

Config lookup is exact by `(N,K,device-name,block-shape)`:

```text
N=<N>,K=<K>,device_name=<token>,dtype=fp8_w8a8,block_shape=[128,128].json
```

Within an existing file, the JSON is a map from string M keys to configurations. The runtime selects the numerically closest M key:

```python
configs[min(configs.keys(), key=lambda x: abs(x - M))]
```

Therefore an entry measured at one M can affect unlisted M values. A file with only a `"1"` key is not a single-token-only configuration.

The generic fallback currently selected for the confirmed shapes is:

```json
{
  "BLOCK_SIZE_M": 64,
  "BLOCK_SIZE_N": 128,
  "BLOCK_SIZE_K": 128,
  "GROUP_SIZE_M": 32,
  "num_warps": 4,
  "num_stages": 2
}
```

### Production data contract

DeepSeek V4 Flash dense weights on this route are block-scaled W8A8, not an epilogue-scaled GEMV:

```text
A:  [M,K], E4M3 FP8, contiguous
B:  [N,K], E4M3 FP8, logical layout; supplied strides must be honored
As: [M,ceil(K/128)], dynamic activation scales
Bs: [ceil(N/128),ceil(K/128)], weight scales
C:  [M,N], BF16 output
block size: [128,128]
```

The kernel's per-K-block operation is:

```text
accumulator += dot(A[:, kblock], B[nblock, kblock])
               * As[:, kblock] * Bs[nblock, kblock]
```

On ROCm, the wrapper converts E8M0 `As` and `Bs` to contiguous FP32 immediately before launching Triton. The benchmark must measure this production-equivalent scale representation or separately measure and record the conversion boundary. It must never replace blockwise scale application with one global epilogue scale.

`VLLM_ROCM_FP8_PADDING=1` is normally enabled. It can leave the logical B shape unchanged while making its row stride larger than K. The existing benchmark's `B.is_contiguous()` assertion does not reproduce that case and must be corrected before benchmarking padded production layouts. Do not materialize `B.contiguous()` merely to make benchmarking easier.

### Confirmed runtime shape inventory

These are actual local W8A8 Block-FP8 wrapper shapes from both PP workers, not inferred global checkpoint dimensions. All use `[128,128]` blocks and BF16 output.

| Priority | `(N,K)` | Corresponding dense role | Observed M values |
|---:|---:|---|---|
| P0 | `(4096,8192)` | `attn.wo_b` | `1,2,3,4,5,6,7,8,16,69,256` |
| P0 | `(4096,4096)` | shared-expert gate/up projection | `1,2,3,4,5,6,7,8,16,69,256` |
| P0 | `(4096,2048)` | shared-expert down projection | `1,2,3,4,5,6,7,8,16,69,256` |
| P0 | `(32768,1024)` | `attn.wq_b` | `1,2,3,4,5,6,7,8,16,69,256` |
| P1 | `(8192,1024)` | indexer `wq_b` | `1,2,3,4,5,6,7,8,256`; PP0 also records `16,69` |
| P1 | `(1536,4096)` | fused WQA/WKV projection | `1,2,3,4,5,6,7,8,16,69,256` |

Phase evidence is deliberately limited:

- `M=256` appears in model-runner profiling/initialization and is a startup/prefill-style guard.
- `M=1..8` appears during CUDA-graph memory profiling/capture and is the primary graph-captured decode domain.
- `M=16` is explicitly seen during sparse-MLA warm-up.
- `M=69` is a real observed launch but has no request-side phase annotation; classify it as indeterminate until a bounded serving capture correlates it with prefill, mixed, or decode work.

These facts do not establish production frequency or per-shape kernel cost. A small request-correlated inventory is required to prioritize among P0 shapes and correctly classify M=69.

## Requirements

### 1. Preserve a baseline and create an execution record

1. Before modifying tracked files, record `git status --short`, repository revision, device name/token, PyTorch version, HIP version, vLLM revision, `VLLM_ROCM_FP8_PADDING`, graph setting, and intended TP/PP/DP topology in a dated file under `tmp/`.
2. Record the generic fallback configuration as the baseline meta-parameter set.
3. Run a minimal GPU smoke test through `.venv/bin/python` confirming the selected target GPU and FP8 E4M3 allocation/launch capability. Do not claim success based only on import availability.
4. Preserve all raw benchmark output, candidate manifests, and serving logs under a unique ignored directory such as `tmp/gfx1151_block_fp8_bench_<timestamp>/`.

### 2. Capture the final production layout facts

1. During one ordinary warmed deployment run, use the already available opt-in Block-FP8 telemetry or a bounded metadata-only launch record to capture the actual data needed for microbenchmark reproduction.
2. Deduplicate records by module or site, M, N, K, block shape, all operand/output dtypes, B/As/Bs strides, config decision, graph mode, and topology; retain count and first/last occurrence.
3. Never dump tensor values, model-weight data, secrets, or per-request profiler traces.
4. Stop after a representative prefill-plus-decode workload has completed without a new record for a complete workload cycle.
5. Classify each retained record as prefill, decode, warm-up, graph-capture, or indeterminate using explicit launcher/request context. Do not label by M alone.
6. Confirm the actual token returned by `get_device_name_as_file_name()` and the active W8A8 Triton route. Do not assume a warning or a metadata inventory proves the active backend.

### 3. Make the existing benchmark fit for evidence collection

Modify `benchmarks/kernels/benchmark_w8a8_block_fp8.py` minimally and compatibly so it can benchmark an explicit target matrix instead of only its DeepSeek-V3 hard-coded list.

1. Add explicit CLI input for one or more `(M,N,K)` targets; accept repeated flags or a machine-readable manifest. The interface must unambiguously represent all six target pairs and their M values.
2. Add options for warm-up iterations, measured iterations, output dtype, scale dtype/path behavior, output directory or JSON path, device index, and B-row padding/stride reproduction.
3. Preserve existing behavior when no explicit target is provided, unless maintaining it would conflict with correctness.
4. Reproduce logical B `[N,K]` with actual supplied strides. Remove the benchmark-only contiguous-B assertion; use a safe strided allocation/view for padded rows. Do not benchmark a contiguous substitute when production uses padding.
5. Support the ROCm production scale route: either generate E8M0 scales and execute the same upcast used by the wrapper, or generate FP32 scales only after recording that timing is kernel-only and separately validating the full wrapper path. The final decision and methodology must be explicit in benchmark output.
6. Run the actual existing kernel without injecting serving-path autotuning, allocations inside the timed region, host synchronizations inside the candidate kernel measurement beyond the deliberate event timing protocol, or weight transposes.
7. Add a correctness mode that compares each tested candidate against `native_w8a8_block_matmul` for a deterministic seed. It must report max absolute and relative error, output dtype, and pass/fail tolerances.
8. Emit machine-readable output per target/candidate including all parameters, random seed, exact B/scale strides, software/hardware identity, warm-up/sample counts, all raw times, median, p10, p90, mean, standard deviation, coefficient of variation, correctness result, and any compile/resource failure.
9. Do not use the existing average-of-ten timing result as the winner-selection metric.

### 4. Add focused tests for the benchmark and layout contract

1. Extend the nearest existing block-FP8 test suite rather than adding broad unrelated tests.
2. Add/extend tests that establish numerical equivalence for `[128,128]` block scales across more than one K block.
3. Include a padded/non-contiguous B case that has the same logical B values as a contiguous reference, and assert equivalent output.
4. Include a final partial N or K block if the underlying kernel/reference supports it; otherwise record the actual divisibility limitation and keep target coverage to valid shapes.
5. Where ROCm FP8 E8M0 support is available, test the E8M0-to-FP32 scale conversion equivalence used by the benchmark/launcher. Skip with a specific capability reason where the platform lacks required dtype support.
6. Add a CPU-only argument/manifest or statistics test only if it isolates benchmark behavior better than the existing tests. It must not require a GPU in the default unit-test path.

### 5. Define the legal candidate sweep

1. Use only kernel parameter combinations legal for `[128,128]` scaling and compatible with the existing kernel.
2. Start from the repository's existing legal candidate family:
   - `BLOCK_SIZE_M` in `{16,32,64,128,256}`;
   - `BLOCK_SIZE_N` in `{32,64,128,256}` only when scale indexing remains correct;
   - `BLOCK_SIZE_K` in `{64,128}` only when compatible with 128-wide K scale grouping;
   - `GROUP_SIZE_M` in `{1,16,32,64}`;
   - `num_warps` in `{4,8}`;
   - `num_stages` in `{2,3,4,5}`.
3. Prune illegal, non-compiling, resource-exhausted, or numerically incorrect candidates and record the reason. Do not treat a failed candidate as a benchmark result.
4. Start with a reduced screening sweep if needed, then remeasure the generic fallback and the top candidates in a full 30-sample statistical round. The candidate-selection procedure must be recorded and identical across comparisons for a target.
5. Keep the GPU otherwise idle, fix device selection, and record clock/power policy if controllable. Run an interleaved fallback/candidate/fallback order or another documented anti-drift procedure.

### 6. Run the target matrix in controlled stages

1. First tune the P0 shapes in this order unless request-correlated occurrence/cost data changes priority:
   - `(4096,8192)`;
   - `(4096,4096)`;
   - `(4096,2048)`;
   - `(32768,1024)`.
2. Then tune `(8192,1024)` and `(1536,4096)`.
3. For each `(N,K)`, first benchmark primary graph-captured decode M values: `1,2,4,8`.
4. Measure `16`, `69`, and `256` as guard values before installing the first JSON file for that `(N,K)`. Do not omit a guard just because it is less latency-critical: nearest-M selection can route it to a decode-oriented config.
5. Measure `3,5,6,7` only when an adjacent-key boundary is ambiguous, observed configurations have different winners, or a candidate would otherwise be selected for a region that was not tested.
6. Include every M key necessary so nearest-M boundaries intentionally cover all observed M values. Document each region of M values and the config that will be selected.
7. Do not benchmark or install a config for unobserved `(8192,4096)` merely because it is present in weight inventory.

### 7. Apply strict performance and numerical acceptance gates

For every target M/configuration that may be committed:

1. Exclude explicit JIT/compile warm-ups from measurement.
2. Use at least 30 measured, synchronized repetitions in the final round. More samples are required if coefficient of variation or fallback/candidate ordering indicates instability.
3. Retain raw samples and report median, p10, p90, mean, standard deviation, coefficient of variation, and percentage change from the generic fallback.
4. Compare against the exact previously selected configuration: initially the generic fallback; for further refinement, also compare with the current candidate winner.
5. Require the candidate to pass reference correctness and to beat baseline beyond timing noise. Define the concrete threshold before the final run as both:
   - candidate median is lower than fallback median by at least 3%; and
   - the improvement is larger than the combined observed variability, with no reversal in the interleaved confirmation measurement.
6. Reject and do not commit configurations that fail compilation, produce resource errors, have incorrect output, show unstable timings, improve only below the acceptance threshold, or regress a guard M selected by the same nearest-key region.
7. Treat a microbenchmark win as shape-local. Do not represent it as a prefill, whole-model, or universal gfx1151 win.

### 8. Add only validated static configuration files

1. Write one file per accepted local `(N,K)` under `vllm/model_executor/layers/quantization/utils/configs/`.
2. Use the exact verified device token and name pattern:

```text
N=<N>,K=<K>,device_name=AMD_Radeon_8060S,dtype=fp8_w8a8,block_shape=[128,128].json
```

3. Use string M keys and only the existing configuration fields:

```json
{
  "1": {
    "BLOCK_SIZE_M": 0,
    "BLOCK_SIZE_N": 0,
    "BLOCK_SIZE_K": 0,
    "GROUP_SIZE_M": 0,
    "num_warps": 0,
    "num_stages": 0
  }
}
```

The zero values above are schema placeholders, not legal values and must never be committed.

4. Include no speculative shape file, blanket Radeon fallback, or values copied from another architecture.
5. Do not edit `fp8_utils.py` to make a file appear selected. The existing filename and nearest-M behavior are the integration contract.

### 9. Validate integration and serving behavior

1. Run focused numerical tests using the project virtual environment:

```bash
cd /a0/usr/projects/vllm
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python -m pytest tests/kernels/quantization/test_block_fp8.py -v
```

Run any added narrow benchmark/test file too, with the exact command recorded.

2. For each installed file, run a focused harness or actual vLLM invocation that confirms:
   - the W8A8 Triton route is selected;
   - the exact file path is loaded;
   - the old missing-file warning is absent for the covered shape;
   - the selected M key is the intended nearest key;
   - uncovered shapes preserve old fallback/config behavior.
3. Compare one target per installed file in eager mode and intended CUDA-graph mode. Confirm output equivalence and no new dynamic allocation, compilation, synchronization, or autotuning occurs in normal serving execution.
4. Run a warmed, fixed serving workload with unchanged model revision, prompt length, generation length, concurrency, topology, memory setting, graph setting, and transport configuration before/after the config addition.
5. Report M=1 latency and larger-batch throughput separately. Record p50/p90/p99 where the serving harness provides them.
6. Do not claim proportional end-to-end gain from kernel timings; RCCL/NCCL was separately significant in the prior profile.

### 10. Produce a reviewable final report

Write a concise report under `tmp/` containing:

- all environment and source revisions;
- final runtime shape/layout inventory and phase classification;
- benchmark command lines and manifests;
- baseline and candidate configurations;
- raw-results file locations and summary statistics;
- numerical test results and tolerances;
- selected JSON contents and nearest-M coverage regions;
- exact integration/serving commands and before/after outcomes;
- rejected candidates/shapes and reasons;
- explicit limitations, including traffic classification uncertainty and transport overlap.

Do not add diary-like detail or raw timing blobs to tracked documentation. Keep artifacts in `tmp/` and only source/config/test changes in the implementation diff.

## Constraints

- Follow `/a0/AGENTS.md` and `/a0/usr/projects/vllm/AGENTS.md`.
- Use `.venv/bin/python` and `uv` workflow; never use system `python3` or bare pip.
- Do not install dependencies, change `agent.py`/`initialize.py`, create a commit, or push without the required separate approval.
- Do not modify AITER, model loaders, generic fallback behavior, tensor layout policy, quantization semantics, PP/RCCL transport, or MoE/FP4 code.
- Preserve blockwise scaling correctness and supplied B strides; no B transpose, B contiguity materialization, or global scale epilogue shortcut.
- Do not copy source structure, constants, benchmark code, or configuration values from non-Apache third-party repositories.
- Keep Python code at 88 columns and match repository style. Use Google-style docstrings only where a docstring is necessary.
- Do not write config JSON until target-hardware data satisfies the acceptance gate.
- Do not treat start-up compilation timestamps, graph-capture occurrence alone, or checkpoint inventory alone as performance rankings.
- Do not expose model tensor values, credentials, or private deployment data in artifacts.

## Key decisions (confirmed)

| Decision | Chosen option | Rejected alternatives |
|---|---|---|
| Work boundary | One focused spec covering benchmark improvements, measurement, validation, and conditional configuration installation | New kernel work; splitting off measurement from config validation; broad serving optimization |
| Initial shapes | Six runtime-observed local pairs: `(1536,4096)`, `(4096,2048)`, `(4096,4096)`, `(4096,8192)`, `(8192,1024)`, `(32768,1024)` | Inventory-only `(8192,4096)`; global checkpoint-shape assumptions |
| Primary M values | `1,2,4,8` | Tuning only M=1; tuning all observed M without staged prioritization |
| Guard M values | `16,69,256` before file installation | Ignoring non-decode/indeterminate M values despite nearest-M selection |
| Extra small-M values | `3,5,6,7` only when needed by winner/boundary evidence | Unconditionally multiplying the sweep cost |
| Evidence standard | >=30 measured samples, raw samples, median/p10/p90/CV, reference correctness, improvement beyond measured noise | One sample; average-only ten-iteration benchmark; copied configuration values |
| Integration standard | Real vLLM selection check in eager and graph modes plus warmed serving before/after | Microbenchmark-only acceptance |
| Artifact location | Spec in `specs/`; raw data/inventory/reports in ignored `tmp/` | Tracked raw benchmark logs; undocumented ad-hoc results |

## Edge cases and failure modes

| Condition | Required behavior |
|---|---|
| GPU unavailable or wrong device token | Stop before tuning; record the probe failure and do not generate configs. |
| ROCm library resolution failure | Reapply/record the required `LD_LIBRARY_PATH`; do not substitute a system Python runtime. |
| New runtime shape appears | Record it as a candidate, but do not expand scope or add a JSON file without a follow-up decision and measurement plan. |
| `M=69` cannot be classified | Retain it as indeterminate guard coverage; do not claim it is decode. |
| Actual B stride differs from synthetic benchmark layout | Fix the benchmark reproduction first; do not use contiguous B results as a proxy. |
| Candidate fails JIT/resources | Record configuration and error; reject it and continue only with legal candidates. |
| Candidate passes microbenchmark but regresses a nearest-key guard M | Add/test a deliberate M boundary or reject the candidate; do not install the file with an unintended region. |
| No candidate passes 3% plus variability gate | Make no JSON/source configuration change; publish the negative result and retain generic fallback. |
| Serving path does not load a new file | Investigate filename token, local N/K, block shape, route, and install location. Do not alter dispatch to force selection. |
| Eager and graph output differ | Reject the configuration and investigate before serving performance measurement. |
| Existing working tree has unrelated changes | Avoid touching or formatting unrelated files; report only intended files. |

## Execution guidance

- **Recommended model:** lower-cost coding model is suitable once the benchmark layout and command contract above are followed exactly.
- **Agent shape:** one implementation session owns benchmark/test/config changes. A separate verification session must independently run the tests, inspect emitted JSON, and confirm the result matrix. A separate reviewer must inspect the final diff against this spec.
- **Parallelism:** runtime inventory capture can run independently of benchmark-harness implementation. Candidate measurement begins only after the target layout facts are available. Sparse-indexer and transport specs remain independent sibling work.
- **Suggested execution order:**
  1. environment/route/layout inventory;
  2. minimal benchmark/test enhancement;
  3. benchmark screening and 30-sample confirmations;
  4. conditional JSON installation;
  5. focused numerical/integration/graph checks;
  6. warmed serving comparison and report.
- **No commit/PR:** do not create a commit, push, or PR in this spec's execution unless the human explicitly requests it and completes the repository's contribution-policy obligations, including duplicate-work checks and review.

## Definition of done

1. The target environment and exact device filename token have been recorded through the vLLM environment.
2. A bounded runtime inventory captures actual layout/phase facts for each targeted shape and records any unknowns.
3. The benchmark can execute explicit targets, production-equivalent scale handling, BF16 output, and actual B stride/padding without forcing B contiguous.
4. Benchmark output includes raw timing samples and required statistical/correctness fields.
5. Focused tests cover multi-block scale correctness and padded/non-contiguous B equivalence, with ROCm conversion coverage or a capability-specific skip.
6. Every added JSON file is for one of the six confirmed local shapes, has the exact verified Radeon 8060S filename token, and has only statistically accepted M-keyed configurations.
7. Every installed configuration passes numerical validation, removes its corresponding missing-config warning through the real route, and behaves correctly in eager and graph modes.
8. A warmed serving before/after run reports M=1 latency and larger-batch throughput separately using unchanged workload conditions.
9. The final report includes commands, environment, raw artifact paths, result tables, accepted/rejected candidates, and limitations.
10. No dispatch, AITER, MoE, quantization, loader, transport, or speculative configuration change is introduced.

## Evaluation criteria

- [ ] `git diff --check` exits `0` for the implementation diff.
- [ ] `.venv/bin/python -m pytest tests/kernels/quantization/test_block_fp8.py -v` exits `0` on the target environment, with the ROCm library path recorded.
- [ ] Any added focused benchmark/unit tests exit `0` with their exact commands documented.
- [ ] The final benchmark command accepts explicit M/N/K targets and produces machine-readable records containing all raw samples, median, p10, p90, mean, standard deviation, CV, stride metadata, configuration, and correctness status.
- [ ] The final target manifest contains no shape outside the six confirmed runtime pairs unless a separately recorded follow-up decision approves it.
- [ ] For every committed configuration entry, a result record demonstrates reference-correct output and >=30 measured samples after warm-up.
- [ ] For every committed configuration entry, the candidate beats the baseline by at least 3% median and exceeds recorded variability in an interleaved confirmation run.
- [ ] Every committed JSON name exactly matches `N=<N>,K=<K>,device_name=AMD_Radeon_8060S,dtype=fp8_w8a8,block_shape=[128,128].json` using the token revalidated during execution.
- [ ] A real vLLM or focused route run proves each committed file is selected and its prior corresponding missing-file warning is absent.
- [ ] Eager and graph-replay output checks pass for every installed shape/M region.
- [ ] The serving comparison uses the same model revision, workload, topology, graph mode, and transport settings before/after, and reports M=1 latency and larger-batch throughput separately.
- [ ] No files outside the benchmark, focused tests, accepted config JSON files, and ignored `tmp/` artifacts are modified, unless the final report explains and a reviewer approves an exception.
- [ ] A human can verify the hardware-dependent timing claims from the documented command lines and raw output artifacts. This is not fully automatable and must be reviewed on the same gfx1151 node.

## Handoff

1. **Execute** — A clean implementation session works from this file alone on branch `gfx1151-block-fp8-benchmarking`. It first captures target layout facts, then makes the minimal benchmark/test improvements and conditionally writes only validated config files.
2. **Verify** — An independent clean session runs the focused tests, repeats the benchmark confirmation for each proposed winner, checks machine-readable artifacts, and observes route selection on the gfx1151 node.
3. **Review** — An adversarial review compares the diff and artifacts to Constraints, especially stride preservation, E8M0/FP32 scale behavior, nearest-M regions, scope exclusions, and the evidence gate.
4. **Gate** — Walk every Evaluation criteria checkbox. Any missing layout fact, failed numerical result, noisy/insufficient timing win, unintended nearest-M regression, or unverified serving selection returns the work to Execute without JSON installation.
5. **Ship** — Only after the human has reviewed every changed line, requested a commit, and completed the repository contribution-policy requirements. The sparse-indexer sibling specs may execute in parallel; transport changes should be reprofiled after this work but are not a prerequisite to this spec.
