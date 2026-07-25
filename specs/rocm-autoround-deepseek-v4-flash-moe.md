# Spec: ROCm AutoRound MoE support for DeepSeek V4 Flash

## Tracking
- Target checkpoint: `Intel/DeepSeek-V4-Flash-W4A16-AutoRound`
- Model card: <https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound>
- Related completed work: `docs/design/autoround-int4-rocm-support-plan.md` (dense AutoRound GPTQ W4A16 on ROCm)
- Related upstream context: model card references vLLM PR #45645.

## Goal

Make the current branch serve `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` on a supported ROCm system, including its routed MoE layers, by using vLLM's existing Triton WNA16 MoE backend rather than NVIDIA-only Marlin. The result must be correct, explicitly bounded to the checkpoint format actually used by this model, and verified on an MI300-class ROCm runner.

## Scope

### In scope

- Auto-detected INC/AutoRound loading of the target checkpoint.
- Correct MoE quantization-method resolution, weight loading, backend selection, conversion, and invocation for its routed-expert layers on ROCm.
- Reuse of `MoeWNA16Method` and `WNA16MoEBackend.TRITON`; make the smallest adapter/dispatch/kernel-format correction necessary after proving the current path is insufficient.
- Unit coverage for the dispatch and backend-selection contract.
- ROCm Triton WNA16 MoE correctness coverage for the target's INT4 geometry.
- A ROCm end-to-end smoke test using the exact target model, subject to an explicitly provisioned large-memory ROCm runner.
- User-facing documentation that names the supported target format and its limits.

### Exact initial support boundary

The supported AutoRound MoE subset is deliberately limited to the target checkpoint's observed contract:

- `quant_method: "auto-round"` and `packing_format: "auto_round:auto_gptq"`.
- Integer W4A16: `bits: 4`, `data_type: "int"`, `sym: true`, `group_size: 128`.
- No activation ordering: `desc_act` absent in the checkpoint and interpreted as `False` by the adapter.
- BF16 or FP16 activations; the target declares `torch_dtype: "bfloat16"`.
- Routed-expert MoE layers with no expert bias and shapes accepted by the selected Triton WNA16 MoE kernel.
- The target architecture as represented by its pinned `config.json`: `DeepseekV4ForCausalLM`, 43 transformer layers, hidden size 4096, 256 routed experts plus one shared expert, six routed experts per token, and routed MoE intermediate size 2048.
- Tensor parallelism size 1 for the first validated release. Any TP > 1 claim requires a separate successful multi-GPU ROCm validation of expert/scale sharding.
- ROCm hardware for which the current vLLM DeepSeek V4 ROCm implementation and Triton WNA16 MoE backend both report eligibility; MI300-class hardware is the initial required validation target.

The model's AutoRound `extra_config` intentionally leaves embedding, `wo_a`, FFN gates, compressor components, and indexer components at 16-bit float. This work must preserve those per-layer overrides rather than treating the model as universally INT4.

### Out of scope

- A new HIP/C++/AITER W4A16 MoE kernel, Marlin-to-HIP port, kernel performance retuning, or performance guarantees.
- `auto_round:auto_awq`, asymmetric zero-point GPTQ, `desc_act=true`, expert biases, W2/W3/W8 AutoRound, activation quantization, FP4/MXFP4/NVFP4, and checkpoint conversion.
- Broad support claims for arbitrary AutoRound MoE checkpoints or all DeepSeek-family architectures.
- Changes to DeepSeek V4 MLA, sparse-attention, cache, routing, or model architecture behavior not required to load and run the target checkpoint.
- TP > 1 validation or support claims.

## Context

### Target checkpoint facts

Read the live model files before implementation and treat them as the source of truth:

- `https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound/raw/main/config.json`
- `https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound/raw/main/README.md`

At spec-writing time, `config.json` declares `auto-round`, `auto_round:auto_gptq`, 4-bit symmetric integer weights, group size 128, `model_free: true`, and AutoRound version 0.14.0. The model card says it was generated in RTN mode from `deepseek-ai/DeepSeek-V4-Flash` and points to a vLLM support PR. Revalidate both files before changing code because the Hub revision may change.

### Existing implementation path

- `vllm/model_executor/layers/quantization/inc/inc.py` recognizes AutoRound as INC.
- `vllm/platforms/rocm.py` currently excludes `inc` from `RocmPlatform.supported_quantization`. This is the proven first blocker: `ModelConfig._verify_quantization()` resolves the checkpoint to `inc` and calls `current_platform.verify_quantization()` before model construction, weight loading, MoE dispatch, or kernel selection. Narrowly permit `inc` on ROCm for the validated subset; do not represent that platform-list change as proof that every INC format is supported.
- `vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py` resolves AutoRound GPTQ MoE through `_resolve_gptq_moe()`.
- The existing resolver first checks Marlin eligibility, then already constructs `MoeWNA16Method` with a GPTQ `MoeWNA16Config` when Marlin is ineligible. Do **not** assume this needs a new ROCm branch. First prove, with focused tests and a target checkpoint load, whether the fallback is selected and whether its layout is loadable.
- `vllm/model_executor/layers/quantization/moe_wna16.py` makes the WNA16 MoE backend selection and owns parameter construction/loading for the fallback layout.
- `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` prioritizes several backends, including Triton. Its Triton incompatibility checks reject AutoAWQ, activation ordering, and expert bias. `MoeWNA16Config` does not itself make Triton incompatible, while it deliberately makes Marlin, batched Marlin, and emulation incompatible with the fallback layout.
- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` contains `TritonWNA16Experts`; inspect its runtime device/config checks before changing them. ROCm-specific Triton invocation handling also exists in `vllm/model_executor/layers/fused_moe/fused_moe.py`.
- DeepSeek V4 ROCm model support lives under `vllm/models/deepseek_v4/amd/`, including the ROCm MLA implementation. This spec does not alter it unless the target's observed load/inference failure conclusively originates there.

### Existing tests and documentation

- Extend `tests/quantization/test_auto_round.py` for INC AutoRound resolution.
- Extend `tests/quantization/test_moe_wna16.py` for WNA16 MoE backend/layout behavior.
- Reuse a nearby MoE kernel test under `tests/kernels/moe/` if it exposes the public Triton WNA16 experts interface; create a narrowly scoped test there only if none fits.
- Update `docs/features/quantization/inc.md` and `docs/features/quantization/README.md` only after the target-model validation passes.

## Requirements

1. **Remove the proven top-level eligibility blocker before model loading.** `RocmPlatform.supported_quantization` in `vllm/platforms/rocm.py` currently omits `inc`, causing `ModelConfig._verify_quantization()` to reject the target with `inc quantization is currently not supported in rocm` before any layer dispatch occurs. Add `inc` to ROCm eligibility only in conjunction with explicit INC-side validation that restricts support to the target-format AutoRound GPTQ W4A16 subset. Add platform/configuration tests covering target acceptance and rejection of unsupported INC configurations. Do not bypass `verify_quantization()` or add a model-name exception.

2. **Establish the post-gate baseline before editing later layers.** On an appropriate ROCm runner, attempt a target-checkpoint model construction/load with automatic quantization detection and capture the actual selected method/backend plus the first error, if any. Inspect representative checkpoint tensors and their loader metadata for one routed-expert layer (`w13`, `w2`, scales, zero-points, and group-index metadata if present). Record whether the current `_resolve_gptq_moe()` fallback reaches `MoeWNA16Method` and which incompatibility/shape/layout condition blocks it.

3. **Preserve the existing correct dispatch where it is already sufficient.** If the post-gate baseline proves the existing Marlin-ineligible fallback creates `MoeWNA16Method` and selects Triton on ROCm, do not add redundant ROCm-only dispatch. Add regression coverage for that behavior and fix only the later failing layer. If a dispatch change is necessary, make it narrowly enforce that the supported AutoRound GPTQ subset reaches `MoeWNA16Method` on ROCm without invoking or silently selecting Marlin.

4. **Make backend choice explicit and safe.** For an eligible target-like routed MoE layer on ROCm, backend selection must resolve to `WNA16MoEBackend.TRITON` and `TritonWNA16Experts`. An unavailable or explicitly requested Marlin backend must fail with an actionable error; it must not be substituted silently. Preserve existing CUDA Marlin selection and CPU/XPU behavior.

5. **Prove and preserve checkpoint layout semantics.** Verify that `MoeWNA16Method.create_weights()`, its weight loader, and `convert_to_wna16_moe_kernel_format()` preserve the target checkpoint's GPTQ sequential int4 packing and grouped scale meaning. The Triton-format tensors must have the expected orientation, dtype/view, and dimensions for fused `w13` and `w2`. Add a deterministic layout test using synthetic target-shaped-or-smaller-but-equivalent data that catches nibble-order or transpose errors. Do not add AutoRound format detection inside the kernel oracle.

6. **Validate Triton WNA16 MoE numerical correctness on ROCm.** Add or extend a public-interface test that uses symmetric group-128 INT4 GPTQ-style expert weights and BF16 and/or FP16 activations. Cover both decode-like and prefill-like token counts and the target-relevant expert configuration where practical. Compare the Triton result against explicit dequantization plus a PyTorch routed-MoE reference. State numerical tolerances in the test. The test must run on ROCm when a supported GPU is present and skip only for absent/ineligible hardware with a precise reason.

7. **Validate the target end to end.** On an MI300-class ROCm runner with sufficient model and KV-cache capacity, load `Intel/DeepSeek-V4-Flash-W4A16-AutoRound`, generate from a fixed short prompt with `temperature=1.0` and `top_p=1.0`, and assert a non-empty completion. Use an existing logging/test seam or a narrowly added observable seam to assert that routed MoE uses Triton WNA16, never Marlin. Use automatic INC/AutoRound detection; do not require users to rewrite checkpoint metadata. Keep the smoke configuration intentionally small (e.g., a short max model length and one request) except where DeepSeek V4 requires a larger setting to initialize.

8. **Keep failure boundaries observable.** Unit-test rejection or explanatory selector failure for at least one unsupported property from each class: `auto_round:auto_awq`, non-symmetric weights/zero points, activation ordering, unsupported expert bias, and an ineligible shape. Do not broaden the backend contract merely to make these cases load.

9. **Document only validated support.** After successful target validation, document the exact supported AutoRound MoE ROCm subset, name the target checkpoint as the initial validated model, state the Triton dependency and TP=1 boundary, and list exclusions. Do not imply that all AutoRound formats or all DeepSeek V4 models are supported.

## Constraints

- Reuse the existing Triton WNA16 MoE infrastructure. A new native ROCm kernel is explicitly excluded.
- The ROCm platform allowlist must not become an unconditional claim that all `inc` models work. Gate the `inc` allowlist change with target-format validation in the INC configuration/dispatch path and preserve clear failures for unsupported formats.
- Do not weaken global Marlin capability validation or change CUDA Marlin behavior.
- Keep checkpoint-format interpretation in the INC quantization adapter and layout conversion in the WNA16 MoE path; do not create target-model-name special cases in a kernel.
- Preserve `INCConfig` handling of the target's 16-bit `extra_config` overrides.
- Follow repository rules: Python through `uv` and `.venv/bin/python`; maintain 88-character Python line length; use existing fixtures/helpers; do not install dependencies without approval.
- Do not add a huge checkpoint to normal unit-test assets or require network/model download for standard CPU-only unit suites.

## Key decisions (confirmed)

- **Primary end-to-end target:** `Intel/DeepSeek-V4-Flash-W4A16-AutoRound`, not a proxy MoE model.
- **Scope shape:** one focused spec; generic mechanisms are included only as required to serve the target.
- **MoE backend:** reuse the existing Triton WNA16 MoE backend on ROCm.
- **Format boundary:** derived from the target model configuration: symmetric GPTQ-style AutoRound W4A16 with group size 128 and no activation ordering.
- **Delivery strategy:** establish whether current fallback routing already works before modifying dispatch; do not encode an unproven routing hypothesis.

## Edge cases & failure modes

- **Marlin unavailable on ROCm:** expected normal path for the target is the WNA16 fallback followed by Triton selection; no Marlin construction must be required.
- **Triton backend unavailable or target geometry rejected:** fail at selection with backend rejection reasons; never fall back to an incompatible layout/backend or run unquantized experts silently.
- **Target weight layout differs from assumed GPTQ sequential packing:** stop before enabling support, add a fixture that reproduces the observed layout, and make the smallest loader/conversion fix with a nibble-level regression test.
- **A model `extra_config` entry marks a module 16-bit:** it must stay on the normal unquantized/16-bit route and must not enter WNA16 MoE/linear handling.
- **`desc_act=true`, zero-point/asymmetric, AutoAWQ, or expert bias:** reject through the existing selector contract with the specific unsupported reason.
- **Large-model resource shortage:** skip only the end-to-end test with an explicit capacity/hardware reason; unit and layout coverage must still run. Do not call a skipped large-model test proof of support.
- **TP > 1:** reject support claims pending an independently validated multi-GPU follow-up.

## Execution guidance

- **Recommended model:** lower-cost coding model (e.g. Sonnet) after initial target checkpoint metadata inspection; escalate only if the baseline exposes ambiguity in DeepSeek V4 loader/model integration.
- **Agent shape:** a single implementation session is appropriate. It should make one ordered pass: baseline inspection, focused unit/layout tests, smallest code fix, ROCm kernel test, target integration test, documentation.
- **Parallelism:** unit-test/design inspection and target checkpoint metadata inspection can proceed in parallel. Code changes depend on the baseline result because a dispatch change may be unnecessary.
- **Do not start with a broad implementation.** First determine whether the existing `_resolve_gptq_moe()` fallback already gives the target `MoeWNA16Method`; the likely work may be backend selection, loader layout, kernel eligibility, or test coverage rather than resolver logic.

## Definition of done

- The exact target checkpoint loads on the designated MI300-class ROCm environment through normal AutoRound/INC detection and generates a non-empty completion from a fixed short prompt.
- Target routed MoE layers select `WNA16MoEBackend.TRITON`/`TritonWNA16Experts`; no Marlin MoE method/kernel is constructed or selected.
- A ROCm correctness test passes against a dequantized reference for target-format W4A16 grouped MoE data.
- Unit tests prove the supported route, fallback/selection behavior, layout conversion, and key rejection boundaries.
- No behavior regression is introduced for CUDA Marlin or existing CPU/XPU quantization paths.
- Documentation accurately states the target-specific ROCm AutoRound MoE support boundary and exclusions.

## Evaluation criteria

- [ ] `curl -L --fail https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound/raw/main/config.json` confirms the implementation target still has `quant_method=auto-round`, `packing_format=auto_round:auto_gptq`, `bits=4`, `group_size=128`, and `sym=true`; any changed value is assessed before implementation proceeds.
- [ ] `.venv/bin/python -m pytest tests/quantization/test_auto_round.py tests/quantization/test_moe_wna16.py -v` exits 0.
- [ ] The added unit tests assert that a target-like ROCm MoE configuration selects `MoeWNA16Method`/Triton without Marlin and that a forced/unavailable Marlin route raises an actionable error.
- [ ] A deterministic test proves the expected transpose/view and all synthetic packed INT4 nibbles/scales survive the target-format conversion path.
- [ ] On supported ROCm hardware, the added or extended WNA16 Triton MoE correctness test exits 0 and compares output with an explicit dequantized reference using test-stated tolerances.
- [ ] On the designated MI300-class ROCm runner, a short offline `vllm_runner` or `vllm serve` smoke test for `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` generates a non-empty completion and records `TRITON` WNA16 MoE selection.
- [ ] The end-to-end run does not report Marlin selection or instantiate a Marlin MoE path for routed experts.
- [ ] `pre-commit run ruff-check --files <changed Python files>` exits 0, and any targeted test added under `tests/kernels/moe/` exits 0 on its applicable environment.
- [ ] No modified file is outside the minimal implementation, tests, and two documentation paths unless the baseline evidence names a necessary additional owner.
- [ ] `docs/features/quantization/inc.md` and `docs/features/quantization/README.md` name only the verified symmetric AutoRound GPTQ W4A16 ROCm MoE subset, target model, Triton backend, and TP=1 limitation.
- [ ] A human reviewer verifies the large-model smoke-test logs and confirms sufficient ROCm VRAM was available; this resource condition cannot be inferred from a unit test.

## Handoff

The branch name is `rocm-autoround-deepseek-v4-flash-moe`.

1. **Execute** — A clean coding session implements the Requirements from this file alone. It first records baseline target-load evidence, then modifies only the demonstrated failing layer or adds regression coverage if no behavior change is required.
2. **Verify** — An independent fresh session runs the commands in Evaluation criteria, including the MI300-class target smoke test, and confirms the Definition of Done is observable. The hardware test must be reported as unavailable rather than fabricated if no qualifying runner exists.
3. **Review** — An adversarial fresh review compares the diff with Constraints, especially preserving CUDA Marlin behavior, maintaining format/layout boundaries, and avoiding model-specific kernel special cases.
4. **Gate** — Walk every Evaluation criteria checkbox. Any failed item returns to Execute with the observed failure and no broadened support claim.
5. **Ship** — After human review of every changed line and successful relevant tests, commit on `rocm-autoround-deepseek-v4-flash-moe`. A future PR must follow the repository's duplicate-work checks and AI-assistance disclosure requirements.

This is a single ordered spec; no sibling spec runs in parallel. The internal metadata-inspection and unit-test work may run concurrently at the start, but all code changes wait for the baseline finding.
