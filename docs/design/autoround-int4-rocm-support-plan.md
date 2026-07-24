# Plan: AutoRound INT4 inference on ROCm without Marlin

## Goal

Enable dense Intel AutoRound `W4A16` checkpoints whose packing format is
`auto_round:auto_gptq` to run on AMD ROCm through vLLM's existing mixed-precision
W4A16 kernel-selection path. The ROCm implementation must select the existing
Triton W4A16 kernel (or an earlier eligible ROCm W4A16 kernel) rather than
requiring NVIDIA-only Marlin.

This is deliberately an implementation plan, not an implementation. It is based
on the checkout at commit `a49d37c6b` and should be revalidated against the
implementation branch before coding.

## Why the current path excludes ROCm

AutoRound is detected as the `inc` quantization method when
`quant_method` is `auto-round`:

1. `vllm/model_executor/layers/quantization/inc/inc.py` constructs `INCConfig`.
2. The default `packing_format` is `auto_round:auto_gptq`; typical symmetric
   AutoRound INT4 checkpoints use `bits: 4`, `group_size: 128`, and `sym: true`.
3. `INCWna16Scheme` in
   `vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py`
   routes non-CPU/XPU dense layers to `INCWNA16LinearScheme`.
4. `INCWNA16LinearScheme._build_gptq_method()` in
   `vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py`
   only instantiates `AutoGPTQLinearMethod` after `check_marlin_supported()`
   succeeds. On ROCm that capability check is the gate that currently prevents
   the AutoRound path from reaching a usable dense linear kernel.
5. `AutoGPTQLinearMethod.__init__()` in
   `vllm/model_executor/layers/quantization/auto_gptq.py` additionally calls
   `verify_marlin_supported()`, making Marlin an unconditional construction-time
   requirement even though its `create_weights()` method uses
   `choose_mp_linear_kernel()`.

The existing ROCm kernel selector already contains the desired non-Marlin
fallback. `vllm/model_executor/kernels/linear/__init__.py` prioritizes
`RDNA3W4A16LinearKernel`, `RDNAHybridW4A16LinearKernel`, and then
`TritonW4A16LinearKernel` for `PlatformEnum.ROCM`. The Triton implementation in
`vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py` supports
symmetric GPTQ `uint4b8`, grouped scales including group size 128, FP16/BF16
activations, and converts the normal GPTQ parameter layout to its ROCm kernel
layout during weight processing.

## Scope

### In scope

- Dense W4A16 AutoRound checkpoints with all of the following properties:
  - `quant_method: "auto-round"`
  - `packing_format: "auto_round:auto_gptq"`, including the absent-field default
  - 4-bit, symmetric (`sym: true`) GPTQ sequential packing
  - group size `-1`, `32`, `64`, `128`, or `256` when the selected ROCm kernel
    accepts the rank-local shape
  - FP16 or BF16 activations
- ROCm devices for which the existing W4A16 selector finds an eligible kernel.
  MI300/Triton is the initial validated target; existing RDNA-specific kernels
  may be selected when they report support.
- Correct selection, loading, and inference for tensor-parallel size 1 first.
- Unit, kernel-correctness, and ROCm integration coverage plus user-facing
  compatibility documentation.

### Out of scope for this change

- Writing or porting Marlin to HIP/ROCm.
- A new AutoRound serialization format or a checkpoint conversion tool.
- `auto_round:auto_awq` support on ROCm. It has different packing and currently
  uses AutoAWQ/Marlin-oriented behavior; plan it separately after the GPTQ path
  is proven.
- AutoRound INT2/INT3/INT8, activation quantization, MXFP4/MXFP8/NVFP4, or
  GGUF.
- MoE AutoRound support. Although the WNA16 MoE oracle has a Triton backend,
  its layout conversion and activation constraints require a separate validated
  work item.
- Performance retuning beyond avoiding the Marlin-only rejection.

## Proposed implementation

### 1. Make dense AutoRound GPTQ use kernel capability, not Marlin capability

Update these files together:

| File | Required edit |
| --- | --- |
| `vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py` | In `INCWNA16LinearScheme._build_gptq_method()`, retain the existing CUDA/Marlin behavior, but add a ROCm-compatible route for `(bits=4, sym=true)` that constructs the GPTQ dense linear method without requiring `check_marlin_supported()`. Do not let `backend: "auto"` imply that Marlin itself must be available on ROCm. Explicit `marlin` backends must continue to fail clearly on ROCm rather than silently selecting Triton. |
| `vllm/model_executor/layers/quantization/auto_gptq.py` | Refactor `AutoGPTQLinearMethod` so construction does not unconditionally call `verify_marlin_supported()`. Preserve Marlin validation only when Marlin is selected, and let `create_weights()` select a supported `MPLinearKernel` via its existing `choose_mp_linear_kernel()` call. The resulting selector error must include the rejected-kernel reasons if no ROCm kernel can implement the layer. |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py` | Do not weaken Marlin capability checks globally. Only adjust call sites or introduce a narrowly named helper if necessary so NVIDIA Marlin contracts remain intact. |

The smallest correct design is to reuse `AutoGPTQLinearMethod`; do **not**
create a second AutoRound-specific ROCm GEMM implementation. Its parameter
creation already produces GPTQ `qweight`, `scales`, `qzeros`, and `g_idx`
objects that the W4A16 selector can consume. `TritonW4A16LinearKernel` already
repackages the GPTQ weight data at load time and ignores `qzeros` for symmetric
`uint4b8`, using the type's fixed zero bias instead.

### 2. Preserve exact format and backend semantics

Implement explicit capability rules and error behavior:

- ROCm auto-selection is valid only for `auto_round:auto_gptq`, 4-bit,
  symmetric, no activation reordering (`desc_act=False` for this AutoRound
  adapter), compatible group size, and selector-supported partition shapes.
- Treat a missing `packing_format` as the existing
  `auto_round:auto_gptq` default; do not change `INCConfig.from_config()`.
- Keep `auto_round:auto_awq` on its current path and report it as unsupported
  on ROCm. Do not attempt to reinterpret AWQ nibbles as GPTQ nibbles.
- If a user forces `backend: "marlin"` or `"gptq:marlin"` on ROCm, raise an
  actionable error naming the unavailable backend. A forced non-Marlin backend
  should either be honored when implemented or fail; never override an explicit
  request with an unexpected kernel.
- Keep CUDA selection and all current CPU/XPU behavior unchanged.
- Preserve normal model-loader and tensor-parallel parameter metadata. In
  particular, test that the ROCm repack receives the expected GPTQ sequential
  nibble ordering, rather than merely accepting a shape-compatible tensor.

### 3. Validate the existing Triton kernel contract before extending it

No Triton kernel change is expected for the initial dense symmetric GPTQ scope.
Before relying on that conclusion, add or extend coverage for these observable
properties of `TritonW4A16LinearKernel`:

- A generated GPTQ-packed int4 weight, grouped scales, and BF16/FP16 activation
  produce the same result as explicit dequantization plus `torch.matmul` within
  a documented numerical tolerance.
- Group size 128 is covered, as that is the primary AutoRound checkpoint
  configuration.
- The load-time repack converts the AutoGPTQ parameter layout to the kernel's
  `[K, N // 8]` layout and preserves all nibbles in sequential GPTQ order.
- Unsupported `g_idx`, non-divisible output width, non-divisible rank-local K,
  unsupported group size, and unsupported activation dtype fail through the
  kernel selector with an explanatory reason.

If this validation reveals a real layout mismatch, make the smallest change in
`triton_w4a16.py` and add a regression test. Do not add format detection to the
kernel: checkpoint-format interpretation belongs in the quantization adapter.

## Test plan

### Unit tests

Extend `tests/quantization/test_auto_round.py` rather than creating another
AutoRound test module.

1. Add a platform-mocked ROCm test for a symmetric `auto_round:auto_gptq`
   layer. Assert that `INCWna16Scheme`/`INCWNA16LinearScheme` returns the dense
   GPTQ method without calling the Marlin-only rejection path.
2. Assert that a ROCm `backend="auto"` configuration has the same result, while
   `backend="marlin"` and `backend="gptq:marlin"` produce a clear unsupported
   error.
3. Retain/extend the existing unsupported-config tests for asymmetric GPTQ,
   AWQ, INT2, and non-ROCm platforms so the scope boundary is enforced.
4. Add an `AutoGPTQLinearMethod` test showing that its construction no longer
   assumes Marlin, while unsupported layers still fail at kernel selection.

### Kernel tests

Create `tests/kernels/quantization/test_triton_w4a16.py` if no nearby W4A16
kernel test is present when implementation begins. Test the public kernel
interface, not Triton internals:

- Parameterize FP16 and BF16, `M` values covering decode and prefill, and group
  sizes `-1`, `32`, `64`, `128`, and `256` where legal.
- Generate symmetric GPTQ sequential-packed int4 tensors, compare against a
  dequantized PyTorch reference, and assert that the tolerance is documented
  in the test.
- Run only when `current_platform.is_rocm()` and a supported ROCm GPU is
  available; retain a small platform-independent layout/repack unit test where
  feasible.

### ROCm end-to-end test

Add a ROCm-marked model test/CI target using the small public checkpoint
`OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc`. Its config is a representative
`bits=4`, `group_size=128`, `sym=true`, `quant_method="auto-round"` checkpoint
and defaults to `auto_round:auto_gptq`.

The test must:

1. Create an `LLM`/`vllm_runner` with the checkpoint on ROCm and generate from
   a fixed prompt.
2. Assert a non-empty completion and inspect the selected dense kernel through
   the existing logging/test seam, confirming it is a ROCm W4A16 kernel and not
   a Marlin kernel.
3. Exercise `--tensor-parallel-size 1` first. Add TP > 1 only after verifying
   scale/qzero sharding for row-parallel layers on at least two AMD GPUs.
4. Skip with a precise reason only when ROCm hardware or the test model is not
   provisioned; do not skip merely because the platform is ROCm.

Run these commands in the project's existing environment:

```bash
.venv/bin/python -m pytest tests/quantization/test_auto_round.py -v
.venv/bin/python -m pytest tests/kernels/quantization/test_triton_w4a16.py -v
.venv/bin/python -m pytest tests/quantization/test_auto_round.py -v \
  -k 'auto_round and rocm'
pre-commit run ruff-check --files \
  vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py \
  vllm/model_executor/layers/quantization/auto_gptq.py \
  tests/quantization/test_auto_round.py \
  tests/kernels/quantization/test_triton_w4a16.py
```

On an MI300 ROCm runner, also record a one-prompt `vllm serve` or offline
inference smoke test using the same model. Compare a fixed greedy output or
logits with a known-good CUDA/CPU reference only if tokenization and runtime
versions are pinned; otherwise require successful generation and the kernel
selection assertion rather than brittle text equality.

## Documentation changes after implementation

| File | Required update |
| --- | --- |
| `docs/features/quantization/inc.md` | Add a ROCm subsection that names the supported initial subset: dense, symmetric AutoRound GPTQ W4A16. Show a ROCm invocation and state the requirements and exclusions. Change the current platform wording so it does not imply all AutoRound formats work everywhere. |
| `docs/features/quantization/README.md` | Add or update the AMD ROCm compatibility-chart entry for the supported AutoRound/INC W4A16 subset. Include a footnote that support is format- and kernel-eligibility-dependent. |

Document that `auto_round:auto_awq`, MoE, non-symmetric checkpoints,
activation-order checkpoints, and shapes rejected by the selected ROCm W4A16
kernel are not part of this first release.

## Risks and decision gates

| Risk | Required mitigation / gate |
| --- | --- |
| AutoRound GPTQ files use a packing variation not accepted by the existing GPTQ repack | Download the exact fixture checkpoint and inspect `qweight`, `scales`, `qzeros`, and `g_idx` metadata before coding. A nibble-level reference test must pass before enabling ROCm. |
| An eager Marlin check is relied on by CUDA callers | Keep Marlin verification when Marlin is selected; add CUDA-focused regression coverage or run the existing AutoGPTQ suite on CUDA where available. |
| ROCm selector accepts some but not all partition shapes | Rely on `MPLinearLayerConfig`/`choose_mp_linear_kernel` for final eligibility and propagate its failure reasons. Do not claim universal ROCm support. |
| Tensor-parallel scale and qzero sharding changes semantics | Ship TP=1 support first. Gate TP>1 on a multi-GPU ROCm test that covers row- and column-parallel layers. |
| The objective expands into AutoAWQ or MoE | Split it into separate design/implementation work; neither has the same dense GPTQ layout and selection contract. |

## Definition of done

- An unmodified representative symmetric AutoRound GPTQ INT4 checkpoint loads
  and generates on a supported ROCm GPU with `--quantization inc` or automatic
  AutoRound detection.
- The selected dense kernel is one of the ROCm W4A16 kernels, never Marlin.
- Unsupported AutoRound formats and forced Marlin requests fail predictably
  with actionable errors.
- New unit and kernel-correctness tests pass, and a provisioned ROCm integration
  test passes.
- CUDA, CPU, and XPU AutoRound regressions relevant to the changed dispatch
  still pass.
- `docs/features/quantization/inc.md` and the quantization compatibility chart
  accurately state the initial ROCm support boundary.

## Research references

- vLLM AutoRound W4A16 introduction: [PR #39778](https://github.com/vllm-project/vllm/pull/39778)
- vLLM AutoRound/INC RFC: [issue #40675](https://github.com/vllm-project/vllm/issues/40675)
- Existing ROCm W4A16 Triton kernel API documentation: [triton_w4a16](https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16/)
- AMD ROCm vLLM GPTQ serving example: [ROCm vLLM optimization guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)
- Intel AutoRound project and export formats: [intel/auto-round](https://github.com/intel/auto-round)
- Representative AutoRound GPTQ checkpoint: [OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc](https://huggingface.co/OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc)
