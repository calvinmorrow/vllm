# Intel Quantization Support

[AutoRound](https://github.com/intel/auto-round) is Intel’s advanced quantization algorithm designed for large language models(LLMs). It produces highly efficient **INT2, INT3, INT4, INT8, MXFP8, MXFP4, NVFP4**, and **GGUF** quantized models, balancing accuracy and inference performance. AutoRound is also part of the [Intel® Neural Compressor](https://github.com/intel/neural-compressor). For a deeper introduction, see the [AutoRound step-by-step guide](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md).

## Key Features

✅ Superior Accuracy Delivers strong performance even at 2–3 bits [example models](https://huggingface.co/collections/OPEA/2-3-bits)

✅ Fast Mixed `Bits`/`Dtypes` Scheme Generation Automatically configure in minutes

✅ Support for exporting **AutoRound, AutoAWQ, AutoGPTQ, and GGUF** formats

✅ **10+ vision-language models (VLMs)** are supported

✅ **Per-layer mixed-bit quantization** for fine-grained control

✅ **RTN (Round-To-Nearest) mode** for quick quantization with slight accuracy loss

✅ **Multiple quantization recipes**: best, base, and light

✅ Advanced utilities such as immediate packing and support for **10+ backends**

## Supported Recipes on Intel Platforms

On Intel platforms, AutoRound recipes are being enabled progressively by format and hardware. Currently, vLLM supports:

- **`W4A16`**: weight-only, 4-bit weights with 16-bit activations
- **`W8A16`**: weight-only, 8-bit weights with 16-bit activations

Additional recipes and formats will be supported in future releases.

## Quantizing a Model

### Installation

```bash
uv pip install auto-round
```

### Quantize with CLI

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --scheme W4A16 \
    --format auto_round \
    --output_dir ./tmp_autoround
```

### Quantize with Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"
autoround = AutoRound(model_name, scheme="W4A16")

# the best accuracy, 4-5X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

# 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
# format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format="auto_round")
```

## Deploying AutoRound Quantized Models in vLLM

```bash
vllm serve Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

## Evaluating the Quantized Model with vLLM

```bash
lm_eval --model vllm \
  --model_args pretrained="Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 128
```

## ROCm Support

vLLM supports dense symmetric AutoRound GPTQ W4A16 (`auto_round:auto_gptq`)
inference on AMD ROCm GPUs through Triton/RDNA W4A16 kernels.

### Supported on ROCm

- **W4A16**: 4-bit symmetric GPTQ (`bits=4`, `sym=true`, `desc_act=false`)
- **W8A16**: 8-bit symmetric GPTQ (`bits=8`, `sym=true`)
- Group sizes: `-1`, `32`, `64`, `128`, `256`
- Activation dtypes: FP16, BF16
- Tensor parallelism: TP=1 (TP>1 requires additional validation)

### Not yet supported on ROCm

- `auto_round:auto_awq` packing format
- Asymmetric quantization (`sym=false`)
- INT2 / INT3 / INT8 weight-only quantization
- Activation reordering (`desc_act=true`)
- Forced `backend="marlin"` (Marlin is NVIDIA-only; use `backend="auto"`)

### MoE (Mixture of Experts) on ROCm

vLLM supports AutoRound MoE models on ROCm through the Triton WNA16 MoE
backend. The following MoE configuration is supported:

- **Packing format**: `auto_round:auto_gptq` only
- **W4A16**: 4-bit symmetric INT4 (`bits=4`, `sym=true`, `group_size=128`)
- **No activation ordering**: `desc_act` must be absent or `false`
- **No expert bias**: expert layers must not use bias terms
- **TP=1**: tensor parallelism > 1 requires additional validation
- **Initial validated model**: `Intel/DeepSeek-V4-Flash-W4A16-AutoRound`

#### Example: ROCm MoE inference

```bash
vllm serve Intel/DeepSeek-V4-Flash-W4A16-AutoRound \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

### Example: ROCm inference

```bash
vllm serve OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```
