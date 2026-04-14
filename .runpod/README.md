# OpenAI-Compatible vLLM Worker

Deploy a configurable `vLLM` serverless worker on RunPod Hub for OpenAI-style chat and completion APIs.

## Best for

- teams that want a reusable `vLLM` starter instead of a model-specific worker,
- users comparing `RTX 4090`, `A100`, and `H100` presets,
- fast iteration on ungated Hugging Face models such as Qwen instruct variants.

## Request shapes

- `prompt` for `/v1/completions`
- `messages` for `/v1/chat/completions`
- `route` + `body` for explicit OpenAI-compatible requests

## Main knobs

- `MODEL_NAME`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `MAX_NUM_SEQS`
- `TENSOR_PARALLEL_SIZE`
- `DTYPE`
- `QUANTIZATION`

If you hit memory pressure, lower `MAX_MODEL_LEN` first. If you want the fastest cold start, use the smoke-test preset or a smaller default model.
