# OpenAI-Compatible vLLM Worker

Deploy a configurable `vLLM` serverless worker on RunPod Hub with:

- OpenAI-style chat and completion routes,
- presets for `RTX 4090`, `A100`, and `H100`-class deployments,
- exposed context-window and memory-tuning knobs,
- a cheap smoke-test profile for Hub validation.

## Input shortcuts

This worker accepts:

- `prompt` for `/v1/completions`
- `messages` for `/v1/chat/completions`
- `route` + `body` for explicit OpenAI-compatible requests

## Good defaults

- `Smoke Test 125M`: fastest validation path
- `Balanced 7B`: sensible one-GPU default
- `H100 / 32B`: larger-context profile for higher-end GPUs

## Main knobs

- `MODEL_NAME`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `MAX_NUM_SEQS`
- `TENSOR_PARALLEL_SIZE`
- `DTYPE`
- `QUANTIZATION`
- `TOKENIZER_MODE`
- `CONFIG_FORMAT`
- `LOAD_FORMAT`

If you hit memory pressure, lower `MAX_MODEL_LEN` first. If you want a faster cold start, use the smoke preset or a smaller default model.
