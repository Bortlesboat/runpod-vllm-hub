# RunPod vLLM Hub Template

[![RunPod](https://api.runpod.io/badge/Bortlesboat/runpod-vllm-hub)](https://console.runpod.io/hub/Bortlesboat/runpod-vllm-hub)

This repo is a practical starter for publishing an OpenAI-compatible `vLLM` worker to the RunPod Hub.

Deploy on RunPod Hub:

- https://console.runpod.io/hub/Bortlesboat/runpod-vllm-hub

Good search terms for this listing:

- `vllm`
- `openai-compatible`
- `serverless`
- `qwen`
- `runpod`

It does four things:

- pins a current official `runpod/worker-v1-vllm` base image,
- exposes the `vLLM` knobs that matter most for real deployments,
- adds current Hub metadata in `.runpod/hub.json` and `.runpod/tests.json`,
- keeps the handler small enough to reason about and extend locally.

## Why this shape

RunPod's current Hub flow indexes GitHub releases, expects `hub.json` and `tests.json` in a `.runpod/` directory, and requires a `handler.py`, `Dockerfile`, and `README.md` in the repo. This template keeps the worker-side code minimal and leans on the official RunPod `vLLM` image instead of rebuilding the whole stack from scratch.

The default deployment path is GPU-first and tuned for:

- cheap smoke tests on `ADA_24` (`RTX 4090`),
- cost-aware production on `AMPERE_80` (`A100`),
- higher-end H100 deployment on `ADA_80_PRO`.

## What the handler supports

The handler starts a local `vLLM` OpenAI server inside the worker container and forwards three input shapes:

1. Plain completions:

```json
{
  "input": {
    "prompt": "Reply in one short sentence.",
    "temperature": 0.2,
    "max_tokens": 128
  }
}
```

2. Chat completions:

```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Summarize why H100s are useful for big models."}
    ],
    "temperature": 0.3,
    "max_tokens": 256
  }
}
```

3. Explicit route and body:

```json
{
  "input": {
    "route": "/v1/chat/completions",
    "body": {
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "messages": [
        {"role": "user", "content": "Say hello."}
      ]
    }
  }
}
```

Streaming is intentionally rejected in this template because the standard RunPod serverless request/response path is synchronous.

## Exposed environment variables

The Hub metadata exposes these `vLLM` controls:

- `MODEL_NAME`
- `HF_TOKEN`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `MAX_NUM_SEQS`
- `TENSOR_PARALLEL_SIZE`
- `DTYPE`
- `QUANTIZATION`
- `TOKENIZER_MODE`
- `CONFIG_FORMAT`
- `LOAD_FORMAT`
- `TRUST_REMOTE_CODE`
- `ENABLE_AUTO_TOOL_CHOICE`
- `TOOL_CALL_PARSER`
- `REASONING_PARSER`
- `ENFORCE_EAGER`
- `OPENAI_SERVED_MODEL_NAME_OVERRIDE`
- `DEFAULT_MAX_TOKENS`
- `MAX_CONCURRENCY`

Three presets are included:

- `Smoke Test 125M`
- `Balanced 7B`
- `H100 / 32B`

## Local verification

The local test suite only checks the template logic, not a live GPU deployment:

```bash
python -m unittest discover -s tests -v
```

## Publish checklist

1. Create a new GitHub repo from this directory.
2. Review `.runpod/hub.json` defaults and branding.
3. Create a GitHub release.
4. Submit the repo through the RunPod Hub console.
5. After approval, iterate by cutting new releases instead of force-editing old tags.
