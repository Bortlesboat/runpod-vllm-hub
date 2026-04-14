import atexit
import json
import os
import subprocess
import threading
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_TOKENS = 256
DEFAULT_HEALTH_TIMEOUT_SECONDS = 900
DEFAULT_REQUEST_TIMEOUT_SECONDS = 1800
DEFAULT_VLLM_PORT = 8000

_SERVER_LOCK = threading.Lock()
_SERVER = None
_REQUEST_LIMIT = max(1, int(os.environ.get("MAX_CONCURRENCY", "30")))
_REQUEST_SEMAPHORE = threading.Semaphore(_REQUEST_LIMIT)


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    raw = _strip_or_none(env.get(name))
    return int(raw) if raw is not None else default


def _env_bool(env: dict[str, str], name: str, default: bool = False) -> bool:
    raw = _strip_or_none(env.get(name))
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def normalize_route(route: str) -> str:
    route = route.strip()
    if not route:
        raise ValueError("`route` cannot be empty.")
    if route == "/health":
        return route
    if not route.startswith("/"):
        route = f"/{route}"
    if not route.startswith("/v1/"):
        route = f"/v1/{route.lstrip('/')}"
    return route


def normalize_input(
    job_input: dict[str, Any],
    *,
    default_model: str,
    default_max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(job_input, dict):
        raise TypeError("RunPod job input must be a JSON object.")

    if job_input.get("healthcheck"):
        return "/health", {}

    if "route" in job_input or "body" in job_input:
        route = normalize_route(str(job_input.get("route", "")))
        body = job_input.get("body", {})
        if not isinstance(body, dict):
            raise TypeError("`body` must be a JSON object when `route` is supplied.")
        body = dict(body)
    elif "messages" in job_input:
        route = "/v1/chat/completions"
        body = dict(job_input)
    elif "prompt" in job_input:
        route = "/v1/completions"
        body = dict(job_input)
    else:
        raise ValueError(
            "Provide either `prompt`, `messages`, or both `route` and `body`."
        )

    if body.get("stream"):
        raise ValueError("Streaming responses are not supported in this template.")

    body.setdefault("model", default_model)
    body.setdefault("max_tokens", default_max_tokens)
    return route, body


def build_vllm_command(env: dict[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    model_name = _strip_or_none(env.get("MODEL_NAME")) or DEFAULT_MODEL_NAME
    port = str(_env_int(env, "VLLM_PORT", DEFAULT_VLLM_PORT))

    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        port,
        "--model",
        model_name,
    ]

    value_flags = {
        "TOKENIZER_NAME": "--tokenizer",
        "TOKENIZER_MODE": "--tokenizer-mode",
        "CONFIG_FORMAT": "--config-format",
        "LOAD_FORMAT": "--load-format",
        "MAX_MODEL_LEN": "--max-model-len",
        "GPU_MEMORY_UTILIZATION": "--gpu-memory-utilization",
        "MAX_NUM_SEQS": "--max-num-seqs",
        "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
        "DTYPE": "--dtype",
        "QUANTIZATION": "--quantization",
        "DOWNLOAD_DIR": "--download-dir",
        "TOOL_CALL_PARSER": "--tool-call-parser",
        "REASONING_PARSER": "--reasoning-parser",
        "OPENAI_SERVED_MODEL_NAME_OVERRIDE": "--served-model-name",
    }

    for env_key, flag in value_flags.items():
        value = _strip_or_none(env.get(env_key))
        if value is not None:
            command.extend([flag, value])

    if _env_bool(env, "TRUST_REMOTE_CODE"):
        command.append("--trust-remote-code")

    if _env_bool(env, "ENABLE_AUTO_TOOL_CHOICE"):
        command.append("--enable-auto-tool-choice")

    if _env_bool(env, "ENFORCE_EAGER"):
        command.append("--enforce-eager")

    return command


def _read_json_response(response) -> dict[str, Any]:
    payload = response.read().decode("utf-8")
    if not payload:
        return {}
    return json.loads(payload)


def _request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: int,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers=headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            return _read_json_response(response)
    except urlerror.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM returned HTTP {exc.code}: {details}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Could not reach local vLLM server: {exc}") from exc


class VLLMServer:
    def __init__(self, env: dict[str, str] | None = None):
        self.env = dict(os.environ if env is None else env)
        self.port = _env_int(self.env, "VLLM_PORT", DEFAULT_VLLM_PORT)
        self.health_timeout = _env_int(
            self.env,
            "HEALTH_TIMEOUT_SECONDS",
            DEFAULT_HEALTH_TIMEOUT_SECONDS,
        )
        self.request_timeout = _env_int(
            self.env,
            "REQUEST_TIMEOUT_SECONDS",
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def ensure_started(self) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return

            command = build_vllm_command(self.env)
            self.process = subprocess.Popen(command, env=self.env)

        self._wait_until_healthy()

    def _wait_until_healthy(self) -> None:
        deadline = time.time() + self.health_timeout
        last_error = "No response from /health yet."

        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup with code {self.process.returncode}."
                )

            try:
                _request_json(
                    "GET",
                    f"{self.base_url}/health",
                    timeout=5,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                time.sleep(2)

        raise TimeoutError(f"Timed out waiting for vLLM healthcheck: {last_error}")

    def shutdown(self) -> None:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                return
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)

    def invoke(self, route: str, body: dict[str, Any]) -> dict[str, Any]:
        self.ensure_started()
        if route == "/health":
            return {"status": "ok", "model": body.get("model")}
        return _request_json(
            "POST",
            f"{self.base_url}{route}",
            body,
            timeout=self.request_timeout,
        )


def get_server() -> VLLMServer:
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is None:
            _SERVER = VLLMServer()
        return _SERVER


def _shutdown_server() -> None:
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is not None:
            _SERVER.shutdown()
            _SERVER = None


atexit.register(_shutdown_server)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    job_input = event.get("input", {})
    default_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    default_max_tokens = _env_int(
        os.environ,
        "DEFAULT_MAX_TOKENS",
        DEFAULT_MAX_TOKENS,
    )
    route, body = normalize_input(
        job_input,
        default_model=default_model,
        default_max_tokens=default_max_tokens,
    )

    with _REQUEST_SEMAPHORE:
        return get_server().invoke(route, body)


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
