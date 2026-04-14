import unittest

import handler


class NormalizeInputTests(unittest.TestCase):
    def test_prompt_shortcut_defaults_model_and_tokens(self):
        route, body = handler.normalize_input(
            {"prompt": "Say hello."},
            default_model="facebook/opt-125m",
            default_max_tokens=64,
        )

        self.assertEqual(route, "/v1/completions")
        self.assertEqual(body["model"], "facebook/opt-125m")
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["prompt"], "Say hello.")

    def test_chat_shortcut_preserves_explicit_model(self):
        route, body = handler.normalize_input(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "max_tokens": 12,
            },
            default_model="facebook/opt-125m",
            default_max_tokens=64,
        )

        self.assertEqual(route, "/v1/chat/completions")
        self.assertEqual(body["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(body["max_tokens"], 12)

    def test_explicit_route_body_is_normalized(self):
        route, body = handler.normalize_input(
            {
                "route": "chat/completions",
                "body": {"messages": [{"role": "user", "content": "Hi"}]},
            },
            default_model="facebook/opt-125m",
            default_max_tokens=64,
        )

        self.assertEqual(route, "/v1/chat/completions")
        self.assertEqual(body["model"], "facebook/opt-125m")
        self.assertEqual(body["max_tokens"], 64)

    def test_streaming_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            handler.normalize_input(
                {"prompt": "Hi", "stream": True},
                default_model="facebook/opt-125m",
                default_max_tokens=64,
            )


class BuildCommandTests(unittest.TestCase):
    def test_build_vllm_command_maps_known_env_vars(self):
        env = {
            "MODEL_NAME": "facebook/opt-125m",
            "VLLM_PORT": "8012",
            "TOKENIZER_MODE": "mistral",
            "CONFIG_FORMAT": "mistral",
            "LOAD_FORMAT": "mistral",
            "MAX_MODEL_LEN": "4096",
            "GPU_MEMORY_UTILIZATION": "0.88",
            "MAX_NUM_SEQS": "64",
            "TENSOR_PARALLEL_SIZE": "2",
            "DTYPE": "bfloat16",
            "QUANTIZATION": "awq",
            "DOWNLOAD_DIR": "/models",
            "TRUST_REMOTE_CODE": "true",
            "ENABLE_AUTO_TOOL_CHOICE": "true",
            "TOOL_CALL_PARSER": "hermes",
            "REASONING_PARSER": "deepseek_r1",
            "ENFORCE_EAGER": "true",
            "OPENAI_SERVED_MODEL_NAME_OVERRIDE": "custom-name",
        }

        command = handler.build_vllm_command(env)

        self.assertEqual(
            command[:8],
            [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--host",
                "0.0.0.0",
                "--port",
                "8012",
                "--model",
            ],
        )
        self.assertIn("facebook/opt-125m", command)
        self.assertIn("--max-model-len", command)
        self.assertIn("4096", command)
        self.assertIn("--tokenizer-mode", command)
        self.assertIn("--config-format", command)
        self.assertIn("--load-format", command)
        self.assertIn("--gpu-memory-utilization", command)
        self.assertIn("0.88", command)
        self.assertIn("--trust-remote-code", command)
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertIn("--tool-call-parser", command)
        self.assertIn("hermes", command)
        self.assertIn("--reasoning-parser", command)
        self.assertIn("deepseek_r1", command)
        self.assertIn("--enforce-eager", command)


if __name__ == "__main__":
    unittest.main()
