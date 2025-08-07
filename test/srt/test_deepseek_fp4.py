import unittest

from sglang.test.test_utils import (
    popen_launch_server,
    kill_process_tree,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
)
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from types import SimpleNamespace

class TestFlashinferCutlassMoe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/DeepSeek-R1-0528-FP4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--quantization",
            "modelopt_fp4",
            "--tp",
            "4",
            "--enable-flashinfer-cutlass-moe",
            "--ep-size",
            "4",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)

class TestFlashinferTrtllmMoe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/DeepSeek-R1-0528-FP4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--quantization",
            "modelopt_fp4",
            "--tp",
            "4",
            "--enable-flashinfer-trtllm-moe",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)

if __name__ == "__main__":
    unittest.main()
