from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoscilab.benchmarking import (
    ResponseCache,
    atomic_write_json,
    completed_result,
    ensure_run_config,
    preflight_openai_endpoint,
    safe_endpoint,
    stable_seed,
)


class BenchmarkingHelpersTest(unittest.TestCase):
    def test_atomic_json_and_completion_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "result.json"
            atomic_write_json(
                target,
                {"status": "completed", "run_fingerprint": "abc", "value": 7},
            )
            self.assertEqual(json.loads(target.read_text())["value"], 7)
            self.assertEqual(completed_result(target, "abc")["value"], 7)
            self.assertIsNone(completed_result(target, "wrong"))

            atomic_write_json(target, {"status": "error", "run_fingerprint": "abc"})
            self.assertIsNone(completed_result(target, "abc"))

    def test_run_config_rejects_incompatible_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = ensure_run_config(directory, {"model": "local-a", "budget": 10})
            self.assertEqual(
                first,
                ensure_run_config(directory, {"model": "local-a", "budget": 10}),
            )
            with self.assertRaises(RuntimeError):
                ensure_run_config(directory, {"model": "local-b", "budget": 10})

    def test_response_cache_preserves_occurrence_order_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.json"
            key = ResponseCache.fingerprint({"prompt": "same prompt"})

            first = ResponseCache(path)
            pos0, value0 = first.lookup(key)
            self.assertEqual((pos0, value0), (0, None))
            first.store(key, pos0, "sample one")
            pos1, value1 = first.lookup(key)
            self.assertEqual((pos1, value1), (1, None))
            first.store(key, pos1, "sample two")

            resumed = ResponseCache(path)
            self.assertEqual(resumed.lookup(key), (0, "sample one"))
            self.assertEqual(resumed.lookup(key), (1, "sample two"))
            self.assertEqual(resumed.lookup(key), (2, None))

    def test_stable_seed_and_endpoint_redaction(self) -> None:
        self.assertEqual(stable_seed("m0", "easy", "v0", 0), stable_seed("m0", "easy", "v0", 0))
        self.assertNotEqual(stable_seed("m0", "easy", "v0", 0), stable_seed("m0", "easy", "v0", 1))
        self.assertEqual(
            safe_endpoint("http://user:secret@127.0.0.1:8000/v1/?token=x"),
            "http://127.0.0.1:8000/v1",
        )

    def test_endpoint_preflight_checks_advertised_model(self) -> None:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"data": [{"id": "local/model"}]}).encode()

        endpoint = "http://127.0.0.1:8000/v1"
        with patch("urllib.request.urlopen", return_value=Response()):
            self.assertEqual(preflight_openai_endpoint(endpoint, "local/model"), ["local/model"])
            with self.assertRaises(RuntimeError):
                preflight_openai_endpoint(endpoint, "wrong/model")


if __name__ == "__main__":
    unittest.main()
