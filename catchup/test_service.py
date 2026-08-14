import json
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.request import Request, urlopen

from catchup.service import (
    CatchupService,
    apply_rolling_window,
    color_for,
    estimate_prompt_tokens,
    hash_snapshot,
    reserved_limit,
)
from catchup.__main__ import build_handler


def wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class HashAndRollTests(unittest.TestCase):
    def test_hash_is_stable_and_ignores_session_metadata(self):
        messages = [{"role": "user", "content": "hello"}]
        left = hash_snapshot(messages, [], {"thinking": False})
        right = hash_snapshot(messages, [], {"thinking": False})
        self.assertTrue(left.startswith("sha256:"))
        self.assertEqual(left, right)
        self.assertNotEqual(left, hash_snapshot(messages, [], {"thinking": True}))
        self.assertNotEqual(left, hash_snapshot(messages, [{"type": "function", "function": {"name": "x"}}], {"thinking": False}))

    def test_rolling_window_keeps_system_and_newest_tail(self):
        messages = (
            [{"role": "system", "content": "sys"}]
            + [{"role": "user", "content": "x" * 40} for _ in range(8)]
        )
        # 1 + 10 tokens each user ≈ 81; cap at 25 keeps system + last two-ish
        rolled = apply_rolling_window(messages, 25)
        self.assertEqual(rolled[0]["role"], "system")
        self.assertLess(len(rolled), len(messages))
        self.assertEqual(rolled[-1], messages[-1])
        self.assertLessEqual(estimate_prompt_tokens(rolled), 25)

    def test_reserved_limit_leaves_headroom(self):
        self.assertEqual(reserved_limit(1000000), 900000)
        self.assertEqual(reserved_limit(0), 0)
        self.assertEqual(reserved_limit(8), 8)

    def test_colors(self):
        self.assertEqual(color_for("idle"), "grey")
        self.assertEqual(color_for("warming"), "orange")
        self.assertEqual(color_for("stale"), "orange")
        self.assertEqual(color_for("warm"), "green")
        self.assertEqual(color_for("error"), "red")


class ServiceTests(unittest.TestCase):
    def test_successful_warmup_turns_green(self):
        seen = []

        def warmup(work):
            seen.append(work["hash"])
            return {"usage": {"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 0}}}

        service = CatchupService(max_context=1000, warmup_fn=warmup)
        first = service.submit({
            "session_id": "eva-dm",
            "reason": "compact",
            "messages": [{"role": "user", "content": "hello world"}],
        })
        self.assertIn(first["color"], {"orange", "green"})
        self.assertTrue(wait_until(lambda: service.get("eva-dm")["state"] == "warm"))
        status = service.get("eva-dm")
        self.assertEqual(status["color"], "green")
        self.assertEqual(status["reason"], "compact")
        self.assertEqual(status["prompt_tokens"], 12)
        self.assertEqual(len(seen), 1)

        again = service.submit({
            "session_id": "eva-dm",
            "messages": [{"role": "user", "content": "hello world"}],
        })
        self.assertEqual(again["color"], "green")
        self.assertEqual(len(seen), 1)

    def test_newer_snapshot_cancels_stale_warmup(self):
        release = []

        def warmup(work):
            if "first" in json.dumps(work["messages"]):
                while not release:
                    time.sleep(0.01)
            return {"usage": {"prompt_tokens": 3}}

        service = CatchupService(warmup_fn=warmup, max_context=10000)
        service.submit({"session_id": "s", "messages": [{"role": "user", "content": "first turn xxx"}]})
        service.submit({"session_id": "s", "messages": [{"role": "user", "content": "second turn yyy"}]})
        release.append(True)
        self.assertTrue(wait_until(lambda: service.get("s")["state"] == "warm"))
        status = service.get("s")
        self.assertEqual(status["color"], "green")
        self.assertIn(hash_snapshot([{"role": "user", "content": "second turn yyy"}]), status["warmed_hash"])

    def test_engine_error_is_red(self):
        def warmup(_work):
            raise RuntimeError("vLLM HTTP 400: context overflow")

        service = CatchupService(warmup_fn=warmup)
        service.submit({"session_id": "s", "messages": [{"role": "user", "content": "boom"}]})
        self.assertTrue(wait_until(lambda: service.get("s")["state"] == "error"))
        self.assertEqual(service.get("s")["color"], "red")
        self.assertIn("400", service.get("s")["error"])

    def test_oversized_prompt_rejected_immediately(self):
        service = CatchupService(max_context=8, warmup_fn=lambda work: {})
        with self.assertRaises(ValueError):
            service.submit({
                "session_id": "s",
                "max_context": 8,
                "messages": [{"role": "user", "content": "x" * 80}],
            })


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.service = CatchupService(
            vllm_url="http://vllm.example/v1",
            warmup_fn=lambda work: {"usage": {"prompt_tokens": 4}},
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(self.service))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_health_snapshot_status(self):
        health = json.loads(urlopen(self.base + "/v1/health", timeout=2).read())
        self.assertTrue(health["ok"])
        req = Request(
            self.base + "/v1/snapshot",
            data=json.dumps({
                "session_id": "eva-dm",
                "reason": "boot",
                "messages": [{"role": "user", "content": "hi"}],
            }).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        posted = json.loads(urlopen(req, timeout=2).read())
        self.assertIn(posted["color"], {"orange", "green"})
        self.assertTrue(wait_until(lambda: json.loads(
            urlopen(self.base + "/v1/status?session_id=eva-dm", timeout=2).read()
        )["color"] == "green"))
        status = json.loads(urlopen(self.base + "/v1/status?session_id=eva-dm", timeout=2).read())
        self.assertEqual(status["state"], "warm")
        self.assertEqual(status["reason"], "boot")


if __name__ == "__main__":
    unittest.main()
