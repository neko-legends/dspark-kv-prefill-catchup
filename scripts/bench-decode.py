#!/usr/bin/env python3
"""C1 streaming decode bench + Prometheus decode-only rate."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


PROMPT = (
    "Write a complete, idiomatic Python implementation of a binary search tree with "
    "insert, delete, search, traversal, height, docstrings, and tests. Code only."
)


def fetch_metrics(base: str) -> dict:
    txt = urllib.request.urlopen(base.replace("/v1", "") + "/metrics", timeout=10).read().decode()
    out = {"gen": 0.0, "decode_s": 0.0, "success": 0.0, "running": 0.0}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:generation_tokens_total{"):
            out["gen"] = float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:request_decode_time_seconds_sum{"):
            out["decode_s"] = float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:request_success_total{"):
            out["success"] += float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:num_requests_running{"):
            out["running"] = float(line.rsplit(" ", 1)[-1])
    return out


def run_once(url: str, model: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t1 = t2 = None
    ntok = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                ntok = chunk["usage"].get("completion_tokens")
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = (delta.get("content") or "") + (delta.get("reasoning") or "")
            if text:
                now = time.monotonic()
                if t1 is None:
                    t1 = now
                t2 = now
    wall = None
    if t1 is not None and ntok and ntok >= 2:
        wall = round((ntok - 1) / max(t2 - t1, 1e-3), 2)
    return {"tok": ntok, "ttft": None if t1 is None else round(t1 - t0, 3), "wall_tok_s": wall}


def summarize(rows: list[dict], key: str) -> dict:
    xs = sorted(row[key] for row in rows if row.get(key) is not None)
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 2) if xs else None,
        "median": xs[len(xs) // 2] if xs else None,
        "sd": round(statistics.pstdev(xs), 2) if len(xs) > 1 else 0,
        "min": xs[0] if xs else None,
        "max": xs[-1] if xs else None,
        "trials": xs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731-ablit-32-32")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    def once(label: str) -> dict | None:
        before = fetch_metrics(args.base_url)
        if before["running"] > 0:
            raise SystemExit(f"cluster busy before {label}")
        client = run_once(args.base_url, args.model, args.max_tokens)
        after = fetch_metrics(args.base_url)
        delta_ok = after["success"] - before["success"]
        delta_s = after["decode_s"] - before["decode_s"]
        delta_gen = after["gen"] - before["gen"]
        server = round(delta_gen / delta_s, 2) if delta_s > 0 else None
        rec = {**client, "server_tok_s": server, "success_delta": delta_ok}
        print(f"{label}: {rec}", flush=True)
        if delta_ok != 1:
            print(f"{label}: dropped (foreign traffic)", flush=True)
            return None
        return rec

    print(f"warmup {args.warmup} then n={args.n}", flush=True)
    for i in range(args.warmup):
        once(f"warmup {i+1}")
    measured = []
    for i in range(args.n):
        rec = once(f"meas {i+1}")
        if rec:
            measured.append(rec)
    out = {
        "server_decode": summarize(measured, "server_tok_s"),
        "client_wall": summarize(measured, "wall_tok_s"),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
