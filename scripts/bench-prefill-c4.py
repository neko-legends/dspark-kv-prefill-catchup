#!/usr/bin/env python3
"""Prefill-rate bench (TTFT at increasing prompt sizes) + C4 concurrent decode bench.

Prefill: unique-nonce prompts (defeats prefix cache), max_tokens=1, streaming;
rate = server-reported prompt_tokens / client TTFT.

C4: 4 concurrent 256-token streams, temp 0; per-stream decode tok/s and aggregate.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
import uuid

WORDS = (
    "amber willow drifted across the quiet harbor while engineers argued about "
    "tensor layouts and the kettle boiled over again".split()
)


def make_prompt(target_tokens: int, nonce: str) -> str:
    # ~1.3 tokens/word for english prose; overshoot then rely on server count
    words_needed = int(target_tokens / 1.15)
    out = [f"nonce-{nonce}"]
    i = 0
    while len(out) < words_needed:
        out.append(WORDS[(i * 7 + 3) % len(WORDS)])
        i += 1
    return " ".join(out)


def post_stream(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = t_last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
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
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if (delta.get("content") or delta.get("reasoning")) and t_first is None:
                t_first = time.monotonic()
            t_last = time.monotonic()
    t_end = time.monotonic()
    return {
        "ttft": (t_first - t0) if t_first else None,
        "total": t_end - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "decode_window": (t_last - t_first) if (t_first and t_last) else None,
    }


def prefill_bench(base: str, model: str, sizes: list[int]) -> list[dict]:
    rows = []
    for size in sizes:
        nonce = uuid.uuid4().hex[:12]
        prompt = make_prompt(size, nonce)
        r = post_stream(base, model, prompt, 1)
        rate = None
        if r["prompt_tokens"] and r["ttft"]:
            rate = round(r["prompt_tokens"] / r["ttft"], 1)
        row = {
            "target": size,
            "prompt_tokens": r["prompt_tokens"],
            "ttft_s": round(r["ttft"], 2) if r["ttft"] else None,
            "prefill_tok_s": rate,
        }
        rows.append(row)
        print(f"prefill target={size}: {row}", flush=True)
    return rows


def c4_bench(base: str, model: str, max_tokens: int) -> dict:
    prompt = (
        "Write a complete, idiomatic Python implementation of a binary search tree "
        "with insert, delete, search, traversal, height, docstrings, and tests. Code only. "
        + uuid.uuid4().hex[:8]
    )
    results: list[dict] = []

    def worker() -> None:
        results.append(post_stream(base, model, prompt, max_tokens))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0

    per_stream = []
    total_completion = 0
    for r in results:
        n = r["completion_tokens"] or 0
        total_completion += n
        if r["decode_window"] and n and n >= 2:
            per_stream.append(round((n - 1) / r["decode_window"], 2))
    agg = round(total_completion / wall, 2) if wall > 0 else None
    out = {
        "streams": len(results),
        "tokens_each": [r["completion_tokens"] for r in results],
        "per_stream_tok_s": per_stream,
        "wall_s": round(wall, 2),
        "aggregate_tok_s": agg,
    }
    print(f"c4: {out}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://forge:18888/v1")
    p.add_argument("--model", default="deepseek-v4-flash-abliterated")
    p.add_argument("--sizes", default="1024,8192,32768,65536,131072")
    p.add_argument("--c4-tokens", type=int, default=256)
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument("--skip-c4", action="store_true")
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    report = {"model": args.model, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if not args.skip_prefill:
        report["prefill"] = prefill_bench(args.base_url, args.model, sizes)
    if not args.skip_c4:
        report["c4"] = c4_bench(args.base_url, args.model, args.c4_tokens)
    print("REPORT_JSON " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
