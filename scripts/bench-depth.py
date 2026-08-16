#!/usr/bin/env python3
"""Decode rate at prompt depth (5k/10k/...) + C4 aggregate.

Decode wall rate = (completion_tokens-1)/(t_last_text - t_first_text), temp 0,
thinking off, streamed. Prefill depth is reported from usage.prompt_tokens.
Run on an idle cluster after a warmup pass.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

FILLER = (
    "The committee reviewed the deployment plan for the tensor-parallel cluster. "
    "Each node contributes memory bandwidth and compute to every decode step, so "
    "the minutes note synchronization overhead, kernel launch shape, and cache "
    "behavior in detail. Appendix {i} restates the throughput budget, the fabric "
    "topology, and the rollback procedure for the quarterly release. "
)


def build_prompt(target_tokens: int) -> str:
    # measured ~5.6 chars/token on this filler; trust usage.prompt_tokens.
    parts = ["Read the following operations log, then summarize the rollback procedure.\n"]
    i = 0
    chars = 0
    want = int(target_tokens * 5.6)
    while chars < want:
        seg = FILLER.format(i=i)
        parts.append(seg)
        chars += len(seg)
        i += 1
    return "".join(parts)


def run_once(url: str, model: str, prompt: str, max_tokens: int,
             thinking: str = "off") -> dict:
    if thinking == "off":
        kwargs: dict = {"thinking": False}
    else:
        kwargs = {"thinking": True, "reasoning_effort": thinking}
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": kwargs,
    }).encode()
    req = urllib.request.Request(url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t1 = t2 = None
    ntok = ptok = None
    with urllib.request.urlopen(req, timeout=900) as response:
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
                ptok = chunk["usage"].get("prompt_tokens")
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
    return {"prompt_tokens": ptok, "completion_tokens": ntok,
            "ttft_s": None if t1 is None else round(t1 - t0, 3), "decode_tok_s": wall}


def c4(url: str, model: str, max_tokens: int, thinking: str = "off") -> dict:
    prompt = "Write a complete, idiomatic Python binary search tree with tests. Code only."
    results: list[dict] = [{} for _ in range(4)]
    t0 = time.monotonic()
    threads = []
    for i in range(4):
        th = threading.Thread(target=lambda j: results.__setitem__(j, run_once(url, model, prompt, max_tokens, thinking)), args=(i,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    t_end = time.monotonic()
    total = sum(r.get("completion_tokens") or 0 for r in results)
    agg = round(total / max(t_end - t0, 1e-3), 2)
    per_stream = [r.get("decode_tok_s") for r in results]
    return {"aggregate_tok_s": agg, "per_stream_decode": per_stream,
            "total_completion_tokens": total, "wall_s": round(t_end - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731-ablit-32-32")
    ap.add_argument("--depths", default="5000,10000")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--skip-c4", action="store_true")
    ap.add_argument("--thinking", default="off", choices=["off", "low", "high", "max"])
    args = ap.parse_args()

    out = {"thinking": args.thinking, "depths": {}, "c4": None}
    for target in [int(d) for d in args.depths.split(",")]:
        prompt = build_prompt(target)
        # one warmup at this depth, then n measured
        warm = run_once(args.base_url, args.model, prompt, args.max_tokens, args.thinking)
        rows = []
        for _ in range(args.n):
            rows.append(run_once(args.base_url, args.model, prompt, args.max_tokens, args.thinking))
        rates = sorted(r["decode_tok_s"] for r in rows if r.get("decode_tok_s"))
        out["depths"][str(target)] = {
            "prompt_tokens_actual": rows[-1].get("prompt_tokens"),
            "rates": rates,
            "median": rates[len(rates) // 2] if rates else None,
            "ttft_s": [r.get("ttft_s") for r in rows],
        }
        print(f"depth {target}: {out['depths'][str(target)]}", flush=True)
    if not args.skip_c4:
        run_once(args.base_url, args.model, "warm", 64, args.thinking)
        out["c4"] = c4(args.base_url, args.model, 256, args.thinking)
        print(f"c4: {out['c4']}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
