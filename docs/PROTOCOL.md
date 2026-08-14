# Catch-up protocol

Harness-agnostic. The sidecar does not know Pi, Eva, or Hermes. It knows an
OpenAI chat body and a session id.

Base URL is the sidecar, not vLLM. Default `http://127.0.0.1:18900`.

## `POST /v1/snapshot`

Push the body the engine will see on the next real local turn.

```json
{
  "session_id": "eva-dm",
  "messages": [{ "role": "user", "content": "hello" }],
  "tools": [],
  "chat_template_kwargs": { "thinking": false },
  "model": "deepseek-v4-flash-0731-ablit-32-32",
  "reason": "turn",
  "max_context": 1000000
}
```

| field | required | notes |
|---|---|---|
| `session_id` | yes | One reserved prefix per id |
| `messages` | yes | Exact OpenAI messages |
| `tools` | no | Included in the hash. Send them if the real turn will. |
| `chat_template_kwargs` | no | Included in the hash (`thinking`, `reasoning_effort`, …) |
| `model` | no | Defaults to sidecar `--model` |
| `reason` | no | `turn` `compact` `boot` `restore` `touch` |
| `max_context` | no | Rolling cap in estimated tokens. Default sidecar `--max-context` |

Response (202 if a warmup was queued, 200 if already warm for this hash):

```json
{
  "session_id": "eva-dm",
  "state": "warming",
  "color": "orange",
  "current_hash": "sha256:…",
  "warmed_hash": "sha256:…",
  "reason": "compact",
  "prompt_estimate": 12840
}
```

A newer snapshot for the same `session_id` **replaces** an in-flight warmup.
Only the latest hash is worth parking.

## `GET /v1/status?session_id=`

```json
{
  "session_id": "eva-dm",
  "state": "warm",
  "color": "green",
  "current_hash": "sha256:…",
  "warmed_hash": "sha256:…",
  "prompt_tokens": 12840,
  "cached_tokens": 0,
  "prompt_estimate": 12840,
  "reason": "compact",
  "error": null,
  "updated_at": 1786660000.1
}
```

Omit `session_id` to list every session the sidecar has seen.

## `GET /v1/health`

```json
{ "ok": true, "vllm": "http://127.0.0.1:18888/v1" }
```

## Hash

SHA-256 of canonical JSON:

```
{ "messages", "tools", "chat_template_kwargs" }
```

Stable key order. No `session_id`, no `reason`, no `max_tokens`. Those are
not part of the prompt.

## Colors

| color | state | rule |
|---|---|---|
| grey | `idle` | no snapshot |
| orange | `warming` | warmup in flight for `current_hash` |
| orange | `stale` | `current_hash != warmed_hash` and not in flight |
| green | `warm` | warmup finished and hashes match |
| red | `error` | last warmup threw; `error` is set |

Green does **not** require `cached_tokens ≈ prompt_tokens` on the warmup
response. That call is the prefill. The KV is warm when it returns 200.

## Rolling window

If the estimated prompt (`ceil(chars/4)`) exceeds `max_context`, drop the
oldest non-system messages until it fits. Keep the first system message.
This changes the hash. The sidecar then does a full background prefill of
the kept tail.

## Warmup request to vLLM

```json
{
  "model": "…",
  "messages": ["…rolled…"],
  "tools": ["…or omitted if empty…"],
  "max_tokens": 1,
  "temperature": 0,
  "stream": false,
  "chat_template_kwargs": { "thinking": false }
}
```

## Errors the sidecar maps to red

- vLLM HTTP 4xx/5xx
- prompt alone ≥ engine context (do not wait in a queue)
- network / timeout
