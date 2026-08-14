# Harness bridges

The sidecar is the contract. Each harness implements a 50-line client:

1. After a turn lands in the session, `POST /v1/snapshot` with the body
   that a *local* turn would send.
2. After compact finishes, do it again (`reason=compact`). The new prefix
   is smaller and must be re-read.
3. After engine boot or a fat foreign job, `reason=restore`.
4. Paint `GET /v1/status` grey / orange / green / red.

If the snapshot is not byte-identical to the later real request, the light
will lie or the cache will miss. Share one prompt builder.

## Pi

Pi owns the session JSONL and Snapcompact. Hook:

- `session` / prompt settled → snapshot the messages Pi would send
- Snapcompact end (instant local archive) → snapshot the compacted
  transcript immediately, even if the user compacted on a hosted model

Do not add a clock or “today is…” to the system prompt only when the
provider is Sparks.

Reference implementation in this house: Eva-core’s
`server/kv-prefill-catchup-service.js` watches the Pi session file Eva
already tracks in `pi-session-refs.json`.

## Eva-core

Eva-core is a Pi (or OMP) harness plus a chat UI.

| piece | what it does |
|---|---|
| `EVA_KV_CATCHUP_URL` | sidecar base, e.g. `http://127.0.0.1:18900` |
| session-file watcher | mtime change → snapshot (`reason=turn`) |
| compact hook | channel compact + auto-compact → `reason=compact` |
| `GET /api/kv-catchup/status` | UI poll |
| context pip | orange → green next to the context % |

Eva often chats on Venice and only sometimes on `sparks/auto`. The watcher
is the point: Sparks plays catch-up in the background.

Nightly memory-world jobs should stay small windows and call
`reason=restore` for the Eva session when the wave ends.

## Hermes (or any other OpenAI client)

Same POST. On session write / compact / boot:

```bash
curl -sS http://127.0.0.1:18900/v1/snapshot \
  -H 'content-type: application/json' \
  -d '{"session_id":"hermes-main","reason":"turn","messages":[...]}'
```

Poll `/v1/status?session_id=hermes-main` for the pip.

If Hermes rebuilds a system prompt every turn, fix that first. Catch-up
cannot save a moving prefix.

## Isolation from other jobs

KV is one LRU lot (~4.8M tokens on the 4× recipe). Budget:

```text
reserved agent window     ≤ 1M
other in-flight work      small isolated prompts
slop                      ~20%
```

A 150k × 8 unique memory-world wave can evict the reserved prefix even
though 1M “should fit.” Cap those jobs. Restore the agent snapshot after.
