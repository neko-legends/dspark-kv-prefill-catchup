"""Pure catch-up logic. No HTTP here so tests stay offline."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REASONS = ("turn", "compact", "boot", "restore", "touch")
COLORS = ("grey", "orange", "green", "red")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def estimate_prompt_tokens(messages: list | None, extra_text: str = "") -> int:
    total = len(extra_text)
    for message in messages or []:
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping):
                    total += len(str(part.get("text") or ""))
                else:
                    total += len(str(part or ""))
        elif content is not None:
            total += len(json.dumps(content, ensure_ascii=False, sort_keys=True))
        for key in ("name", "tool_call_id"):
            total += len(str(message.get(key) or "")) if isinstance(message, Mapping) else 0
        if isinstance(message, Mapping) and message.get("tool_calls"):
            total += len(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
    return max(1, (total + 3) // 4)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def normalize_snapshot(body: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(body or {})
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("snapshot requires a non-empty messages array")
    tools = payload.get("tools")
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    kwargs = payload.get("chat_template_kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    session_id = _clean(payload.get("session_id"))
    if not session_id:
        raise ValueError("snapshot requires session_id")
    reason = _clean(payload.get("reason")).lower() or "turn"
    if reason not in REASONS:
        reason = "turn"
    max_context = int(payload.get("max_context") or 0)
    return {
        "session_id": session_id,
        "messages": messages,
        "tools": tools,
        "chat_template_kwargs": kwargs,
        "model": _clean(payload.get("model")),
        "reason": reason,
        "max_context": max_context,
    }


def hash_snapshot(messages: list, tools: list | None = None, chat_template_kwargs: Mapping | None = None) -> str:
    blob = json.dumps(
        _canonical({
            "messages": messages,
            "tools": tools or [],
            "chat_template_kwargs": chat_template_kwargs or {},
        }),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def apply_rolling_window(messages: list, max_context: int) -> list:
    limit = int(max_context or 0)
    if limit <= 0 or estimate_prompt_tokens(messages) <= limit:
        return list(messages)
    kept = list(messages)
    system = []
    if kept and isinstance(kept[0], Mapping) and kept[0].get("role") == "system":
        system = [kept.pop(0)]
    while kept and estimate_prompt_tokens(system + kept) > limit:
        # Drop the oldest non-system message. Never drop the last user turn.
        if len(kept) <= 1:
            break
        kept.pop(0)
    return system + kept


def color_for(state: str) -> str:
    if state == "warm":
        return "green"
    if state == "error":
        return "red"
    if state in {"warming", "stale"}:
        return "orange"
    return "grey"


@dataclass
class SessionState:
    session_id: str
    state: str = "idle"
    current_hash: str | None = None
    warmed_hash: str | None = None
    prompt_estimate: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    reason: str = ""
    error: str | None = None
    updated_at: float = 0.0
    generation: int = 0

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("generation", None)
        payload["color"] = color_for(self.state)
        return payload


WarmupFn = Callable[[dict[str, Any]], Mapping[str, Any]]


class CatchupService:
    def __init__(
        self,
        *,
        vllm_url: str = "",
        model: str = "",
        max_context: int = 1_000_000,
        timeout_s: float = 1800.0,
        warmup_fn: WarmupFn | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.vllm_url = vllm_url.rstrip("/")
        self.model = model
        self.max_context = int(max_context or 1_000_000)
        self.timeout_s = float(timeout_s)
        self._warmup_fn = warmup_fn or self._http_warmup
        self._now = now or time.time
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str = "") -> dict[str, Any] | list[dict[str, Any]]:
        with self._lock:
            if session_id:
                session = self._sessions.get(session_id)
                if not session:
                    return SessionState(session_id=session_id, updated_at=self._now()).public()
                return session.public()
            return [session.public() for session in self._sessions.values()]

    def submit(self, body: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = normalize_snapshot(body)
        limit = snapshot["max_context"] or self.max_context
        rolled = apply_rolling_window(snapshot["messages"], limit)
        digest = hash_snapshot(rolled, snapshot["tools"], snapshot["chat_template_kwargs"])
        estimate = estimate_prompt_tokens(rolled)
        if estimate >= limit:
            raise ValueError(
                f"prompt is about {estimate} tokens and the reserved window is {limit}"
            )
        with self._lock:
            session = self._sessions.setdefault(snapshot["session_id"], SessionState(session_id=snapshot["session_id"]))
            session.current_hash = digest
            session.prompt_estimate = estimate
            session.reason = snapshot["reason"]
            session.updated_at = self._now()
            session.error = None
            if session.warmed_hash == digest and session.state == "warm":
                return session.public()
            session.state = "warming"
            session.generation += 1
            generation = session.generation
            work = {
                "session_id": snapshot["session_id"],
                "messages": rolled,
                "tools": snapshot["tools"],
                "chat_template_kwargs": snapshot["chat_template_kwargs"],
                "model": snapshot["model"] or self.model,
                "hash": digest,
                "generation": generation,
            }
        thread = threading.Thread(target=self._run_warmup, args=(work,), daemon=True)
        thread.start()
        with self._lock:
            return self._sessions[snapshot["session_id"]].public()

    def _run_warmup(self, work: dict[str, Any]) -> None:
        error_text = None
        prompt_tokens = None
        cached_tokens = None
        try:
            result = self._warmup_fn(work)
            usage = result.get("usage") if isinstance(result, Mapping) else {}
            usage = usage or {}
            prompt_tokens = _as_int(usage.get("prompt_tokens"))
            details = usage.get("prompt_tokens_details") or {}
            cached_tokens = _as_int(details.get("cached_tokens"))
        except Exception as exc:  # noqa: BLE001 — surface any engine failure as red
            error_text = str(exc) or exc.__class__.__name__
        with self._lock:
            session = self._sessions.get(work["session_id"])
            if not session or session.generation != work["generation"]:
                return
            session.prompt_tokens = prompt_tokens
            session.cached_tokens = cached_tokens
            session.updated_at = self._now()
            if error_text:
                session.state = "error"
                session.error = error_text
                return
            if session.current_hash != work["hash"]:
                session.state = "stale"
                return
            session.warmed_hash = work["hash"]
            session.state = "warm"
            session.error = None

    def _http_warmup(self, work: dict[str, Any]) -> Mapping[str, Any]:
        if not self.vllm_url:
            raise RuntimeError("CATCHUP_VLLM_URL is not set")
        payload: dict[str, Any] = {
            "model": work["model"],
            "messages": work["messages"],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
        if work.get("tools"):
            payload["tools"] = work["tools"]
        if work.get("chat_template_kwargs"):
            payload["chat_template_kwargs"] = work["chat_template_kwargs"]
        request = Request(
            f"{self.vllm_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code}: {detail[:400]}") from exc
        except URLError as exc:
            raise RuntimeError(f"vLLM unreachable: {exc.reason}") from exc


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
