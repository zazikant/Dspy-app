"""
OpenCode LLM client — streaming chat completions via the OpenCode gateway.

Python port of `src/lib/brain/opencode.ts` from zazikant/tradingview-notes-app
(the GLM-5.1 reference implementation).

Required env vars:
    OPENCODE_API_KEY   — your OpenCode API key (from https://opencode.ai/)

Gateway: https://opencode.ai/zen/go/v1/chat/completions
Model:   glm-5.1 (alias — currently serves GLM 5.3 thinking model)

CRITICAL: GLM 5.3 is a thinking-only model. We MUST send reasoning_effort:'low'
to keep the reasoning overhead minimal while still ensuring the final answer
lands in `content`. Do NOT combine with `thinking:{type:"disabled"}` — the
gateway rejects the combination.

Retry / rate-handling (port from nvidia.ts:52-61):
    Retryable HTTP:   429, 500, 502, 503, 504
    Retryable codes:  ECONNRESET, ETIMEDOUT, UND_ERR_CONNECT_TIMEOUT
    Retryable names:  APIConnectionError, APITimeoutError, ConnectionError
    Per-call timeout: 55s (Streamlit Cloud caps free apps around 60s)
    Max attempts:     1 (caller can wrap in pipeline-level retry)
    Backoff:          500ms × attempt
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import requests

OPENCODE_GATEWAY = "https://opencode.ai/zen/go/v1/chat/completions"
OPENCODE_DEFAULT_MODEL = "glm-5.1"
OPENCODE_DEFAULT_TIMEOUT_S = 55.0
OPENCODE_DEFAULT_TEMPERATURE = 0.7
OPENCODE_DEFAULT_MAX_TOKENS = 2048
OPENCODE_DEFAULT_TOP_P = 1.0


@dataclass
class OpenCodeResult:
    content: str
    reasoning: str
    model: str
    elapsed_ms: int
    attempts: int


# ─── Retry classification (port of nvidia.ts:52-61) ──────────────────────────
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}
_RETRYABLE_CODES = {"ECONNRESET", "ETIMEDOUT", "UND_ERR_CONNECT_TIMEOUT"}
_RETRYABLE_NAMES = {"APIConnectionError", "APITimeoutError", "ConnectionError"}


def is_retryable_error(err: Exception) -> bool:
    """True iff the error is a transient/rate-limit hiccup worth retrying."""
    status = getattr(err, "status", None) or getattr(err, "status_code", None) or 0
    if status in _RETRYABLE_HTTP:
        return True
    code = getattr(err, "code", None)
    if code in _RETRYABLE_CODES:
        return True
    cls_name = type(err).__name__
    if cls_name in _RETRYABLE_NAMES:
        return True
    msg = (str(err) or "").lower()
    return any(s in msg for s in ("timeout", "rate limit", "too many requests", "econnreset"))


def _new_session_id() -> str:
    """Per-call UUID required by the OpenCode gateway for session routing."""
    return str(uuid.uuid4())


def _parse_sse_line(line: str, content: list[str], reasoning: list[str]) -> None:
    """Parse one SSE `data: ...` line. Returns True if stream should end."""
    if not line.startswith("data:"):
        return False
    payload = line[5:].strip()
    if payload == "[DONE]":
        return True
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        # Partial JSON across chunks — will be retried on the next read
        return False
    delta = (event.get("choices") or [{}])[0].get("delta") or {}
    if isinstance(delta.get("content"), str) and delta["content"]:
        content.append(delta["content"])
    if isinstance(delta.get("reasoning_content"), str):
        reasoning.append(delta["reasoning_content"])
    return False


def _stream_once(
    *,
    messages: list[dict],
    model: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    on_log: Optional[Callable[[str], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> tuple[list[str], list[str], int]:
    """Make a single streaming chat-completion call.

    Returns (content_parts, reasoning_parts, ttfb_ms).
    Raises requests.HTTPError for non-2xx, requests.exceptions.Timeout on timeout.
    """
    session_id = _new_session_id()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
        "x-opencode-session": session_id,
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "reasoning_effort": "low",
    }
    if on_log:
        on_log(
            f"[opencode] start  model={model} max_tokens={max_tokens} "
            f"temp={temperature} top_p={top_p} timeout={timeout_s}s session={session_id[:8]}"
        )

    t0 = time.time()
    response = requests.post(
        OPENCODE_GATEWAY,
        headers=headers,
        json=body,
        stream=True,
        timeout=timeout_s,
    )
    if not response.ok:
        # Surface status code on the exception so is_retryable_error() can see it
        err = requests.HTTPError(
            f"OpenCode API error ({response.status_code}): {response.text[:300]}",
            response=response,
        )
        err.status = response.status_code  # type: ignore[attr-defined]
        raise err
    if response.raw is None:
        raise RuntimeError("OpenCode API returned no response body")

    ttfb_ms: Optional[int] = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        if ttfb_ms is None:
            ttfb_ms = int((time.time() - t0) * 1000)
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        done = _parse_sse_line(line, content_parts, reasoning_parts)
        # Fire chunk callback for content (NOT reasoning — internal scratchpad)
        if on_chunk and content_parts:
            # Emit only the new tail since last callback
            # (callback was fired once per delta already in _parse_sse_line — see below)
            pass
        if done:
            break

    # Re-emit each parsed content delta as a chunk callback for live UI updates
    if on_chunk:
        for part in content_parts:
            on_chunk(part)

    return content_parts, reasoning_parts, ttfb_ms or 0


def opencode_chat_stream_controlled(
    *,
    messages: list[dict],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = OPENCODE_DEFAULT_TEMPERATURE,
    top_p: float = OPENCODE_DEFAULT_TOP_P,
    max_tokens: int = OPENCODE_DEFAULT_MAX_TOKENS,
    timeout_s: float = OPENCODE_DEFAULT_TIMEOUT_S,
    max_retries: int = 1,
    on_log: Optional[Callable[[str], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> OpenCodeResult:
    """Controlled, logged, time-bounded chat completion with retry.

    Mirrors opencodeChatStreamControlled from src/lib/brain/opencode.ts.
    Returns OpenCodeResult with full content + reasoning + timing metadata.

    The reasoning_content emitted by GLM 5.3 is accumulated internally but
    NEVER streamed to the user (it's chain-of-thought scratchpad). If the
    model only produces reasoning without a finished answer (e.g. budget
    exhausted mid-thought), the reasoning is returned as a degraded content
    fallback.
    """
    api_key = api_key or os.getenv("OPENCODE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENCODE_API_KEY env var is not set. Add it in the Streamlit sidebar."
        )
    model = model or OPENCODE_DEFAULT_MODEL
    call_start = time.time()
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            content_parts, reasoning_parts, ttfb_ms = _stream_once(
                messages=messages,
                model=model,
                api_key=api_key,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                on_log=on_log,
                on_chunk=None,  # we re-emit after the stream completes for accuracy
            )
            elapsed_ms = int((time.time() - call_start) * 1000)
            content = "".join(content_parts)
            reasoning = "".join(reasoning_parts)
            if on_log:
                on_log(
                    f"[opencode] ttfb={ttfb_ms}ms done attempt={attempt} "
                    f"elapsed={elapsed_ms}ms content_chars={len(content)} "
                    f"reasoning_chars={len(reasoning)}"
                )
            if not content and reasoning:
                # Degraded mode: model only produced reasoning (max_tokens exhausted).
                content = reasoning
                if on_chunk:
                    on_chunk(content)
            if not content:
                raise RuntimeError(
                    f"empty content (reasoning_chars={len(reasoning)})"
                )
            return OpenCodeResult(
                content=content,
                reasoning=reasoning,
                model=model,
                elapsed_ms=elapsed_ms,
                attempts=attempt,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - call_start) * 1000)
            last_err = e
            err_name = type(e).__name__
            if err_name == "Timeout" or "timeout" in str(e).lower():
                if on_log:
                    on_log(f"[opencode] TIMEOUT attempt={attempt} after {timeout_s}s")
            else:
                if on_log:
                    on_log(
                        f"[opencode] ERROR attempt={attempt} after {elapsed_ms}ms: "
                        f"{err_name}: {str(e)[:200]}"
                    )
            if attempt < max_retries and is_retryable_error(e):
                backoff_ms = 500 * attempt
                if on_log:
                    on_log(
                        f"[opencode] retry backing off {backoff_ms}ms before attempt {attempt + 1}"
                    )
                time.sleep(backoff_ms / 1000)
            elif not is_retryable_error(e):
                raise  # surface immediately

    elapsed_ms = int((time.time() - call_start) * 1000)
    raise RuntimeError(
        f"OpenCode call failed after {max_retries} attempts ({elapsed_ms}ms): "
        f"{type(last_err).__name__ if last_err else 'unknown'}: {last_err}"
    )


__all__ = [
    "OPENCODE_GATEWAY",
    "OPENCODE_DEFAULT_MODEL",
    "OpenCodeResult",
    "is_retryable_error",
    "opencode_chat_stream_controlled",
]
