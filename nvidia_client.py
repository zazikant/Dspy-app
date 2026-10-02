"""
NVIDIA NIM LLM client — streaming chat completions with auto-continue on truncation.

Python port of the TypeScript `nvidia-client.ts` pattern from ax-translator /
google-ads-subagent-vercel / gsap-animation-pipeline. Each call:
  - Streams the response (chunk-by-chunk) from openai/gpt-oss-20b
  - Has a hard per-call timeout (DEFAULT_TIMEOUT_S)
  - Retries once with backoff on transient failures (429 / 5xx / network)
  - Auto-continues when finish_reason === 'length' — automatically sends another
    call with the partial output appended as an assistant message + a generic
    "continue from where you left off" user prompt, then concatenates.
    gpt-oss-20b has a 4096-token output limit per call; with 7 continuation
    rounds (8 total calls × 4096 = 32768 tokens), we achieve ~32K tokens of
    effective output capacity — the same budget the UI advertises.
  - Emits structured log lines via on_log so the Streamlit status widget
    can show progress

Required env vars:
    NVIDIA_NIM_API_KEY   — your NVIDIA NIM API key (from https://build.nvidia.com/)

Gateway: https://integrate.api.nvidia.com/v1/chat/completions
Model:   openai/gpt-oss-20b (20B params, fast ~2-4s TTFB)

Retry / rate-handling (ported from nvidia.ts):
    Retryable HTTP:   429, 500, 502, 503, 504
    Retryable codes:  ECONNRESET, ETIMEDOUT, UND_ERR_CONNECT_TIMEOUT
    Retryable names:  APIConnectionError, APITimeoutError, ConnectionError
    Per-call timeout: 55s (Streamlit Cloud caps free apps around 60s)
    Max attempts:     1 per round (pipeline-level retry handles additional)
    Backoff:          500ms × attempt
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

NVIDIA_GATEWAY = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_DEFAULT_MODEL = "openai/gpt-oss-20b"
NVIDIA_DEFAULT_TIMEOUT_S = 55.0
NVIDIA_DEFAULT_TEMPERATURE = 0.5
NVIDIA_DEFAULT_TOP_P = 1.0

# gpt-oss-20b has a hard 4096-token output limit per call. We cap max_tokens
# at this value regardless of what the caller requests — sending higher values
# would either be silently clamped by the API or cause a 400 error.
# The auto-continue loop (below) chains multiple 4096-token calls to produce
# longer outputs transparently.
MODEL_MAX_TOKENS_CAP = 4096

# Max auto-continue rounds when the model returns finish_reason === 'length'.
# Each round re-calls the model with the partial output appended as an
# assistant message, asking it to continue. 7 rounds gives 8 total calls
# × 4096 tokens = 32768 tokens of effective output capacity — matching
# the "32k output" the UI advertises.
DEFAULT_MAX_CONTINUATIONS = 7

# Generic continuation prompt used when the model returns finish_reason:'length'.
# This does NOT modify the caller's system/user prompts — it's a fixed
# instruction appended only when a continuation round is needed. Works for
# both free-form text AND structured output: the model sees its partial
# output in the assistant message and continues from exactly where it left off.
CONTINUE_USER_PROMPT = (
    "Continue your previous response from exactly where you left off. "
    "Do not repeat any text you have already produced. "
    "Do not add any preamble, acknowledgements, or summary — "
    "output only the continuation."
)


@dataclass
class NvidiaResult:
    content: str
    reasoning: str
    model: str
    elapsed_ms: int
    attempts: int
    # Number of continuation rounds that were triggered (0 if the model
    # finished in one call).
    continuations: int
    # True if the model exhausted all continuations and is STILL truncated.
    truncated: bool


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


def _parse_sse_line(line: str, content_parts: list[str], reasoning_parts: list[str]) -> Optional[str]:
    """Parse one SSE `data: ...` line.

    Returns:
        'done'        — if the line is [DONE] (stream end sentinel)
        'length'      — if finish_reason === 'length' (truncated)
        'stop'        — if finish_reason === 'stop' (natural end)
        None          — if the line is a delta chunk (content/reasoning appended)
    """
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "done"
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        # Partial JSON across chunks — will be retried on the next read
        return None
    choice = (event.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    if isinstance(delta.get("content"), str) and delta["content"]:
        content_parts.append(delta["content"])
    if isinstance(delta.get("reasoning_content"), str):
        reasoning_parts.append(delta["reasoning_content"])
    # Capture finish_reason — this is the key field that drives auto-continue.
    finish_reason = choice.get("finish_reason")
    if finish_reason:
        return finish_reason
    return None


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
) -> tuple[list[str], list[str], int, Optional[str]]:
    """Make a single streaming chat-completion call.

    Returns (content_parts, reasoning_parts, ttfb_ms, finish_reason).
    Raises requests.HTTPError for non-2xx, requests.exceptions.Timeout on timeout.
    """
    # Cap max_tokens at the model's hard limit (4096 for gpt-oss-20b).
    effective_max_tokens = min(max_tokens, MODEL_MAX_TOKENS_CAP)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": effective_max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        # gpt-oss-20b is a reasoning model. reasoning_effort:'low' keeps TTFB
        # fast (~2-4s) while still producing a small reasoning trail for
        # debugging. NVIDIA's API accepts this param for reasoning models and
        # silently ignores it for non-reasoning models.
        "reasoning_effort": "low",
    }
    if on_log:
        on_log(
            f"[nvidia] start  model={model} max_tokens={effective_max_tokens} "
            f"temp={temperature} top_p={top_p} timeout={timeout_s}s"
        )

    t0 = time.time()
    response = requests.post(
        NVIDIA_GATEWAY,
        headers=headers,
        json=body,
        stream=True,
        timeout=timeout_s,
    )
    if not response.ok:
        # Surface status code on the exception so is_retryable_error() can see it
        err = requests.HTTPError(
            f"NVIDIA API error ({response.status_code}): {response.text[:300]}",
            response=response,
        )
        err.status = response.status_code  # type: ignore[attr-defined]
        raise err
    if response.raw is None:
        raise RuntimeError("NVIDIA API returned no response body")

    ttfb_ms: Optional[int] = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: Optional[str] = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        if ttfb_ms is None:
            ttfb_ms = int((time.time() - t0) * 1000)
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        result = _parse_sse_line(line, content_parts, reasoning_parts)
        # Fire chunk callback for content deltas (live UI updates)
        if on_chunk and content_parts:
            new_chunk = content_parts[-1]
            on_chunk(new_chunk)
        if result == "done":
            break
        elif result and result != "stop":
            # finish_reason captured (e.g. 'length', 'content_filter')
            finish_reason = result

    # Reasoning-as-content fallback: if the model returned only
    # reasoning_content (no content), surface it as content instead of
    # returning empty.
    if not content_parts and reasoning_parts:
        content_parts[:] = ["".join(reasoning_parts)]
        if on_chunk:
            on_chunk(content_parts[0])

    return content_parts, reasoning_parts, ttfb_ms or 0, finish_reason


def nvidia_chat_stream_controlled(
    *,
    messages: list[dict],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = NVIDIA_DEFAULT_TEMPERATURE,
    top_p: float = NVIDIA_DEFAULT_TOP_P,
    max_tokens: int = MODEL_MAX_TOKENS_CAP,
    timeout_s: float = NVIDIA_DEFAULT_TIMEOUT_S,
    max_retries: int = 1,
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
    on_log: Optional[Callable[[str], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> NvidiaResult:
    """Controlled, logged, time-bounded chat completion with retry + auto-continue.

    Mirrors nvidiaChatCompletion from ax-translator/src/lib/nvidia-client.ts.
    Returns NvidiaResult with full content + reasoning + timing metadata.

    Auto-continue behavior:
        When the model returns `finish_reason: "length"`, it means the output
        was truncated at max_tokens mid-generation. Instead of returning a
        truncated response, we automatically send another call with the partial
        output appended as an assistant message + a generic "continue from
        where you left off" user prompt, then concatenate. The caller sees
        continuous content with no visible boundary between the original call
        and continuation(s). Capped at max_continuations (default 7) to
        achieve ~32K tokens of effective output capacity with gpt-oss-20b's
        4096-token per-call limit.
    """
    api_key = api_key or os.getenv("NVIDIA_NIM_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_NIM_API_KEY env var is not set. Add it in the Streamlit sidebar."
        )
    model = model or NVIDIA_DEFAULT_MODEL
    call_start = time.time()

    # Strip the litellm provider prefix (e.g. "nvidia_nim/openai/gpt-oss-20b"
    # → "openai/gpt-oss-20b") — we call NVIDIA's API directly, not via litellm.
    if model.startswith("nvidia_nim/"):
        model = model[len("nvidia_nim/"):]

    if on_log:
        on_log(
            f"[nvidia] start  model={model} max_tokens={min(max_tokens, MODEL_MAX_TOKENS_CAP)} "
            f"temp={temperature} timeout={timeout_s}s max_continuations={max_continuations}"
        )

    # Accumulate across the original call + any continuation rounds.
    full_content_parts: list[str] = []
    full_reasoning_parts: list[str] = []
    attempts_used = 0
    continuations = 0
    still_truncated = False
    last_err: Optional[Exception] = None

    # The messages array may grow across continuation rounds: each round
    # appends the assistant's partial output + the generic continue prompt.
    round_messages = list(messages)

    # Loop: original call (round 0) + up to max_continuations continuation rounds.
    for round_idx in range(max_continuations + 1):
        round_content_parts: list[str] = []
        round_reasoning_parts: list[str] = []
        round_finish_reason: Optional[str] = None
        round_success = False

        # ─── Per-round retry loop (handles transient errors) ───────────
        for attempt in range(1, max_retries + 1):
            try:
                content_parts, reasoning_parts, ttfb_ms, finish_reason = _stream_once(
                    messages=round_messages,
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                    on_log=on_log,
                    on_chunk=on_chunk,
                )
                attempts_used += 1
                round_content_parts = content_parts
                round_reasoning_parts = reasoning_parts
                round_finish_reason = finish_reason
                round_success = True

                elapsed_ms = int((time.time() - call_start) * 1000)
                round_content = "".join(round_content_parts)
                total_so_far = len("".join(full_content_parts)) + len(round_content)
                if on_log:
                    on_log(
                        f"[nvidia] round={round_idx} done  ttfb={ttfb_ms}ms "
                        f"elapsed={elapsed_ms}ms content_chars={len(round_content)} "
                        f"(total={total_so_far}) finish_reason={round_finish_reason or 'n/a'}"
                    )
                break  # success — exit retry loop, move to continuation check
            except Exception as e:
                elapsed_ms = int((time.time() - call_start) * 1000)
                last_err = e
                err_name = type(e).__name__
                if err_name == "Timeout" or "timeout" in str(e).lower():
                    if on_log:
                        on_log(f"[nvidia] TIMEOUT round={round_idx} attempt={attempt} after {timeout_s}s")
                else:
                    if on_log:
                        on_log(
                            f"[nvidia] ERROR round={round_idx} attempt={attempt} after {elapsed_ms}ms: "
                            f"{err_name}: {str(e)[:200]}"
                        )
                # Non-retryable errors surface immediately.
                if not is_retryable_error(e):
                    if round_idx == 0:
                        raise  # no content at all — surface the error
                    break  # we have partial content — fall through to truncated return
                if attempt < max_retries:
                    backoff_ms = 500 * attempt
                    if on_log:
                        on_log(f"[nvidia] retry backing off {backoff_ms}ms before attempt {attempt + 1}")
                    time.sleep(backoff_ms / 1000)

        if not round_success:
            # Round failed — if this is round 0, throw the error (no content at all).
            # If we already have partial content from earlier rounds, return it
            # with truncated=True so the caller knows the output is incomplete.
            if round_idx == 0:
                elapsed_ms = int((time.time() - call_start) * 1000)
                raise RuntimeError(
                    f"NVIDIA call failed after {max_retries} attempts ({elapsed_ms}ms): "
                    f"{type(last_err).__name__ if last_err else 'unknown'}: {last_err}"
                )
            still_truncated = True
            if on_log:
                on_log(f"[nvidia] TRUNCATED at round {round_idx} — upstream error after {continuations} continuation(s)")
            break

        # Accumulate content/reasoning across rounds.
        full_content_parts.extend(round_content_parts)
        full_reasoning_parts.extend(round_reasoning_parts)

        if round_finish_reason != "length":
            # Model finished naturally — no continuation needed.
            still_truncated = False
            break

        # finish_reason === 'length' → output was truncated.
        # If we have continuation budget left, append the partial output as
        # an assistant message + the generic continue prompt, and loop again.
        round_content = "".join(round_content_parts)
        if round_idx >= max_continuations or not round_content:
            still_truncated = True
            if on_log:
                on_log(
                    f"[nvidia] TRUNCATED after {round_idx + 1} round(s) — "
                    f"exhausted max_continuations={max_continuations}. Output ends mid-structure."
                )
            break

        continuations += 1
        total_chars = len("".join(full_content_parts))
        if on_log:
            on_log(
                f"[nvidia] continue  round={round_idx + 1}/{max_continuations} — "
                f"model hit max_tokens, resuming from char {total_chars}"
            )

        # Build the next round's messages: original + assistant's partial + continue prompt.
        round_messages = list(messages) + [
            {"role": "assistant", "content": round_content},
            {"role": "user", "content": CONTINUE_USER_PROMPT},
        ]

    full_content = "".join(full_content_parts)
    full_reasoning = "".join(full_reasoning_parts)

    if not full_content:
        elapsed_ms = int((time.time() - call_start) * 1000)
        raise RuntimeError(
            f"NVIDIA produced no content after {attempts_used} attempt(s) ({elapsed_ms}ms): "
            f"{type(last_err).__name__ if last_err else 'unknown'}: {last_err}"
        )

    elapsed_ms = int((time.time() - call_start) * 1000)
    if on_log:
        on_log(
            f"[nvidia] done   elapsed={elapsed_ms}ms content_chars={len(full_content)} "
            f"reasoning_chars={len(full_reasoning)} attempts={attempts_used} "
            f"continuations={continuations} truncated={still_truncated}"
        )

    return NvidiaResult(
        content=full_content,
        reasoning=full_reasoning,
        model=model,
        elapsed_ms=elapsed_ms,
        attempts=attempts_used,
        continuations=continuations,
        truncated=still_truncated,
    )


__all__ = [
    "NVIDIA_GATEWAY",
    "NVIDIA_DEFAULT_MODEL",
    "MODEL_MAX_TOKENS_CAP",
    "DEFAULT_MAX_CONTINUATIONS",
    "NvidiaResult",
    "is_retryable_error",
    "nvidia_chat_stream_controlled",
]
