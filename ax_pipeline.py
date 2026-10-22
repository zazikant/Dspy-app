"""
AX DSPy quality gate — ported from atomic-graph-opencode (the "Bible"
AX DSPy reference implementation), serving this repo's output: the Streamlit
Scene → Prompt / PRD generator running on NVIDIA NIM (gpt-oss-20b,
4096-token cap + auto-continue) and OpenCode (GLM 5.3) transports.

The Bible pipeline is: EXTRACT → LINK → VALIDATE → REFINE with a
validate → refine loop gated by a 0.75 threshold. This module provides the
VALIDATE / REFINE layer for this app's generation workflow (v1 generation
and every Grok-refinement round both pass through it):

- AX_VALIDATE_SYSTEM_PROMPT — the Bible's VALIDATE rubric (weighted axes,
  exact JSON format, scoring bands, "Be FAIR") adapted to prompt/PRD output
- AX_REFINE_SYSTEM_PROMPT  — the Bible's REFINE prompt ("fix the identified
  issues while preserving what already works") for prompt/PRD repair
- compile_refine_prompt()   — DSPy-style Module.compile(): surgical fix
  context compiled from the full error history (word-to-word)
- get_stage_temperature()   — validate=0.1, refine=0.2 (Bible verbatim)
- calculate_max_tokens()    — Bible token budget (validate=1024 fixed;
  other stages floor 2048, cap 8192, estimate CJK/2 + other/4)
- parse_llm_json() + recover_truncated_json() — the Bible's robust JSON
  parsing chain for the validate response (4 strategies + 3-layer recovery)
- call_with_retry_awareness() — rate-limit / transient error detection with
  ErrorEntry history (feeds compile_refine_prompt on the next attempt)
- run_quality_gate()        — the Bible run() state machine
  (validate → refine → revalidate; VALIDATION_THRESHOLD 0.75,
  MAX_ITERATIONS 3; results carried FORWARD between stages) with a
  wall-clock budget so Streamlit hosting stays responsive

Echo detection (isEcho) is intentionally NOT ported: the output domain
(long-form prompts / PRD documents) never echoes the input domain (short
natural-language directions), and degenerate outputs are already caught by
is_valid_output() in app.py's retry wrapper.

Score boundary: the Bible's 0.0-1.0 score is mapped to the UI's 0-100
display at the boundary (score_to_100 = round(score * 100)), matching the
other AX ports (ax-opencode-translator 4444813, ax-translator a1d6651,
gsap-animation-pipeline 4ceae52, tradingview-notes-app 2b8c4dd).

Refine context guard: refine only runs when the current output fits in
REFINE_MAX_CONTEXT_CHARS — a surgical edit needs the FULL document in
context, and a partial-context rewrite would silently drop the unseen tail.
Outputs too large for refine still get scored + annotated (validate-only),
and the validator's issues double as ready-to-paste Grok feedback.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from nvidia_client import (
    nvidia_chat_stream_controlled as _nvidia_chat,
    MODEL_MAX_TOKENS_CAP as _NVIDIA_MODEL_MAX_TOKENS,
)
from opencode_client import (
    opencode_chat_stream_controlled as _opencode_chat,
)

# ─── Constants (from atomic-graph-opencode) ──────────────────────────────────

VALIDATION_THRESHOLD = 0.75  # Minimum validation score (0.0-1.0) to accept output
MAX_ITERATIONS = 3           # Maximum validate → refine loops

# Bible validate step: `calculateMaxTokens('', 'validate')` returns 1024.
VALIDATE_MAX_TOKENS = 1024

# Refine is a surgical edit of the FULL document — it only runs when the
# whole current output fits in context (see module docstring).
REFINE_MAX_CONTEXT_CHARS = 12000

# Direction truncation for gate prompts — the Bible truncates the original
# notes at 3000 chars in its validate step (`rawNotes.slice(0, 3000)`).
DIRECTIONS_TRUNCATE_CHARS = 3000

# ─── DSPy-style Error Tracking (from atomic-graph-opencode) ─────────────────


@dataclass
class ErrorEntry:
    """One entry per failed/errored call — feeds compile_refine_prompt()."""
    attempt: int
    stage: str            # "generate" | "validate" | "refine"
    error: str
    issues: Optional[List[str]] = None


# ─── Stage-Specific System Prompts (Bible structure, this repo's domain) ────
# Same structure word-to-word as the Bible's VALIDATE_SYSTEM_PROMPT:
# weighted axes, exact JSON format, scoring bands, and the "Be FAIR"
# instruction. Axes adapted from the Bible's semantic-fidelity rubric to
# prompt / PRD generation.

AX_VALIDATE_SYSTEM_PROMPT = """You are a prompt engineering quality reviewer. Evaluate the provided generated output and respond in JSON format.

Evaluate on these criteria (weight: direction fidelity > guideline adherence > completeness > actionability > structure):
1. DIRECTION FIDELITY: Does the output preserve the user's requested scene/feature meaning? Minor wording changes don't lower score.
2. GUIDELINE ADHERENCE: Does it follow the mode's system-prompt guidelines (composition/lighting/pose for image; shot/motion/temporal arc for video; ✅/⚠️/❌ markers and lean decisions for PRD; full spec depth for Exhaustive PRD)?
3. COMPLETENESS: Are all requested elements present? Nothing critical omitted or truncated mid-structure?
4. ACTIONABILITY: Ready-to-use as-is for the target consumer (image/video model or developer) — specific and concrete, no vague platitudes?
5. STRUCTURAL INTEGRITY: Required sections and markers all present (PRD: every numbered section plus Architecture Graveyard; prompts: one well-structured ready-to-use prompt)?

Respond in this exact JSON format:
{
  "score": 0.85,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1"]
}

Scoring:
- 0.90-1.00: Faithful representation, minor wording differences only
- 0.75-0.89: Minor gaps
- 0.50-0.74: Significant gaps
- 0.00-0.49: Major problems

Be FAIR — clear directions + good output = high score. Don't invent reasons to lower it.
Only list issues affecting MEANING or STRUCTURE."""

# Bible REFINE structure word-to-word: "You are a ... refinement engine.
# Fix the identified issues while preserving what already works. Output ONLY
# the corrected ... CRITICAL OUTPUT RULES: ... INFER and add it ... No
# speculative additions." Adapted to plain-text prompt/PRD repair.

AX_REFINE_SYSTEM_PROMPT = """You are a prompt and PRD refinement engine. Fix the identified issues while preserving what already works.
Output ONLY the corrected output as plain text — no preamble, no acknowledgement, no explanation, and no code fence around the whole output.

CRITICAL OUTPUT RULES:
- Preserve the user's directions and the previous version's wording where accurate — only fix CONTENT or STRUCTURAL problems.
- Missing required section or element? INFER it from the directions and add it.
- Vague or generic phrasing? Replace it with specific, concrete detail.
- Truncated or incomplete structure? COMPLETE it.
- No speculative additions — never invent requirements the user did not ask for.
- Keep the complete structure required for this mode (all required sections, markers, and format rules).
- Return the FULL corrected output with unchanged parts as-is — never a diff, never a summary of changes."""

# ─── compile_refine_prompt — pure function (DSPy-like) ──────────────────────
# Like DSPy's Module.compile() — produces focused prompts based on error
# history. This is the ONLY place refinement context construction happens.
# Ported word-to-word from atomic-graph-opencode's compileRefinePrompt().


def compile_refine_prompt(
    issues: List[str],
    error_history: List[ErrorEntry],
    stage: str,
) -> str:
    # No errors yet — this is initial context
    if len(error_history) == 0 and len(issues) == 0:
        return f"Initial {stage} request for prompt generation"

    # Build surgical context from error history (Mode B: surgical fix)
    latest_error = error_history[-1] if len(error_history) > 0 else None
    previous_errors = "\n".join(
        f"  Attempt {e.attempt} | {e.stage}: {e.error[:200]}"
        for e in error_history[:-1]
    )

    issue_context = (
        "\nValidation issues to fix:\n" + "\n".join(f"  - {i}" for i in issues)
        if len(issues) > 0
        else ""
    )

    error_context = (
        f"\nLatest error (attempt {latest_error.attempt}, stage {latest_error.stage}): {latest_error.error[:300]}\n"
        + (f"Issues: {', '.join(latest_error.issues)}" if latest_error.issues else "")
        if latest_error is not None
        else ""
    )

    previous_context = (
        f"\nPrevious errors — do NOT repeat these patterns:\n{previous_errors}"
        if len(previous_errors) > 0
        else ""
    )

    return f'Refinement context for stage "{stage}":{issue_context}{error_context}{previous_context}'


# ─── Stage-Specific Temperatures (from atomic-graph-opencode's axPipeline) ──
# Lower temperature = more deterministic/focused output.


def get_stage_temperature(stage: str) -> float:
    if stage == "generate":
        return 0.3   # Focused, deterministic generation
    if stage == "validate":
        return 0.1   # Very deterministic — objective quality assessment
    if stage == "refine":
        return 0.2   # Slightly creative — targeted fixes
    return 0.3


# ─── Token estimation + budget (from atomic-graph-opencode) ─────────────────


def estimate_tokens(text: str) -> int:
    """CJK chars ≈ 1 token per 2 chars; other chars ≈ 1 token per 4 chars."""
    cjk_chars = sum(
        1 for ch in text
        if "\u4e00" <= ch <= "\u9fff"      # CJK unified
        or "\u3040" <= ch <= "\u309f"      # Hiragana
        or "\u30a0" <= ch <= "\u30ff"      # Katakana
        or "\uac00" <= ch <= "\ud7af"      # Hangul
    )
    other_chars = len(text) - cjk_chars
    return math.ceil(cjk_chars / 2 + other_chars / 4)


def calculate_max_tokens(
    input_text: str,
    stage: str = "translate",
    is_same_language: bool = False,
) -> int:
    """Bible calculateMaxTokens: validate → fixed 1024; others floor 2048,
    cap 8192, same-language transformation gets the 4× multiplier.
    Values above NVIDIA's 4096 per-call cap are handled transparently by
    nvidia_client's auto-continue loop."""
    # Validation returns small JSON — fixed 1024 is enough
    if stage == "validate":
        return 1024

    input_tokens = estimate_tokens(input_text)
    # Same-language transformation produces structured output ≈ 3-5× input
    multiplier = 4 if (stage == "translate" and is_same_language) else 1.5
    output_tokens = math.ceil(input_tokens * multiplier)
    return max(2048, min(8192, output_tokens))


# ─── Robust JSON Parser (ported word-to-word from atomic-graph-opencode) ────


def parse_llm_json(raw: str) -> Dict:
    """Robust JSON parser for LLM outputs: strips fences, extracts balanced
    JSON, fixes trailing commas, falls back to greedy regex."""
    cleaned = raw.strip()

    fence = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    # Strategy 1: Try direct parse of the full text
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract balanced JSON using bracket matching.
    json_str = _extract_balanced_json(cleaned)
    if json_str:
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Fix common issues like trailing commas
    if json_str:
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: Try the greedy regex as a last resort before failing
    greedy = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if greedy:
        try:
            return json.loads(greedy.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(
        f"Failed to parse LLM JSON response. Output length: {len(raw)} chars.\n"
        f"First 300 chars: {raw[:300]}\nLast 300 chars: {raw[-300:]}"
    )


def _extract_balanced_json(text: str) -> Optional[str]:
    """Extract a balanced JSON object or array using bracket matching that
    respects string literals (ported word-to-word)."""
    start_curly = text.find("{")
    start_square = text.find("[")

    if start_curly == -1 and start_square == -1:
        return None
    if start_curly == -1:
        start_idx, open_ch, close_ch = start_square, "[", "]"
    elif start_square == -1:
        start_idx, open_ch, close_ch = start_curly, "{", "}"
    else:
        start_idx = min(start_curly, start_square)
        open_ch = "{" if start_idx == start_curly else "["
        close_ch = "}" if start_idx == start_curly else "]"

    depth = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == open_ch or ch == "{" or ch == "[":
            if ch == open_ch and depth == 0:
                depth = 1
            elif depth > 0:
                depth += 1
        elif ch == close_ch or ch == "}" or ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]

    return None


def recover_truncated_json(raw: str) -> Optional[Dict]:
    """Attempt to recover useful data from a truncated JSON response.
    Three-layer recovery chain, ported word-to-word: closeOpenBrackets →
    truncateToLastCompleteElement → recoverFieldsIndividually."""
    cleaned = raw.strip()

    fence = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    json_start = cleaned.find("{")
    if json_start == -1:
        return None
    cleaned = cleaned[json_start:]

    attempt1 = _close_open_brackets(cleaned)
    if attempt1 is not None:
        return attempt1

    attempt2 = _truncate_to_last_complete_element(cleaned)
    if attempt2 is not None:
        return attempt2

    attempt3 = _recover_fields_individually(cleaned)
    if attempt3 is not None:
        return attempt3

    return None


def _close_open_brackets(cleaned: str) -> Optional[Dict]:
    curly_depth = square_depth = 0
    in_string = escape = False

    for ch in cleaned:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            curly_depth += 1
        elif ch == "}":
            curly_depth -= 1
        elif ch == "[":
            square_depth += 1
        elif ch == "]":
            square_depth -= 1

    if curly_depth == 0 and square_depth == 0:
        return None

    attempt = _find_last_safe_truncation(cleaned)
    if attempt is None:
        return None

    new_curly = new_square = 0
    in_str = esc = False
    for ch in attempt:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            new_curly += 1
        elif ch == "}":
            new_curly -= 1
        elif ch == "[":
            new_square += 1
        elif ch == "]":
            new_square -= 1

    attempt += "]" * new_square
    attempt += "}" * new_curly
    attempt = re.sub(r",\s*([}\]])", r"\1", attempt)

    try:
        return json.loads(attempt)
    except (json.JSONDecodeError, ValueError):
        return None


def _find_last_safe_truncation(text: str) -> Optional[str]:
    """Find the last position where truncation keeps the text structurally
    safe (ported word-to-word from the Bible's recovery chain)."""
    string_state = [False] * len(text)
    is_str = is_esc = False
    for j, ch in enumerate(text):
        if is_esc:
            is_esc = False
            string_state[j] = is_str
            continue
        if ch == "\\" and is_str:
            is_esc = True
            string_state[j] = is_str
            continue
        if ch == '"':
            is_str = not is_str
        string_state[j] = is_str

    ends_in_string = string_state[-1]

    if ends_in_string:
        string_start_quote = -1
        for j in range(len(text) - 1, -1, -1):
            if text[j] == '"':
                state_before = string_state[j - 1] if j > 0 else False
                state_after = string_state[j]
                if not state_before and state_after:
                    string_start_quote = j
                    break

        if string_start_quote >= 0:
            before_quote = string_start_quote - 1
            while before_quote >= 0 and text[before_quote].isspace():
                before_quote -= 1

            if before_quote >= 0:
                prev_char = text[before_quote]
                if prev_char == ":":
                    obj_open_idx = -1
                    depth = 0
                    for k in range(before_quote, -1, -1):
                        if text[k] == "}" and not string_state[k]:
                            depth += 1
                        elif text[k] == "{" and not string_state[k]:
                            if depth == 0:
                                obj_open_idx = k
                                break
                            depth -= 1
                    if obj_open_idx >= 0:
                        before_obj = obj_open_idx - 1
                        while before_obj >= 0 and text[before_obj].isspace():
                            before_obj -= 1
                        if before_obj >= 0 and text[before_obj] == ",":
                            return re.sub(r",\s*$", "", text[:before_obj])
                        if before_obj >= 0 and text[before_obj] == "[":
                            return text[:before_obj + 1]
                        if before_obj >= 0 and text[before_obj] == "{":
                            return text[:before_obj + 1]
                        return re.sub(r",\s*$", "", text[:obj_open_idx])
                    return re.sub(r",\s*$", "", text[:before_quote])
                if prev_char == "," or prev_char == "[":
                    cut = before_quote + (1 if prev_char == "[" else 0)
                    return re.sub(r",\s*$", "", text[:cut])
            return re.sub(r",\s*$", "", text[:string_start_quote])

    i = len(text) - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0:
        return None

    ch = text[i]

    if (ch == "}" or ch == "]") and not string_state[i]:
        end = i
        k = i - 1
        while k >= 0 and text[k].isspace():
            k -= 1
        if k >= 0 and text[k] == ",":
            end = k
        return re.sub(r",\s*$", "", text[:end + 1])

    if ch == '"' and not string_state[i]:
        return re.sub(r",\s*$", "", text[:i + 1])

    if ch.isdigit():
        num_end = i
        j = i
        while j >= 0 and (text[j].isdigit() or text[j] in ".eE+-"):
            j -= 1
        before_num = j
        while before_num >= 0 and text[before_num].isspace():
            before_num -= 1
        if before_num >= 0 and text[before_num] in ":,[":
            return re.sub(r",\s*$", "", text[:num_end + 1])

    for j in range(len(text) - 1, -1, -1):
        if (text[j] == "}" or text[j] == "]") and not string_state[j]:
            return re.sub(r",\s*$", "", text[:j + 1])

    return None


def _truncate_to_last_complete_element(cleaned: str) -> Optional[Dict]:
    string_state = [False] * len(cleaned)
    is_str = is_esc = False
    for j, ch in enumerate(cleaned):
        if is_esc:
            is_esc = False
            string_state[j] = is_str
            continue
        if ch == "\\" and is_str:
            is_esc = True
            string_state[j] = is_str
            continue
        if ch == '"':
            is_str = not is_str
        string_state[j] = is_str

    last_complete_obj_end = -1
    for j in range(len(cleaned) - 1, -1, -1):
        if cleaned[j] == "}" and not string_state[j]:
            last_complete_obj_end = j
            break

    last_complete_arr_end = -1
    for j in range(len(cleaned) - 1, -1, -1):
        if cleaned[j] == "]" and not string_state[j]:
            last_complete_arr_end = j
            break

    best_idx = max(last_complete_obj_end, last_complete_arr_end)
    if best_idx <= 0:
        return None

    truncated = re.sub(r",\s*$", "", cleaned[:best_idx + 1])

    curly = square = 0
    in_str = esc = False
    for ch in truncated:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            curly += 1
        elif ch == "}":
            curly -= 1
        elif ch == "[":
            square += 1
        elif ch == "]":
            square -= 1

    truncated += "]" * square
    truncated += "}" * curly
    truncated = re.sub(r",\s*([}\]])", r"\1", truncated)

    try:
        return json.loads(truncated)
    except (json.JSONDecodeError, ValueError):
        return None


def _recover_fields_individually(cleaned: str) -> Optional[Dict]:
    """Last-resort field-by-field recovery for the validate JSON shape
    (score / issues / suggestions) — ported word-to-word."""
    try:
        result: Dict = {}

        score_match = re.search(r'"score"\s*:\s*([\d.]+)', cleaned)
        if score_match:
            result["score"] = float(score_match.group(1))

        issues_match = re.search(r'"issues"\s*:\s*\[', cleaned)
        if issues_match:
            issues_start = cleaned.find("[", cleaned.find('"issues"'))
            recovered_issues = _recover_array_elements(cleaned, issues_start)
            if recovered_issues:
                result["issues"] = recovered_issues

        suggestions_match = re.search(r'"suggestions"\s*:\s*\[', cleaned)
        if suggestions_match:
            suggestions_start = cleaned.find("[", cleaned.find('"suggestions"'))
            recovered_suggestions = _recover_array_elements(cleaned, suggestions_start)
            if recovered_suggestions:
                result["suggestions"] = recovered_suggestions

        if len(result) > 0:
            return result
    except Exception:
        pass

    return None


def _recover_array_elements(text: str, array_start_index: int) -> Optional[List]:
    if array_start_index == -1:
        return None

    elements: List = []
    depth = 0
    obj_start = -1
    in_string = False
    escape = False

    for i in range(array_start_index, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start != -1:
                obj_str = text[obj_start:i + 1]
                try:
                    elements.append(json.loads(obj_str))
                except (json.JSONDecodeError, ValueError):
                    pass
                obj_start = -1

    return elements


# ─── call_with_retry_awareness (ported from atomic-graph-opencode) ──────────


def call_with_retry_awareness(
    fn: Callable[[], object],
    error_history: List[ErrorEntry],
    attempt: int,
    phase_label: str,
    stage: str,
    log: Optional[Callable[[str], None]] = None,
):
    """Wrap a callable with rate-limit / transient error detection.
    Records errors to errorHistory for DSPy-style compile_refine_prompt().
    Ported word-to-word from atomic-graph-opencode's callWithRetryAwareness().
    """
    try:
        return fn()
    except Exception as error:
        msg = str(error)

        # Track error in history for DSPy-style compiled prompts
        error_history.append(ErrorEntry(attempt=attempt, stage=stage, error=msg))

        is_rate_limit = (
            "rate limit" in msg
            or "Rate limit" in msg
            or "429" in msg
            or "retry attempt" in msg
            or "too many requests" in msg
        )

        is_transient = (
            "server error" in msg
            or "Server Error" in msg
            or "502" in msg
            or "503" in msg
            or "504" in msg
            or "temporary" in msg
        )

        if is_rate_limit:
            log(
                f"[AX Pipeline] retrying | iteration={attempt} — Rate limited during {phase_label} — "
                f"all proxy retries exhausted. Please wait and try again."
            )
        elif is_transient:
            log(
                f"[AX Pipeline] retrying | iteration={attempt} — Transient server error during {phase_label} — "
                f"all proxy retries exhausted. Try again shortly."
            )

        raise error


# ─── Result shapes ───────────────────────────────────────────────────────────


@dataclass
class IterationLog:
    """Bible IterationLog fields (word-to-word)."""
    iteration: int
    phase: str            # "validating" | "refining" | "complete"
    score: float
    passed: bool
    issues: Optional[List[str]] = None
    detail: Optional[str] = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class AxGateResult:
    """Quality-gate outcome — the Bible PipelineResult fields mapped onto
    this repo's text output (result → output, 0-1 score kept on the Bible
    scale; UI maps to 0-100 via score_to_100)."""
    output: str
    score: float
    attempts: int
    passed: bool
    issues: List[str]
    suggestions: List[str]
    iterations: List[IterationLog]
    refinements: int          # number of accepted refine rounds (0 = output untouched)
    budget_exceeded: bool     # True if the wall-clock budget stopped the loop early


def score_to_100(score: float) -> int:
    """Bible 0.0-1.0 → UI 0-100 boundary mapping (Math.round(score*100))."""
    return int(score * 100 + 0.5)


# ─── Transport adapter ───────────────────────────────────────────────────────


def make_call_llm(provider: str, api_key: str, model: str) -> Callable[..., str]:
    """Build the gate's single LLM entry point on top of this repo's
    existing streaming clients (same transports the app already uses).

    - NVIDIA NIM: per-call 4096-token cap + auto-continue chaining — the
      Bible's stage-based max_tokens values (up to 8192) are safe.
    - OpenCode: direct gateway call (GLM 5.3, reasoning_effort=low).
    """
    if provider == "OpenCode (GLM 5.3)":
        def call_llm(system_prompt: str, user_content: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.3) -> str:
            result = _opencode_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                api_key=api_key,
                model=model,
                max_tokens=max_tokens if max_tokens is not None else 8192,
                temperature=temperature,
            )
            return result.content
    else:
        def call_llm(system_prompt: str, user_content: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.3) -> str:
            result = _nvidia_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                api_key=api_key,
                model=model,
                max_tokens=max_tokens if max_tokens is not None else _NVIDIA_MODEL_MAX_TOKENS,
                temperature=temperature,
                top_p=1.0,
            )
            return result.content

    return call_llm


# ─── Stage prompt builders (Bible validate/refine user-prompt structure) ────


def _build_validate_user_prompt(output: str, original_input: str, mode: str) -> str:
    """Ported word-to-word from the Bible's validate step user prompt:
    axes → scoring bands → Be FAIR → original input (3000-char truncation)
    → artifact → Return JSON."""
    directions = (
        original_input[:DIRECTIONS_TRUNCATE_CHARS] + "\n… (directions truncated)"
        if len(original_input) > DIRECTIONS_TRUNCATE_CHARS
        else original_input
    )
    # Long artifacts (32k-token Exhaustive PRDs) are windowed for the
    # validator — the Bible windows its notes the same way.
    output_windowed = (
        output[:6000] + "\n… (output truncated)"
        if len(output) > 6000
        else output
    )

    return f"""Evaluate this generated output for quality and direction fidelity.

Axes (weight: direction fidelity > guideline adherence > completeness > actionability > structure):
1. DIRECTION FIDELITY: Does the output preserve the user's requested meaning? Minor wording changes don't lower score.
2. GUIDELINE ADHERENCE: Does it follow the mode's system-prompt guidelines for "{mode}"?
3. COMPLETENESS: Are all requested elements present? Nothing critical omitted or truncated mid-structure?
4. ACTIONABILITY: Ready-to-use as-is for the target consumer — specific and concrete, no vague platitudes?
5. STRUCTURAL INTEGRITY: Required sections and markers all present and complete?

Scoring:
- 0.90-1.00: Faithful representation, minor wording differences only
- 0.75-0.89: Minor gaps
- 0.50-0.74: Significant gaps
- 0.00-0.49: Major problems

Be FAIR — clear directions + good output = high score. Don't invent reasons to lower it.

ORIGINAL DIRECTIONS:
{directions}

GENERATED OUTPUT:
\"\"\"
{output_windowed}
\"\"\"

Return JSON: {{ "score": 0.85, "issues": ["..."], "suggestions": ["..."] }}
Only list issues affecting MEANING or STRUCTURE."""


def _build_refine_user_prompt(
    output: str,
    original_input: str,
    mode: str,
    issues: List[str],
    fix_context: str,
) -> str:
    """Ported word-to-word from the Bible's refine step user prompt:
    Rules → compiled fix context → Return the FULL corrected artifact →
    Issues → Current artifact."""
    directions = (
        original_input[:DIRECTIONS_TRUNCATE_CHARS] + "\n… (directions truncated)"
        if len(original_input) > DIRECTIONS_TRUNCATE_CHARS
        else original_input
    )
    issue_bullets = "\n".join(f"- {i}" for i in issues)

    return f"""Fix ONLY these specific issues in the generated output. Mode: {mode}

Rules:
- Preserve the user's directions and existing wording where accurate
- Only fix CONTENT or STRUCTURAL problems, not style
- Missing required element? INFER and add it.
- Vague or generic phrasing? Replace with specific, concrete detail.
- Truncated or incomplete structure? COMPLETE it.
- No speculative additions.
- Keep the complete structure required for this mode.
- Return the FULL corrected output — unchanged parts as-is.
- {fix_context}

USER DIRECTIONS:
{directions}

CURRENT OUTPUT:
{output}

Issues:
{issue_bullets}

Return ONLY the full corrected output text — no explanations, no code fence around the whole output."""


def _strip_outer_fence(text: str) -> str:
    """Light guard: if the refine output came back wrapped in a single code
    fence, unwrap it. (The refine system prompt already forbids this.)"""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            return stripped[first_newline + 1:-3].strip()
    return stripped


# ─── The quality gate (Bible run() state machine, validate→refine tail) ─────
# The Bible's run(): extract → link → validate → (if fail) refine → validate
# → done, with results carried FORWARD between stages. Here generation
# already happened in app.py (it consumes the first attempt, like the
# Bible's extract), so the gate starts at resume_from = "validate" and runs:
# validate → (if score < 0.75) refine → revalidate → done.


def run_quality_gate(
    output: str,
    original_input: str,
    mode: str,
    call_llm: Callable[..., str],
    on_log: Optional[Callable[[str], None]] = None,
    time_budget_s: float = 120.0,
    threshold: float = VALIDATION_THRESHOLD,
    iterations: int = MAX_ITERATIONS,
) -> AxGateResult:
    started = time.time()
    error_history: List[ErrorEntry] = []
    iter_logs: List[IterationLog] = []

    score = 0.0
    attempt = 1  # generation consumed the first attempt (Bible's extract slot)
    current_issues: List[str] = []
    current_suggestions: List[str] = []
    current_output = output
    refinements = 0
    budget_exceeded = False

    def emit(phase: str, it: int, sc: float, passed: bool,
             issues: Optional[List[str]] = None, detail: Optional[str] = None) -> None:
        if on_log:
            line = f"[AX gate] {phase} | iteration={it} score={sc:.2f} passed={passed}"
            if detail:
                line += f" — {detail}"
            on_log(line)

    def within_budget() -> bool:
        return (time.time() - started) < time_budget_s

    def _validate_once() -> Dict:
        prompt = _build_validate_user_prompt(current_output, original_input, mode)
        raw = call_llm(
            AX_VALIDATE_SYSTEM_PROMPT,
            prompt,
            max_tokens=VALIDATE_MAX_TOKENS,
            temperature=get_stage_temperature("validate"),
        )
        try:
            parsed = parse_llm_json(raw)
        except ValueError:
            recovered = recover_truncated_json(raw)
            if recovered is None:
                raise
            parsed = recovered
        # Bible clamp, word-to-word:
        return {
            "score": min(1, max(0, parsed["score"])) if isinstance(parsed.get("score"), (int, float)) else 0,
            "issues": parsed.get("issues") if isinstance(parsed.get("issues"), list) else [],
            "suggestions": parsed.get("suggestions") if isinstance(parsed.get("suggestions"), list) else [],
        }

    def _refine_once() -> str:
        fix_context = compile_refine_prompt(current_issues, error_history, "refine")
        prompt = _build_refine_user_prompt(
            current_output, original_input, mode, current_issues, fix_context
        )
        refine_budget = calculate_max_tokens(current_output, "refine")
        raw = call_llm(
            AX_REFINE_SYSTEM_PROMPT,
            prompt,
            max_tokens=refine_budget,
            temperature=get_stage_temperature("refine"),
        )
        return _strip_outer_fence(raw)

    # State machine: validate → refine → validate → … (Bible run() loop)
    resume_from = "validate"
    while attempt < iterations and score < threshold and resume_from != "done":
        # ─── Stage: VALIDATE ─────────────────────────────────────────
        if resume_from == "validate":
            if not within_budget():
                budget_exceeded = True
                emit("validating", attempt, score, False,
                     current_issues, "Time budget reached — returning best result so far")
                break
            emit("validating", attempt, 0, False)
            try:
                validation = call_with_retry_awareness(
                    _validate_once, error_history, attempt, "Validate", "validate", on_log,
                )
                score = validation["score"]
                current_issues = validation["issues"]
                current_suggestions = validation["suggestions"]

                passed = score >= threshold
                emit("validating", attempt, score, passed, validation["issues"])

                iter_logs.append(IterationLog(
                    iteration=attempt,
                    phase="complete" if passed else "validating",
                    score=score,
                    passed=passed,
                    issues=validation["issues"],
                ))

                if passed:
                    resume_from = "done"
                elif attempt < iterations:
                    resume_from = "refine"
                else:
                    resume_from = "done"
            except Exception:
                # Bible fallback, word-to-word: estimated score keeps the
                # pipeline moving instead of dying on a validator error.
                score = 0.7
                current_issues = ["Validation step failed — using estimated score"]

                iter_logs.append(IterationLog(
                    iteration=attempt,
                    phase="validating",
                    score=score,
                    passed=score >= threshold,
                    issues=current_issues,
                ))

                if score >= threshold:
                    resume_from = "done"
                elif attempt < iterations:
                    resume_from = "refine"
                else:
                    resume_from = "done"

        # ─── Stage: REFINE (DSPy-compiled prompt) ────────────────────
        if resume_from == "refine":
            if not within_budget():
                budget_exceeded = True
                emit("refining", attempt, score, False,
                     current_issues, "Time budget reached — returning validated (unrefined) result")
                break
            if len(current_output) > REFINE_MAX_CONTEXT_CHARS:
                # Surgical edit needs the FULL document in context — a
                # partial-context rewrite would drop the unseen tail.
                emit("refining", attempt, score, False,
                     current_issues,
                     f"Output too large for refine ({len(current_output)} chars) — validate-only")
                resume_from = "done"
            else:
                attempt += 1
                emit("refining", attempt, score, False)
                try:
                    refined = call_with_retry_awareness(
                        _refine_once, error_history, attempt, "Refine", "refine", on_log,
                    )
                    # Truncation guard: a collapsed/truncated rewrite is worse
                    # than the validated original — keep the original instead.
                    if refined and len(refined.strip()) >= max(100, len(current_output.strip()) * 0.5):
                        current_output = refined
                        refinements += 1
                    else:
                        emit("refining", attempt, score, False,
                             current_issues, "Refine output incomplete — keeping validated original")
                    # After refine, loop back to validate (like repo 2's revalidate)
                    resume_from = "validate"
                except Exception:
                    # Bible behavior: keep current result, loop back to validate
                    resume_from = "validate"

    passed = score >= threshold
    emit("complete", attempt, score, passed,
         None, f"{refinements} refine round(s)" + (" · budget capped" if budget_exceeded else ""))

    return AxGateResult(
        output=current_output,
        score=score,
        attempts=attempt,
        passed=passed,
        issues=current_issues,
        suggestions=current_suggestions,
        iterations=iter_logs,
        refinements=refinements,
        budget_exceeded=budget_exceeded,
    )


__all__ = [
    "VALIDATION_THRESHOLD",
    "MAX_ITERATIONS",
    "VALIDATE_MAX_TOKENS",
    "REFINE_MAX_CONTEXT_CHARS",
    "AX_VALIDATE_SYSTEM_PROMPT",
    "AX_REFINE_SYSTEM_PROMPT",
    "ErrorEntry",
    "IterationLog",
    "AxGateResult",
    "compile_refine_prompt",
    "get_stage_temperature",
    "estimate_tokens",
    "calculate_max_tokens",
    "parse_llm_json",
    "recover_truncated_json",
    "call_with_retry_awareness",
    "make_call_llm",
    "run_quality_gate",
    "score_to_100",
]
