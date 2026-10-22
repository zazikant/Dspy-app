Never Change this code, any changes leads to abrupt behaviour of Dspy.

Only replace model names from openrouter as and when required.

## AX DSPy Quality Gate (ported from atomic-graph-opencode)

`ax_pipeline.py` ports the AX DSPy quality gate word-to-word from
[zazikant/atomic-graph-opencode](https://github.com/zazikant/atomic-graph-opencode)
(the reference "Bible" implementation) — the same port already shipped in
ax-opencode-translator (4444813), ax-translator (a1d6651),
gsap-animation-pipeline (4ceae52) and tradingview-notes-app (2b8c4dd).

Every v1 generation AND every Grok-refinement round now passes through:

```
generate ──► VALIDATE (rubric LLM, temp 0.1, 1024 tok)
                │  score ≥ 0.75 ──► accept (badge: AX NN/100)
                ▼  score < 0.75
             REFINE (compiled fix context, temp 0.2)   ── max 3 iterations
                │                                       ── threshold 0.75
                └────► re-validate
```

Ported verbatim from the Bible:

- `compile_refine_prompt()` — DSPy-style `Module.compile()`: surgical fix
  context compiled from the full `ErrorEntry` history
- `call_with_retry_awareness()` — rate-limit / transient error detection
  with error history (feeds the compiled refine prompt)
- `parse_llm_json()` + `recover_truncated_json()` — 4-strategy parsing +
  3-layer truncation recovery chain for the validator JSON
- `get_stage_temperature()` — validate=0.1, refine=0.2 (Bible verbatim)
- `calculate_max_tokens()` — Bible token budget (validate=1024 fixed;
  others floor 2048 / cap 8192, CJK/2 + other/4 estimation)
- `run_quality_gate()` — the Bible `run()` state machine: validate → refine
  → revalidate, threshold 0.75, MAX_ITERATIONS 3, results carried forward
- VALIDATE rubric structure — weighted axes, exact JSON format, scoring
  bands, "Be FAIR" (axes adapted to prompt/PRD output)

Repo-specific adaptations:

- Refine runs only when the full output fits the refine context window
  (`REFINE_MAX_CONTEXT_CHARS`) — a partial-context rewrite would drop the
  unseen tail. Oversized outputs still get scored + annotated.
- A wall-clock budget (`AX_GATE_TIME_BUDGET_S = 120`) caps the gate so
  Streamlit hosting stays responsive; on expiry the best-so-far result is
  returned and flagged "budget-capped".
- Gate failures never break generation — the raw output always survives.
- Validator issues below threshold are surfaced as Grok-ready feedback.
- Echo detection (`isEcho`) intentionally not ported: prompt/PRD output
  never echoes the short input directions; degenerate output is already
  caught by `is_valid_output()`.
- Score displayed 0-100 (`round(score * 100)` boundary mapping, like the
  other AX ports).

Also fixed here: `_RATE_LIMIT_RE` NameError in the refine error handler
(renamed to the actually-defined `_RATE_LIMIT_TEXT`), and `requests` is
now declared explicitly in requirements.txt.
