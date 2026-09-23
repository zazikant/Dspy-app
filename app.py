import streamlit as st
import dspy
import os
import re
import time

from opencode_client import (
    opencode_chat_stream_controlled as _opencode_chat,
    OPENCODE_DEFAULT_MODEL as _OPENCODE_DEFAULT_MODEL,
)

st.set_page_config(
    page_title="Scene → Prompt Generator",
    page_icon="🎨",
    layout="centered"
)

# ====================== CLIPBOARD HELPER ======================
def copy_button(text: str, label: str = "📋 Copy to Clipboard"):
    """Clipboard copy that works for arbitrarily large outputs (32k+).

    Key fix: full text is written into a hidden <textarea> via html.escape()
    at render time. The JS reads from that DOM node — not a JS string literal —
    so there is no browser template-literal size cap or Streamlit iframe
    serialization truncation regardless of output length.
    """
    import streamlit.components.v1 as components
    import html as html_mod
    encoded = html_mod.escape(text, quote=True)
    btn_id = f"copy-btn-{abs(hash(text)) % 1000000}"
    ta_id  = f"copy-ta-{abs(hash(text)) % 1000000}"
    components.html(
        f"""
        <textarea id="{ta_id}"
            style="position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;"
        >{encoded}</textarea>
        <button id="{btn_id}"
            style="background:#1f77b4;color:white;border:none;padding:10px 20px;
                   border-radius:6px;font-size:14px;cursor:pointer;width:100%;margin-top:4px;">
            {label}
        </button>
        <script>
        (function() {{
            var btn = document.getElementById('{btn_id}');
            var ta  = document.getElementById('{ta_id}');
            btn.addEventListener('click', function() {{
                var txt = ta.value;
                function markDone() {{
                    btn.innerText = '✅ Copied!';
                    btn.style.background = '#2d6a2d';
                    setTimeout(function() {{
                        btn.innerText = '{label}';
                        btn.style.background = '#1f77b4';
                    }}, 2000);
                }}
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(txt).then(markDone).catch(function() {{
                        ta.style.cssText = 'position:static;width:100%;height:2px;';
                        ta.select();
                        document.execCommand('copy');
                        ta.style.cssText = 'position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;';
                        markDone();
                    }});
                }} else {{
                    ta.style.cssText = 'position:static;width:100%;height:2px;';
                    ta.select();
                    document.execCommand('copy');
                    ta.style.cssText = 'position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;';
                    markDone();
                }}
            }});
        }})();
        </script>
        """,
        height=60,
    )

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("### 🎛️ Prompt Mode")
    mode = st.radio(
        label="Select mode:",
        options=["🎨 Image Prompt", "🎬 Video Scene Prompt", "🧠 Software PRD Prompt", "📐 Exhaustive PRD (32k)"],
        index=0,
        horizontal=False,
        help="Switch between Image, Video, or PRD prompt generation. Same Grok refinement workflow throughout."
    )
    st.divider()

    st.markdown("### 🔌 Provider")
    provider = st.radio(
        "Select Provider",
        options=["NVIDIA NIM", "OpenCode (GLM 5.1)"],
        index=0,
        help="NVIDIA NIM: 32k output, Nemotron 3 Super 120B default. OpenCode: GLM 5.1 via opencode.ai gateway, thinking model."
    )

    if provider == "NVIDIA NIM":
        nvidia_key_input = st.text_input(
            "NVIDIA NIM API Key",
            type="password",
            value=os.getenv("NVIDIA_NIM_API_KEY", ""),
            help="Paste your NVIDIA NIM API key here."
        )
        if st.button("✅ Apply NVIDIA Key", type="primary", use_container_width=True):
            if nvidia_key_input.strip():
                os.environ["NVIDIA_NIM_API_KEY"] = nvidia_key_input.strip()
                st.cache_resource.clear()
                st.success("✅ NVIDIA Key applied")
                st.rerun()
            else:
                st.error("Please enter your NVIDIA NIM API key.")
        opencode_key_input = ""

        nvidia_model_options = {
            "Nemotron 3 Super 120B (32k output, default)": "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
            "GPT OSS 20B (32k output, fallback)": "nvidia_nim/openai/gpt-oss-20b",
            "Custom Model": "custom",
        }
        selected_model_label = st.selectbox(
            "Select NVIDIA NIM Model",
            options=list(nvidia_model_options.keys()),
            index=0,
            help="Nemotron 3 Super 120B is the primary; GPT OSS 20B is the fast fallback. Both support 32k output."
        )
        if selected_model_label == "Custom Model":
            model_name = st.text_input("Custom Model Name", value="nvidia_nim/nvidia/nemotron-3-super-120b-a12b")
        else:
            model_name = nvidia_model_options[selected_model_label]
        st.caption("⚡ 32,000 output token limit · Hosted on NVIDIA infrastructure")
    else:
        # OpenCode (GLM 5.1)
        nvidia_key_input = ""
        opencode_key_input = st.text_input(
            "OpenCode API Key",
            type="password",
            value=os.getenv("OPENCODE_API_KEY", ""),
            help="Paste your OpenCode API key here. Get one at https://opencode.ai/"
        )
        if st.button("✅ Apply OpenCode Key", type="primary", use_container_width=True):
            if opencode_key_input.strip():
                os.environ["OPENCODE_API_KEY"] = opencode_key_input.strip()
                st.cache_resource.clear()
                st.success("✅ OpenCode Key applied")
                st.rerun()
            else:
                st.error("Please enter your OpenCode API key.")

        opencode_model_options = {
            "GLM 5.1 (thinking, reasoning_effort=low)": "glm-5.1",
            "Custom Model": "custom",
        }
        selected_model_label = st.selectbox(
            "Select OpenCode Model",
            options=list(opencode_model_options.keys()),
            index=0,
            help="GLM 5.1 is a thinking model — the opencode gateway requires reasoning_effort='low'."
        )
        if selected_model_label == "Custom Model":
            model_name = st.text_input("Custom Model Name", value="glm-5.1")
        else:
            model_name = opencode_model_options[selected_model_label]
        st.caption("🧠 GLM 5.3 thinking model via opencode.ai gateway · reasoning_effort=low")

    if st.button("✅ Apply Model", type="primary", use_container_width=True):
        st.cache_resource.clear()
        st.success(f"✅ Model set to: {selected_model_label}")
        st.rerun()

    module_type = st.selectbox(
        "Reasoning Mode",
        options=["Predict", "ChainOfThought"],
        index=1,
        help="ChainOfThought usually gives better results"
    )

    st.divider()
    if st.button("🔁 Force Reload Model", use_container_width=True, help="Clear all cached resources and reload"):
        st.cache_resource.clear()
        st.success("Cache cleared — reloading...")
        st.rerun()

# ====================== DYNAMIC TITLE ======================
is_video_mode    = mode == "🎬 Video Scene Prompt"
is_prd_mode      = mode == "🧠 Software PRD Prompt"
is_prd_exhaustive = mode == "📐 Exhaustive PRD (32k)"
is_image_mode    = mode == "🎨 Image Prompt"

if is_video_mode:
    st.title("🎬 Scene to Video Prompt Generator")
    st.markdown("Ultra-detailed prompts for Sora / Kling / Runway / Wan — motion, timing, camera + Grok-powered iterative refinement")
elif is_prd_mode:
    st.title("🧠 Software PRD Meta-Prompt Generator")
    st.markdown(
        "Converge on a bulletproof technical architecture through iterative Grok refinement — "
        "each round locks in confirmed patterns and buries dead weight permanently."
    )
elif is_prd_exhaustive:
    st.title("📐 Exhaustive PRD Generator (32k)")
    st.markdown(
        "No token limits. Every node, every edge, every interface contract written in full. "
        "The Architecture Graveyard grows without bound. "
        "Designed for any architecture — microservices, agent pipelines, event-driven systems, ETL, web apps, and more. "
        "**Requires NVIDIA NIM (32k output).** Each Grok round expands depth — nothing is summarised, everything is spelled out."
    )
else:
    st.title("🎨 Scene to Image Prompt Generator")
    st.markdown("Ultra-detailed prompts for Flux / SD3 / SDXL + Grok-powered iterative refinement")

# ====================== DSPy SETUP ======================
# NOTE: do NOT use @st.cache_resource here. dspy.ChainOfThought / dspy.Predict
# carry internal _thread.RLock objects that cannot be pickled, which makes
# Streamlit's cache_resource fail with: "cannot pickle '_thread.RLock' object".
# The function is cheap (no I/O, just class instantiation) so it's safe to call
# on every rerun. LM init inside dspy.LM() is also lightweight.
#
# Returns: (run_fn, module_or_None, load_error)
#   - run_fn(user_input: str) -> str   — the unified callable the UI uses
#   - module_or_None                  — the dspy module (NVIDIA only), for debugging
#   - load_error: str | None          — set if init failed
def get_generator(module_type: str, model_name: str, mode: str, api_key: str, provider: str):
    if not api_key:
        return lambda u: (_ for _ in ()).throw(RuntimeError("No API key set.")), None, "No API key set."

    is_opencode = provider == "OpenCode (GLM 5.1)"

    try:
        # ── IMAGE SIGNATURE ─────────────────────────────
        if mode == "🎨 Image Prompt":
            class SceneToImagePrompt(dspy.Signature):
                """
                You are an expert image prompt engineer specializing in turning loose or explicit user directions into ultra-detailed, vivid, high-quality prompts for Flux, SD3, SDXL, Pony, etc.

ALWAYS follow these guidelines:
                - Strong cinematic composition and camera angles
                - Rich pose, body language, and clothing details (especially sheer/translucent fabrics)
                - Seductive atmosphere with professional lighting, shadows, and skin texture
                - Anatomically realistic + high-end erotic photography style
                - Tasteful yet explicit when appropriate
                - Output a ready-to-use, well-structured detailed prompt (80-200 words)
                """
                user_directions: str = dspy.InputField(desc="Original scene + previous prompt + all Grok feedback accumulated")
                detailed_prompt: str = dspy.OutputField(desc="Final optimized image generation prompt")

            sig = SceneToImagePrompt

        # ── VIDEO SIGNATURE ──────────────────────────────
        elif mode == "🎬 Video Scene Prompt":
            class SceneToVideoPrompt(dspy.Signature):
                """
                You are an expert video prompt engineer specializing in turning loose or explicit user directions into ultra-detailed, motion-rich prompts for text-to-video models like Sora, Kling, Runway Gen-4, Wan, and Hailuo.

ALWAYS follow these guidelines:
                - Describe the shot type and camera movement (e.g. slow dolly-in, handheld tracking shot, bird's eye crane descent, Dutch angle push)
                - Specify subject motion and body language over time (e.g. slowly turns head, fabric ripples as she walks, hair catches the breeze)
                - Define the temporal arc: how the scene opens, progresses, and ends within the clip
                - Include lighting evolution if relevant (e.g. golden hour light shifting to deep shadow, flickering neon reflecting off wet skin)
                - Capture atmosphere, texture, and mood in motion (e.g. steam rising, fabric clinging, shallow depth of field pulling focus)
                - Suggest clip duration and pacing feel (e.g. 6-second slow burn, 12-second continuous take, rhythmic cuts implied)
                - Tasteful yet explicit motion details when appropriate
                - Output a ready-to-use, well-structured detailed video prompt (80-220 words)
                """
                user_directions: str = dspy.InputField(desc="Original scene + previous prompt + all Grok feedback accumulated")
                detailed_prompt: str = dspy.OutputField(desc="Final optimized video generation prompt")

            sig = SceneToVideoPrompt

        # ── PRD SIGNATURE ────────────────────────────────
        elif mode == "🧠 Software PRD Prompt":
            class SoftwareToPRDPrompt(dspy.Signature):
                """
                You are a senior software architect and technical product strategist. Your job is to take a raw feature idea or problem statement and produce a comprehensive, opinionated PRD meta-prompt — a living technical document that sharpens itself with every round of expert feedback.

Think like someone who has seen every naive approach fail and every clever pattern succeed. Be decisive. Name the architecture. Commit to the stack. Call out the anti-patterns. And crucially: be willing to KILL components that don't survive scrutiny.

THE THREE MARKERS — use them rigorously on every component, tool, and decision:

  ✅ CONFIRMED ARCHITECTURE
     — This pattern/component has been reinforced across multiple Grok rounds. It is locked in.
       Never remove or question it in future versions. Build on it.

  ⚠️ CHALLENGED
     — Grok has questioned this component but hasn't killed it yet. It must be explicitly
       justified with a concrete reason in this version, or promoted to ❌ REMOVED.
       A ⚠️ CHALLENGED item that cannot be justified this round becomes ❌ REMOVED next round.

  ❌ REMOVED
     — Grok has repeatedly challenged this and it has failed to justify its existence.
       Move it immediately to the ARCHITECTURE GRAVEYARD. It must NEVER reappear in any
       future section of the PRD. Do not soften this — dead weight stays buried.

ALWAYS structure your output as a complete PRD meta-prompt covering ALL of the following sections:

1. PROBLEM STATEMENT
   - Crisp one-paragraph definition of what is being solved and why naive approaches break down

2. CORE ARCHITECTURE DECISION
   - Name the primary architectural pattern chosen — mark it ✅ CONFIRMED if reinforced
   - State WHY this pattern wins over the alternatives considered
   - Explicitly name patterns that are ❌ REMOVED and must never return

3. TECH STACK & TOOLING
   - Every component must carry exactly one marker: ✅ CONFIRMED, ⚠️ CHALLENGED, or ❌ REMOVED
   - ⚠️ CHALLENGED components must include a one-line justification or be killed this round
   - ❌ REMOVED components must not appear here — they go only in the Graveyard

4. DATA MODEL & FLOW
   - Key entities and their relationships
   - How data moves through the system end-to-end
   - Any transformation or enrichment steps

5. WORKFLOW & SEQUENCE
   - Step-by-step operational flow a developer would implement
   - Name every LangGraph node explicitly with edges (e.g. pdf_loader → ocr_detector → text_extractor → llm_extractor → validator → formatter)
   - Define the LangGraph state object fields (TypedDict)
   - Decision points, branching logic, error handling strategy

6. INTERFACE CONTRACTS
   - API shape with key endpoints or function signatures — mark any ⚠️ CHALLENGED
   - Input validation strategy
   - Response structure and error codes

7. OPEN QUESTIONS & NEXT REFINEMENT TARGETS
   - What is still unresolved
   - Which ⚠️ CHALLENGED decisions Grok should stress-test next
   - Hypotheses worth challenging

8. ARCHITECTURE GRAVEYARD
   - Every component ever marked ❌ REMOVED, listed with a one-line reason why it was killed
   - This section only ever grows — nothing leaves the Graveyard
   - Format: "❌ [Component name] — [reason killed]"
   - If no components have been removed yet, write: "No casualties yet — first round."

RULES:
- A leaner PRD that makes fewer decisions confidently beats a bloated one that lists every option
- If Grok challenged something and you cannot justify it in one concrete sentence, kill it
- Every version must have FEWER ⚠️ CHALLENGED items than the previous version
- The Graveyard must grow with each Grok round or you are not being decisive enough
- Output the full PRD meta-prompt as a well-structured document (250-600 words)
- It must be immediately usable as context for a developer or the next Grok refinement round

TONE: Opinionated, specific, architect-grade. No vague platitudes. Every sentence either names something concrete or makes a decision.

CRITICAL: You MUST always return the full PRD document. Never return None, empty string, or partial output.
If the input contains ratings, scores, or review-style feedback mixed with architectural suggestions,
extract ONLY the architectural suggestions and apply them. Ignore scores, praise, and meta-commentary.
Focus solely on: what to add, what to kill, what to confirm, what to challenge.
                """
                user_directions: str = dspy.InputField(
                    desc="Original feature/problem description + previous PRD meta-prompt + architectural feedback from Grok. NOTE: extract only architectural decisions from the feedback — ignore any ratings, scores, or review commentary."
                )
                detailed_prompt: str = dspy.OutputField(
                    desc="Full PRD meta-prompt with ✅ CONFIRMED / ⚠️ CHALLENGED / ❌ REMOVED markers on every component, plus Architecture Graveyard. Must never be empty or None."
                )

            sig = SoftwareToPRDPrompt

        # ── EXHAUSTIVE PRD SIGNATURE ─────────────────────
        elif mode == "📐 Exhaustive PRD (32k)":
            class ExhaustivePRDPrompt(dspy.Signature):
                """
You are a principal engineer writing a technical specification that a developer can implement without asking a single follow-up question. No prose. No story. No scene-setting. Every token spent must be a decision, a field name, a type, an edge, an error code, or a constraint.

YOU HAVE A 32,000 TOKEN OUTPUT BUDGET. SPEND IT ON SPEC DEPTH, NOT NARRATIVE WIDTH.
More tokens = more fields defined, more edge cases covered, more code written, more error paths named.
NOT more sentences explaining what a database is.

THE THREE MARKERS — apply to every component, library, pattern, and decision:
  ✅ CONFIRMED — locked in, build on it, never re-debate
  ⚠️ CHALLENGED — survives this round only with a one-line concrete justification; unkillable items become ❌ next round
  ❌ REMOVED — dead, goes only in Graveyard, never referenced again

GROK FEEDBACK RULE: If the input contains Grok feedback, extract ONLY architectural decisions.
Strip all scores, ratings, praise, and meta-commentary. Apply only: what to add, kill, confirm, or challenge.

═══════════════════════════════════════════════════════════════
REQUIRED SECTIONS — write every one, every time, in full
═══════════════════════════════════════════════════════════════

## 1. PROBLEM STATEMENT [3-5 sentences MAX]
- Sentence 1: What breaks without this system (specific failure mode, not generic pain)
- Sentence 2: Why the naive/obvious approach fails (name the approach, name the failure)
- Sentence 3: The exact constraint that makes this hard (scale, latency, consistency, auth, etc.)
- Sentence 4-5 (optional): What "solved" looks like in measurable terms

NO PARAGRAPHS. NO BACKGROUND. If it doesn't name a concrete failure or constraint, cut it.

## 2. CORE ARCHITECTURE DECISION
Format strictly as:
  CHOSEN: [Pattern name] ✅ CONFIRMED — [one sentence: why it wins on the specific constraint above]
  KILLED: ❌ [Alternative] — [one sentence: specific reason it fails on THIS problem]
  KILLED: ❌ [Alternative] — [one sentence: specific reason it fails on THIS problem]
  COMMITMENT: [The one architectural invariant that must never be violated]

## 3. TECH STACK & TOOLING
One line per component. Format:
  [Library/Tool] vX.Y ✅/⚠️/❌ — [exact role in this system] | [why this over the obvious alternative]
  ⚠️ items MUST include: "Survives because: [one concrete reason]"
  ❌ items must NOT appear here — Graveyard only.

## 4. DATA CONTRACTS & SCHEMAS
Write the actual code. Every field must have:
  - Name, type, constraints (min/max/regex/enum), nullable?, default, which component writes it, which reads it
  Format as Python TypedDict or Pydantic BaseModel with Field() annotations.
  No field descriptions in prose — annotate inline with comments.
  Cover: primary state object, every entity passed between nodes/services, every DB table schema.

## 5. COMPONENT MAP & EXECUTION FLOW
First: ASCII node graph showing every component, every directed edge, every conditional branch.
  Format: [node_name] --condition--> [next_node] or END
  Every branch must be named. No implicit "then it continues".

Then: For EACH node/service/stage, write a spec block:
  NODE: node_name
  INPUT:  field: type  # constraint
  OUTPUT: field: type  # constraint
  PROCESS:
    1. [Exact operation — name the function/method/API call]
    2. [Exact operation]
    ...
  ERROR HANDLING:
    [ErrorType] → [exact action: retry N times / transition to X node / raise / log + skip]
  STATE MUTATIONS: [list every GraphState field this node reads and writes]
  INVARIANTS: [what must be true before and after this node runs]

## 6. INTERFACE CONTRACTS
Write actual signatures. No pseudocode — valid Python/TypeScript/SQL.
  For every external interface:
    - Full function/method signature with types
    - Preconditions (what must be true before calling)
    - Postconditions (what is guaranteed on success)
    - Every exception/error type it raises and why
    - HTTP: method, path, request schema, response schema, all error codes with meanings

## 7. FAILURE MODES & RECOVERY PATHS
Table format:
  FAILURE | DETECTION | RECOVERY ACTION | STATE AFTER RECOVERY | PREVENTS
  One row per distinct failure mode. Be exhaustive — at least 8 rows.
  Include: auth expiry, rate limits, partial writes, schema mismatch, timeout, poison pill records, OOM.

## 8. OPEN DECISIONS [max 5 items]
Format strictly:
  ❓ [Decision title]
  Options: A) [option] — [tradeoff] | B) [option] — [tradeoff]
  Kill if: [condition under which one option is immediately eliminated]
  Decide by: [what test or metric resolves this]

No open-ended questions. Every item must have a decision path.

## 9. ARCHITECTURE GRAVEYARD
  ❌ [Component] — [exact round killed] — [one-line kill reason]
  This section only grows. Nothing leaves. No softening.
  First round with no kills: write "No casualties — [name the weakest ⚠️ item and what would kill it]"

═══════════════════════════════════════════════════════════
ABSOLUTE RULES
═══════════════════════════════════════════════════════════
- Problem Statement ≤ 5 sentences. Violation = rewrite it.
- Every node in section 5 gets a full spec block. No exceptions.
- Every field in section 4 has a type and constraint. "string" alone is not a type.
- No sentence starts with "This system", "The goal", "In order to", or "We need to".
- No section may contain only prose where code or a table would serve.
- ⚠️ CHALLENGED count must decrease each version. If it doesn't, you are not deciding.
- NEVER return None, empty string, or truncated output.
                """
                user_directions: str = dspy.InputField(
                    desc="Feature/problem description + optional previous PRD + optional Grok feedback. Extract only architectural decisions from feedback — strip all scores, ratings, and commentary."
                )
                detailed_prompt: str = dspy.OutputField(
                    desc="Complete exhaustive PRD spec. Every node fully specced. Every field typed. Every failure mode named. Every interface contracted. ✅/⚠️/❌ on every decision. Graveyard at end. Never empty, never truncated."
                )

            sig = ExhaustivePRDPrompt

        else:
            return lambda u: (_ for _ in ()).throw(RuntimeError(f"Unknown mode: {mode}")), None, f"Unknown mode: {mode}"

        # ── Build the unified run_fn ──────────────────────
        if is_opencode:
            # OpenCode: bypass dspy entirely. Build messages list from signature
            # docstring + user_input. GLM 5.3 already thinks internally — using
            # dspy.ChainOfThought on top would double-charge reasoning tokens.
            system_prompt = (sig.__doc__ or "").strip()
            if module_type == "ChainOfThought":
                # Inject an explicit CoT prefix so GLM reasons before answering
                system_prompt = (
                    "Before producing your final answer, reason step by step about "
                    "the requirements. Then output the final response in the exact "
                    "format the system prompt specifies.\n\n" + system_prompt
                )

            def run_fn(user_input: str) -> str:
                """Direct OpenCode call — no dspy, no litellm, no RLock."""
                result = _opencode_chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    api_key=api_key,
                    model=model_name,
                    max_tokens=8192,
                    temperature=0.7,
                )
                return result.content

            return run_fn, None, None
        else:
            # NVIDIA NIM: dspy.LM + dspy.Predict / ChainOfThought
            # - Drop `stream=True` — litellm's streaming path creates _thread.RLock
            #   objects that fail to pickle across Streamlit reruns
            # - Drop `reasoning_effort` — NVIDIA NIM rejects it for nemotron-3-super
            lm = dspy.LM(
                model_name,
                api_base="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
                max_tokens=32000,
                temperature=0.5,
                top_p=1.0,
            )
            # dspy.configure(lm=lm) sets a global LM and avoids dspy.context's
            # thread-local locks. We never enter a dspy.context(...) block again.
            dspy.configure(lm=lm)
            module = dspy.ChainOfThought(sig) if module_type == "ChainOfThought" else dspy.Predict(sig)

            def run_fn(user_input: str) -> str:
                """dspy call — LM set globally via dspy.configure()."""
                return module(user_directions=user_input).detailed_prompt

            return run_fn, module, None
    except Exception as e:
        return lambda u: (_ for _ in ()).throw(e), None, str(e)

        # ── IMAGE SIGNATURE ─────────────────────────────
        if mode == "🎨 Image Prompt":
            class SceneToImagePrompt(dspy.Signature):
                """
                You are an expert image prompt engineer specializing in turning loose or explicit user directions into ultra-detailed, vivid, high-quality prompts for Flux, SD3, SDXL, Pony, etc.

ALWAYS follow these guidelines:
                - Strong cinematic composition and camera angles
                - Rich pose, body language, and clothing details (especially sheer/translucent fabrics)
                - Seductive atmosphere with professional lighting, shadows, and skin texture
                - Anatomically realistic + high-end erotic photography style
                - Tasteful yet explicit when appropriate
                - Output a ready-to-use, well-structured detailed prompt (80-200 words)
                """
                user_directions: str = dspy.InputField(desc="Original scene + previous prompt + all Grok feedback accumulated")
                detailed_prompt: str = dspy.OutputField(desc="Final optimized image generation prompt")

            sig = SceneToImagePrompt

        # ── VIDEO SIGNATURE ──────────────────────────────
        elif mode == "🎬 Video Scene Prompt":
            class SceneToVideoPrompt(dspy.Signature):
                """
                You are an expert video prompt engineer specializing in turning loose or explicit user directions into ultra-detailed, motion-rich prompts for text-to-video models like Sora, Kling, Runway Gen-4, Wan, and Hailuo.

ALWAYS follow these guidelines:
                - Describe the shot type and camera movement (e.g. slow dolly-in, handheld tracking shot, bird's eye crane descent, Dutch angle push)
                - Specify subject motion and body language over time (e.g. slowly turns head, fabric ripples as she walks, hair catches the breeze)
                - Define the temporal arc: how the scene opens, progresses, and ends within the clip
                - Include lighting evolution if relevant (e.g. golden hour light shifting to deep shadow, flickering neon reflecting off wet skin)
                - Capture atmosphere, texture, and mood in motion (e.g. steam rising, fabric clinging, shallow depth of field pulling focus)
                - Suggest clip duration and pacing feel (e.g. 6-second slow burn, 12-second continuous take, rhythmic cuts implied)
                - Tasteful yet explicit motion details when appropriate
                - Output a ready-to-use, well-structured detailed video prompt (80-220 words)
                """
                user_directions: str = dspy.InputField(desc="Original scene + previous prompt + all Grok feedback accumulated")
                detailed_prompt: str = dspy.OutputField(desc="Final optimized video generation prompt")

            sig = SceneToVideoPrompt

        # ── PRD SIGNATURE ────────────────────────────────
        elif mode == "🧠 Software PRD Prompt":
            class SoftwareToPRDPrompt(dspy.Signature):
                """
                You are a senior software architect and technical product strategist. Your job is to take a raw feature idea or problem statement and produce a comprehensive, opinionated PRD meta-prompt — a living technical document that sharpens itself with every round of expert feedback.

Think like someone who has seen every naive approach fail and every clever pattern succeed. Be decisive. Name the architecture. Commit to the stack. Call out the anti-patterns. And crucially: be willing to KILL components that don't survive scrutiny.

THE THREE MARKERS — use them rigorously on every component, tool, and decision:

  ✅ CONFIRMED ARCHITECTURE
     — This pattern/component has been reinforced across multiple Grok rounds. It is locked in.
       Never remove or question it in future versions. Build on it.

  ⚠️ CHALLENGED
     — Grok has questioned this component but hasn't killed it yet. It must be explicitly
       justified with a concrete reason in this version, or promoted to ❌ REMOVED.
       A ⚠️ CHALLENGED item that cannot be justified this round becomes ❌ REMOVED next round.

  ❌ REMOVED
     — Grok has repeatedly challenged this and it has failed to justify its existence.
       Move it immediately to the ARCHITECTURE GRAVEYARD. It must NEVER reappear in any
       future section of the PRD. Do not soften this — dead weight stays buried.

ALWAYS structure your output as a complete PRD meta-prompt covering ALL of the following sections:

1. PROBLEM STATEMENT
   - Crisp one-paragraph definition of what is being solved and why naive approaches break down

2. CORE ARCHITECTURE DECISION
   - Name the primary architectural pattern chosen — mark it ✅ CONFIRMED if reinforced
   - State WHY this pattern wins over the alternatives considered
   - Explicitly name patterns that are ❌ REMOVED and must never return

3. TECH STACK & TOOLING
   - Every component must carry exactly one marker: ✅ CONFIRMED, ⚠️ CHALLENGED, or ❌ REMOVED
   - ⚠️ CHALLENGED components must include a one-line justification or be killed this round
   - ❌ REMOVED components must not appear here — they go only in the Graveyard

4. DATA MODEL & FLOW
   - Key entities and their relationships
   - How data moves through the system end-to-end
   - Any transformation or enrichment steps

5. WORKFLOW & SEQUENCE
   - Step-by-step operational flow a developer would implement
   - Name every LangGraph node explicitly with edges (e.g. pdf_loader → ocr_detector → text_extractor → llm_extractor → validator → formatter)
   - Define the LangGraph state object fields (TypedDict)
   - Decision points, branching logic, error handling strategy

6. INTERFACE CONTRACTS
   - API shape with key endpoints or function signatures — mark any ⚠️ CHALLENGED
   - Input validation strategy
   - Response structure and error codes

7. OPEN QUESTIONS & NEXT REFINEMENT TARGETS
   - What is still unresolved
   - Which ⚠️ CHALLENGED decisions Grok should stress-test next
   - Hypotheses worth challenging

8. ARCHITECTURE GRAVEYARD
   - Every component ever marked ❌ REMOVED, listed with a one-line reason why it was killed
   - This section only ever grows — nothing leaves the Graveyard
   - Format: "❌ [Component name] — [reason killed]"
   - If no components have been removed yet, write: "No casualties yet — first round."

RULES:
- A leaner PRD that makes fewer decisions confidently beats a bloated one that lists every option
- If Grok challenged something and you cannot justify it in one concrete sentence, kill it
- Every version must have FEWER ⚠️ CHALLENGED items than the previous version
- The Graveyard must grow with each Grok round or you are not being decisive enough
- Output the full PRD meta-prompt as a well-structured document (250-600 words)
- It must be immediately usable as context for a developer or the next Grok refinement round

TONE: Opinionated, specific, architect-grade. No vague platitudes. Every sentence either names something concrete or makes a decision.

CRITICAL: You MUST always return the full PRD document. Never return None, empty string, or partial output.
If the input contains ratings, scores, or review-style feedback mixed with architectural suggestions,
extract ONLY the architectural suggestions and apply them. Ignore scores, praise, and meta-commentary.
Focus solely on: what to add, what to kill, what to confirm, what to challenge.
                """
                user_directions: str = dspy.InputField(
                    desc="Original feature/problem description + previous PRD meta-prompt + architectural feedback from Grok. NOTE: extract only architectural decisions from the feedback — ignore any ratings, scores, or review commentary."
                )
                detailed_prompt: str = dspy.OutputField(
                    desc="Full PRD meta-prompt with ✅ CONFIRMED / ⚠️ CHALLENGED / ❌ REMOVED markers on every component, plus Architecture Graveyard. Must never be empty or None."
                )

            sig = SoftwareToPRDPrompt

        # ── EXHAUSTIVE PRD SIGNATURE ─────────────────────
        elif mode == "📐 Exhaustive PRD (32k)":
            class ExhaustivePRDPrompt(dspy.Signature):
                """
You are a principal engineer writing a technical specification that a developer can implement without asking a single follow-up question. No prose. No story. No scene-setting. Every token spent must be a decision, a field name, a type, an edge, an error code, or a constraint.

YOU HAVE A 32,000 TOKEN OUTPUT BUDGET. SPEND IT ON SPEC DEPTH, NOT NARRATIVE WIDTH.
More tokens = more fields defined, more edge cases covered, more code written, more error paths named.
NOT more sentences explaining what a database is.

THE THREE MARKERS — apply to every component, library, pattern, and decision:
  ✅ CONFIRMED — locked in, build on it, never re-debate
  ⚠️ CHALLENGED — survives this round only with a one-line concrete justification; unkillable items become ❌ next round
  ❌ REMOVED — dead, goes only in Graveyard, never referenced again

GROK FEEDBACK RULE: If the input contains Grok feedback, extract ONLY architectural decisions.
Strip all scores, ratings, praise, and meta-commentary. Apply only: what to add, kill, confirm, or challenge.

═══════════════════════════════════════════════════════
REQUIRED SECTIONS — write every one, every time, in full
═══════════════════════════════════════════════════════

## 1. PROBLEM STATEMENT [3-5 sentences MAX]
- Sentence 1: What breaks without this system (specific failure mode, not generic pain)
- Sentence 2: Why the naive/obvious approach fails (name the approach, name the failure)
- Sentence 3: The exact constraint that makes this hard (scale, latency, consistency, auth, etc.)
- Sentence 4-5 (optional): What "solved" looks like in measurable terms

NO PARAGRAPHS. NO BACKGROUND. If it doesn't name a concrete failure or constraint, cut it.

## 2. CORE ARCHITECTURE DECISION
Format strictly as:
  CHOSEN: [Pattern name] ✅ CONFIRMED — [one sentence: why it wins on the specific constraint above]
  KILLED: ❌ [Alternative] — [one sentence: specific reason it fails on THIS problem]
  KILLED: ❌ [Alternative] — [one sentence: specific reason it fails on THIS problem]
  COMMITMENT: [The one architectural invariant that must never be violated]

## 3. TECH STACK & TOOLING
One line per component. Format:
  [Library/Tool] vX.Y ✅/⚠️/❌ — [exact role in this system] | [why this over the obvious alternative]
  ⚠️ items MUST include: "Survives because: [one concrete reason]"
  ❌ items must NOT appear here — Graveyard only.

## 4. DATA CONTRACTS & SCHEMAS
Write the actual code. Every field must have:
  - Name, type, constraints (min/max/regex/enum), nullable?, default, which component writes it, which reads it
  Format as Python TypedDict or Pydantic BaseModel with Field() annotations.
  No field descriptions in prose — annotate inline with comments.
  Cover: primary state object, every entity passed between nodes/services, every DB table schema.

## 5. COMPONENT MAP & EXECUTION FLOW
First: ASCII node graph showing every component, every directed edge, every conditional branch.
  Format: [node_name] --condition--> [next_node] or END
  Every branch must be named. No implicit "then it continues".

Then: For EACH node/service/stage, write a spec block:
  NODE: node_name
  INPUT:  field: type  # constraint
  OUTPUT: field: type  # constraint
  PROCESS:
    1. [Exact operation — name the function/method/API call]
    2. [Exact operation]
    ...
  ERROR HANDLING:
    [ErrorType] → [exact action: retry N times / transition to X node / raise / log + skip]
  STATE MUTATIONS: [list every GraphState field this node reads and writes]
  INVARIANTS: [what must be true before and after this node runs]

## 6. INTERFACE CONTRACTS
Write actual signatures. No pseudocode — valid Python/TypeScript/SQL.
  For every external interface:
    - Full function/method signature with types
    - Preconditions (what must be true before calling)
    - Postconditions (what is guaranteed on success)
    - Every exception/error type it raises and why
    - HTTP: method, path, request schema, response schema, all error codes with meanings

## 7. FAILURE MODES & RECOVERY PATHS
Table format:
  FAILURE | DETECTION | RECOVERY ACTION | STATE AFTER RECOVERY | PREVENTS
  One row per distinct failure mode. Be exhaustive — at least 8 rows.
  Include: auth expiry, rate limits, partial writes, schema mismatch, timeout, poison pill records, OOM.

## 8. OPEN DECISIONS [max 5 items]
Format strictly:
  ❓ [Decision title]
  Options: A) [option] — [tradeoff] | B) [option] — [tradeoff]
  Kill if: [condition under which one option is immediately eliminated]
  Decide by: [what test or metric resolves this]

No open-ended questions. Every item must have a decision path.

## 9. ARCHITECTURE GRAVEYARD
  ❌ [Component] — [exact round killed] — [one-line kill reason]
  This section only grows. Nothing leaves. No softening.
  First round with no kills: write "No casualties — [name the weakest ⚠️ item and what would kill it]"

═══════════════════════════════════════════
ABSOLUTE RULES
═══════════════════════════════════════════
- Problem Statement ≤ 5 sentences. Violation = rewrite it.
- Every node in section 5 gets a full spec block. No exceptions.
- Every field in section 4 has a type and constraint. "string" alone is not a type.
- No sentence starts with "This system", "The goal", "In order to", or "We need to".
- No section may contain only prose where code or a table would serve.
- ⚠️ CHALLENGED count must decrease each version. If it doesn't, you are not deciding.
- NEVER return None, empty string, or truncated output.
                """
                user_directions: str = dspy.InputField(
                    desc="Feature/problem description + optional previous PRD + optional Grok feedback. Extract only architectural decisions from feedback — strip all scores, ratings, and commentary."
                )
                detailed_prompt: str = dspy.OutputField(
                    desc="Complete exhaustive PRD spec. Every node fully specced. Every field typed. Every failure mode named. Every interface contracted. ✅/⚠️/❌ on every decision. Graveyard at end. Never empty, never truncated."
                )

            sig = ExhaustivePRDPrompt

        # ── PRD SIGNATURE ────────────────────────────────
        elif mode == "🧠 Software PRD Prompt":
            class SoftwareToPRDPrompt(dspy.Signature):
                """
                You are a senior software architect and technical product strategist. Your job is to take a raw feature idea or problem statement and produce a comprehensive, opinionated PRD meta-prompt — a living technical document that sharpens itself with every round of expert feedback.

Think like someone who has seen every naive approach fail and every clever pattern succeed. Be decisive. Name the architecture. Commit to the stack. Call out the anti-patterns. And crucially: be willing to KILL components that don't survive scrutiny.

THE THREE MARKERS — use them rigorously on every component, tool, and decision:

  ✅ CONFIRMED ARCHITECTURE
     — This pattern/component has been reinforced across multiple Grok rounds. It is locked in.
       Never remove or question it in future versions. Build on it.

  ⚠️ CHALLENGED
     — Grok has questioned this component but hasn't killed it yet. It must be explicitly
       justified with a concrete reason in this version, or promoted to ❌ REMOVED.
       A ⚠️ CHALLENGED item that cannot be justified this round becomes ❌ REMOVED next round.

  ❌ REMOVED
     — Grok has repeatedly challenged this and it has failed to justify its existence.
       Move it immediately to the ARCHITECTURE GRAVEYARD. It must NEVER reappear in any
       future section of the PRD. Do not soften this — dead weight stays buried.

ALWAYS structure your output as a complete PRD meta-prompt covering ALL of the following sections:

1. PROBLEM STATEMENT
   - Crisp one-paragraph definition of what is being solved and why naive approaches break down

2. CORE ARCHITECTURE DECISION
   - Name the primary architectural pattern chosen — mark it ✅ CONFIRMED if reinforced
   - State WHY this pattern wins over the alternatives considered
   - Explicitly name patterns that are ❌ REMOVED and must never return

3. TECH STACK & TOOLING
   - Every component must carry exactly one marker: ✅ CONFIRMED, ⚠️ CHALLENGED, or ❌ REMOVED
   - ⚠️ CHALLENGED components must include a one-line justification or be killed this round
   - ❌ REMOVED components must not appear here — they go only in the Graveyard

4. DATA MODEL & FLOW
   - Key entities and their relationships
   - How data moves through the system end-to-end
   - Any transformation or enrichment steps

5. WORKFLOW & SEQUENCE
   - Step-by-step operational flow a developer would implement
   - Name every component, service, node, or stage explicitly with its connections and transitions
   - Define all state fields, message schemas, or data contracts passed between steps
   - Decision points, branching logic, error handling strategy

6. INTERFACE CONTRACTS
   - API shape with key endpoints or function signatures — mark any ⚠️ CHALLENGED
   - Input validation strategy
   - Response structure and error codes

7. OPEN QUESTIONS & NEXT REFINEMENT TARGETS
   - What is still unresolved
   - Which ⚠️ CHALLENGED decisions Grok should stress-test next
   - Hypotheses worth challenging

8. ARCHITECTURE GRAVEYARD
   - Every component ever marked ❌ REMOVED, listed with a one-line reason why it was killed
   - This section only ever grows — nothing leaves the Graveyard
   - Format: "❌ [Component name] — [reason killed]"
   - If no components have been removed yet, write: "No casualties yet — first round."

RULES:
- A leaner PRD that makes fewer decisions confidently beats a bloated one that lists every option
- If Grok challenged something and you cannot justify it in one concrete sentence, kill it
- Every version must have FEWER ⚠️ CHALLENGED items than the previous version
- The Graveyard must grow with each Grok round or you are not being decisive enough
- Output the full PRD meta-prompt as a well-structured document (250-600 words)
- It must be immediately usable as context for a developer or the next Grok refinement round

TONE: Opinionated, specific, architect-grade. No vague platitudes. Every sentence either names something concrete or makes a decision.

CRITICAL: You MUST always return the full PRD document. Never return None, empty string, or partial output.
If the input contains ratings, scores, or review-style feedback mixed with architectural suggestions,
extract ONLY the architectural suggestions and apply them. Ignore scores, praise, and meta-commentary.
Focus solely on: what to add, what to kill, what to confirm, what to challenge.
                """
                user_directions: str = dspy.InputField(
                    desc="Original feature/problem description + previous PRD meta-prompt + architectural feedback from Grok. NOTE: extract only architectural decisions from the feedback — ignore any ratings, scores, or review commentary."
                )
                detailed_prompt: str = dspy.OutputField(
                    desc="Full PRD meta-prompt with ✅ CONFIRMED / ⚠️ CHALLENGED / ❌ REMOVED markers on every component, plus Architecture Graveyard. Must never be empty or None."
                )

            sig = SoftwareToPRDPrompt

        else:
            return None, None, "Unknown mode."

        if module_type == "ChainOfThought":
            module = dspy.ChainOfThought(sig)
        else:
            module = dspy.Predict(sig)

        return module, lm, None
    except Exception as e:
        return None, None, str(e)

# Pass provider + API key into cache key so any change busts the cache
if provider == "OpenCode (GLM 5.1)":
    _api_key = os.getenv("OPENCODE_API_KEY", "")
else:
    _api_key = os.getenv("NVIDIA_NIM_API_KEY", "")
run_fn, generator, load_error = get_generator(
    module_type,
    model_name,
    mode,
    _api_key,
    provider,
)

# ====================== SESSION STATE ======================
img_state  = st.session_state.setdefault("img",  {"prompt_history": [], "last_prompt": "", "original_input": ""})
vid_state  = st.session_state.setdefault("vid",  {"prompt_history": [], "last_prompt": "", "original_input": ""})
prd_state  = st.session_state.setdefault("prd",  {"prompt_history": [], "last_prompt": "", "original_input": ""})
eprd_state = st.session_state.setdefault("eprd", {"prompt_history": [], "last_prompt": "", "original_input": ""})

if is_prd_exhaustive:
    state = eprd_state
elif is_prd_mode:
    state = prd_state
elif is_video_mode:
    state = vid_state
else:
    state = img_state

# ====================== HELPER ======================
def is_valid_output(text):
    """Guard against None, empty, or suspiciously short output."""
    return text and isinstance(text, str) and len(text.strip()) > 100

def render_output(text, is_prd):
    """Render full output — no truncation."""
    if is_prd:
        st.markdown(text)
    else:
        st.code(text, language=None)

# ====================== RETRY WRAPPER ======================
# Mirrors ax-translator's adaptive backoff strategy (nvidia-client.ts:144-208,
# page.tsx:727-786) but adapted for dspy.LM's higher-level call interface.
#
# Why this exists: NVIDIA NIM enforces per-minute / per-day rate limits.
# Large 32k-output requests are the first to get throttled, and a single 429
# used to kill the whole Exhaustive PRD round. ax-translator learned this
# the hard way and solved it with three layered defenses:
#
#   1. Per-call retry with backoff   (nvidia-client.ts:158-202)
#   2. Inter-chunk adaptive cooldown (page.tsx:741-786)
#   3. Echo detection on retry       (translation-pipeline.ts:77)
#
# Since dspy.LM hides the raw HTTP layer, we wrap the whole `generator()`
# call instead of the stream. The trade-off: each retry re-runs the full
# chain-of-thought, which is expensive — so we cap retries at 3 and use
# longer backoffs than ax-translator's 500ms (which sits on top of a
# 10s-call ceiling).

# ─── Retryable-error classification ──────────────────────────────────────────
# Ported from tradingview-notes-app-nvidia/src/lib/brain/nvidia.ts:52-61.
# Only retryable errors are retried; everything else surfaces to the UI
# immediately so the user sees a real error instead of burning time on
# backoffs that won't help.
_RETRYABLE_HTTP  = {429, 500, 502, 503, 504}
_RETRYABLE_CODES = {"ECONNRESET", "ETIMEDOUT", "UND_ERR_CONNECT_TIMEOUT"}
_RETRYABLE_NAMES = {"APIConnectionError", "APITimeoutError", "ConnectionError"}
_RETRYABLE_MSG   = re.compile(r"rate.?limit|too many requests|timeout|econnreset", re.I)

# Pre-compiled error text patterns for snappy matching
_RATE_LIMIT_TEXT = re.compile(r"rate.?limit|429|too many requests|quota|throttl", re.I)
_TRANSIENT_TEXT  = re.compile(r"timeout|connection.?reset|aborted|stream|502|503|504|server error", re.I)
_EMPTY_TEXT      = re.compile(r"finish_reason.*length|truncat|max_tokens|empty content", re.I)
_AUTH_TEXT       = re.compile(r"401|403|unauthor|invalid.*key|expired|user not found", re.I)


def _classify_error(err_msg: str) -> str:
    """Return one of: 'rate_limit', 'transient', 'auth', 'empty', 'fatal'."""
    if _AUTH_TEXT.search(err_msg):
        return "auth"
    if _RATE_LIMIT_TEXT.search(err_msg):
        return "rate_limit"
    if _EMPTY_TEXT.search(err_msg):
        return "empty"          # likely max_tokens/finish_reason=length
    if _TRANSIENT_TEXT.search(err_msg):
        return "transient"
    return "fatal"


def call_generator_with_retry(run_fn, user_input, max_retries: int = 3, status=None):
    """Run a generator callable with classification-aware retry + adaptive backoff.

    Provider-agnostic: works for both dspy modules (NVIDIA NIM) and the
    raw OpenCode client. The `run_fn` is just `callable(str) -> str`.

    Ported from tradingview-notes-app-nvidia/src/lib/brain/nvidia.ts and
    ax-translator's adaptive cooldown (page.tsx:727-786).

    Args:
        run_fn:     callable(user_input: str) -> str  — the unified generator
        user_input: the user_directions string passed to the signature
        max_retries: total attempts = max_retries + 1 (default 3 retries = 4 attempts)
        status:     optional st.status for live progress logging

    Error classification (see _classify_error):
        - 'auth'        → no retry, surface immediately (wrong key)
        - 'fatal'       → no retry, surface immediately (4xx other than 429, bad request)
        - 'rate_limit'  → retry with 30s/60s backoff (NVIDIA free tier quota)
        - 'empty'       → retry with 15s backoff (model hit max_tokens mid-response)
        - 'transient'   → retry with 0.5s × attempt backoff (502/503/504/timeout)

    Returns:
        The model's output string on success.

    Raises:
        RuntimeError with the last error message after exhausting retries,
        or surfaces auth/fatal errors immediately without retrying.
    """
    last_err = None

    for attempt in range(max_retries + 1):
        attempt_label = f"Attempt {attempt + 1}/{max_retries + 1}"
        if status:
            status.update(label=f"🔄 {attempt_label}…")
        try:
            output = run_fn(user_input)
            if is_valid_output(output):
                if status and attempt > 0:
                    status.update(label=f"✅ Succeeded on {attempt_label}")
                return output
            last_err = "empty or truncated output"
            kind = "empty"
            if status:
                status.update(label=f"⚠️ {attempt_label} returned {last_err}")
        except Exception as e:
            last_err = str(e)
            kind = _classify_error(last_err)
            if status:
                status.update(label=f"⚠️ {attempt_label} [{kind}]: {last_err[:120]}…")

        # Don't retry auth/fatal — surface immediately.
        if kind in ("auth", "fatal"):
            raise RuntimeError(last_err)

        if attempt >= max_retries:
            break

        # Backoff by error kind (NVIDIA integrate / OpenCode gateway behavior):
        if kind == "rate_limit":
            backoff = 60 if attempt == max_retries - 1 else 30
        elif kind == "empty":
            backoff = 15
        elif kind == "transient":
            backoff = 0.5 * (attempt + 1)   # 0.5s, 1s, 1.5s
        else:
            backoff = 2

        if status:
            status.update(label=f"⏳ {attempt_label} failed ({kind}). Waiting {backoff}s…")
        time.sleep(backoff)

    raise RuntimeError(f"Generator failed after {max_retries + 1} attempts. Last error: {last_err}")

# ====================== MAIN UI ======================
if is_prd_mode or is_prd_exhaustive:
    st.subheader("1. Describe Your Software Feature or Problem")
    if is_prd_exhaustive:
        st.markdown(
            "Describe your system — the more detail you give, the better the first draft. "
            "The generator produces a **fully exhaustive PRD**: every component named and described, "
            "every flow and decision mapped, full data schemas and interface contracts, and an Architecture Graveyard "
            "that only grows. Each Grok round makes the document **longer and more precise** — nothing is summarised, "
            "nothing is dropped. Recommended with NVIDIA NIM for the full 32k output budget."
        )
    else:
        st.markdown(
            "Write what you're trying to build. The generator produces an opinionated PRD with full architecture, "
            "stack, data model, and workflow — with every component marked ✅ / ⚠️ / ❌. "
            "Each Grok round locks in survivors and buries the rest in the **Architecture Graveyard** permanently."
        )
else:
    st.subheader("1. Initial Scene Description")

placeholder_map = {
    "🎨 Image Prompt": "beautiful woman in red lace lingerie, sitting on bed, legs slightly apart, soft warm lighting, low camera angle...",
    "🎬 Video Scene Prompt": "a woman walks slowly through a rain-soaked alley at night, neon signs reflecting off wet pavement, camera tracks her from behind at ground level...",
    "🧠 Software PRD Prompt": (
        "I want to build a natural language interface that lets non-technical users query our PostgreSQL database by typing plain English questions. "
        "The system should handle ambiguous phrasing, multi-table joins, and return results in a human-readable summary alongside the raw data..."
    ),
    "📐 Exhaustive PRD (32k)": (
        "Describe your system — any architecture works. Examples:\n"
        "• A microservices e-commerce backend with order, inventory, payment, and notification services communicating over Kafka...\n"
        "• A multi-agent research assistant where a planner agent delegates to a web search agent, a summariser agent, and a citation validator agent...\n"
        "• An Airflow ETL pipeline that pulls from 3 APIs, normalises into a star schema, and loads into BigQuery with SLA alerting...\n"
        "• A FastAPI backend with JWT auth, role-based access, background job processing, and a PostgreSQL read replica...\n"
        "The more detail you give upfront, the richer the first draft."
    )
}

user_input = st.text_area(
    "Feature / problem description:" if (is_prd_mode or is_prd_exhaustive) else "Your directions:",
    placeholder=placeholder_map[mode],
    height=160 if (is_prd_mode or is_prd_exhaustive) else 140
)

col1, col2 = st.columns(2)
with col1:
    if is_prd_exhaustive:
        btn_label = "📐 Generate Exhaustive PRD v1 (32k)"
    elif is_prd_mode:
        btn_label = "🧠 Generate Initial PRD Meta-Prompt (v1)"
    else:
        btn_label = "✨ Generate Initial Prompt (v1)"
    if st.button(btn_label, type="primary", use_container_width=True):
        if provider == "OpenCode (GLM 5.1)" and not os.getenv("OPENCODE_API_KEY"):
            st.error("Please apply OpenCode API key first.")
        elif provider == "NVIDIA NIM" and not os.getenv("NVIDIA_NIM_API_KEY"):
            st.error("Please apply NVIDIA NIM API key first.")
        elif not user_input.strip():
            st.error("Please describe your feature / scene!")
        elif run_fn is None:
            st.error(f"Generator error: {load_error}")
        else:
            with st.status(f"Generating v1 with {selected_model_label}…", expanded=True) as status:
                try:
                    output = call_generator_with_retry(
                        run_fn, user_input.strip(), max_retries=3, status=status
                    )
                except Exception as e:
                    err = str(e)
                    status.update(label="❌ v1 failed", state="error")
                    if "401" in err or "AuthenticationError" in err or "User not found" in err:
                        st.error("❌ Invalid or expired API key. Please paste a fresh key in the sidebar and click Apply.")
                    elif _RATE_LIMIT_TEXT.search(err):
                        st.error(
                            f"❌ Rate limited after 4 attempts (60s final backoff). "
                            "Wait a minute and try again, or pick a different model."
                        )
                    else:
                        st.error(f"❌ {err}")
                else:
                    state["original_input"] = user_input.strip()
                    state["last_prompt"] = output
                    state["prompt_history"] = [{
                        "version": 1,
                        "prompt": output,
                        "feedback_used": "Initial generation — no Grok feedback yet"
                    }]
                    status.update(label="✅ v1 ready", state="complete")
                    render_output(output, is_prd_mode or is_prd_exhaustive)
                    with st.expander("📋 Copy v1 for Grok", expanded=True):
                        st.code(output, language=None)
                        copy_button(output, "📋 Copy v1 to Clipboard")

with col2:
    if st.button("🔄 Reset Everything", type="secondary", use_container_width=True):
        state["prompt_history"] = []
        state["last_prompt"] = ""
        state["original_input"] = ""
        st.rerun()

# ====================== GROK REFINEMENT ======================
st.subheader("2. Iterative Refinement with Grok (Manual)")

if is_prd_mode or is_prd_exhaustive:
    if is_prd_exhaustive:
        st.markdown(
            "Paste the full PRD into Grok → tell it to: expand every node, add any missing edges, "
            "challenge every ⚠️ item, kill what cannot be defended, and push every section to exhaustive detail. "
            "Paste Grok's reply here — the next version will be **longer** than this one."
        )
        st.info(
            "💡 **Paste architectural + structural feedback** — new nodes to add, edges to define, "
            "fields to add to State, interfaces to specify, or components to kill. "
            "The generator will expand every section. Nothing shrinks between versions.",
            icon="ℹ️"
        )
    else:
        st.markdown(
            "Paste the PRD into Grok → ask it to challenge every ⚠️ CHALLENGED component, "
            "suggest what should be ❌ REMOVED, and reinforce what deserves ✅ CONFIRMED → paste reply here."
        )
        st.info(
            "💡 **Paste architectural feedback only** — what to add, kill, confirm, or challenge. "
            "Strip out any ratings, scores, or review commentary before pasting. "
            "The generator ignores scores and only acts on architectural decisions.",
            icon="ℹ️"
        )
else:
    st.markdown("Copy the latest prompt → paste into Grok → ask for improvements → paste Grok's reply here")

placeholder_feedback_map = {
    "🎨 Image Prompt": "Grok said: Add more dramatic rim lighting, make the fabric more sheer and clinging, use a lower camera angle, emphasize skin glow and subtle sweat beads...",
    "🎬 Video Scene Prompt": "Grok said: Add a slow rack focus from foreground rain to her face, extend the dolly move, add breath mist in cold air...",
    "🧠 Software PRD Prompt": (
        "Architectural feedback to apply:\n"
        "- KILL spaCy and Transformers — LLM handles extraction better, move to Graveyard\n"
        "- KILL PyPDF2 — abandoned, pdfplumber wins, confirm it\n"
        "- CONFIRM the chosen orchestrator/framework\n"
        "- CONFIRM GPT-4o-mini + Pydantic structured output as extraction core\n"
        "- Name every component explicitly with its connections and transitions\n"
        "- Add fallback/error handling branch for failed processing\n"
        "- Define all state fields and data contracts passed between components"
    ),
    "📐 Exhaustive PRD (32k)": (
        "Paste Grok's structural + architectural feedback here. Examples of what to include:\n"
        "- Add a dead-letter service between the validator and the sink — captures failed records with reason codes\n"
        "- CONFIRM Kafka as the message bus — remove RabbitMQ from consideration, move to Graveyard\n"
        "- Expand the payment service: add refund handler, webhook receiver, and idempotency key store\n"
        "- Define the full message schema: order_id: UUID, customer_id: str, items: list[LineItem], total: Decimal\n"
        "- KILL the in-process job runner — move to Celery with Redis broker, justify with scale argument\n"
        "- Map routing: validator → [record.is_valid] → enrichment_stage | [not valid] → dead_letter_stage"
    )
}

if is_prd_exhaustive:
    _fb_label = "Grok's feedback (node expansions, new edges, state fields, components to kill):"
    _fb_height = 240
elif is_prd_mode:
    _fb_label = "Grok's feedback (architectural suggestions only — strip ratings/scores):"
    _fb_height = 180
else:
    _fb_label = "Grok's feedback (paste entire response or key suggestions):"
    _fb_height = 160

grok_feedback = st.text_area(
    _fb_label,
    placeholder=placeholder_feedback_map[mode],
    height=_fb_height
)

if is_prd_exhaustive:
    refine_label = "📐 Expand PRD — Next Exhaustive Version"
elif is_prd_mode:
    refine_label = "🚀 Generate Next PRD Version with Grok Feedback"
else:
    refine_label = "🚀 Generate Next Version with Grok Feedback"
if st.button(refine_label, type="primary", use_container_width=True):
    if not state["last_prompt"]:
        st.error("Generate v1 first!")
    elif not grok_feedback.strip():
        st.error("Paste Grok feedback first!")
    else:
        next_v = len(state["prompt_history"]) + 1
        with st.status(f"Creating v{next_v} …", expanded=True) as status:
            try:
                if is_prd_exhaustive:
                    enhanced_input = f"""ORIGINAL PROBLEM:
{state['original_input']}

PREVIOUS SPEC (v{len(state['prompt_history'])}):
{state['last_prompt']}

GROK FEEDBACK — extract ONLY the architectural decisions below.
Strip all scores, ratings, praise, and meta-commentary before applying.
Apply only: what to add, what to kill, what to confirm, what to challenge.
{grok_feedback.strip()}

MANDATORY FOR v{len(state['prompt_history']) + 1}:
- GROK FILTERING: Ignore any sentence from the feedback that contains a score, rating, percentage, or review commentary. Extract only: new components to add, components to kill, decisions to confirm, decisions to challenge.
- PROBLEM STATEMENT: Max 5 sentences. If the previous version was longer, cut it down. No background prose.
- DATA CONTRACTS: Every new field mentioned in feedback must appear in section 4 with full type + constraint. No field without a type. No type without a constraint.
- COMPONENT MAP: Add every new node/edge mentioned in feedback to the ASCII graph. Every new node gets a full NODE spec block — INPUT, OUTPUT, PROCESS steps, ERROR HANDLING, STATE MUTATIONS, INVARIANTS.
- FAILURE MODES: Add any new failure mode surfaced in feedback as a table row. Minimum 8 rows total.
- INTERFACE CONTRACTS: Every new interface mentioned in feedback written as actual code signature, not prose.
- ⚠️ CHALLENGED: Every ⚠️ item from v{len(state['prompt_history'])} must be confirmed ✅ or killed ❌ — none survive unchanged.
- GRAVEYARD: Must have more entries than v{len(state['prompt_history'])}. If nothing new was killed, kill the weakest ⚠️ item and explain why.
- NEVER: grow the Problem Statement, write prose where a table or code block would serve, or return partial output."""
                elif is_prd_mode:
                    enhanced_input = f"""Original feature / problem description:
{state['original_input']}

Previous PRD meta-prompt (v{len(state['prompt_history'])}):
{state['last_prompt']}

Architectural feedback to apply (extract ONLY the architectural decisions below — ignore any ratings, scores, or review-style commentary):
{grok_feedback.strip()}

INSTRUCTIONS FOR THIS VERSION:
- Promote every pattern the feedback reinforced to ✅ CONFIRMED ARCHITECTURE
- Move every component the feedback challenged to ⚠️ CHALLENGED — justify in one sentence or kill it
- Move every component the feedback killed to ❌ REMOVED and add it to the Architecture Graveyard with a reason
- The Graveyard must be larger than the previous version's Graveyard
- Every ⚠️ CHALLENGED item from the previous version must either be confirmed or removed — none survive unchanged
- The overall stack must be LEANER than v{len(state['prompt_history'])}
- Return the COMPLETE PRD document. Never return partial output or None."""
                else:
                    enhanced_input = f"""Original scene description:
{state['original_input']}

Previous best prompt (v{len(state['prompt_history'])}):
{state['last_prompt']}

Grok has repeatedly suggested the following improvements across feedback:
{grok_feedback.strip()}

Create the strongest next version. Incorporate all the valuable patterns and elements Grok has been emphasizing."""

                output = call_generator_with_retry(
                    run_fn, enhanced_input, max_retries=3, status=status
                )

                state["prompt_history"].append({
                    "version": next_v,
                    "prompt": output,
                    "feedback_used": grok_feedback.strip()[:300] + "..."
                })
                state["last_prompt"] = output
                status.update(label=f"✅ v{next_v} generated", state="complete")
                render_output(output, is_prd_mode or is_prd_exhaustive)
                with st.expander(f"📋 Copy v{next_v} for Grok", expanded=True):
                    st.code(output, language=None)
                    copy_button(output, f"📋 Copy v{next_v} to Clipboard")

            except Exception as e:
                    err = str(e)
                    status.update(label=f"❌ v{next_v} failed", state="error")
                    if "401" in err or "AuthenticationError" in err or "User not found" in err:
                        st.error("❌ Invalid or expired API key. Please paste a fresh key in the sidebar and click Apply.")
                    elif _RATE_LIMIT_RE.search(err):
                        st.error(
                            f"❌ Rate limited after 4 attempts. "
                            "Wait a minute and try again, or pick a different model."
                        )
                    else:
                        st.error(f"Error: {err}")

# ====================== HISTORY ======================
if state["prompt_history"]:
    if is_prd_exhaustive:
        history_label = "📐 Exhaustive PRD Evolution"
    elif is_prd_mode:
        history_label = "📜 PRD Evolution"
    else:
        history_label = "📜 Prompt Evolution"
    st.subheader(history_label)

    if is_prd_mode:
        st.caption(
            "✅ = locked in | ⚠️ = on trial | ❌ = buried in Graveyard. "
            "Each version must be leaner than the last."
        )
    elif is_prd_exhaustive:
        st.caption(
            "✅ = locked in | ⚠️ = on trial | ❌ = buried in Graveyard. "
            "Each version must be LONGER and MORE DETAILED than the last. Graveyard only grows."
        )

    for item in reversed(state["prompt_history"]):
        with st.expander(f"Version {item['version']}"):
            if is_prd_mode or is_prd_exhaustive:
                st.markdown(item['prompt'])
            else:
                st.code(item['prompt'], language=None)
            copy_button(item['prompt'], f"📋 Copy v{item['version']} to Clipboard")
            st.caption(f"Based on: {item['feedback_used']}")

mode_label_map = {
    "🎨 Image Prompt": "Image (Flux/SD3/SDXL)",
    "🎬 Video Scene Prompt": "Video (Sora/Kling/Runway)",
    "🧠 Software PRD Prompt": "PRD — ✅ locks in · ⚠️ on trial · ❌ buried",
    "📐 Exhaustive PRD (32k)": "Exhaustive PRD — every node · every edge · Graveyard grows forever"
}
token_note = "32k output tokens"
st.caption(
    f"Mode: {mode_label_map[mode]} • {provider} · {selected_model_label} ({token_note}) + Grok manual refinement • "
    "Graveyard only grows · Stack only shrinks · Confidence compounds"
)
