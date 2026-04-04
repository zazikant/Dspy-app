import streamlit as st
import dspy
import os

st.set_page_config(
    page_title="Scene → Prompt Generator",
    page_icon="🎨",
    layout="centered"
)

# ====================== CLIPBOARD HELPER ======================
def copy_button(text: str, label: str = "📋 Copy to Clipboard"):
    """Render a real clipboard copy button for arbitrarily long text."""
    import streamlit.components.v1 as components
    # Escape backticks and backslashes so the JS template literal is safe
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(
        f"""
        <textarea id="copy-area" style="position:absolute;left:-9999px;top:-9999px;">{{}}</textarea>
        <button onclick="
            var txt = `{safe_text}`;
            navigator.clipboard.writeText(txt).then(function() {{
                this.innerText = '✅ Copied!';
                this.style.background = '#2d6a2d';
                setTimeout(() => {{ this.innerText = '{label}'; this.style.background = '#1f77b4'; }}, 2000);
            }}.bind(this)).catch(function() {{
                var ta = document.createElement('textarea');
                ta.value = txt;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                this.innerText = '✅ Copied!';
                this.style.background = '#2d6a2d';
                setTimeout(() => {{ this.innerText = '{label}'; this.style.background = '#1f77b4'; }}, 2000);
            }}.bind(this));
        "
        style="
            background:#1f77b4;
            color:white;
            border:none;
            padding:10px 20px;
            border-radius:6px;
            font-size:14px;
            cursor:pointer;
            width:100%;
            margin-top:4px;
        ">{label}</button>
        """,
        height=55,
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
    provider = st.selectbox(
        "Select Provider",
        options=["OpenRouter", "NVIDIA NIM"],
        index=0,
        help="OpenRouter uses free community models. NVIDIA NIM uses high-output models (32k tokens) — no API key required."
    )

    is_nvidia = provider == "NVIDIA NIM"

    # ── API Key (OpenRouter only) ────────────────────────────
    if not is_nvidia:
        api_key_input = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=os.getenv("OPENROUTER_API_KEY", "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"),
            help="Paste your OpenRouter API key here."
        )
        if st.button("✅ Apply API Key", type="primary", use_container_width=True):
            if api_key_input.strip():
                os.environ["OPENROUTER_API_KEY"] = api_key_input.strip()
                st.cache_resource.clear()
                st.success("✅ API Key applied")
                st.rerun()
            else:
                st.error("Please enter an API key.")
    else:
        api_key_input = ""
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

    # ── Model lists ──────────────────────────────────────────
    openrouter_model_options = {
        "Qwen3.6 Plus Preview (Best overall)": "openrouter/qwen/qwen3.6-plus-preview:free",
        "GLM 4.5 Air (Default)": "openrouter/z-ai/glm-4.5-air:free",
        "Step 3.5 Flash (Fast & strong)": "openrouter/stepfun/step-3.5-flash:free",
        "Arcee Trinity Large Preview (Creative)": "openrouter/arcee-ai/trinity-large-preview:free",
        "MiniMax M2.5": "openrouter/minimax/minimax-m2.5:free",
        "Llama 3.3 70B Instruct": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "Custom Model": "custom"
    }

    nvidia_model_options = {
        "Kimi K2.5 (32k output)": "nvidia_nim/moonshotai/kimi-k2.5",
        "MiniMax M2.5 (32k output)": "nvidia_nim/minimaxai/minimax-m2.5",
        "GPT OSS 120B (32k output)": "nvidia_nim/openai/gpt-oss-120b",
    }

    if is_nvidia:
        model_options = nvidia_model_options
        selected_model_label = st.selectbox(
            "Select NVIDIA NIM Model",
            options=list(nvidia_model_options.keys()),
            index=0,
            help="All NVIDIA NIM models support up to 32,000 output tokens."
        )
        model_name = nvidia_model_options[selected_model_label]
        st.caption("⚡ 32,000 output token limit · Hosted on NVIDIA infrastructure")
    else:
        model_options = openrouter_model_options
        selected_model_label = st.selectbox(
            "Select Model",
            options=list(openrouter_model_options.keys()),
            index=1,
        )
        if selected_model_label == "Custom Model":
            model_name = st.text_input("Custom Model Name", value="openrouter/z-ai/glm-4.5-air:free")
        else:
            model_name = openrouter_model_options[selected_model_label]

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
# API key is now part of the cache key — changing it busts the cache automatically
@st.cache_resource(show_spinner="Loading DSPy...")
def get_generator(module_type: str, model_name: str, mode: str, api_key: str, provider: str):
    is_nvidia = provider == "NVIDIA NIM"

    if not api_key:
        return None, None, "No API key set."

    try:
        if is_nvidia:
            lm = dspy.LM(
                model_name,
                api_base="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
                max_tokens=32000,
                temperature=0.7
            )
        else:
            lm = dspy.LM(
                model_name,
                api_base="https://openrouter.ai/api/v1",
                api_key=api_key,
                max_tokens=2048,
                temperature=0.7
            )

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
_api_key = os.getenv("NVIDIA_NIM_API_KEY", "") if is_nvidia else os.getenv("OPENROUTER_API_KEY", "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")
generator, lm, load_error = get_generator(
    module_type,
    model_name,
    mode,
    _api_key,
    provider
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
    """Render output in correct format per mode."""
    if is_prd:
        st.markdown(text)
    else:
        st.code(text, language=None)

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
        if is_nvidia and not os.getenv("NVIDIA_NIM_API_KEY"):
            st.error("Please apply NVIDIA NIM API key first.")
        elif not is_nvidia and not os.getenv("OPENROUTER_API_KEY"):
            st.error("Please apply OpenRouter API key first.")
        elif not user_input.strip():
            st.error("Please describe your feature / scene!")
        elif generator is None:
            st.error(f"Generator error: {load_error}")
        else:
            with st.spinner(f"Generating v1 with {selected_model_label}..."):
                try:
                    with dspy.context(lm=lm):
                        result = generator(user_directions=user_input.strip())

                    output = result.detailed_prompt

                    if not is_valid_output(output):
                        st.error(
                            "⚠️ The model returned an empty or invalid response. "
                            "Try switching to ChainOfThought mode in the sidebar, or try a different model."
                        )
                    else:
                        state["original_input"] = user_input.strip()
                        state["last_prompt"] = output
                        state["prompt_history"] = [{
                            "version": 1,
                            "prompt": output,
                            "feedback_used": "Initial generation — no Grok feedback yet"
                        }]
                        st.success("✅ v1 ready!")
                        render_output(output, is_prd_mode or is_prd_exhaustive)
                        with st.expander("📋 Copy v1 for Grok", expanded=True):
                            st.code(output, language=None)
                            copy_button(output, "📋 Copy v1 to Clipboard")
                except Exception as e:
                    st.error(str(e))

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
        with st.spinner(f"Creating v{next_v} ..."):
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

                with dspy.context(lm=lm):
                    result = generator(user_directions=enhanced_input)

                output = result.detailed_prompt

                if not is_valid_output(output):
                    st.error(
                        f"⚠️ v{next_v} came back empty or malformed. "
                        "This usually means the feedback contained too much review commentary and not enough architectural direction. "
                        "Try stripping ratings/scores from the feedback and resubmitting, "
                        "or switch to ChainOfThought mode in the sidebar."
                    )
                else:
                    state["prompt_history"].append({
                        "version": next_v,
                        "prompt": output,
                        "feedback_used": grok_feedback.strip()[:300] + "..."
                    })
                    state["last_prompt"] = output
                    st.success(f"✅ v{next_v} generated!")
                    render_output(output, is_prd_mode or is_prd_exhaustive)
                    with st.expander(f"📋 Copy v{next_v} for Grok", expanded=True):
                        st.code(output, language=None)
                        copy_button(output, f"📋 Copy v{next_v} to Clipboard")

            except Exception as e:
                st.error(f"Error: {str(e)}")

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
token_note = "32k output tokens" if is_nvidia else "2k output tokens"
st.caption(
    f"Mode: {mode_label_map[mode]} • {provider} · {selected_model_label} ({token_note}) + Grok manual refinement • "
    "Graveyard only grows · Stack only shrinks · Confidence compounds"
)
