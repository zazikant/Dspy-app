import streamlit as st
import dspy
import os

st.set_page_config(
    page_title="Scene → Prompt Generator",
    page_icon="🎨",
    layout="centered"
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
            value=os.getenv("OPENROUTER_API_KEY", ""),
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
        st.info("🟢 NVIDIA NIM models require no API key.", icon="ℹ️")

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
        "Kimi K2.5 (32k output)": "nvidia_nim/moonshotai/kimi-k2",
        "MiniMax M2.5 (32k output)": "nvidia_nim/minimax/minimax-text-01",
        "GPT OSS 120B (32k output)": "nvidia_nim/deepseek/deepseek-r1-0528",
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
        "Designed for LangGraph systems with complex branching, conditional edges, sub-graphs, and multi-agent coordination. "
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

    if not is_nvidia and not api_key:
        return None, None, "No API key set."

    try:
        if is_nvidia:
            lm = dspy.LM(
                model_name,
                api_base="https://integrate.api.nvidia.com/v1",
                api_key="no-key-required",
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

        # ── EXHAUSTIVE PRD SIGNATURE ─────────────────────
        elif mode == "📐 Exhaustive PRD (32k)":
            class ExhaustivePRDPrompt(dspy.Signature):
                """
                You are a senior software architect and technical product strategist. Your job is to take a raw feature idea or problem statement and produce a comprehensive, opinionated PRD meta-prompt — a living technical document that sharpens itself with every round of expert feedback.

Think like someone who has seen every naive approach fail and every clever pattern succeed. Be decisive. Name the architecture. Commit to the stack. Call out the anti-patterns. And crucially: be willing to KILL components that don't survive scrutiny.

YOU HAVE A 32,000 TOKEN OUTPUT BUDGET. USE IT.
Do not summarise. Do not truncate. Do not write "see above" or "as mentioned". Every section must be written in full, every time.
Every component, every decision, every data field, every interface — spelled out completely.
If a previous version had 8 sections, this version must have at least 8 — likely more, and each one longer.
The Graveyard must grow with every round. If nothing was killed, challenge the weakest ⚠️ item and kill it.
A short PRD here is a failure. You have the token budget — spend it on precision and completeness.

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
   - Name every component, node, service, or stage explicitly with its connections and transitions
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
- You have 32,000 output tokens — there is no excuse for a thin or vague section
- Write every section exhaustively: every component named, every field defined, every decision explained
- Every version must have FEWER ⚠️ CHALLENGED items than the previous version
- The Graveyard must grow with each Grok round or you are not being decisive enough
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
                    desc="Full exhaustive PRD with ✅ CONFIRMED / ⚠️ CHALLENGED / ❌ REMOVED markers on every component, plus Architecture Graveyard. Use the full 32k token budget — every section written completely, nothing summarised. Must never be empty or None."
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
_api_key = "" if is_nvidia else os.getenv("OPENROUTER_API_KEY", "")
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
        if not is_nvidia and not os.getenv("OPENROUTER_API_KEY"):
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
                        st.button(
                            "📋 Copy v1 for Grok",
                            on_click=lambda: st.clipboard(output) or st.toast("Copied to clipboard!")
                        )
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
        "- CONFIRM LangGraph as orchestrator\n"
        "- CONFIRM GPT-4o-mini + Pydantic structured output as extraction core\n"
        "- Name the LangGraph nodes explicitly: pdf_loader → ocr_detector → text_extractor → llm_extractor → validator → formatter\n"
        "- Add OCR fallback branch for scanned PDFs using pdf2image + Tesseract\n"
        "- Define the state TypedDict fields between nodes"
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
                    enhanced_input = f"""Original feature / problem description:
{state['original_input']}

Previous Exhaustive PRD (v{len(state['prompt_history'])}):
{state['last_prompt']}

Structural and architectural feedback to apply:
{grok_feedback.strip()}

MANDATORY INSTRUCTIONS FOR THIS VERSION:
- This version MUST be longer and more detailed than v{len(state['prompt_history'])} — no section may shrink
- Add every new node mentioned in feedback to the Node Catalogue with ALL required fields fully written out
- Add every new conditional edge to the Conditional Edge Map with its full boolean condition
- Expand the State TypedDict with any new fields — include type, comment, and which node writes it
- Promote every reinforced component to ✅ CONFIRMED with a full explanatory paragraph
- Move every challenged component to ⚠️ CHALLENGED with a multi-sentence defence, or kill it immediately
- Move every killed component to ❌ REMOVED — add to Graveyard with round number and exact kill reason
- The Architecture Graveyard must be larger than v{len(state['prompt_history'])}'s Graveyard
- Every ⚠️ CHALLENGED item from the previous version must be confirmed or removed — none survive unchanged
- Return the COMPLETE exhaustive PRD. Never truncate. Never summarise. Never return partial output."""
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
                    st.button(
                        f"📋 Copy v{next_v} for Grok",
                        on_click=lambda: st.clipboard(output) or st.toast("Copied!")
                    )

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
