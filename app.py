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

    # ── Mode Toggle ────────────────────────────────────
    st.markdown("### 🎛️ Prompt Mode")
    mode = st.radio(
        label="Select mode:",
        options=["🎨 Image Prompt", "🎬 Video Scene Prompt", "🧠 Software PRD Prompt"],
        index=0,
        horizontal=False,
        help="Switch between Image, Video, or PRD prompt generation. Same Grok refinement workflow throughout."
    )
    st.divider()
    # ───────────────────────────────────────────────────

    api_key_input = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Paste your OpenRouter API key here."
    )

    if st.button("✅ Apply API Key", type="primary", use_container_width=True):
        if api_key_input.strip():
            os.environ["OPENROUTER_API_KEY"] = api_key_input.strip()
            st.success("✅ API Key applied")
            st.rerun()
        else:
            st.error("Please enter an API key.")

    model_options = {
        "Qwen3.6 Plus Preview (Best overall)": "openrouter/qwen/qwen3.6-plus-preview:free",
        "GLM 4.5 Air (Default)": "openrouter/z-ai/glm-4.5-air:free",
        "Step 3.5 Flash (Fast & strong)": "openrouter/stepfun/step-3.5-flash:free",
        "Arcee Trinity Large Preview (Creative)": "openrouter/arcee-ai/trinity-large-preview:free",
        "MiniMax M2.5": "openrouter/minimax/minimax-m2.5:free",
        "Llama 3.3 70B Instruct": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "Custom Model": "custom"
    }

    selected_model_label = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=1,
    )

    if selected_model_label == "Custom Model":
        model_name = st.text_input("Custom Model Name", value="openrouter/z-ai/glm-4.5-air:free")
    else:
        model_name = model_options[selected_model_label]

    if st.button("✅ Apply Model", type="primary", use_container_width=True):
        st.success(f"✅ Model set to: {selected_model_label}")
        st.rerun()

    module_type = st.selectbox(
        "Reasoning Mode",
        options=["Predict", "ChainOfThought"],
        index=0,
        help="ChainOfThought usually gives better results"
    )

# ====================== DYNAMIC TITLE ======================
is_video_mode = mode == "🎬 Video Scene Prompt"
is_prd_mode   = mode == "🧠 Software PRD Prompt"
is_image_mode = mode == "🎨 Image Prompt"

if is_video_mode:
    st.title("🎬 Scene to Video Prompt Generator")
    st.markdown("Ultra-detailed prompts for Sora / Kling / Runway / Wan — motion, timing, camera + Grok-powered iterative refinement")
elif is_prd_mode:
    st.title("🧠 Software PRD Meta-Prompt Generator")
    st.markdown(
        "Converge on a bulletproof technical architecture through iterative Grok refinement — "
        "each round reinforces the strongest patterns until the stack, workflow, and data model are locked in."
    )
else:
    st.title("🎨 Scene to Image Prompt Generator")
    st.markdown("Ultra-detailed prompts for Flux / SD3 / SDXL + Grok-powered iterative refinement")

# ====================== DSPy SETUP ======================
@st.cache_resource(show_spinner="Loading DSPy...")
def get_generator(module_type: str, model_name: str, mode: str):
    if not os.getenv("OPENROUTER_API_KEY"):
        return None, None, "No API key set."

    try:
        lm = dspy.LM(
            model_name,
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
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
        else:
            class SoftwareToPRDPrompt(dspy.Signature):
                """
                You are a senior software architect and technical product strategist. Your job is to take a raw feature idea or problem statement and produce a comprehensive, opinionated PRD meta-prompt — a living technical document that sharpens itself with every round of expert feedback.

Think like someone who has seen every naive approach fail and every clever pattern succeed. Be decisive. Name the architecture. Commit to the stack. Call out the anti-patterns.

ALWAYS structure your output as a complete PRD meta-prompt covering ALL of the following sections:

1. PROBLEM STATEMENT
   - Crisp one-paragraph definition of what is being solved and why naive approaches break down

2. CORE ARCHITECTURE DECISION
   - Name the primary architectural pattern chosen (e.g. event-driven pipeline, query chain, agent loop, CQRS, etc.)
   - State WHY this pattern wins over the alternatives that were considered
   - Call out any pattern that must be explicitly avoided and why

3. TECH STACK & TOOLING
   - Backend: language, framework, key libraries
   - Data layer: database type, ORM/query strategy, caching approach
   - Integrations: APIs, queues, external services
   - Infrastructure: deployment, observability

4. DATA MODEL & FLOW
   - Key entities and their relationships
   - How data moves through the system end-to-end (ingestion → processing → storage → retrieval)
   - Any transformation or enrichment steps

5. WORKFLOW & SEQUENCE
   - Step-by-step operational flow a developer would implement
   - Decision points, branching logic, error handling strategy
   - State management approach

6. INTERFACE CONTRACTS
   - API shape (REST / GraphQL / RPC) with key endpoints or function signatures
   - Input validation strategy
   - Response structure and error codes

7. OPEN QUESTIONS & NEXT REFINEMENT TARGETS
   - What is still unresolved
   - Which decisions Grok should stress-test next
   - Hypotheses worth challenging

TONE: Opinionated, specific, architect-grade. No vague platitudes. Every sentence should either name something concrete or make a decision. When Grok feedback has reinforced a pattern across multiple rounds, mark it ✅ CONFIRMED ARCHITECTURE — this signals it has survived scrutiny and is locked in.

Output the full PRD meta-prompt as a well-structured document (200-500 words). It must be immediately usable as context for a developer or for the next Grok refinement round.
                """
                user_directions: str = dspy.InputField(
                    desc="Original feature/problem description + previous PRD meta-prompt + all Grok architectural feedback accumulated across rounds"
                )
                detailed_prompt: str = dspy.OutputField(
                    desc="Full PRD meta-prompt: architecture decisions, tech stack, data model, workflow, interface contracts, and open questions — sharpened by all Grok feedback so far"
                )

            sig = SoftwareToPRDPrompt

        if module_type == "ChainOfThought":
            module = dspy.ChainOfThought(sig)
        else:
            module = dspy.Predict(sig)

        return module, lm, None
    except Exception as e:
        return None, None, str(e)

generator, lm, load_error = get_generator(module_type, model_name, mode)

# ====================== SESSION STATE ======================
# Separate history per mode so switching never mixes them
img_state = st.session_state.setdefault("img", {"prompt_history": [], "last_prompt": "", "original_input": ""})
vid_state = st.session_state.setdefault("vid", {"prompt_history": [], "last_prompt": "", "original_input": ""})
prd_state = st.session_state.setdefault("prd", {"prompt_history": [], "last_prompt": "", "original_input": ""})

if is_prd_mode:
    state = prd_state
elif is_video_mode:
    state = vid_state
else:
    state = img_state

# ====================== MAIN UI ======================
if is_prd_mode:
    st.subheader("1. Describe Your Software Feature or Problem")
    st.markdown(
        "Write what you're trying to build — as rough or as detailed as you like. "
        "The generator will produce an opinionated PRD meta-prompt with full architecture, "
        "stack, data model, and workflow. Then paste it into Grok for critique. "
        "Each round locks in the patterns Grok keeps reinforcing — marked ✅ CONFIRMED ARCHITECTURE."
    )
else:
    st.subheader("1. Initial Scene Description")

placeholder_map = {
    "🎨 Image Prompt": "beautiful woman in red lace lingerie, sitting on bed, legs slightly apart, soft warm lighting, low camera angle...",
    "🎬 Video Scene Prompt": "a woman walks slowly through a rain-soaked alley at night, neon signs reflecting off wet pavement, camera tracks her from behind at ground level...",
    "🧠 Software PRD Prompt": (
        "I want to build a natural language interface that lets non-technical users query our PostgreSQL database by typing plain English questions. "
        "The system should handle ambiguous phrasing, multi-table joins, and return results in a human-readable summary alongside the raw data..."
    )
}

user_input = st.text_area(
    "Your directions:" if not is_prd_mode else "Feature / problem description:",
    placeholder=placeholder_map[mode],
    height=160 if is_prd_mode else 140
)

col1, col2 = st.columns(2)
with col1:
    btn_label = "🧠 Generate Initial PRD Meta-Prompt (v1)" if is_prd_mode else "✨ Generate Initial Prompt (v1)"
    if st.button(btn_label, type="primary", use_container_width=True):
        if not os.getenv("OPENROUTER_API_KEY"):
            st.error("Please apply OpenRouter API key first.")
        elif not user_input.strip():
            st.error("Please describe your feature / scene!")
        elif generator is None:
            st.error(f"Generator error: {load_error}")
        else:
            spinner_msg = f"Generating v1 with {selected_model_label}..."
            with st.spinner(spinner_msg):
                try:
                    with dspy.context(lm=lm):
                        result = generator(user_directions=user_input.strip())

                    state["original_input"] = user_input.strip()
                    state["last_prompt"] = result.detailed_prompt
                    state["prompt_history"] = [{
                        "version": 1,
                        "prompt": result.detailed_prompt,
                        "feedback_used": "Initial generation — no Grok feedback yet"
                    }]

                    st.success("✅ v1 ready!")
                    if is_prd_mode:
                        st.markdown(result.detailed_prompt)
                    else:
                        st.code(result.detailed_prompt, language=None)

                    st.button(
                        "📋 Copy v1 for Grok",
                        on_click=lambda: st.clipboard(result.detailed_prompt) or st.toast("Copied to clipboard!")
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

if is_prd_mode:
    st.markdown(
        "Paste the PRD meta-prompt into Grok → ask it to critique the architecture, suggest better patterns, "
        "challenge tech choices → paste Grok's full reply here. "
        "The generator will absorb all feedback and promote repeatedly-confirmed decisions to ✅ CONFIRMED ARCHITECTURE."
    )
else:
    st.markdown("Copy the latest prompt → paste into Grok → ask for improvements → paste Grok's reply here")

placeholder_feedback_map = {
    "🎨 Image Prompt": "Grok said: Add more dramatic rim lighting, make the fabric more sheer and clinging, use a lower camera angle, emphasize skin glow and subtle sweat beads...",
    "🎬 Video Scene Prompt": "Grok said: Add a slow rack focus from foreground rain to her face, extend the dolly move, add breath mist in cold air...",
    "🧠 Software PRD Prompt": (
        "Grok said: Ditch the raw SQL generation approach — use a SQL Query Chain (LangChain) instead so each sub-question is decomposed, "
        "validated, and executed independently before results are merged. Add a semantic caching layer (Redis + vector similarity) "
        "so repeated conceptually-similar questions bypass the LLM entirely. Schema introspection should happen once at startup and be stored "
        "in a structured context object passed into every chain call, not re-fetched per query..."
    )
}

grok_feedback = st.text_area(
    "Grok's feedback (paste entire response or key suggestions):",
    placeholder=placeholder_feedback_map[mode],
    height=180 if is_prd_mode else 160
)

refine_label = "🚀 Generate Next PRD Version with Grok Feedback" if is_prd_mode else "🚀 Generate Next Version with Grok Feedback"
if st.button(refine_label, type="primary", use_container_width=True):
    if not state["last_prompt"]:
        st.error("Generate v1 first!")
    elif not grok_feedback.strip():
        st.error("Paste Grok feedback first!")
    else:
        next_v = len(state["prompt_history"]) + 1
        with st.spinner(f"Creating v{next_v} ..."):
            try:
                if is_prd_mode:
                    enhanced_input = f"""Original feature / problem description:
{state['original_input']}

Previous PRD meta-prompt (v{len(state['prompt_history'])}):
{state['last_prompt']}

Grok's architectural feedback and suggestions (absorb ALL of this — promote repeatedly-confirmed patterns to ✅ CONFIRMED ARCHITECTURE):
{grok_feedback.strip()}

Produce the strongest next version of the PRD meta-prompt. Lock in every pattern Grok has reinforced. Sharpen open questions. Replace any section that Grok challenged with the superior approach it suggested."""
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

                state["prompt_history"].append({
                    "version": next_v,
                    "prompt": result.detailed_prompt,
                    "feedback_used": grok_feedback.strip()[:300] + "..."
                })
                state["last_prompt"] = result.detailed_prompt

                st.success(f"✅ v{next_v} generated!")
                if is_prd_mode:
                    st.markdown(result.detailed_prompt)
                else:
                    st.code(result.detailed_prompt, language=None)

                st.button(
                    f"📋 Copy v{next_v} for Grok",
                    on_click=lambda: st.clipboard(result.detailed_prompt) or st.toast("Copied!")
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ====================== HISTORY ======================
if state["prompt_history"]:
    history_label = "📜 PRD Evolution" if is_prd_mode else "📜 Prompt Evolution"
    st.subheader(history_label)

    if is_prd_mode:
        st.caption(
            "Each version absorbs more Grok critique. Decisions marked ✅ CONFIRMED ARCHITECTURE "
            "have survived multiple rounds of challenge — treat them as locked."
        )

    for item in reversed(state["prompt_history"]):
        with st.expander(f"Version {item['version']}"):
            if is_prd_mode:
                st.markdown(item['prompt'])
            else:
                st.code(item['prompt'], language=None)
            st.caption(f"Based on: {item['feedback_used']}")

mode_label_map = {
    "🎨 Image Prompt": "Image (Flux/SD3/SDXL)",
    "🎬 Video Scene Prompt": "Video (Sora/Kling/Runway)",
    "🧠 Software PRD Prompt": "PRD (Architecture → Confirmed Stack)"
}
st.caption(
    f"Mode: {mode_label_map[mode]} • {selected_model_label} + Grok manual refinement • "
    "Signature stays constant • Confidence compounds with each Grok dump"
)
