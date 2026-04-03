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
        options=["🎨 Image Prompt", "🎬 Video Scene Prompt"],
        index=0,
        horizontal=False,
        help="Switch between generating image prompts or video scene prompts. Same workflow, different AI focus."
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
        index=1,   # GLM 4.5 Air
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

if is_video_mode:
    st.title("🎬 Scene to Video Prompt Generator")
    st.markdown("Ultra-detailed prompts for Sora / Kling / Runway / Wan — motion, timing, camera + Grok-powered iterative refinement")
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

        else:  # Video Scene Prompt
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

        if module_type == "ChainOfThought":
            module = dspy.ChainOfThought(sig)
        else:
            module = dspy.Predict(sig)

        return module, lm, None
    except Exception as e:
        return None, None, str(e)

generator, lm, load_error = get_generator(module_type, model_name, mode)

# ====================== SESSION STATE ======================
# Separate history per mode so switching doesn't mix them up
img_state = st.session_state.setdefault("img", {"prompt_history": [], "last_prompt": "", "original_input": ""})
vid_state = st.session_state.setdefault("vid", {"prompt_history": [], "last_prompt": "", "original_input": ""})
state = vid_state if is_video_mode else img_state

# ====================== MAIN UI ======================
st.subheader("1. Initial Scene Description")

placeholder = (
    "a woman walks slowly through a rain-soaked alley at night, neon signs reflecting off wet pavement, camera tracks her from behind at ground level..."
    if is_video_mode else
    "beautiful woman in red lace lingerie, sitting on bed, legs slightly apart, soft warm lighting, low camera angle..."
)

user_input = st.text_area(
    "Your directions:",
    placeholder=placeholder,
    height=140
)

col1, col2 = st.columns(2)
with col1:
    if st.button("✨ Generate Initial Prompt (v1)", type="primary", use_container_width=True):
        if not os.getenv("OPENROUTER_API_KEY"):
            st.error("Please apply OpenRouter API key first.")
        elif not user_input.strip():
            st.error("Please describe your scene!")
        elif generator is None:
            st.error(f"Generator error: {load_error}")
        else:
            with st.spinner(f"Generating v1 with {selected_model_label}..."):
                try:
                    with dspy.context(lm=lm):
                        result = generator(user_directions=user_input.strip())

                    state["original_input"] = user_input.strip()
                    state["last_prompt"] = result.detailed_prompt
                    state["prompt_history"] = [{
                        "version": 1,
                        "prompt": result.detailed_prompt,
                        "feedback_used": "Initial generation - no Grok feedback yet"
                    }]

                    st.success("✅ v1 ready!")
                    st.code(result.detailed_prompt, language=None)

                    st.button("📋 Copy v1 for Grok",
                             on_click=lambda: st.clipboard(result.detailed_prompt) or st.toast("Copied to clipboard!"))
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
st.markdown("Copy the latest prompt → paste into Grok → ask for improvements → paste Grok's reply here")

grok_feedback = st.text_area(
    "Grok's feedback (paste entire response or key suggestions):",
    placeholder=(
        "Grok said: Add a slow rack focus from foreground rain to her face, extend the dolly move, add breath mist in cold air..."
        if is_video_mode else
        "Grok said: Add more dramatic rim lighting, make the fabric more sheer and clinging, use a lower camera angle, emphasize skin glow and subtle sweat beads..."
    ),
    height=160
)

if st.button("🚀 Generate Next Version with Grok Feedback", type="primary", use_container_width=True):
    if not state["last_prompt"]:
        st.error("Generate v1 first!")
    elif not grok_feedback.strip():
        st.error("Paste Grok feedback first!")
    else:
        with st.spinner(f"Creating v{len(state['prompt_history'])+1} ..."):
            try:
                enhanced_input = f"""Original scene description:
{state['original_input']}

Previous best prompt (v{len(state['prompt_history'])}):
{state['last_prompt']}

Grok has repeatedly suggested the following improvements across feedback:
{grok_feedback.strip()}

Create the strongest next version. Incorporate all the valuable patterns and elements Grok has been emphasizing."""

                with dspy.context(lm=lm):
                    result = generator(user_directions=enhanced_input)

                new_version = len(state["prompt_history"]) + 1
                state["prompt_history"].append({
                    "version": new_version,
                    "prompt": result.detailed_prompt,
                    "feedback_used": grok_feedback.strip()[:300] + "..."
                })
                state["last_prompt"] = result.detailed_prompt

                st.success(f"✅ v{new_version} generated!")
                st.code(result.detailed_prompt, language=None)

                st.button(f"📋 Copy v{new_version} for Grok",
                         on_click=lambda: st.clipboard(result.detailed_prompt) or st.toast("Copied!"))
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ====================== HISTORY ======================
if state["prompt_history"]:
    st.subheader("📜 Prompt Evolution")
    for item in reversed(state["prompt_history"]):
        with st.expander(f"Version {item['version']}"):
            st.code(item['prompt'], language=None)
            st.caption(f"Based on: {item['feedback_used']}")

mode_label = "Video (Sora/Kling/Runway)" if is_video_mode else "Image (Flux/SD3/SDXL)"
st.caption(f"Mode: {mode_label} • {selected_model_label} + Grok manual refinement • Signature stays constant • Quality compounds with each Grok dump")
