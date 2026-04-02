import streamlit as st
import dspy
import os

st.set_page_config(
    page_title="AI Image Prompt Studio",
    page_icon="🎨",
    layout="centered"
)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Settings")

    page = st.radio(
        "Page",
        ["🎬 Scene → Image Prompt", "🎥 Video Script → Scenes"],
        label_visibility="collapsed"
    )

    st.divider()

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
        index=1,
        help="ChainOfThought usually gives better results"
    )

# ====================== DSPy SETUP ======================
@st.cache_resource(show_spinner="Loading DSPy...")
def get_lm(model_name: str):
    if not os.getenv("OPENROUTER_API_KEY"):
        return None, "No API key set."
    try:
        lm = dspy.LM(
            model_name,
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=2048,
            temperature=0.7
        )
        return lm, None
    except Exception as e:
        return None, str(e)

lm, load_error = get_lm(model_name)

# ====================== DSPy SIGNATURES ======================

class SceneToImagePrompt(dspy.Signature):
    """
    You are an expert image prompt engineer specializing in turning loose or explicit user directions
    into ultra-detailed, vivid, high-quality prompts for Flux, SD3, SDXL, Pony, etc.

    ALWAYS follow these guidelines:
    - Strong cinematic composition and camera angles
    - Rich pose, body language, and clothing details (especially sheer/translucent fabrics)
    - Seductive atmosphere with professional lighting, shadows, and skin texture
    - Anatomically realistic + high-end erotic photography style
    - Tasteful yet explicit when appropriate
    - Output a ready-to-use, well-structured detailed prompt (80-200 words)
    """
    user_directions: str = dspy.InputField(
        desc="Original scene + previous prompt + all Grok feedback accumulated"
    )
    detailed_prompt: str = dspy.OutputField(
        desc="Final optimized image generation prompt"
    )


class VideoScriptToScenePrompts(dspy.Signature):
    """
    You are an expert at analyzing video scripts and breaking them into distinct visual scenes.
    For each scene, write a concise, self-contained image prompt (20-40 words) capturing:
    - Setting and environment
    - Subjects and their action/pose
    - Mood and atmosphere
    - Lighting style
    Each prompt must be ready to paste directly into an image generation tool like Flux or SDXL.
    Output ONLY a numbered list, one prompt per scene — no headers, no extra text.
    """
    video_script: str = dspy.InputField(
        desc="Full video script text to be broken down into distinct visual scenes"
    )
    scene_prompts: str = dspy.OutputField(
        desc=(
            "Numbered list of brief image prompts, one per scene. "
            "Each 20-40 words, self-contained, and image-generation ready. "
            "Format: '1. [prompt]\\n2. [prompt]\\n...'"
        )
    )


def get_module(signature_class, module_type):
    if module_type == "ChainOfThought":
        return dspy.ChainOfThought(signature_class)
    return dspy.Predict(signature_class)


# ====================== SESSION STATE ======================
for key, default in [
    ("prompt_history", []),
    ("last_prompt", ""),
    ("original_input", ""),
    ("scene_list", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ====================== PAGE 1: Scene → Image Prompt ======================
if page == "🎬 Scene → Image Prompt":
    st.title("🎨 Scene to Image Prompt Generator")
    st.markdown("Ultra-detailed prompts for Flux / SD3 / SDXL + Grok-powered iterative refinement")

    st.subheader("1. Initial Scene Description")
    user_input = st.text_area(
        "Your directions:",
        placeholder="beautiful woman in red lace lingerie, sitting on bed, legs slightly apart, soft warm lighting, low camera angle...",
        height=140
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Generate Initial Prompt (v1)", type="primary", use_container_width=True):
            if not os.getenv("OPENROUTER_API_KEY"):
                st.error("Please apply OpenRouter API key first.")
            elif not user_input.strip():
                st.error("Please describe your scene!")
            elif lm is None:
                st.error(f"Generator error: {load_error}")
            else:
                with st.spinner(f"Generating v1 with {selected_model_label}..."):
                    try:
                        module = get_module(SceneToImagePrompt, module_type)
                        with dspy.context(lm=lm):
                            result = module(user_directions=user_input.strip())

                        st.session_state.original_input = user_input.strip()
                        st.session_state.last_prompt = result.detailed_prompt
                        st.session_state.prompt_history = [{
                            "version": 1,
                            "prompt": result.detailed_prompt,
                            "feedback_used": "Initial generation — no Grok feedback yet"
                        }]
                        st.success("✅ v1 ready!")
                        st.code(result.detailed_prompt, language=None)
                    except Exception as e:
                        st.error(str(e))

    with col2:
        if st.button("🔄 Reset Everything", type="secondary", use_container_width=True):
            st.session_state.prompt_history = []
            st.session_state.last_prompt = ""
            st.session_state.original_input = ""
            st.rerun()

    # ====================== GROK REFINEMENT ======================
    st.subheader("2. Iterative Refinement with Grok (Manual)")
    st.markdown(
        "Copy the latest prompt → paste into Grok → ask for improvements → paste Grok's reply here. "
        "**Each iteration accumulates all prior feedback — quality compounds.**"
    )

    grok_feedback = st.text_area(
        "Grok's feedback (paste entire response or key suggestions):",
        placeholder="Grok said: Add more dramatic rim lighting, make the fabric more sheer and clinging, use a lower camera angle, emphasize skin glow and subtle sweat beads...",
        height=160
    )

    if st.button("🚀 Generate Next Version with Grok Feedback", type="primary", use_container_width=True):
        if not st.session_state.last_prompt:
            st.error("Generate v1 first!")
        elif not grok_feedback.strip():
            st.error("Paste Grok feedback first!")
        elif lm is None:
            st.error(f"Generator error: {load_error}")
        else:
            version_num = len(st.session_state.prompt_history) + 1
            with st.spinner(f"Creating v{version_num}..."):
                try:
                    # Accumulate ALL prior Grok feedback rounds (not just latest)
                    prior_feedback_items = [
                        item for item in st.session_state.prompt_history
                        if item["feedback_used"] != "Initial generation — no Grok feedback yet"
                    ]
                    if prior_feedback_items:
                        accumulated = "\n\n---\n\n".join(
                            f"Round {i+1} feedback:\n{item['feedback_used']}"
                            for i, item in enumerate(prior_feedback_items)
                        )
                        accumulated += f"\n\n---\n\nRound {len(prior_feedback_items)+1} feedback (latest):\n{grok_feedback.strip()}"
                    else:
                        accumulated = grok_feedback.strip()

                    enhanced_input = (
                        f"Original scene description:\n{st.session_state.original_input}\n\n"
                        f"Previous best prompt (v{len(st.session_state.prompt_history)}):\n{st.session_state.last_prompt}\n\n"
                        f"All accumulated Grok feedback across iterations "
                        f"(themes that repeat across rounds are highest priority):\n{accumulated}\n\n"
                        f"Create the strongest next version. "
                        f"Patterns Grok has emphasized multiple times must dominate the result."
                    )

                    module = get_module(SceneToImagePrompt, module_type)
                    with dspy.context(lm=lm):
                        result = module(user_directions=enhanced_input)

                    st.session_state.prompt_history.append({
                        "version": version_num,
                        "prompt": result.detailed_prompt,
                        "feedback_used": grok_feedback.strip()
                    })
                    st.session_state.last_prompt = result.detailed_prompt

                    st.success(f"✅ v{version_num} generated!")
                    st.code(result.detailed_prompt, language=None)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # ====================== HISTORY ======================
    if st.session_state.prompt_history:
        st.subheader("📜 Prompt Evolution")
        for item in reversed(st.session_state.prompt_history):
            with st.expander(f"Version {item['version']}"):
                st.code(item['prompt'], language=None)
                st.caption(f"Based on: {item['feedback_used']}")

    st.caption(
        "Free models via OpenRouter + Grok manual refinement • "
        "All Grok feedback accumulates across iterations • Quality compounds with each round"
    )


# ====================== PAGE 2: Video Script → Scenes ======================
elif page == "🎥 Video Script → Scenes":
    st.title("🎥 Video Script → Scene Prompts")
    st.markdown(
        "Break a video script into individual scene prompts — "
        "each ready to drop into **Scene → Image Prompt** as your starting description."
    )

    video_script = st.text_area(
        "Paste your video script:",
        placeholder=(
            "Scene 1: The camera opens on a dimly lit penthouse at dusk...\n"
            "Scene 2: She walks toward the floor-to-ceiling window, silhouetted by city lights...\n"
            "Scene 3: Close-up on her hand tracing the condensation on a glass..."
        ),
        height=300
    )

    if st.button("🎬 Break Into Scene Prompts", type="primary", use_container_width=True):
        if not os.getenv("OPENROUTER_API_KEY"):
            st.error("Please apply OpenRouter API key first.")
        elif not video_script.strip():
            st.error("Please paste a video script!")
        elif lm is None:
            st.error(f"Generator error: {load_error}")
        else:
            with st.spinner(f"Analyzing script with {selected_model_label}..."):
                try:
                    module = get_module(VideoScriptToScenePrompts, module_type)
                    with dspy.context(lm=lm):
                        result = module(video_script=video_script.strip())

                    raw_lines = result.scene_prompts.strip().split("\n")
                    scenes = []
                    for line in raw_lines:
                        line = line.strip()
                        if not line:
                            continue
                        import re
                        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                        if cleaned:
                            scenes.append(cleaned)

                    st.session_state.scene_list = scenes
                    st.success(f"✅ Found {len(scenes)} scenes!")
                except Exception as e:
                    st.error(str(e))

    if st.session_state.scene_list:
        st.subheader(f"📋 {len(st.session_state.scene_list)} Scene Prompts")
        st.markdown(
            "Copy any prompt below → switch to **Scene → Image Prompt** → "
            "paste it as your scene description → generate a full detailed prompt."
        )
        for i, scene in enumerate(st.session_state.scene_list, 1):
            with st.expander(f"Scene {i}", expanded=True):
                st.code(scene, language=None)

        if st.button("🔄 Clear Scenes", type="secondary", use_container_width=True):
            st.session_state.scene_list = []
            st.rerun()

    st.caption(
        "Free models via OpenRouter • Scene prompts are 20-40 words — "
        "seed them into Scene → Image Prompt for full 80-200 word detail"
    )