import streamlit as st
import dspy
import os
import re

st.set_page_config(
    page_title="Scene → Image Prompt Generator",
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
        index=1,
        help="ChainOfThought usually gives better results"
    )

# ====================== DSPy SETUP ======================
@st.cache_resource(show_spinner="Loading DSPy...")
def get_generator(module_type: str, model_name: str):
    if not os.getenv("OPENROUTER_API_KEY"):
        return None, None, None, "No API key set."

    try:
        lm = dspy.LM(
            model_name,
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=2048,
            temperature=0.7
        )

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

        class VideoScriptToScenePrompts(dspy.Signature):
            """
            You are an expert image prompt engineer specializing in turning video scripts into ultra-detailed, vivid, high-quality scene prompts for Flux, SD3, SDXL, Pony, etc.

ALWAYS follow these guidelines:
            - Strong cinematic composition and camera angles
            - Rich pose, body language, and clothing details (especially sheer/translucent fabrics)
            - Seductive atmosphere with professional lighting, shadows, and skin texture
            - Anatomically realistic + high-end erotic photography style
            - Tasteful yet explicit when appropriate
            - Output each scene as a ready-to-use, well-structured prompt (20-40 words) as a numbered list
            """
            video_script: str = dspy.InputField(
                desc="Full video script + previous scene prompts + all Grok feedback accumulated"
            )
            scene_prompts: str = dspy.OutputField(
                desc="Numbered list, one scene per line. Example: 1. wide shot of... 2. close-up of..."
            )

        if module_type == "ChainOfThought":
            scene_module = dspy.ChainOfThought(SceneToImagePrompt)
            video_module = dspy.ChainOfThought(VideoScriptToScenePrompts)
        else:
            scene_module = dspy.Predict(SceneToImagePrompt)
            video_module = dspy.Predict(VideoScriptToScenePrompts)

        return scene_module, video_module, lm, None
    except Exception as e:
        return None, None, None, str(e)

generator, video_generator, lm, load_error = get_generator(module_type, model_name)

# ====================== SESSION STATE ======================
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = ""
if 'original_input' not in st.session_state:
    st.session_state.original_input = ""
if 'scene_history' not in st.session_state:
    st.session_state.scene_history = []
if 'last_scenes' not in st.session_state:
    st.session_state.last_scenes = []
if 'original_script' not in st.session_state:
    st.session_state.original_script = ""


def parse_scenes(raw: str):
    """Parse numbered scene list from model output using 3 fallback strategies."""
    # Strategy 1: split by newlines, strip leading numbers
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    scenes = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)\-\:]\s*", "", line).strip()
        if cleaned:
            scenes.append(cleaned)

    # Strategy 2: split inline by "N. " pattern
    if not scenes:
        parts = re.split(r"\d+[\.\)]\s+", raw)
        scenes = [p.strip() for p in parts if p.strip()]

    # Strategy 3: whole output as one scene
    if not scenes and raw:
        scenes = [raw]

    return scenes


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
            elif generator is None:
                st.error(f"Generator error: {load_error}")
            else:
                with st.spinner(f"Generating v1 with {selected_model_label}..."):
                    try:
                        with dspy.context(lm=lm):
                            result = generator(user_directions=user_input.strip())

                        st.session_state.original_input = user_input.strip()
                        st.session_state.last_prompt = result.detailed_prompt
                        st.session_state.prompt_history = [{
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
            st.session_state.prompt_history = []
            st.session_state.last_prompt = ""
            st.session_state.original_input = ""
            st.rerun()

    # ====================== GROK REFINEMENT ======================
    st.subheader("2. Iterative Refinement with Grok (Manual)")
    st.markdown("Copy the latest prompt → paste into Grok → ask for improvements → paste Grok's reply here")

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
        else:
            with st.spinner(f"Creating v{len(st.session_state.prompt_history)+1} ..."):
                try:
                    enhanced_input = f"""Original scene description:
{st.session_state.original_input}

Previous best prompt (v{len(st.session_state.prompt_history)}):
{st.session_state.last_prompt}

Grok has repeatedly suggested the following improvements across feedback:
{grok_feedback.strip()}

Create the strongest next version. Incorporate all the valuable patterns and elements Grok has been emphasizing."""

                    with dspy.context(lm=lm):
                        result = generator(user_directions=enhanced_input)

                    new_version = len(st.session_state.prompt_history) + 1
                    st.session_state.prompt_history.append({
                        "version": new_version,
                        "prompt": result.detailed_prompt,
                        "feedback_used": grok_feedback.strip()[:300] + "..." 
                    })
                    st.session_state.last_prompt = result.detailed_prompt

                    st.success(f"✅ v{new_version} generated!")
                    st.code(result.detailed_prompt, language=None)

                    st.button("📋 Copy v{new_version} for Grok", 
                             on_click=lambda: st.clipboard(result.detailed_prompt) or st.toast("Copied!"))
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # ====================== HISTORY ======================
    if st.session_state.prompt_history:
        st.subheader("📜 Prompt Evolution")
        for item in reversed(st.session_state.prompt_history):
            with st.expander(f"Version {item['version']}"):
                st.code(item['prompt'], language=None)
                st.caption(f"Based on: {item['feedback_used']}")

    st.caption("MiniMax M2.5 Free + Grok manual refinement • Signature stays constant • Quality compounds with each Grok dump")


# ====================== PAGE 2: Video Script → Scenes ======================
elif page == "🎥 Video Script → Scenes":
    st.title("🎥 Video Script → Scene Prompts")
    st.markdown(
        "Break a video script into individual NSFW scene prompts — "
        "each ready to drop into **Scene → Image Prompt** as your starting description."
    )

    st.subheader("1. Paste Your Video Script")
    video_script_input = st.text_area(
        "Your video script:",
        placeholder=(
            "Scene 1: The camera opens on a dimly lit penthouse at dusk...\n"
            "Scene 2: She walks toward the floor-to-ceiling window, silhouetted by city lights...\n"
            "Scene 3: Close-up on her hand tracing the condensation on a glass..."
        ),
        height=300
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Generate Scene Prompts (v1)", type="primary", use_container_width=True):
            if not os.getenv("OPENROUTER_API_KEY"):
                st.error("Please apply OpenRouter API key first.")
            elif not video_script_input.strip():
                st.error("Please paste a video script!")
            elif video_generator is None:
                st.error(f"Generator error: {load_error}")
            else:
                with st.spinner(f"Analyzing script with {selected_model_label}..."):
                    try:
                        with dspy.context(lm=lm):
                            result = video_generator(video_script=video_script_input.strip())

                        scenes = parse_scenes(result.scene_prompts.strip())
                        scenes_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(scenes))

                        st.session_state.original_script = video_script_input.strip()
                        st.session_state.last_scenes = scenes
                        st.session_state.scene_history = [{
                            "version": 1,
                            "scenes": scenes,
                            "feedback_used": "Initial generation - no Grok feedback yet"
                        }]

                        st.success(f"✅ v1 ready — {len(scenes)} scenes!")
                        for i, scene in enumerate(scenes, 1):
                            with st.expander(f"Scene {i}", expanded=True):
                                st.code(scene, language=None)

                        st.button("📋 Copy all scenes for Grok",
                                 on_click=lambda: st.clipboard(scenes_text) or st.toast("Copied!"))
                    except Exception as e:
                        st.error(str(e))

    with col2:
        if st.button("🔄 Reset Everything", type="secondary", use_container_width=True):
            st.session_state.scene_history = []
            st.session_state.last_scenes = []
            st.session_state.original_script = ""
            st.rerun()

    # ====================== GROK REFINEMENT ======================
    st.subheader("2. Iterative Refinement with Grok (Manual)")
    st.markdown("Copy the scene list → paste into Grok → ask for improvements → paste Grok's reply here")

    grok_scene_feedback = st.text_area(
        "Grok's feedback (paste entire response or key suggestions):",
        placeholder="Grok said: Scene 3 needs more explicit detail on the clothing, add dramatic rim lighting to Scene 1, Scene 5 should emphasize body language more...",
        height=160
    )

    if st.button("🚀 Generate Next Version with Grok Feedback", type="primary", use_container_width=True):
        if not st.session_state.last_scenes:
            st.error("Generate v1 first!")
        elif not grok_scene_feedback.strip():
            st.error("Paste Grok feedback first!")
        elif video_generator is None:
            st.error(f"Generator error: {load_error}")
        else:
            version_num = len(st.session_state.scene_history) + 1
            with st.spinner(f"Creating v{version_num}..."):
                try:
                    prev_scenes_text = "\n".join(
                        f"{i+1}. {s}" for i, s in enumerate(st.session_state.last_scenes)
                    )

                    enhanced_input = f"""Original video script:
{st.session_state.original_script}

Previous best scene prompts (v{len(st.session_state.scene_history)}):
{prev_scenes_text}

Grok has repeatedly suggested the following improvements across feedback:
{grok_scene_feedback.strip()}

Create the strongest next version. Incorporate all the valuable patterns and elements Grok has been emphasizing."""

                    with dspy.context(lm=lm):
                        result = video_generator(video_script=enhanced_input)

                    scenes = parse_scenes(result.scene_prompts.strip())
                    scenes_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(scenes))

                    st.session_state.scene_history.append({
                        "version": version_num,
                        "scenes": scenes,
                        "feedback_used": grok_scene_feedback.strip()[:300] + "..."
                    })
                    st.session_state.last_scenes = scenes

                    st.success(f"✅ v{version_num} generated — {len(scenes)} scenes!")
                    for i, scene in enumerate(scenes, 1):
                        with st.expander(f"Scene {i}", expanded=True):
                            st.code(scene, language=None)

                    st.button(f"📋 Copy all v{version_num} scenes for Grok",
                             on_click=lambda: st.clipboard(scenes_text) or st.toast("Copied!"))
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # ====================== HISTORY ======================
    if st.session_state.scene_history:
        st.subheader("📜 Scene List Evolution")
        for item in reversed(st.session_state.scene_history):
            with st.expander(f"Version {item['version']} — {len(item['scenes'])} scenes"):
                for i, scene in enumerate(item['scenes'], 1):
                    st.markdown(f"**{i}.** {scene}")
                st.caption(f"Based on: {item['feedback_used']}")

    st.caption(
        "Free models via OpenRouter + Grok manual refinement • "
        "Scene prompts are 20-40 words — seed them into Scene → Image Prompt for full 80-200 word detail"
    )