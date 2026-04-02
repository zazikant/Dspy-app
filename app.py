import streamlit as st
import dspy
import os

st.set_page_config(
    page_title="Scene → Image Prompt Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Scene to Image Prompt Generator")
st.markdown("Ultra-detailed prompts for Flux / SD3 / SDXL + Grok-powered iterative refinement")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Settings")

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
        "Google Gemini Flash 1.5 (Recommended)": "openrouter/google/gemini-flash-1.5:free",
        "MiniMax M2.5 (Free)": "openrouter/minimax/minimax-m2.5:free",
        "Z-AI GLM-4.5-Air Free": "openrouter/z-ai/glm-4.5-air:free",
        "Meta Llama 3.3 70B": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "Qwen3 235B Thinking": "openrouter/qwen/qwen3-235b-thinking:free",
        "Custom Model": "custom"
    }

    selected_model_label = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=1,   # MiniMax M2.5 default
    )

    if selected_model_label == "Custom Model":
        model_name = st.text_input("Custom Model Name", value="openrouter/minimax/minimax-m2.5:free")
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
        return None, None, "No API key set."

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
            You are an expert image prompt engineer.You are an expert at turning loose, casual, or explicit user directions
    into ultra-detailed, vivid, high-quality prompts for modern text-to-image models
    (Flux, SD3, SDXL, Pony, etc.).

    ALWAYS follow these guidelines:
    - Start with strong cinematic composition (e.g. low-angle shot, dramatic perspective)
    - Emphasize body language, pose details, clothing (especially sheer/translucent fabrics)
    - Build seductive/mood-rich atmosphere with lighting, shadows, skin texture
    - Keep it anatomically realistic + professional erotic photography style
    - Tasteful yet explicit when appropriate — avoid vulgar/cartoon unless user says so
    - Output format: comma-separated descriptive tags (80–150 words), ready to copy-paste

    Examples:

    taste="photorealistic",
                user_input="a cat sitting on a windowsill",
                enhanced_prompt="You are an expert image prompt engineer. Transform this image concept into a photorealistic AI image generation prompt.

    **Input**:
    - Image Concept: "a cat sitting on a windowsill"
    - Quality Style: "photorealistic"

    **Output**:
    Ultra realistic photograph of a ginger cat sitting on a sunlit windowsill, detailed fur texture with individual hairs visible, sharp focus on the subject's eyes, natural lighting with soft shadows, 85mm lens with shallow depth of field, visible dust particles in the light beam, detailed wood grain on the windowsill, warm color temperature"


taste="photorealistic",
                user_input="a mountain landscape at sunset",
                enhanced_prompt="You are an expert image prompt engineer. Transform this image concept into a photorealistic AI image generation prompt.

**Input**:
- Image Concept: "a mountain landscape at sunset"
- Quality Style: "photorealistic"

**Output**:
Breathtaking photorealistic landscape of snow-capped mountains at golden hour, warm sunset colors reflecting on a serene lake, volumetric lighting with sun rays breaking through clouds, ultra high detail with visible rock textures and snow crystals, 8k resolution, deep depth of field with foreground elements to establish scale, atmospheric perspective on distant peaks"
            """
            user_directions: str = dspy.InputField(desc="Original scene + previous prompt + all Grok feedback accumulated")

            detailed_prompt: str = dspy.OutputField(desc="Final optimized image generation prompt")

        if module_type == "ChainOfThought":
            module = dspy.ChainOfThought(SceneToImagePrompt)
        else:
            module = dspy.Predict(SceneToImagePrompt)

        return module, lm, None
    except Exception as e:
        return None, None, str(e)

generator, lm, load_error = get_generator(module_type, model_name)

# ====================== SESSION STATE ======================
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = ""
if 'original_input' not in st.session_state:
    st.session_state.original_input = ""

# ====================== MAIN UI ======================
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
st.markdown("Copy the latest prompt → paste into Grok → ask for improvements → paste Grok’s reply here")

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