import streamlit as st
import dspy
import os

# Page config
st.set_page_config(
    page_title="Scene → Image Prompt Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Scene to Image Prompt Generator")
st.markdown("Turn casual descriptions into ultra-detailed prompts for Flux, SD3, SDXL, etc.")

# ====================== SIDEBAR SETTINGS ======================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key
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

    # Model Selection (includes Z-AI GLM-4.5-Air Free)
    model_options = {
        "Google Gemini Flash 1.5 (Recommended)": "openrouter/google/gemini-flash-1.5:free",
        "Z-AI GLM-4.5-Air Free": "openrouter/z-ai/glm-4.5-air:free",
        "Meta Llama 3.3 70B": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "Qwen3 235B Thinking": "openrouter/qwen/qwen3-235b-thinking:free",
        "Custom Model": "custom"
    }
    
    selected_model_label = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=1,   # Default = Z-AI GLM-4.5-Air Free
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

# ====================== DSPy SETUP (Thread-Safe) ======================
@st.cache_resource(show_spinner="Loading DSPy generator...")
def get_generator(module_type: str, model_name: str):
    if not os.getenv("OPENROUTER_API_KEY"):
        return None, "No API key set. Please apply API key first."

    try:
        # Create LM fresh every time (avoids thread configure conflict)
        lm = dspy.LM(
            model_name,
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=2048,
            temperature=0.75
        )

        # === YOUR EXACT SIGNATURE ===
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
            user_directions: str = dspy.InputField(desc="Free-form directions from the user (e.g. 'she spreads legs wide, wears translucent, low camera angle')")

            detailed_prompt: str = dspy.OutputField(desc="Full, optimized image generation prompt")

        # Create module (we use dspy.context later for safety)
        if module_type == "ChainOfThought":
            module = dspy.ChainOfThought(SceneToImagePrompt)
        else:
            module = dspy.Predict(SceneToImagePrompt)

        return module, lm, None   # return module + lm for context usage

    except Exception as e:
        return None, None, str(e)

# Get the cached components
generator, lm, load_error = get_generator(module_type, model_name)

# ====================== MAIN UI ======================
st.subheader("Describe your scene")
user_input = st.text_area(
    "Your directions:",
    placeholder="beautiful woman in red lace lingerie, sitting on bed, legs slightly apart, soft warm lighting, low camera angle...",
    height=180
)

if st.button("✨ Generate Detailed Prompt", type="primary", use_container_width=True):
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("⚠️ Please enter and Apply your OpenRouter API key first.")
    elif not user_input.strip():
        st.error("⚠️ Please describe your scene!")
    elif generator is None:
        st.error(f"❌ Failed to load generator: {load_error}")
    else:
        with st.spinner(f"Generating with {module_type} using {selected_model_label}..."):
            try:
                # Use dspy.context to avoid thread configure error
                with dspy.context(lm=lm):
                    result = generator(user_directions=user_input.strip())
                
                st.success("✅ Prompt generated successfully!")
                
                st.subheader("📋 Your Optimized Image Prompt")
                st.code(result.detailed_prompt, language=None)
                
                st.download_button(
                    label="⬇️ Download Prompt",
                    data=result.detailed_prompt,
                    file_name="detailed_image_prompt.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Generation error: {str(e)}")

st.caption("Z-AI GLM-4.5-Air Free supported • Use Apply buttons after changes")