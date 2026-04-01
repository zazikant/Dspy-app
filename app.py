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
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        help="Enter your OpenRouter API key here."
    )
    
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
        st.success("API Key set ✓")
    else:
        st.warning("Enter API key to enable generation.")

    # Model Name Input
    model_name = st.text_input(
        "OpenRouter Model Name",
        value="openai/arcee-ai/trinity-large-preview:free",
        help="Example: openai/arcee-ai/trinity-large-preview:free or google/gemini-flash-1.5"
    )

    # Module Type: Predict or ChainOfThought
    module_type = st.selectbox(
        "Reasoning Mode",
        options=["Predict", "ChainOfThought"],
        index=1,  # Default to ChainOfThought (better quality)
        help="ChainOfThought usually gives richer and better prompts"
    )

# ====================== DSPy SETUP ======================
@st.cache_resource(show_spinner="Loading DSPy generator...")
def load_generator(module_type: str, model_name: str):
    if not os.getenv("OPENROUTER_API_KEY"):
        return None
    
    try:
        lm = dspy.LM(
            model_name,
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=2048,
            temperature=0.75
        )
        dspy.configure(lm=lm)

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

        # Choose module based on dropdown
        if module_type == "ChainOfThought":
            return dspy.ChainOfThought(SceneToImagePrompt)
        else:
            return dspy.Predict(SceneToImagePrompt)

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load the generator
generator = load_generator(module_type, model_name)

# ====================== MAIN UI ======================
st.subheader("Describe your scene")
user_input = st.text_area(
    "Your directions:",
    placeholder="beautiful woman in red lace lingerie, sitting on bed, legs slightly apart, soft warm lighting, low camera angle...",
    height=180
)

col1, col2 = st.columns([3, 1])
with col1:
    generate_button = st.button("✨ Generate Detailed Prompt", type="primary", use_container_width=True)

if generate_button:
    if not api_key:
        st.error("⚠️ Please enter your OpenRouter API key in the sidebar.")
    elif not model_name:
        st.error("⚠️ Please enter a model name.")
    elif not user_input.strip():
        st.error("⚠️ Please enter a scene description!")
    elif generator is None:
        st.error("❌ Could not load the generator. Check API key and model name.")
    else:
        with st.spinner(f"Generating with {module_type} using {model_name}..."):
            try:
                result = generator(user_directions=user_input.strip())
                
                st.success("✅ Prompt generated successfully!")
                
                st.subheader("📋 Optimized Image Prompt")
                st.code(result.detailed_prompt, language=None)
                
                st.download_button(
                    label="⬇️ Download Prompt as .txt",
                    data=result.detailed_prompt,
                    file_name="detailed_image_prompt.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Generation error: {str(e)}")

st.caption("Powered by DSPy + OpenRouter • Change model or mode anytime in sidebar")