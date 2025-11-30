"""
The web interface for the AI Image Generator. This is what users actually see
and interact with - pretty straightforward Streamlit app that makes everything
easy to use!
"""

import streamlit as st
import os
import time
from PIL import Image
from image_generator import ImageGenerator, ImageStorage
import torch
from io import BytesIO


# Set up the page to look nice
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "generator" not in st.session_state:
    st.session_state.generator = None
if "storage" not in st.session_state:
    st.session_state.storage = ImageStorage()
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []


@st.cache_resource
def load_generator():
    """
    Loads the image generator. The @st.cache_resource decorator means it only
    loads once and then reuses it - saves a ton of time since loading the model
    takes a while!
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ImageGenerator(device=device)


def main():
    """
    The main function that runs everything. Sets up the UI, handles user input,
    and coordinates the image generation. Pretty much the brain of the app!
    """
    
    # Big header at the top
    st.markdown('<p class="main-header">🎨 AI-Powered Image Generator</p>', 
                unsafe_allow_html=True)
    
    # All the settings go in the sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Show what hardware we're running on
        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.info(f"Running on: **{device}**")
        
        if not torch.cuda.is_available():
            st.warning("⚠️ GPU not detected. Generation will be slower on CPU.")
        
        st.markdown("---")
        
        # Generation parameters
        st.subheader("Generation Parameters")
        
        num_images = st.slider(
            "Number of Images",
            min_value=1,
            max_value=4,
            value=1,
            help="Number of images to generate per prompt"
        )
        
        style = st.selectbox(
            "Style Guidance",
            ["photorealistic", "artistic", "cartoon", "oil painting", 
             "watercolor", "digital art"],
            help="Artistic style for generated images"
        )
        
        quality_boost = st.checkbox(
            "Quality Enhancement",
            value=True,
            help="Add quality descriptors to improve image quality"
        )
        
        num_inference_steps = st.slider(
            "Inference Steps",
            min_value=20,
            max_value=100,
            value=50,
            help="More steps = better quality but takes longer. 50 is usually a good balance!"
        )
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            help="How much should it listen to your prompt? 7.5 is the sweet spot, but experiment!"
        )
        
        width = st.selectbox(
            "Image Width",
            [512, 768, 1024],
            index=0,
            help="Width of generated images"
        )
        
        height = st.selectbox(
            "Image Height",
            [512, 768, 1024],
            index=0,
            help="Height of generated images"
        )
        
        use_seed = st.checkbox("Use Random Seed", value=False)
        seed = None
        if use_seed:
            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=2147483647,
                value=42,
                help="Random seed for reproducibility"
            )
        
        st.markdown("---")
        
        # Negative prompt
        st.subheader("Negative Prompt")
        negative_prompt = st.text_area(
            "Things to avoid",
            value="blurry, low quality, distorted, deformed, ugly, bad anatomy",
            help="Describe what you don't want in the image",
            height=100
        )
        
        st.markdown("---")
        
        # Ethical AI notice
        st.info("""
        **Ethical AI Use:**
        - Generated images are watermarked
        - Inappropriate content is filtered
        - Images are for creative and educational purposes
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Prompt")
        
        prompt = st.text_area(
            "Describe the image you want to generate",
            placeholder="e.g., a futuristic city at sunset with flying cars and neon lights",
            height=100
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            generate_button = st.button(
                "🎨 Generate Images",
                type="primary",
                use_container_width=True
            )
        
        with col_btn2:
            clear_button = st.button(
                "🗑️ Clear",
                use_container_width=True
            )
        
        if clear_button:
            st.session_state.generated_images = []
            st.rerun()
    
    with col2:
        st.header("💡 Prompt Tips")
        st.markdown("""
        **Best Practices:**
        - Be specific and descriptive
        - Include style, mood, and details
        - Mention lighting and composition
        - Use quality terms (optional)
        
        **Examples:**
        - "Portrait of a robot in Van Gogh style, vibrant colors, detailed brushstrokes"
        - "Futuristic cityscape at sunset, neon lights, cyberpunk aesthetic, highly detailed"
        - "Cute cartoon cat wearing sunglasses, beach background, sunny day"
        """)
    
    # Generate images
    if generate_button:
        if not prompt:
            st.error("⚠️ Please enter a prompt!")
        else:
            # Load the model if we haven't already (only happens once)
            if st.session_state.generator is None:
                with st.spinner("Loading model (first time only - this might take a minute)..."):
                    st.session_state.generator = load_generator()
            
            generator = st.session_state.generator
            
            # Check if the prompt is appropriate
            is_safe, reason = generator.filter_inappropriate_content(prompt)
            
            if not is_safe:
                st.error(f"⚠️ {reason}")
                st.info("Please modify your prompt to remove inappropriate content.")
            else:
                # Make the prompt better with quality terms and style hints
                enhanced_prompt = generator.enhance_prompt(
                    prompt, style=style, quality_boost=quality_boost
                )
                
                # Show what the enhanced prompt looks like (in case you're curious!)
                with st.expander("🔍 Enhanced Prompt (click to see what we're actually sending to the model)"):
                    st.text(enhanced_prompt)
                
                # Generate images
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                try:
                    status_text.text("Generating images... This might take a bit, especially on CPU!")
                    
                    # This is where the magic happens - actually generate the images
                    images = generator.generate(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        num_images=num_images,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        seed=seed
                    )
                    
                    # Add watermarks to all images (good practice!)
                    watermarked_images = []
                    for img in images:
                        watermarked = generator.add_watermark(img)
                        watermarked_images.append(watermarked)
                    
                    elapsed_time = time.time() - start_time
                    
                    progress_bar.progress(100)
                    status_text.text(f"✅ Generation complete! ({elapsed_time:.1f}s)")
                    
                    # Display images
                    st.session_state.generated_images = watermarked_images
                    
                    # Save everything to disk so you don't lose your creations!
                    storage = st.session_state.storage
                    saved_paths = []
                    
                    for i, img in enumerate(watermarked_images):
                        filename = f"image_{int(time.time())}_{i}"
                        filepath, _ = storage.save_image(
                            img,
                            prompt=enhanced_prompt,
                            filename=filename,
                            format="PNG",
                            metadata={
                                "original_prompt": prompt,  # Save your original prompt too
                                "style": style,
                                "num_inference_steps": num_inference_steps,
                                "guidance_scale": guidance_scale,
                                "width": width,
                                "height": height,
                                "seed": seed,
                                "generation_time": elapsed_time  # How long it took
                            }
                        )
                        saved_paths.append(filepath)
                    
                    st.success(f"Images saved successfully! Check the gallery below to see them.")
                    
                except Exception as e:
                    st.error(f"❌ Error generating images: {str(e)}")
                    st.exception(e)
                
                finally:
                    progress_bar.empty()
    
    # Display generated images
    if st.session_state.generated_images:
        st.markdown("---")
        st.header("🖼️ Generated Images")
        
        num_cols = min(len(st.session_state.generated_images), 2)
        cols = st.columns(num_cols)
        
        for idx, img in enumerate(st.session_state.generated_images):
            with cols[idx % num_cols]:
                st.image(img, use_container_width=True)
                
                # Download button
                buf = BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                st.download_button(
                    label=f"📥 Download Image {idx + 1}",
                    data=img_bytes,
                    file_name=f"generated_image_{idx + 1}.png",
                    mime="image/png",
                    key=f"download_{idx}"
                )
        
        # Gallery view
        st.markdown("---")
        st.header("📚 Image Gallery")
        
        gallery_images = st.session_state.storage.list_images()
        
        if gallery_images:
            st.info(f"Total images in gallery: {len(gallery_images)}")
            
            # Show recent images
            num_recent = st.slider("Show recent images", 1, min(10, len(gallery_images)), 5)
            
            recent_images = gallery_images[:num_recent]
            
            for img_data in recent_images:
                with st.expander(f"📷 {img_data['filename']} - {img_data['timestamp'][:19]}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if os.path.exists(img_data['filepath']):
                            st.image(img_data['filepath'], use_container_width=True)
                    
                    with col2:
                        st.json(img_data)
                        
                        if os.path.exists(img_data['filepath']):
                            with open(img_data['filepath'], "rb") as f:
                                st.download_button(
                                    label="📥 Download",
                                    data=f.read(),
                                    file_name=img_data['filename'],
                                    mime="image/png",
                                    key=f"gallery_download_{img_data['filename']}"
                                )
        else:
            st.info("No images in gallery yet. Generate some images to see them here! 🎨")


if __name__ == "__main__":
    main()

