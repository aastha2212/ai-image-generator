# AI-Powered Image Generator

Turn your words into images! This is a text-to-image generator built with Stable Diffusion that lets you create pretty much anything you can imagine. Just type what you want to see, tweak some settings, and watch it come to life.

I built this as a project to learn about generative AI and to create something actually useful. It's got a nice web interface, some smart prompt engineering tricks, and it tries to be responsible about how it's used.

## 🎯 What This Does

So basically, this project:
- Takes whatever you type and turns it into an image (pretty cool, right?)
- Uses Stable Diffusion (the open-source model everyone's talking about)
- Lets you adjust all the knobs and dials to get exactly what you want
- Automatically makes your prompts better with some quality magic
- Saves everything with metadata so you can remember how you made it
- Has some basic safety checks and watermarks everything

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │  ← User Interface
└────────┬────────┘
         │
┌────────▼────────┐
│ ImageGenerator  │  ← Core Generation Logic
└────────┬────────┘
         │
┌────────▼────────┐
│ Stable Diffusion│  ← AI Model (HuggingFace)
└────────┬────────┘
         │
┌────────▼────────┐
│  ImageStorage   │  ← Save & Metadata Management
└─────────────────┘
```

### Key Components

1. **ImageGenerator** (`image_generator.py`): Core generation engine
   - Model loading and management
   - Prompt enhancement
   - Image generation
   - Watermarking
   - Content filtering

2. **Streamlit Interface** (`app.py`): Web-based UI
   - Prompt input
   - Parameter adjustment
   - Real-time generation
   - Image gallery

3. **ImageStorage** (`image_generator.py`): Storage system
   - Image saving (PNG/JPEG)
   - Metadata management
   - Gallery functionality

## 🚀 Getting Started

### What You'll Need

- Python 3.8 or newer (I used 3.10, but 3.8+ should work fine)
- A GPU is nice to have (makes things way faster), but CPU works too
- At least 8GB of RAM (16GB is better if you can swing it)
- About 10GB of free space (the model is about 4GB, plus room for generated images)

### Installation (It's Pretty Simple)

1. **Get the code**
   ```bash
   git clone <repository-url>
   cd assignment
   ```

2. **Set up a virtual environment** (trust me, you want this)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install all the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **The model downloads automatically**
   - First time you run it, it'll download the Stable Diffusion model
   - It's about 4GB, so make sure you have a decent internet connection
   - After that, it's cached locally so you don't have to download again

### GPU Setup (Makes Everything Way Faster)

If you've got an NVIDIA GPU, here's how to set it up:

1. **Install CUDA Toolkit** (if you don't have it already)
   - Head over to: https://developer.nvidia.com/cuda-downloads
   - Get CUDA 11.8 or newer
   - Follow their installation instructions

2. **Check if PyTorch sees your GPU**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Shows your GPU name
   ```

3. **If PyTorch doesn't see your GPU**, reinstall with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### CPU Setup (It Works, Just Slower)

Don't have a GPU? No worries, it'll still work! Just be patient:
- On CPU, expect 30-60 seconds per image (vs 5-10 seconds on GPU)
- Make sure you have at least 8GB of RAM
- Stick with 512x512 images to keep things reasonable

## 💻 How to Use It

### Starting It Up

Just run:
```bash
streamlit run app.py
```

It'll pop open in your browser automatically at `http://localhost:8501`. Pretty straightforward!

### The Basics

1. **Type what you want** in the big text box
   - Try something like "a futuristic city at sunset"
   - The more details you give, the better it usually turns out

2. **Play with the settings** (they're in the sidebar):
   - How many images do you want? (1-4)
   - What style? (photorealistic, artistic, cartoon, etc.)
   - Quality boost on or off? (usually leave it on)
   - Inference steps: more = better quality but slower (50 is good)
   - Guidance scale: how much should it listen to you? (7.5 is the sweet spot)
   - Image size: bigger = better but takes longer

3. **Hit "Generate Images"** and wait
   - It'll show you progress as it works
   - On GPU this is pretty quick, on CPU grab a coffee

4. **Check out your images!**
   - They're automatically saved
   - Download buttons right there
   - Check the gallery to see everything you've made

### Example Prompts

**Photorealistic:**
```
A serene mountain landscape at sunrise, misty valleys, golden hour lighting, 
highly detailed, professional photography, 4K resolution
```

**Artistic:**
```
Portrait of a robot in Van Gogh style, vibrant colors, expressive brushstrokes, 
impressionist painting, museum quality
```

**Fantasy:**
```
Magical forest with glowing mushrooms, fairy lights, ethereal atmosphere, 
cinematic lighting, highly detailed, fantasy art
```

**Sci-Fi:**
```
Futuristic cyberpunk cityscape at night, neon signs, flying vehicles, 
rain-soaked streets, Blade Runner aesthetic, highly detailed
```

### Tips for Better Prompts

Here's what I've learned works well:

1. **Be specific!** Don't just say "a cat" - say "a fluffy orange cat sitting on a windowsill, golden hour lighting, cozy atmosphere"
2. **Quality words help**: Throw in "highly detailed", "4K", "professional photography" - the model loves these
3. **Style matters**: Want it to look like a painting? Say so! "Van Gogh style", "watercolor", "digital art"
4. **Think about composition**: "wide angle", "close-up", "bird's eye view" - it actually listens to this stuff
5. **Lighting is huge**: "golden hour", "studio lighting", "dramatic shadows" - makes a big difference
6. **Negative prompts are your friend**: Tell it what you DON'T want - "blurry, low quality, distorted"

### Advanced Features

**Negative Prompts:**
- Describe what you don't want in the image
- Example: "blurry, low quality, distorted, text, watermark"

**Seed Control:**
- Enable "Use Random Seed" for reproducible results
- Same seed + same prompt = same image

**Multiple Images:**
- Generate 1-4 images per prompt
- Useful for exploring variations

## 🛠️ Technology Stack

- **Framework**: PyTorch 2.0+
- **Model**: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
- **Libraries**:
  - `diffusers`: HuggingFace diffusion models
  - `transformers`: Model utilities
  - `streamlit`: Web interface
  - `PIL/Pillow`: Image processing
  - `accelerate`: Model optimization

## 📊 Hardware Requirements

### Minimum (CPU)
- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB free
- Generation time: ~30-60s per image

### Recommended (GPU)
- GPU: NVIDIA with 6GB+ VRAM (RTX 3060 or better)
- RAM: 16GB
- Storage: 20GB free (for model cache)
- Generation time: ~5-10s per image

### Optimal
- GPU: NVIDIA with 12GB+ VRAM (RTX 3080/4080 or better)
- RAM: 32GB
- Storage: 50GB+ free
- Generation time: ~3-5s per image

## 🔒 Ethical AI Use

### Content Filtering

The application includes basic content filtering:
- Filters inappropriate keywords
- Blocks explicit or violent content
- Prompts users to modify inappropriate requests

### Watermarking

All generated images are automatically watermarked:
- "AI Generated" text in bottom-right corner
- Semi-transparent overlay
- Indicates AI origin

### Responsible Use Guidelines

1. **Respect Copyright**: Generated images may be inspired by training data
2. **No Misrepresentation**: Don't claim AI-generated images as original artwork
3. **Appropriate Content**: Use for creative and educational purposes
4. **Privacy**: Don't generate images of real people without consent
5. **Transparency**: Disclose AI generation when sharing images

## 📁 Project Structure

```
assignment/
├── app.py                 # Streamlit web interface
├── image_generator.py     # Core generation logic
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── generated_images/      # Generated images (created at runtime)
    ├── *.png             # Generated images
    └── metadata/          # JSON metadata files
```

## 🎨 Model Details

**Stable Diffusion v1.5**
- Architecture: Latent Diffusion Model
- Parameters: ~860M
- Training: LAION-5B dataset
- License: CreativeML Open RAIL-M
- Resolution: 512x512 (supports up to 1024x1024)

### How It Works

1. **Text Encoding**: Prompt is encoded using CLIP text encoder
2. **Latent Space**: Generation happens in compressed latent space
3. **Diffusion Process**: Iterative denoising (50 steps by default)
4. **Decoder**: Latent representation decoded to full-resolution image
5. **Post-processing**: Watermarking and quality enhancement

## ⚠️ Things to Know (The Reality Check)

1. **It's not instant**:
   - On CPU: 30-60 seconds per image (yeah, it's slow)
   - On GPU: 5-10 seconds (much better!)
   - Bigger images = more time

2. **Memory hungry**:
   - GPU needs at least 6GB VRAM (8GB+ is better)
   - CPU needs 8GB+ RAM
   - The model itself is about 4GB

3. **It's not perfect**:
   - Complex scenes can be hit or miss
   - Text in images? Don't count on it being readable
   - Sometimes the details are a bit wonky

4. **Content filtering is basic**:
   - I added some keyword filtering, but it's not foolproof
   - You might get unexpected results sometimes
   - Good prompts = good results (most of the time)

5. **Hardware matters**:
   - CPU works but it's slow
   - You'll need disk space for the model
   - First run needs internet to download everything

## 🔮 Future Improvements

1. **Model Enhancements**:
   - Fine-tuning on custom datasets
   - Support for multiple models (SDXL, etc.)
   - Inpainting and outpainting features
   - Style transfer capabilities

2. **Features**:
   - Image-to-image translation
   - Prompt suggestions and autocomplete
   - Batch processing
   - API endpoint for programmatic access
   - User accounts and saved prompts

3. **Performance**:
   - Model quantization for faster inference
   - Better memory management
   - Caching for repeated prompts
   - Distributed generation

4. **UI/UX**:
   - Real-time preview
   - Prompt templates
   - Image editing tools
   - Comparison view

5. **Ethical AI**:
   - Advanced content filtering
   - Customizable watermarking
   - Usage analytics
   - Attribution tracking

## 🐛 When Things Go Wrong

### Model Won't Download
- **What's happening**: Download is super slow or fails
- **Try this**: Use the HuggingFace CLI to download manually:
  ```bash
  huggingface-cli login
  # Then download the model manually
  ```

### Running Out of Memory
- **What's happening**: "CUDA out of memory" error
- **Try this**: 
  - Use smaller images (512x512 instead of 1024x1024)
  - Generate one image at a time
  - The CPU offloading is already enabled, but you might need to close other apps

### It's So Slow!
- **What's happening**: Takes forever to generate
- **Try this**: 
  - Check if you're actually using GPU (the sidebar will tell you)
  - Reduce inference steps to 20-30 (faster but slightly lower quality)
  - Stick with 512x512 images

### Can't Import Stuff
- **What's happening**: Python can't find the packages
- **Try this**: 
  ```bash
   pip install --upgrade -r requirements.txt
   ```
  Make sure your virtual environment is activated!

## 📝 License Stuff

This project uses Stable Diffusion v1.5, which has its own license (CreativeML Open RAIL-M). Basically, you can use it for most things, but check the full license if you're doing something commercial.

## 👤 Getting in Touch

Questions? Issues? Want to chat about the project?
- Email: aastha2212chaurasia@gmail.com
- When you reach out, include your contact info and when you'd be available to start

## 🙏 Shoutouts

Huge thanks to:
- HuggingFace for making the diffusers library (makes this so much easier!)
- Stability AI for creating Stable Diffusion in the first place
- RunwayML for the model checkpoint
- Streamlit for making web apps actually doable


