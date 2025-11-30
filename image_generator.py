import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import time
from typing import List, Optional, Tuple
import json
from datetime import datetime


class ImageGenerator:
    
    def __init__(self, device: Optional[str] = None, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Set up the image generator. If you don't specify a device, it'll 
        automatically figure out if you have a GPU or need to use CPU.
        
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.pipe = None
        self._load_model()
        
    def _load_model(self):
        print(f"Loading model on {self.device}...")
        
        # GPUs are faster with float16, CPUs need float32 for stability
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            # Load the actual model 
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,  # We handle filtering ourselves, so skip the built-in one
                requires_safety_checker=False
            )
            
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Try to use xformers for memory efficiency (if you have it installed)
            if self.device == "cuda":
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass
            
            # Move everything to the right device (GPU or CPU)
            self.pipe = self.pipe.to(self.device)
            
            # Some memory tricks to keep things running smoothly
            if self.device == "cuda":
                # Offload parts to CPU when not in use - saves VRAM!
                self.pipe.enable_model_cpu_offload()
            else:
                # On CPU, slice the attention to avoid running out of RAM
                self.pipe.enable_attention_slicing()
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def enhance_prompt(self, prompt: str, style: str = "photorealistic", 
                      quality_boost: bool = True) -> str:
        """
        Adds quality terms and style hints that help the model understand 
        what you really want. It's like giving 
        the AI better instructions.
        
        """
        enhanced = prompt
        
        if quality_boost:
            quality_terms = [
                "highly detailed",
                "professional photography",
                "4K resolution",
                "sharp focus",
                "masterpiece",
                "best quality"
            ]
            enhanced = ", ".join(quality_terms) + ", " + enhanced
        
        style_terms = {
            "photorealistic": "photorealistic, realistic, detailed",
            "artistic": "artistic, creative, stylized",
            "cartoon": "cartoon style, animated, colorful",
            "oil painting": "oil painting, classical art style",
            "watercolor": "watercolor painting, soft colors",
            "digital art": "digital art, concept art, detailed"
        }
        
        if style in style_terms:
            enhanced += ", " + style_terms[style]
        
        return enhanced
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        
        if self.pipe is None:
            raise RuntimeError("Oops! Model isn't loaded yet. Something went wrong.")
        
        # If you didn't specify what to avoid, we'll use some sensible defaults
        if not negative_prompt:
            negative_prompt = (
                "blurry, low quality, distorted, deformed, ugly, "
                "bad anatomy, bad proportions, extra limbs, "
                "duplicate, mutilated, watermark, signature"
            )
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        images = []
        
        # Generate each image one by one
        for i in range(num_images):
            print(f"Generating image {i+1}/{num_images}...")
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
            
            images.append(result.images[0])
        
        return images
    
    def add_watermark(self, image: Image.Image, text: str = "AI Generated") -> Image.Image:
        watermarked = image.copy()
        draw = ImageDraw.Draw(watermarked)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        margin = 10
        position = (watermarked.width - text_width - margin, 
                   watermarked.height - text_height - margin)
        
        overlay = Image.new('RGBA', watermarked.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        padding = 4
        overlay_draw.rectangle(
            [position[0] - padding, position[1] - padding,
             position[0] + text_width + padding, position[1] + text_height + padding],
            fill=(0, 0, 0, 128)
        )
        watermarked = Image.alpha_composite(
            watermarked.convert('RGBA'), overlay
        ).convert('RGB')
        
        draw = ImageDraw.Draw(watermarked)
        draw.text(position, text, fill=(255, 255, 255), font=font)
        
        return watermarked
    
    def filter_inappropriate_content(self, prompt: str) -> Tuple[bool, str]:
        """
        A simple check to make sure we're not generating anything inappropriate.

        """
        # Basic list of things we want to avoid
        inappropriate_keywords = [
            "explicit", "nude", "naked", "nsfw", "sexual",
            "violence", "gore", "blood", "weapon", "gun"
        ]
        
        # Check case-insensitively
        prompt_lower = prompt.lower()
        for keyword in inappropriate_keywords:
            if keyword in prompt_lower:
                return False, f"Prompt contains inappropriate content: {keyword}"
        
        # Looks good!
        return True, ""


class ImageStorage:
    """
    Takes care of generated images and keeping track of all the
    metadata (prompt, settings, etc.).
    """
    
    def __init__(self, base_dir: str = "generated_images"):
        
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "metadata"), exist_ok=True)
    
    def save_image(
        self,
        image: Image.Image,
        prompt: str,
        filename: Optional[str] = None,
        format: str = "PNG",
        metadata: Optional[dict] = None
    ) -> Tuple[str, str]:
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}"
        
        # Figure out the file extension
        ext = "png" if format.upper() == "PNG" else "jpg"
        filepath = os.path.join(self.base_dir, f"{filename}.{ext}")
        
        # Actually save the image
        if format.upper() == "JPEG":
            image.save(filepath, "JPEG", quality=95)  # 95 is a good balance of quality/size
        else:
            image.save(filepath, "PNG")  # PNG is lossless, so always perfect quality
        
        metadata_dict = {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "filename": f"{filename}.{ext}",
            "format": format,
            **(metadata or {})  
        }
        
        metadata_filepath = os.path.join(
            self.base_dir, "metadata", f"{filename}_metadata.json"
        )
        
        # Write metadata as formatted JSON
        with open(metadata_filepath, "w") as f:
            json.dump(metadata_dict, f, indent=2)
        
        return filepath, metadata_filepath
    
    def list_images(self) -> List[dict]:
    
        images = []
        metadata_dir = os.path.join(self.base_dir, "metadata")
        
        # Go through all the metadata files
        if os.path.exists(metadata_dir):
            for filename in os.listdir(metadata_dir):
                if filename.endswith("_metadata.json"):
                    metadata_path = os.path.join(metadata_dir, filename)
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        # Make sure the actual image file still exists
                        image_path = os.path.join(
                            self.base_dir, metadata["filename"]
                        )
                        if os.path.exists(image_path):
                            metadata["filepath"] = image_path
                            images.append(metadata)
        
        # Sort by timestamp, newest first
        return sorted(images, key=lambda x: x["timestamp"], reverse=True)

