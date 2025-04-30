#!/usr/bin/env python3
"""
Local Image Generator using SDXL with direct BitsAndBytes 8-bit integration
Optimized for RTX 4060 with 8GB VRAM
"""

import os
import time
from pathlib import Path

class BitsAndBytesImageGenerator:
    """
    Class for generating images using Stable Diffusion XL
    with direct BitsAndBytes integration for reduced memory usage
    """
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0", device="cuda", 
                 output_dir="generated_images"):
        """
        Initialize the quantized image generator
        
        Parameters:
        - model_id: The Stable Diffusion model to use (default: SDXL)
        - device: "cuda" for GPU, "cpu" for CPU
        - output_dir: Directory to save generated images
        """
        self.model_id = model_id
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load models on first use to avoid unnecessary memory usage
        self.pipe = None
        
        print(f"BitsAndBytes image generator initialized with model: {model_id}")
        print(f"Images will be saved to: {self.output_dir}")
        print(f"Using device: {device}")
        
    def _load_model(self):
        """Load the SDXL model with BitsAndBytes optimizations"""
        if self.pipe is None:
            try:
                # Import libraries
                import torch
                from diffusers import DiffusionPipeline
                
                print(f"Loading model {self.model_id} with BitsAndBytes optimizations...")
                
                # First, load the model in half precision
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
                
                # Do not move to device yet - we'll quantize first
                
                # Import bitsandbytes for the conversion
                import bitsandbytes as bnb
                
                print("Converting Linear modules to 8-bit precision...")
                
                # Function to recursively convert linear layers to 8-bit
                def convert_to_8bit(module):
                    for name, child in list(module.named_children()):
                        if isinstance(child, torch.nn.Linear):
                            # Skip text embedding and final projection layers for better quality
                            if any(x in name for x in ["text_model", "lm_head", "proj_out"]):
                                continue
                            
                            # Create 8-bit linear layer
                            in_features = child.in_features
                            out_features = child.out_features
                            bias = child.bias is not None
                            
                            # Create 8-bit replacement
                            eightbit_linear = bnb.nn.Linear8bitLt(
                                in_features, 
                                out_features,
                                bias=bias,
                                has_fp16_weights=False,  # No need to keep fp16 weights
                                threshold=6.0  # Default threshold
                            )
                            
                            # Copy weights and biases
                            with torch.no_grad():
                                eightbit_linear.weight.copy_(child.weight)
                                if bias:
                                    eightbit_linear.bias.copy_(child.bias)
                            
                            # Replace the layer
                            setattr(module, name, eightbit_linear)
                        else:
                            # Recursively convert child modules
                            convert_to_8bit(child)
                
                # Convert UNet to 8-bit
                print("Converting UNet to 8-bit...")
                convert_to_8bit(self.pipe.unet)
                
                # For SDXL, also convert the text_encoder if it's big
                if hasattr(self.pipe, "text_encoder_2"):
                    print("Converting text_encoder_2 to 8-bit...")
                    convert_to_8bit(self.pipe.text_encoder_2)
                
                # Apply memory optimizations
                self.pipe.enable_vae_slicing()            # Slice VAE operations
                self.pipe.enable_attention_slicing("max")  # Maximum attention slicing
                
                # Try to enable xformers if available for more memory efficiency
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("Using xformers for memory-efficient attention")
                except Exception as e:
                    print(f"Xformers not available: {e}")
                    print("Using standard attention")
                
                # Now move to device
                self.pipe = self.pipe.to(self.device)
                
                # Set default dimensions for SDXL
                self.default_width = 768
                self.default_height = 768
                
                print("Model loaded successfully with 8-bit optimizations")
                
            except ImportError as e:
                print(f"Error: Required libraries not installed. {e}")
                print("Please install with: pip install diffusers transformers torch bitsandbytes")
                print("For additional memory efficiency: pip install xformers")
                raise
            except Exception as e:
                print(f"Error loading model: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    def generate_image(self, prompt, output_path=None, negative_prompt=None, 
                       width=None, height=None, num_inference_steps=25, 
                       guidance_scale=7.5, seed=None):
        """
        Generate an image using optimized SDXL
        
        Parameters:
        - prompt: Text prompt for image generation
        - output_path: Path to save the image (if None, a path will be generated)
        - negative_prompt: Things to avoid in the image
        - width: Image width (if None, uses 768)
        - height: Image height (if None, uses 768)
        - num_inference_steps: Number of denoising steps
        - guidance_scale: How closely to follow the prompt (higher = more faithful)
        - seed: Random seed for reproducibility
        
        Returns:
        - Path to the generated image
        """
        try:
            # Load model if not already loaded
            self._load_model()
            
            # Set dimensions to default if not specified
            if width is None:
                width = self.default_width
            if height is None:
                height = self.default_height
            
            # Set a random seed if not provided
            if seed is None:
                import random
                seed = random.randint(0, 2147483647)
                
            # Create a generator for reproducibility
            import torch
            generator = torch.manual_seed(seed)
            
            # Optimal negative prompt for educational images if none provided
            if negative_prompt is None:
                negative_prompt = (
                    "low quality, blurry, distorted, deformed, disfigured, "
                    "bad anatomy, text, watermark, signature, border, frame"
                )
            
            # Generate the image
            print(f"Generating image for prompt: {prompt}")
            print(f"Using dimensions: {width}x{height}, steps: {num_inference_steps}")
            start_time = time.time()
            
            # Clear CUDA cache before generation if using GPU
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()
            
            # Generate image with lower memory settings
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            elapsed = time.time() - start_time
            print(f"Image generation took {elapsed:.2f} seconds")
            
            # Handle output path
            if output_path is None:
                # Generate a filename based on the prompt
                sanitized_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " _-").strip()
                sanitized_prompt = sanitized_prompt.replace(" ", "_")
                output_path = self.output_dir / f"{sanitized_prompt}_{seed}.png"
            else:
                # Ensure output_path is a Path object
                output_path = Path(output_path)
                
                # If output_path is just a filename with no directory, put it in self.output_dir
                if not output_path.parent or output_path.parent == Path('.'):
                    output_path = self.output_dir / output_path
            
            # Ensure the output directory exists
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Save the image
            image.save(output_path)
            print(f"Image saved to {output_path}")
            
            # Clear CUDA cache after generation if using GPU
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error generating image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_image_prompt(self, spanish_word, english_translation):
        """
        Generate a prompt for Stable Diffusion based on the Spanish word
        
        Parameters:
        - spanish_word: The Spanish word
        - english_translation: English translation
        
        Returns:
        - A prompt string optimized for Stable Diffusion
        """
        word_descriptor = english_translation.strip()
        
        # Simple prompt template for SDXL
        template = (
            f"A clear, simple illustration of '{word_descriptor}' for language learning. "
            f"Educational style, minimalist, centered composition, vibrant colors."
        )
        
        return template

    def close(self):
        """Free up GPU memory by deleting the model"""
        if self.pipe is not None:
            import gc
            
            # Move model to CPU first (helps with CUDA memory management)
            self.pipe = self.pipe.to("cpu")
            
            # Delete the pipeline and clear CUDA cache
            del self.pipe
            self.pipe = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available and using GPU
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    print("CUDA memory cleared")
                except ImportError:
                    pass
            
            print("Model unloaded")


# Example usage
if __name__ == "__main__":
    # Example usage of the class
    generator = BitsAndBytesImageGenerator()
    
    # Example with a Spanish word
    spanish_word = "perro"
    english_translation = "dog"
    
    # Generate a prompt
    prompt = generator.generate_image_prompt(spanish_word, english_translation)
    print(f"Generated prompt: {prompt}")
    
    # Specify full output path to ensure it's saved properly
    current_dir = Path(__file__).parent.absolute()
    output_path = current_dir / "test_perro.png"
    
    # Generate the image
    generator.generate_image(
        prompt=prompt, 
        output_path=output_path,
        # Use moderate settings for SDXL on 8GB VRAM
        width=768,
        height=768,
        num_inference_steps=25
    )
    
    print(f"Image generated at {output_path}")
    
    # Free up GPU memory
    generator.close()
