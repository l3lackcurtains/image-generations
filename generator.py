import torch
from diffusers import FluxPipeline
import os
import uuid
from pathlib import Path
import gc

# Device selection and optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# CUDA optimizations if available
if device == "cuda":
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    gc.collect()

# Set dtypes based on device
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Load models
pipe_schnell = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", 
    torch_dtype=dtype,
    cache_dir="./local_models"
)

pipe_dev = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=dtype,
    cache_dir="./local_models"
)

# Configure device placement
if device == "cuda":
    pipe_schnell.enable_model_cpu_offload()
    pipe_dev.enable_model_cpu_offload()

def generate_image(prompt, height=1024, width=1024, model="flux-schnell"):
    """
    Generate an image using the specified model
    
    Args:
        prompt (str): Text prompt for image generation
        height (int): Height of the generated image
        width (int): Width of the generated image
        model (str): Model to use ('flux-schnell' or 'flux-dev')
    
    Returns:
        PIL.Image: Generated image
    """
    # Generate a random seed for each call to ensure different results
    random_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device="cuda" if device == "cuda" else None).manual_seed(random_seed)
    
    # Print seed for reproducibility if needed later
    print(f"Using seed: {random_seed}")
    
    try:
        if model == "flux-schnell":
            image = pipe_schnell(
                prompt,
                height=height, 
                width=width,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=512,
                generator=generator
            ).images[0]
            
        elif model == "flux-dev":
            image = pipe_dev(
                prompt,
                height=height, 
                width=width,
                guidance_scale=3.5,
                num_inference_steps=20,
                max_sequence_length=512,
                generator=generator
            ).images[0]
            
        else:
            raise ValueError(f"Unsupported model: {model}")
            
    finally:
        # Always clean up CUDA memory after generation
        if device == "cuda":
            torch.cuda.empty_cache()
            
    return image

