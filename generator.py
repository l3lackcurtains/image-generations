import torch
from diffusers import FluxPipeline
import os
import uuid
from pathlib import Path

# Load the models once when the script starts
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_schnell = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", 
    torch_dtype=torch.bfloat16,
    cache_dir="./local_models"
)
pipe_dev = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16,
    cache_dir="./local_models"
)
if device == "cuda":
    pipe_schnell.enable_model_cpu_offload()
    pipe_dev.enable_model_cpu_offload()

def generate_image(prompt, height=1024, width=1024, model="flux"):
    """
    Wrapper function for image generation that supports multiple models
    
    Args:
        prompt (str): Text prompt for image generation
        height (int): Height of the generated image
        width (int): Width of the generated image
        model (str): Model to use for generation ('flux' or 'flux-dev')
    
    Returns:
        PIL.Image: Generated image
    """
    if model == "flux":
        return _generate_flux_image(prompt, height, width, pipe_schnell)
    elif model == "flux-dev":
        return _generate_flux_image(prompt, height, width, pipe_dev)
    else:
        raise ValueError(f"Unsupported model: {model}")

def _generate_flux_image(prompt, height, width, pipe):
    """Internal function for FLUX model generation"""
    generator = torch.Generator(device).manual_seed(0)
    
    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=generator
    ).images[0]
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate random filename and save image
    random_filename = f"{uuid.uuid4().hex}.png"
    image.save(results_dir / random_filename)
    
    return image

# Example usage
if __name__ == "__main__":
    prompt = "A cat holding a sign that says hello world"
    generate_image(prompt, height=1024, width=1024, model="flux")
    # Example of using the dev model
    # generate_image(prompt, height=1024, width=1024, model="flux-dev")
   