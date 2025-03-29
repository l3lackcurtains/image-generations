import torch
from diffusers import FluxPipeline

# Enable memory efficient attention
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Load with half precision to save memory
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.float16,  # Use float16 instead of bfloat16
    cache_dir="./local_models",
    device_map="balanced"  # Changed from "auto" to "balanced"
)

# Clear CUDA cache before running
torch.cuda.empty_cache()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=512,  # Reduced from 512
    width=512,   # Reduced from 512
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-dev.png")
