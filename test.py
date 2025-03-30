import torch
from diffusers import FluxPipeline
import gc


import torch
import diffusers

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Diffusers version: {diffusers.__version__}")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Enable memory efficient attention if using CUDA
if device == "cuda":
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Clear CUDA cache before loading model
    torch.cuda.empty_cache()
    gc.collect()

# Set dtype based on device
dtype = torch.float16 if device == "cuda" else torch.float32

# Load model with appropriate settings
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=dtype,
    cache_dir="./local_models"
)

# Move to device if not using device_map
pipe = pipe.to(device)

prompt = "A cat holding a sign that says hello world"

# Configure generator based on device
generator = torch.Generator(device).manual_seed(0) if device == "cuda" else torch.Generator().manual_seed(0)

image = pipe(
    prompt,
    height=512,
    width=512,
    generator=generator
).images[0]

# Clear CUDA cache after generation if using CUDA
if device == "cuda":
    torch.cuda.empty_cache()

image.save("flux-dev.png")
print(f"Image saved as flux-dev.png using {device}")
