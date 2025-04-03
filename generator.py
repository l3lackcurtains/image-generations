import torch
from diffusers import FluxPipeline
import gc
from pathlib import Path

class ImageGenerator:
    def __init__(self, local_model_directory="./local_models"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This application requires a CUDA-enabled GPU.")
            
        self.local_model_directory = local_model_directory
        self.device = "cuda"
        self.dtype = torch.bfloat16
        
        print(f"Initializing ImageGenerator with device: {self.device}")
        self._setup_device()
        self._load_models()
        self._configure_models()

    def _setup_device(self):
        """Configure device-specific optimizations"""
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        gc.collect()

    def _load_models(self):
        """Load and configure the models"""
        self.pipe_schnell = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path=f"{self.local_model_directory}/black-forest-labs/FLUX.1-schnell",
            torch_dtype=self.dtype,
        )
        
        self.pipe_dev = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path=f"{self.local_model_directory}/black-forest-labs/FLUX.1-dev",
            torch_dtype=self.dtype,
        )

        if self.device == "cuda":
            self.pipe_schnell.enable_model_cpu_offload()
            self.pipe_dev.enable_model_cpu_offload()

    def _configure_models(self):
        """Configure models for optimized performance without compilation"""
        import torch
        
        if torch.__version__ >= "2.0.0":
            print("Using PyTorch 2.0+ optimizations without compilation")
            # Enable additional optimizations available in PyTorch 2.0+
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)
        else:
            print("Using default PyTorch optimizations")

    def generate(self, prompt, height=1024, width=1024, model="flux-schnell"):
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
        random_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cuda" if self.device == "cuda" else None).manual_seed(random_seed)
        
        print(f"Using seed: {random_seed}")
        
        try:
            if model == "flux-schnell":
                image = self.pipe_schnell(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=512,
                    generator=generator
                ).images[0]
                
            elif model == "flux-dev":
                image = self.pipe_dev(
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
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
        return image

    def get_system_info(self):
        """Return system information including GPU status"""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available()
        }
        
        if self.device == "cuda":
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda
            })
            
        return info

# Make the class available for import
__all__ = ['ImageGenerator']
