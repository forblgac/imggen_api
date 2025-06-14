# Configuration settings

# Model to use for image generation
SDXL_MODEL_NAME = "stabilityai/sdxl-turbo"
# SDXL_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0" # Example for full SDXL

# Torch settings
TORCH_DTYPE = "float16" # or "float32" if float16 is not supported or causes issues
VARIANT = "fp16" # or None if not using a specific variant like fp16

# API Server settings
API_HOST = "127.0.0.1"
API_PORT = 8000
