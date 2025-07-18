# Configuration settings

# --- Model Loading Configuration ---
# Set to True to load models from a local directory, False to load from Hugging Face Hub.
LOAD_MODELS_FROM_LOCAL = True

# Directory where local checkpoint models are stored.
CHECKPOINT_MODELS_DIR = "./models/checkpoints"
# Directory where local LoRA models are stored.
LORA_MODELS_DIR = "./models/loras"

# Default model identifier.
# - This should be the filename (without .safetensors extension)
#   of the default checkpoint model in CHECKPOINT_MODELS_DIR.
# - If LOAD_MODELS_FROM_LOCAL is False: This is the Hugging Face Hub model identifier
#   (e.g., "stabilityai/sdxl-turbo").
DEFAULT_MODEL_IDENTIFIER = "waiNSFWIllustrious_v140"
# Example for a local default model: DEFAULT_MODEL_IDENTIFIER = "waiNSFWIllustrious_v140"
# Example for a full SDXL model from Hub: DEFAULT_MODEL_IDENTIFIER = "stabilityai/stable-diffusion-xl-base-1.0"

# Optional: Specific configurations for local models (e.g., prediction_type for v-prediction models).
# Keys are model filenames without the .safetensors extension.
# Example: {"my_v_pred_model": {"prediction_type": "v_prediction"}}
LOCAL_MODEL_CONFIGS = {
    "waiNSFWIllustrious_v140": {"prediction_type": "epsilon"}  # Example for your model
}

# Torch settings
TORCH_DTYPE = "float16"  # or "float32" if float16 is not supported or causes issues
VARIANT = "fp16"  # or None if not using a specific variant like fp16

# API Server settings
API_HOST = "127.0.0.1"
API_PORT = 8000
