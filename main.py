from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.schedulers import EulerDiscreteScheduler # Import EulerDiscreteScheduler
from PIL import Image
import io
import base64
import logging # Import logging
import os
import glob

import config # Import the configuration

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting Image Generation API...")

# Determine torch_dtype from config
if config.TORCH_DTYPE == "float16":
    torch_dtype = torch.float16
elif config.TORCH_DTYPE == "float32":
    torch_dtype = torch.float32
else:
    logger.warning(f"TORCH_DTYPE '{config.TORCH_DTYPE}' in config.py is not recognized. Defaulting to float16.")
    torch_dtype = torch.float16

SUPPORTED_PIPELINES = {}
logger.info(f"Attempting to load models. LOAD_MODELS_FROM_LOCAL: {config.LOAD_MODELS_FROM_LOCAL}")

if config.LOAD_MODELS_FROM_LOCAL:
    logger.info(f"Loading models from local directory: {config.LOCAL_MODELS_DIR}")
    if not os.path.isdir(config.LOCAL_MODELS_DIR):
        logger.warning(f"Local models directory not found: {config.LOCAL_MODELS_DIR}. No local models will be loaded.")
    else:
        local_model_paths = glob.glob(os.path.join(config.LOCAL_MODELS_DIR, "*.safetensors"))
        if not local_model_paths:
            logger.warning(f"No .safetensors files found in {config.LOCAL_MODELS_DIR}")
        
        for model_path in local_model_paths:
            model_filename = os.path.basename(model_path)
            model_key = model_filename[:-len(".safetensors")] # Remove .safetensors extension
            logger.info(f"Attempting to load local model '{model_key}' from {model_path}...")
            try:
                pipeline = AutoPipelineForText2Image.from_single_file(
                    model_path,
                    torch_dtype=torch_dtype
                    # variant is typically not used with from_single_file for .safetensors
                )

                # Apply model-specific configurations like prediction_type
                model_specific_config = config.LOCAL_MODEL_CONFIGS.get(model_key)
                if model_specific_config and "prediction_type" in model_specific_config:
                    custom_prediction_type = model_specific_config["prediction_type"]
                    logger.info(f"Applying custom prediction_type '{custom_prediction_type}' for local model '{model_key}'.")
                    
                    # Get the original scheduler's config and update prediction_type
                    scheduler_config = dict(pipeline.scheduler.config) # Make a mutable copy
                    scheduler_config["prediction_type"] = custom_prediction_type
                    
                    # Create a new scheduler instance with the updated prediction_type
                    # Using EulerDiscreteScheduler as it's common and supports this.
                    try:
                        new_scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
                        pipeline.scheduler = new_scheduler
                        logger.info(f"Successfully applied prediction_type '{custom_prediction_type}' to scheduler for model '{model_key}'.")
                    except Exception as scheduler_ex:
                        logger.error(f"Failed to create or apply new scheduler with custom prediction_type for model '{model_key}': {scheduler_ex}", exc_info=True)
                else:
                    # Log the default prediction type if no custom one is set
                    default_pred_type = pipeline.scheduler.config.get("prediction_type")
                    logger.info(f"Using default prediction_type '{default_pred_type if default_pred_type else 'None (or not applicable)'}' for local model '{model_key}'.")

                if torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")
                    logger.info(f"Moved local model '{model_key}' to GPU.")
                else:
                    logger.warning(f"CUDA not available for local model '{model_key}'. Running on CPU might be slow.")
                SUPPORTED_PIPELINES[model_key] = pipeline
                logger.info(f"Successfully loaded local model '{model_key}'.")
            except Exception as e:
                logger.error(f"Failed to load local model '{model_key}' from {model_path}: {e}", exc_info=True)
        
        if config.DEFAULT_MODEL_IDENTIFIER not in SUPPORTED_PIPELINES:
            logger.warning(f"Default local model '{config.DEFAULT_MODEL_IDENTIFIER}' specified in config was not found or failed to load from {config.LOCAL_MODELS_DIR}.")

else: # Load from Hugging Face Hub
    hub_model_id = config.DEFAULT_MODEL_IDENTIFIER
    # Use the last part of the Hub ID as a key, or the full ID if it's simple
    model_key = hub_model_id.split('/')[-1] if '/' in hub_model_id else hub_model_id
    logger.info(f"Attempting to load model '{model_key}' (Hub ID: {hub_model_id}) from Hugging Face Hub...")
    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            hub_model_id,
            torch_dtype=torch_dtype,
            variant=config.VARIANT if config.VARIANT else None
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            logger.info(f"Moved Hub model '{model_key}' to GPU.")
        else:
            logger.warning(f"CUDA not available for Hub model '{model_key}'. Running on CPU might be slow.")
        SUPPORTED_PIPELINES[model_key] = pipeline
        logger.info(f"Successfully loaded model '{model_key}' (Hub ID: {hub_model_id}).")
    except Exception as e:
        logger.error(f"Failed to load model '{model_key}' (Hub ID: {hub_model_id}) from Hugging Face Hub: {e}", exc_info=True)

if not SUPPORTED_PIPELINES:
    logger.error("No models were successfully loaded. The API will not be able to generate images. Please check your configuration and model files.")
else:
    logger.info(f"Available models for generation: {list(SUPPORTED_PIPELINES.keys())}")
    # Log if the default model (from config) is available
    if config.LOAD_MODELS_FROM_LOCAL: # Default is a local key
        if config.DEFAULT_MODEL_IDENTIFIER in SUPPORTED_PIPELINES:
             logger.info(f"Default model for local loading '{config.DEFAULT_MODEL_IDENTIFIER}' is available.")
        # Warning already logged if not found
    else: # Default is a Hub ID, its key should be in supported pipelines
        hub_default_key = config.DEFAULT_MODEL_IDENTIFIER.split('/')[-1] if '/' in config.DEFAULT_MODEL_IDENTIFIER else config.DEFAULT_MODEL_IDENTIFIER
        if hub_default_key in SUPPORTED_PIPELINES:
            logger.info(f"Default Hub model '{hub_default_key}' (from ID '{config.DEFAULT_MODEL_IDENTIFIER}') is available.")
        # Error already logged if Hub loading failed for this


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    guidance_scale: float | None = None  # Will be defaulted in endpoint if None
    num_inference_steps: int | None = None # Will be defaulted in endpoint if None


@app.get("/")
async def root():
    if not SUPPORTED_PIPELINES:
        return {"message": "Image Generation API is running, but no models are currently loaded. Please check server logs and configuration."}
    return {"message": f"Image Generation API is running. Use /generate/{{model_name}} to create images. Currently supported models: {list(SUPPORTED_PIPELINES.keys())}"}


@app.post("/generate/{model_name}")
async def generate_image(model_name: str, request: ImageRequest):
    logger.info(f"Received request for model: {model_name} with prompt: '{request.prompt[:50]}...'")
    if model_name not in SUPPORTED_PIPELINES:
        logger.warning(f"Attempted to use unsupported model: {model_name}")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Supported models: {list(SUPPORTED_PIPELINES.keys())}")

    pipeline = SUPPORTED_PIPELINES[model_name]
    
    # Determine generation parameters, applying defaults if not provided
    is_turbo_model = "turbo" in model_name.lower() # Basic check, might need refinement
    
    current_guidance_scale = request.guidance_scale
    if current_guidance_scale is None:
        current_guidance_scale = 0.0 if is_turbo_model else 7.5
        logger.info(f"guidance_scale not provided, defaulting to {current_guidance_scale} for model '{model_name}'.")

    current_num_steps = request.num_inference_steps
    if current_num_steps is None:
        current_num_steps = 1 if is_turbo_model else 25 # SDXL Turbo often uses 1, full SDXL might use 20-50
        logger.info(f"num_inference_steps not provided, defaulting to {current_num_steps} for model '{model_name}'.")

    logger.info(f"Generating image with model '{model_name}', prompt='{request.prompt[:50]}...', guidance_scale={current_guidance_scale}, num_inference_steps={current_num_steps}")

    try:
        image = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            guidance_scale=current_guidance_scale,
            num_inference_steps=current_num_steps
        ).images[0]

        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info(f"Successfully generated image for model: {model_name} with prompt: '{request.prompt[:50]}...'")
        return {"image_base64": img_str, "model_used": model_name}
    except Exception as e:
        # Consider more specific error handling if needed
        logger.error(f"Error generating image for model {model_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # This allows running the app with `python main.py`
    # Uses host and port from config.py
    logger.info(f"Starting Uvicorn server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_config=None) # Disable uvicorn's default logging to use ours

