from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import io
import base64
import logging # Import logging

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
    # Default or raise error if misconfigured
    logger.warning(f"TORCH_DTYPE '{config.TORCH_DTYPE}' in config.py is not recognized. Defaulting to float16.")
    torch_dtype = torch.float16


# Load the SDXL model using settings from config.py
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    config.SDXL_MODEL_NAME,
    torch_dtype=torch_dtype,
    variant=config.VARIANT if config.VARIANT else None # Pass None if VARIANT is empty or None
)
logger.info(f"Loaded model {config.SDXL_MODEL_NAME} with dtype {config.TORCH_DTYPE} and variant '{config.VARIANT}'.")

# Check if CUDA is available and move the pipeline to GPU
if torch.cuda.is_available():
    pipeline_text2image = pipeline_text2image.to("cuda")
    logger.info("CUDA is available. Model moved to GPU.")
else:
    logger.warning("CUDA not available, running on CPU. This might be slow.")

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    # Default guidance_scale and num_inference_steps might need adjustment based on the model in config
    # For sdxl-turbo, these defaults are fine. For full SDXL, they might change.
    guidance_scale: float = 0.0 if "turbo" in config.SDXL_MODEL_NAME else 7.5
    num_inference_steps: int = 1 if "turbo" in config.SDXL_MODEL_NAME else 25


@app.get("/")
async def root():
    return {"message": "Image Generation API is running. Use /generate/{model_name} to create images. Currently supported: sdxl"}

# For now, we only have one pipeline loaded based on config.SDXL_MODEL_NAME
# This will serve as the pipeline for the 'sdxl' model_name endpoint.
# Future enhancements could load multiple models into a dictionary of pipelines.
SUPPORTED_PIPELINES = {
    "sdxl": pipeline_text2image # The pipeline loaded from config
}

@app.post("/generate/{model_name}")
async def generate_image(model_name: str, request: ImageRequest):
    logger.info(f"Received request for model: {model_name} with prompt: '{request.prompt[:50]}...'")
    if model_name not in SUPPORTED_PIPELINES:
        logger.warning(f"Attempted to use unsupported model: {model_name}")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Supported models: {list(SUPPORTED_PIPELINES.keys())}")

    pipeline = SUPPORTED_PIPELINES[model_name]
    
    # Adjust defaults based on model_name if necessary in the future.
    # For now, ImageRequest defaults are based on config.SDXL_MODEL_NAME which aligns with the 'sdxl' model_name.
    # If config.SDXL_MODEL_NAME was, for example, a non-turbo SDXL, the ImageRequest defaults would suit that.

    try:
        image = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            guidance_scale=request.guidance_scale, # These defaults are now set in ImageRequest based on config
            num_inference_steps=request.num_inference_steps # These defaults are now set in ImageRequest based on config
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

