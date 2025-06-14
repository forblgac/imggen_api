from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
    LCMScheduler,
    DDIMScheduler,
)
from PIL import Image
import io
import base64
import logging
import os
import glob
from typing import Optional

import config

app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Starting Image Generation API...")

if config.TORCH_DTYPE == "float16":
    torch_dtype = torch.float16
elif config.TORCH_DTYPE == "float32":
    torch_dtype = torch.float32
else:
    logger.warning(
        f"TORCH_DTYPE '{config.TORCH_DTYPE}' in config.py is not recognized. Defaulting to float16."
    )
    torch_dtype = torch.float16

SUPPORTED_PIPELINES = {}
logger.info(
    f"Attempting to load models. LOAD_MODELS_FROM_LOCAL: {config.LOAD_MODELS_FROM_LOCAL}"
)

if config.LOAD_MODELS_FROM_LOCAL:
    logger.info(f"Loading models from local directory: {config.LOCAL_MODELS_DIR}")
    if not os.path.isdir(config.LOCAL_MODELS_DIR):
        logger.warning(
            f"Local models directory not found: {config.LOCAL_MODELS_DIR}. No local models will be loaded."
        )
    else:
        local_model_paths = glob.glob(
            os.path.join(config.LOCAL_MODELS_DIR, "*.safetensors")
        )
        if not local_model_paths:
            logger.warning(f"No .safetensors files found in {config.LOCAL_MODELS_DIR}")

        for model_path in local_model_paths:
            model_filename = os.path.basename(model_path)
            model_key = model_filename[: -len(".safetensors")]
            logger.info(
                f"Attempting to load local model '{model_key}' from {model_path}..."
            )
            try:
                # Use StableDiffusionXLPipeline.from_single_file for .safetensors
                pipeline = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch_dtype,
                    variant=config.VARIANT if config.VARIANT else None,
                    use_safetensors=True,
                )

                model_specific_config = config.LOCAL_MODEL_CONFIGS.get(model_key)
                if model_specific_config and "prediction_type" in model_specific_config:
                    custom_prediction_type = model_specific_config["prediction_type"]
                    logger.info(
                        f"Applying custom prediction_type '{custom_prediction_type}' for local model '{model_key}'."
                    )
                    scheduler_config = dict(pipeline.scheduler.config)
                    scheduler_config["prediction_type"] = custom_prediction_type
                    SchedulerClass = pipeline.scheduler.__class__
                    try:
                        new_scheduler = SchedulerClass.from_config(scheduler_config)
                        pipeline.scheduler = new_scheduler
                        logger.info(
                            f"Successfully applied prediction_type '{custom_prediction_type}' using {SchedulerClass.__name__} for model '{model_key}'."
                        )
                    except Exception as scheduler_ex:
                        logger.error(
                            f"Failed to create or apply new scheduler with custom prediction_type for model '{model_key}': {scheduler_ex}",
                            exc_info=True,
                        )
                else:
                    default_pred_type = pipeline.scheduler.config.get("prediction_type")
                    logger.info(
                        f"Using default prediction_type '{default_pred_type if default_pred_type else 'None (or not applicable)'}' for local model '{model_key}'."
                    )

                if torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")
                    logger.info(f"Moved local model '{model_key}' to GPU.")
                else:
                    logger.warning(
                        f"CUDA not available for local model '{model_key}'. Running on CPU might be slow."
                    )
                SUPPORTED_PIPELINES[model_key] = pipeline
                logger.info(f"Successfully loaded local model '{model_key}'.")
            except Exception as e:
                logger.error(
                    f"Failed to load local model '{model_key}' from {model_path}: {e}",
                    exc_info=True,
                )

        if config.DEFAULT_MODEL_IDENTIFIER not in SUPPORTED_PIPELINES:
            logger.warning(
                f"Default local model '{config.DEFAULT_MODEL_IDENTIFIER}' specified in config was not found or failed to load from {config.LOCAL_MODELS_DIR}."
            )

else:  # Load from Hugging Face Hub
    hub_model_id = config.DEFAULT_MODEL_IDENTIFIER
    model_key = hub_model_id.split("/")[-1] if "/" in hub_model_id else hub_model_id
    logger.info(
        f"Attempting to load model '{model_key}' (Hub ID: {hub_model_id}) from Hugging Face Hub..."
    )
    try:
        # Using AutoPipelineForText2Image for hub models is more flexible
        pipeline = AutoPipelineForText2Image.from_pretrained(
            hub_model_id,
            torch_dtype=torch_dtype,
            variant=config.VARIANT if config.VARIANT else None,
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            logger.info(f"Moved Hub model '{model_key}' to GPU.")
        else:
            logger.warning(
                f"CUDA not available for Hub model '{model_key}'. Running on CPU might be slow."
            )
        SUPPORTED_PIPELINES[model_key] = pipeline
        logger.info(
            f"Successfully loaded model '{model_key}' (Hub ID: {hub_model_id})."
        )
    except Exception as e:
        logger.error(
            f"Failed to load model '{model_key}' (Hub ID: {hub_model_id}) from Hugging Face Hub: {e}",
            exc_info=True,
        )

if not SUPPORTED_PIPELINES:
    logger.error(
        "No models were successfully loaded. The API will not be able to generate images. Please check your configuration and model files."
    )
else:
    logger.info(f"Available models for generation: {list(SUPPORTED_PIPELINES.keys())}")
    if config.LOAD_MODELS_FROM_LOCAL:
        if config.DEFAULT_MODEL_IDENTIFIER in SUPPORTED_PIPELINES:
            logger.info(
                f"Default model for local loading '{config.DEFAULT_MODEL_IDENTIFIER}' is available."
            )
    else:
        hub_default_key = (
            config.DEFAULT_MODEL_IDENTIFIER.split("/")[-1]
            if "/" in config.DEFAULT_MODEL_IDENTIFIER
            else config.DEFAULT_MODEL_IDENTIFIER
        )
        if hub_default_key in SUPPORTED_PIPELINES:
            logger.info(
                f"Default Hub model '{hub_default_key}' (from ID '{config.DEFAULT_MODEL_IDENTIFIER}') is available."
            )


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    guidance_scale: Optional[float] = 1.0
    num_inference_steps: Optional[int] = 8
    scheduler: Optional[str] = None  # e.g., "LCM", "DDIM"


@app.get("/")
async def root():
    if not SUPPORTED_PIPELINES:
        return {
            "message": "Image Generation API is running, but no models are currently loaded. Please check server logs and configuration."
        }
    return {
        "message": f"Image Generation API is running. Use /generate/{{model_name}} to create images. Currently supported models: {list(SUPPORTED_PIPELINES.keys())}"
    }


@app.post("/generate/{model_name}")
async def generate_image(model_name: str, request: ImageRequest):
    logger.info(
        f"Received request for model: {model_name} with prompt: '{request.prompt[:50]}...'"
    )
    if model_name not in SUPPORTED_PIPELINES:
        logger.warning(f"Attempted to use unsupported model: {model_name}")
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Supported models: {list(SUPPORTED_PIPELINES.keys())}",
        )

    pipeline = SUPPORTED_PIPELINES[model_name]

    # --- Scheduler Selection ---
    original_scheduler = pipeline.scheduler
    if request.scheduler:
        scheduler_name = request.scheduler.upper()
        logger.info(f"Request received for scheduler: {scheduler_name}")
        try:
            if scheduler_name == "LCM":
                pipeline.scheduler = LCMScheduler.from_config(original_scheduler.config)
                logger.info("Switched to LCMScheduler.")
            elif scheduler_name == "DDIM":
                pipeline.scheduler = DDIMScheduler.from_config(
                    original_scheduler.config
                )
                logger.info("Switched to DDIMScheduler.")
            else:
                logger.warning(
                    f"Unsupported scheduler '{request.scheduler}' requested. Using the model's default scheduler."
                )
        except Exception as e:
            logger.error(f"Failed to switch scheduler: {e}", exc_info=True)
            # Revert to original scheduler on failure
            pipeline.scheduler = original_scheduler

    # --- Generation Parameters ---
    # Use values from the request, which now have defaults in Pydantic model
    current_guidance_scale = request.guidance_scale
    current_num_steps = request.num_inference_steps

    logger.info(
        f"Generating image with model '{model_name}', prompt='{request.prompt[:50]}...', guidance_scale={current_guidance_scale}, num_inference_steps={current_num_steps}, scheduler='{pipeline.scheduler.__class__.__name__}'"
    )

    try:
        image = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            guidance_scale=current_guidance_scale,
            num_inference_steps=current_num_steps,
        ).images[0]

        # Restore the original scheduler after generation
        pipeline.scheduler = original_scheduler
        logger.info("Restored original scheduler.")

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info(
            f"Successfully generated image for model: {model_name} with prompt: '{request.prompt[:50]}...'"
        )
        return {"image_base64": img_str, "model_used": model_name}
    except Exception as e:
        logger.error(
            f"Error generating image for model {model_name}: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Uvicorn server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_config=None)
