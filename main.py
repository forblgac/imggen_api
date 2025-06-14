from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
AVAILABLE_LORAS = {}


def load_models():
    logger.info(
        f"Attempting to load models. LOAD_MODELS_FROM_LOCAL: {config.LOAD_MODELS_FROM_LOCAL}"
    )
    if config.LOAD_MODELS_FROM_LOCAL:
        # --- Load Checkpoint Models ---
        logger.info(f"Loading checkpoint models from: {config.CHECKPOINT_MODELS_DIR}")
        if not os.path.isdir(config.CHECKPOINT_MODELS_DIR):
            logger.warning(
                f"Checkpoint directory not found: {config.CHECKPOINT_MODELS_DIR}"
            )
        else:
            checkpoint_paths = glob.glob(
                os.path.join(config.CHECKPOINT_MODELS_DIR, "*.safetensors")
            )
            for model_path in checkpoint_paths:
                model_key = os.path.basename(model_path)[: -len(".safetensors")]
                logger.info(f"Attempting to load checkpoint model '{model_key}'...")
                try:
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch_dtype,
                        variant=config.VARIANT if config.VARIANT else None,
                        use_safetensors=True,
                    )
                    # Apply model-specific configs
                    model_specific_config = config.LOCAL_MODEL_CONFIGS.get(
                        model_key, {}
                    )
                    if "prediction_type" in model_specific_config:
                        # ... (prediction_type logic remains the same)
                        pass

                    if torch.cuda.is_available():
                        pipeline.to("cuda")
                    SUPPORTED_PIPELINES[model_key] = pipeline
                    logger.info(f"Successfully loaded checkpoint model '{model_key}'.")
                except Exception as e:
                    logger.error(
                        f"Failed to load checkpoint model '{model_key}': {e}",
                        exc_info=True,
                    )

        # --- Scan for LoRA Models ---
        logger.info(f"Scanning for LoRA models in: {config.LORA_MODELS_DIR}")
        if not os.path.isdir(config.LORA_MODELS_DIR):
            logger.warning(f"LoRA directory not found: {config.LORA_MODELS_DIR}")
        else:
            lora_paths = glob.glob(
                os.path.join(config.LORA_MODELS_DIR, "*.safetensors")
            )
            for lora_path in lora_paths:
                lora_key = os.path.basename(lora_path)[: -len(".safetensors")]
                AVAILABLE_LORAS[lora_key] = lora_path
                logger.info(f"Found LoRA: '{lora_key}'")


load_models()

if not SUPPORTED_PIPELINES:
    logger.error("No checkpoint models were successfully loaded.")
else:
    logger.info(f"Available checkpoint models: {list(SUPPORTED_PIPELINES.keys())}")
    logger.info(f"Available LoRA models: {list(AVAILABLE_LORAS.keys())}")


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    guidance_scale: Optional[float] = 1.0
    num_inference_steps: Optional[int] = 8
    scheduler: Optional[str] = None
    lora_model: Optional[str] = None
    lora_scale: Optional[float] = 0.8


@app.get("/")
async def root():
    return {
        "message": "Image Generation API",
        "supported_models": list(SUPPORTED_PIPELINES.keys()),
        "available_loras": list(AVAILABLE_LORAS.keys()),
    }


@app.post("/generate/{model_name}")
async def generate_image(model_name: str, request: ImageRequest):
    if model_name not in SUPPORTED_PIPELINES:
        raise HTTPException(status_code=404, detail="Base model not found.")

    pipeline = SUPPORTED_PIPELINES[model_name]
    original_scheduler = pipeline.scheduler

    try:
        # --- LoRA Loading ---
        if request.lora_model:
            if request.lora_model in AVAILABLE_LORAS:
                lora_path = AVAILABLE_LORAS[request.lora_model]
                logger.info(f"Loading LoRA '{request.lora_model}' from {lora_path}")
                pipeline.load_lora_weights(lora_path)
                # fuse/unfuse can be called here if needed, for now just load
            else:
                logger.warning(f"LoRA model '{request.lora_model}' not found.")

        # --- Scheduler Selection ---
        if request.scheduler:
            # ... (scheduler logic remains the same)
            pass

        # --- Generation ---
        image = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            cross_attention_kwargs={"scale": request.lora_scale}
            if request.lora_model
            else None,
        ).images[0]

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        logger.error(f"Error during image generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image generation failed.")
    finally:
        # --- Cleanup ---
        pipeline.scheduler = original_scheduler
        if request.lora_model and request.lora_model in AVAILABLE_LORAS:
            pipeline.unload_lora_weights()
            logger.info(f"Unloaded LoRA '{request.lora_model}'.")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Uvicorn server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_config=None)
