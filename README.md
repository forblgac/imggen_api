# Image Generation API

This is a FastAPI-based API server for generating images using various diffusion models.

## Features

- Supports SDXL models, loadable from Hugging Face Hub or a local directory (`.safetensors` files).
- Configurable model settings and loading strategy via `config.py`.
- Generic endpoint `/generate/{model_name}` for easy extension to other models.
- Basic logging for requests and server operations.

## Model Configuration

The API can load models either from Hugging Face Hub or from a local directory containing `.safetensors` files. This behavior is controlled by settings in `config.py`:

-   `LOAD_MODELS_FROM_LOCAL` (boolean):
    -   Set to `True` to load models from the directory specified by `LOCAL_MODELS_DIR`.
    -   Set to `False` (default) to load the default model from Hugging Face Hub.
-   `LOCAL_MODELS_DIR` (string):
    -   Specifies the path to your local models directory (e.g., `./models`).
    -   Place your `.safetensors` model files in this directory.
    -   Each model will be available in the API using its filename (without the `.safetensors` extension) as the `{model_name}` in the endpoint path.
-   `DEFAULT_MODEL_IDENTIFIER` (string):
    -   If `LOAD_MODELS_FROM_LOCAL` is `True`, this should be the filename (without extension) of the model in `LOCAL_MODELS_DIR` that you want to be considered the primary or default (e.g., `"my_sdxl_model"`). The server will log if this model is successfully loaded.
    -   If `LOAD_MODELS_FROM_LOCAL` is `False`, this is the Hugging Face Hub identifier for the model to load (e.g., `"stabilityai/sdxl-turbo"`).
-   `TORCH_DTYPE` (string): Sets the torch dtype for model loading (e.g., `"float16"` or `"float32"`).
-   `VARIANT` (string): Specifies the model variant if loading from Hugging Face Hub (e.g., `"fp16"`). Not typically used for `from_single_file` local loading.

### Using Local Models

To ensure correct behavior for certain local models (e.g., v-prediction models), you can provide specific configurations via the `LOCAL_MODEL_CONFIGS` dictionary in `config.py`. The key should be the model's filename (without `.safetensors` extension), and the value is a dictionary of configurations. Currently, `"prediction_type"` is supported.

Example in `config.py` for a v-prediction model named `my_v_pred_model.safetensors`:
```python
LOCAL_MODEL_CONFIGS = {
    "my_v_pred_model": {"prediction_type": "v_prediction"},
    # Add other model-specific configs here
}
```
If a `prediction_type` is specified, the server will attempt to reconfigure the model's scheduler accordingly.

**Steps to use local models:**

1.  Create a directory (e.g., `mkdir models` in the project root).
2.  Place your `.safetensors` model files into this directory.
3.  In `config.py`:
    *   Set `LOAD_MODELS_FROM_LOCAL = True`.
    *   Set `LOCAL_MODELS_DIR` to the path of your models directory (e.g., `LOCAL_MODELS_DIR = "./models"`).
    *   Optionally, set `DEFAULT_MODEL_IDENTIFIER` to the filename (without extension) of one of your local models if you want to explicitly note a default.

The server will attempt to load all `.safetensors` files from the specified directory. You can then use the filename (without extension) as the `{model_name}` in API requests.

## Setup with uv

This project uses `uv` for Python environment and package management.

1.  **Install uv**: If you don't have `uv` installed, follow the instructions at [astral.sh/uv/install](https://astral.sh/uv/install.sh) or run `pip install uv`.

2.  **Clone the repository** (if applicable).

3.  **Create and activate a virtual environment**:
    ```bash
    uv venv  # This creates a .venv directory
    source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
    ```

4.  **Install dependencies**:
    Once the virtual environment is activated, install the dependencies from the lock file:
    ```bash
    uv pip sync uv.lock
    ```

5.  **Run the server**:
    With the virtual environment activated:
    ```bash
    python main.py
    ```
    This will use the host and port defined in `config.py`.

    For development with auto-reload (also with the virtual environment activated):
    ```bash
    uvicorn main:app --reload
    ```
    (Note: You can also define scripts in `pyproject.toml` under `[project.scripts]` and run them using `uv run <script_name>`, e.g., `uv run dev` or `uv run start` if you uncomment and configure them.)

## Usage

Send a POST request to the `/generate/{model_name}` endpoint. The available `{model_name}` values depend on your configuration (Hub model or local model filenames). Check the server startup logs or the root `/` endpoint to see the list of currently loaded and available models.

### Request Body

The request body should be a JSON object with the following fields:

-   `prompt` (string, required): The text prompt for image generation.
-   `negative_prompt` (string, optional): A negative prompt to guide the generation away from certain concepts.
-   `guidance_scale` (float, optional): Controls how much the prompt influences the generation. 
    If not provided, defaults will be applied based on the model type (e.g., `0.0` for "turbo" models, `7.5` for standard models).
-   `num_inference_steps` (int, optional): The number of steps for the diffusion process.
    If not provided, defaults will be applied based on the model type (e.g., `1` for "turbo" models, `25` for standard models).

Example using `curl` (assuming a model named `sdxl-turbo` is loaded, either from Hub or locally):

```bash
curl -X POST "http://127.0.0.1:8000/generate/sdxl-turbo" \
-H "Content-Type: application/json" \
-d '{"prompt": "A futuristic cat astronaut"}'
```

If you had a local model `models/my_custom_model.safetensors` and configured local loading, you would use:

```bash
curl -X POST "http://127.0.0.1:8000/generate/my_custom_model" \
-H "Content-Type: application/json" \
-d '{"prompt": "Epic fantasy landscape"}'
```

This will return a JSON response containing the base64 encoded image.
