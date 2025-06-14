# Image Generation API

This is a FastAPI-based API server for generating images using various diffusion models.

## Features

- Supports SDXL (initially `sdxl-turbo` by default).
- Configurable model settings via `config.py`.
- Generic endpoint `/generate/{model_name}` for easy extension to other models.
- Basic logging for requests and server operations.

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

Send a POST request to the `/generate/{model_name}` endpoint.

Example using `curl` for SDXL:

```bash
curl -X POST "http://127.0.0.1:8000/generate/sdxl" \
-H "Content-Type: application/json" \
-d '{"prompt": "A futuristic cat astronaut"}'
```

This will return a JSON response containing the base64 encoded image.
