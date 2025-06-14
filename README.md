# Image Generation API

This is a FastAPI-based API server for generating images using various diffusion models.

## Features

- Supports SDXL (initially `sdxl-turbo` by default).
- Configurable model settings via `config.py`.
- Generic endpoint `/generate/{model_name}` for easy extension to other models.
- Basic logging for requests and server operations.

## Setup

1.  Clone the repository (if applicable).
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the server:
    ```bash
    python main.py
    ```
    or for development with auto-reload:
    ```bash
    uvicorn main:app --reload
    ```

## Usage

Send a POST request to the `/generate/{model_name}` endpoint.

Example using `curl` for SDXL:

```bash
curl -X POST "http://127.0.0.1:8000/generate/sdxl" \
-H "Content-Type: application/json" \
-d '{"prompt": "A futuristic cat astronaut"}'
```

This will return a JSON response containing the base64 encoded image.
