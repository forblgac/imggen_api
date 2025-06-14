import pytest
from httpx import AsyncClient
from main import app # Assuming your FastAPI app instance is named 'app' in main.py
from unittest.mock import Mock # For creating mock objects
from PIL import Image # For creating a dummy image
import io
import base64

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

# Helper to create a dummy PIL Image
def create_dummy_pil_image():
    img = Image.new('RGB', (60, 30), color = 'red')
    return img

# Mock pipeline class
class MockImageGenerationPipeline:
    def __init__(self, expected_prompt=None, expected_negative_prompt=None, expected_guidance_scale=None, expected_num_steps=None):
        self.expected_prompt = expected_prompt
        self.expected_negative_prompt = expected_negative_prompt
        self.expected_guidance_scale = expected_guidance_scale
        self.expected_num_steps = expected_num_steps

    def __call__(self, prompt, negative_prompt=None, guidance_scale=None, num_inference_steps=None):
        # Optionally, assert that the pipeline was called with expected arguments
        if self.expected_prompt is not None:
            assert prompt == self.expected_prompt
        # Add more assertions if needed for other params
        
        # Simulate returning a list of images, like the actual pipeline
        mock_image_result = Mock()
        mock_image_result.images = [create_dummy_pil_image()]
        return mock_image_result

async def test_read_root():
    """Test that the root endpoint returns a successful response and expected message structure."""
    # Temporarily mock SUPPORTED_PIPELINES to ensure consistent root message for test
    # This also simulates the state where at least one model is loaded.
    original_pipelines = app.dependency_overrides.get('main.SUPPORTED_PIPELINES')
    app.dependency_overrides['main.SUPPORTED_PIPELINES'] = lambda: {"mock_model": Mock() }

    async with AsyncClient(app=app, base_url="http://127.0.0.1:8000") as client:
        response = await client.get("/")
    
    assert response.status_code == 200
    json_response = response.json()
    assert "message" in json_response
    assert "Currently supported models: ['mock_model']" in json_response["message"]
    print(f"Root endpoint response (mocked): {json_response}")

    # Clean up/restore original overrides if any
    if original_pipelines is not None:
        app.dependency_overrides['main.SUPPORTED_PIPELINES'] = original_pipelines
    else:
        del app.dependency_overrides['main.SUPPORTED_PIPELINES']

async def test_generate_image_mocked(monkeypatch):
    """Test the /generate/{model_name} endpoint with a mocked pipeline."""
    mock_pipeline_instance = MockImageGenerationPipeline()
    
    # Use monkeypatch to replace the SUPPORTED_PIPELINES in the main module for this test
    monkeypatch.setattr("main.SUPPORTED_PIPELINES", {"mock_model": mock_pipeline_instance})

    async with AsyncClient(app=app, base_url="http://127.0.0.1:8000") as client:
        response = await client.post("/generate/mock_model", json={"prompt": "a test prompt"})
    
    assert response.status_code == 200
    json_response = response.json()
    assert "image_base64" in json_response
    assert "model_used" in json_response
    assert json_response["model_used"] == "mock_model"

    # Check if the base64 string is plausible (non-empty)
    assert len(json_response["image_base64"]) > 0

    # Optional: try to decode and check if it's a valid image
    try:
        img_bytes = base64.b64decode(json_response["image_base64"])
        img = Image.open(io.BytesIO(img_bytes))
        assert img.format == "PNG"
        assert img.size == (60, 30) # Matches dummy image
    except Exception as e:
        pytest.fail(f"Failed to decode or validate base64 image: {e}")

async def test_generate_image_model_not_found(monkeypatch):
    """Test requesting a model_name that is not loaded."""
    monkeypatch.setattr("main.SUPPORTED_PIPELINES", {"actual_model": MockImageGenerationPipeline()})

    async with AsyncClient(app=app, base_url="http://127.0.0.1:8000") as client:
        response = await client.post("/generate/non_existent_model", json={"prompt": "a test prompt"})
    
    assert response.status_code == 404
    json_response = response.json()
    assert "detail" in json_response
    assert "Model 'non_existent_model' not found" in json_response["detail"]

# Note: To test actual model loading logic (local or hub) without full generation,
# you would need to mock `AutoPipelineForText2Image.from_pretrained` and 
# `AutoPipelineForText2Image.from_single_file` within the `main` module during app startup.
# This is more complex and typically involves conftest.py or session-scoped fixtures.
