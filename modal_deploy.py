"""Modal production deployment configuration for Gemini 3 Pro pipeline"""

import modal

# Create Modal stub
stub = modal.Stub("gemini3-pipeline")

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "wget")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "wget -q https://github.com/visioncortex/vtracer/releases/download/v0.6.1/vtracer-linux-x64 -O /usr/local/bin/vtracer",
        "chmod +x /usr/local/bin/vtracer"
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "MODEL_VOLUME_PATH": "/models",
        "API_OUTPUT_DIR": "/tmp/gemini3_output"
    })
)

# Network volume for model weights (persistent storage)
model_volume = modal.NetworkFileSystem.from_name("gemini3-models", create_if_missing=True)


@stub.function(
    image=image,
    gpu=modal.gpu.A10G(),
    network_file_systems={"/models": model_volume},
    timeout=300,  # 5 minutes max execution time
    container_idle_timeout=60,  # Scale to zero after 60s idle
    secrets=[modal.Secret.from_name("gemini3-secrets")]  # For HuggingFace tokens if needed
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI application for Modal deployment"""
    from src.api.server import app
    return app


@stub.function(
    image=image,
    gpu=modal.gpu.A10G(),
    network_file_systems={"/models": model_volume},
    timeout=600  # 10 minutes for model setup
)
def setup_models():
    """Setup models on network volume (run once)"""
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = Path("scripts/setup_model_volume.py")
    
    if not script_path.exists():
        raise FileNotFoundError(
            f"Model setup script not found: {script_path}\n"
            f"Current directory: {Path.cwd()}"
        )
    
    print(f"[Models] Starting model setup from: {script_path}")
    print(f"[Models] Target volume: /models")
    
    try:
        result = subprocess.run(
            [
                sys.executable,  # Use same Python as current process
                str(script_path),
                "--volume-path", "/models"
            ],
            check=False,  # We handle return code
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Log output
        if result.stdout:
            print("[Models] STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("[Models] STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Model setup failed with exit code {result.returncode}\n"
                f"Error output:\n{result.stderr}"
            )
        
        print("[Models] ✓ Model setup completed successfully")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Model setup timed out after 10 minutes. "
            "Check network connection and model server availability."
        )
    except Exception as e:
        raise RuntimeError(f"Model setup failed: {str(e)}")


@stub.function(
    image=image,
    gpu=modal.gpu.A10G(),
    network_file_systems={"/models": model_volume},
    timeout=300
)
def process_image_async(image_bytes: bytes, palette_hex_list: list = None):
    """
    Async function to process image (called by job queue)
    
    Args:
        image_bytes: Image file bytes
        palette_hex_list: Optional palette colors
    
    Returns:
        Dict with svg_xml and metadata
    """
    import tempfile
    from src.pipeline.orchestrator import Gemini3Pipeline
    
    pipeline = Gemini3Pipeline()
    
    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(image_bytes)
        input_path = tmp.name
    
    try:
        # Process image
        svg_xml, metadata = pipeline.process_image(
            input_image_path=input_path,
            palette_hex_list=palette_hex_list
        )
        
        return {
            "svg_xml": svg_xml,
            "metadata": metadata
        }
    finally:
        # Clean up temp file
        import os
        if os.path.exists(input_path):
            os.unlink(input_path)


if __name__ == "__main__":
    # Deploy to Modal
    # Run: modal deploy modal_deploy.py
    print("Deploy to Modal with: modal deploy modal_deploy.py")
    print("Setup models with: modal run modal_deploy.py::setup_models")




