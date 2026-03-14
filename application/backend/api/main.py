"""
TOKKATOT Cloud API - Ensemble Model Verification Service
"""

import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Dict
import torch
import os
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), 'templates'))

# Import the detector
try:
    from ..services.inference import ChickenDiseaseDetector
except ImportError:
    # Handle if run directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from backend.services.inference import ChickenDiseaseDetector

app = FastAPI(
    title="TOKKATOT Cloud API",
    description="Ensemble AI verification service for chicken disease detection.",
    version="1.0.0"
)

# Global detector instance
detector = None

def get_detector():
    """Lazy load detector."""
    global detector
    if detector is None:
        # Try application/ensemble_model.pth first
        app_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'inferences', 'ensemble_model.pth'))
        dev_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'development', 'outputs', 'inferences', 'ensemble_model.pth'))
        model_path = os.environ.get("ENSEMBLE_MODEL_PATH", app_model_path)
        if not os.path.exists(model_path):
            # Fallback to app_model_path if env var is set incorrectly
            if os.path.exists(app_model_path):
                model_path = app_model_path
            elif os.path.exists(dev_model_path):
                model_path = dev_model_path
            else:
                raise RuntimeError(f"Model not found. Checked: {app_model_path}, {dev_model_path}, env: {model_path}")
        detector = ChickenDiseaseDetector(
            model_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    return detector

@app.get("/api/v1/health")
async def health_check():
    """Check system health."""
    return {"status": "ok", "model_loaded": detector is not None}

@app.post("/api/v1/verify")
async def verify_sample(file: UploadFile = File(...), det: ChickenDiseaseDetector = Depends(get_detector)):
    """
    Verify a fecal sample using the full ensemble model.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Save image to demo_uploads with unique filename
        import uuid
        import shutil
        upload_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'demo_uploads')
        upload_dir = os.path.abspath(upload_dir)
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"demo_{uuid.uuid4().hex[:8]}.png"
        save_path = os.path.join(upload_dir, filename)
        image.save(save_path)

        # Run prediction
        result = det.predict(image, return_details=True)
        result['uploaded_image_path'] = save_path

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to TOKKATOT API. Use /docs for API documentation."}

