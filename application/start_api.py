#!/usr/bin/env python3
"""
Launch script for the TOKKATOT AI API.
"""

import uvicorn
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Start TOKKATOT AI API")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--model", default="application/inferences/ensemble_model.pth", help="Path to ensemble model")
    args = parser.parse_args()
    
    # Set environment variable for the API to find the model
    os.environ["ENSEMBLE_MODEL_PATH"] = str(Path(args.model).resolve())
    
    print(f"Starting TOKKATOT AI API on {args.host}:{args.port}")
    print(f"Using model: {os.environ['ENSEMBLE_MODEL_PATH']}")
    
    # Import app here to ensure environment variable is set
    from backend.api.main import app
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
