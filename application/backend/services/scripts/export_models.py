"""
Export script to convert the trained EfficientNetB0 classification model
and a lightweight YOLOv8n detection model to ONNX format.

These ONNX models can then be used as input for the Hailo compiler to generate
HEF files for the Raspberry Pi AI HAT+.
"""

import torch
import onnx
from pathlib import Path
import os
from ultralytics import YOLO

from models import EfficientNetB0Classifier

def export_efficientnet_to_onnx(
    pytorch_checkpoint_path: Path,
    output_dir: Path,
    img_size: int = 224,
    num_classes: int = 4
):
    """
    Loads a PyTorch EfficientNetB0 model and converts it to ONNX format.

    Args:
        pytorch_checkpoint_path (Path): Path to the PyTorch .pth checkpoint.
        output_dir (Path): Directory to save the ONNX model.
        img_size (int): The input image size for the model.
        num_classes (int): The number of output classes.
    """
    print(f"Starting ONNX export for EfficientNetB0 model from {pytorch_checkpoint_path.name}...")

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{pytorch_checkpoint_path.stem}.onnx"

    # 1. Load PyTorch Model
    print("\nStep 1: Loading PyTorch EfficientNetB0 model...")
    if not pytorch_checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {pytorch_checkpoint_path}")
        print("Please ensure you have trained the model and the checkpoint exists.")
        print("Creating a dummy checkpoint for demonstration purposes...")
        model = EfficientNetB0Classifier(num_classes=num_classes)
        torch.save({'model_state_dict': model.state_dict()}, pytorch_checkpoint_path)
        print(f"✓ Dummy checkpoint created at {pytorch_checkpoint_path}")


    device = torch.device('cpu')
    model = EfficientNetB0Classifier(num_classes=num_classes)
    checkpoint = torch.load(pytorch_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ PyTorch EfficientNetB0 model loaded successfully.")

    # 2. Convert PyTorch to ONNX
    print("\nStep 2: Converting EfficientNetB0 to ONNX...")
    dummy_input = torch.randn(1, 3, img_size, img_size, requires_grad=True, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=12, # Target opset 12 for broader compatibility with older tools
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✓ ONNX model saved to {onnx_path}")
    
    # 3. Verification (optional, but good practice)
    print("\nStep 3: Verifying the ONNX model...")
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid.")
    except Exception as e:
        print(f"❌ ONNX model validation failed: {e}")

    print("\n✅ EfficientNetB0 ONNX export complete!")
    return onnx_path


def export_yolov8n_to_onnx(
    output_dir: Path,
    img_size: int = 640
):
    """
    Loads a pre-trained YOLOv8n model and converts it to ONNX format.

    Args:
        output_dir (Path): Directory to save the ONNX model.
        img_size (int): The input image size for the model.
    """
    print(f"Starting ONNX export for YOLOv8n model...")

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "yolov8n.onnx"

    # 1. Load pre-trained YOLOv8n model
    print("\nStep 1: Loading pre-trained YOLOv8n model...")
    # Using 'yolov8n.pt' to load the nano version of YOLOv8
    model = YOLO("yolov8n.pt") 
    print("✓ YOLOv8n model loaded successfully.")

    # 2. Export to ONNX
    print("\nStep 2: Converting YOLOv8n to ONNX...")
    # The export method handles the conversion
    # Note: 'f' is not a valid argument for YOLO.export. We move the file afterwards.
    exported_path = model.export(format="onnx", imgsz=img_size, opset=12, simplify=True, dynamic=True)
    
    # Move the exported file to the desired output directory
    import shutil
    shutil.move(exported_path, onnx_path)
    
    print(f"✓ YOLOv8n ONNX model saved to {onnx_path}")

    # 3. Verification (optional, but good practice)
    print("\nStep 3: Verifying the ONNX model...")
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid.")
    except Exception as e:
        print(f"❌ ONNX model validation failed: {e}")
    
    print("\n✅ YOLOv8n ONNX export complete!")
    return onnx_path


if __name__ == '__main__':
    # Define paths
    EFFICIENTNET_CHECKPOINT = Path('outputs/checkpoints/EfficientNetB0_best.pth')
    ONNX_OUTPUT_DIR = Path('outputs/onnx')
    
    # Export EfficientNetB0 classification model
    efficientnet_onnx_path = export_efficientnet_to_onnx(
        pytorch_checkpoint_path=EFFICIENTNET_CHECKPOINT,
        output_dir=ONNX_OUTPUT_DIR,
        img_size=224,
        num_classes=4
    )
    print(f"\nEfficientNetB0 ONNX model available at: {efficientnet_onnx_path}")

    # Export YOLOv8n detection model
    yolov8n_onnx_path = export_yolov8n_to_onnx(
        output_dir=ONNX_OUTPUT_DIR,
        img_size=640 # Common input size for YOLO models
    )
    print(f"\nYOLOv8n ONNX model available at: {yolov8n_onnx_path}")
