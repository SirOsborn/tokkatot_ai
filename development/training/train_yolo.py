#!/usr/bin/env python3
"""
Train YOLOv8n for fecal sample detection on conveyor belt.

2-class detection:
  - healthy_feces (class 0): healthy samples → continue monitoring
  - suspicious_feces (class 1): potentially diseased → send to cloud

Usage:
    python train_yolo.py
    python train_yolo.py --epochs 100 --device cuda --batch 32
    python train_yolo.py --resume  # resume from last checkpoint
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(args):
    """Train YOLOv8n on the Tokkatot fecal dataset."""
    
    print("=" * 60)
    print("TOKKATOT YOLO TRAINING")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Data config: {args.data}")
    print("=" * 60)
    
    # Load model
    if args.resume:
        model = YOLO("runs/detect/train/weights/last.pt")
        print("Resuming from last checkpoint...")
    else:
        model = YOLO("yolov8n.pt")
        print("Starting from pretrained YOLOv8n...")
    
    # Parse cache argument
    cache_val = args.cache
    if cache_val.lower() == "true": cache_val = True
    elif cache_val.lower() == "false": cache_val = False
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs/detect",
        name="train",
        exist_ok=True,
        workers=args.workers,   # Explicitly set workers (safer for Windows)
        cache=cache_val,        # Use RAM/Disk caching if requested
        
        # Optimization
        patience=20,           # Early stopping patience
        lr0=0.01,              # Initial learning rate
        lrf=0.01,              # Final learning rate factor
        
        # Augmentation (good for fecal images)
        hsv_h=0.015,           # Hue augmentation
        hsv_s=0.7,             # Saturation augmentation
        hsv_v=0.4,             # Value augmentation
        degrees=15.0,          # Rotation
        flipud=0.5,            # Vertical flip
        fliplr=0.5,            # Horizontal flip
        mosaic=1.0,            # Mosaic augmentation
        
        # Logging
        plots=True,
        save=True,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: runs/detect/train/weights/best.pt")
    print(f"Last model: runs/detect/train/weights/last.pt")
    print(f"\nTo deploy:")
    print(f"  cp runs/detect/train/weights/best.pt ../../application/yolov8_custom.pt")
    print("=" * 60)
    
    # Validate on test set
    print("\nRunning validation on test set...")
    model_best = YOLO("runs/detect/train/weights/best.pt")
    metrics = model_best.val(data=args.data, split="test")
    
    print(f"\nTest Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n for Tokkatot")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device: auto, cuda, cpu, 0, 1, etc.")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--cache", type=str, default="False", help="Cache images for faster training (True, False, 'ram', 'disk')")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    args = parser.parse_args()
    
    # Resolve device
    if args.device == "auto":
        import torch
        args.device = "0" if torch.cuda.is_available() else "cpu"
    
    train(args)


if __name__ == "__main__":
    main()
