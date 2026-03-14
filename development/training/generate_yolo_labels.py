#!/usr/bin/env python3
"""
Generate YOLO-format labels from classification folder structure.

Maps folder-based classification dataset to YOLO bounding box labels:
  - Healthy/         → class 0 (healthy_feces)
  - Coccidiosis/     → class 1 (suspicious_feces)
  - Salmonella/      → class 1 (suspicious_feces)
  - New Castle Disease/ → class 1 (suspicious_feces)

Since each image contains a single fecal sample filling the frame,
bounding boxes are set to cover the full image (0.5 0.5 1.0 1.0).

Usage:
    python generate_yolo_labels.py
    python generate_yolo_labels.py --data-dir ../archive/data --margin 0.05
"""

import argparse
from pathlib import Path

# Class mapping: folder name → YOLO class ID
# 0 = healthy_feces, 1 = suspicious_feces
CLASS_MAP = {
    "Healthy": 0,
    "Coccidiosis": 1,
    "Salmonella": 1,
    "New Castle Disease": 1,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def generate_labels(data_dir: Path, margin: float = 0.05):
    """
    Generate YOLO label .txt files for each image in the dataset.
    
    Args:
        data_dir: Root data directory containing train/val/test splits
        margin: Margin to inset from full image (0.05 = 5% border padding)
    """
    splits = ["train", "val", "test"]
    
    # Calculate bbox with margin
    # Full image = 0.5 0.5 1.0 1.0, with margin we reduce the box slightly
    cx, cy = 0.5, 0.5
    w = 1.0 - (2 * margin)
    h = 1.0 - (2 * margin)
    
    total_labels = 0
    stats = {"healthy_feces": 0, "suspicious_feces": 0}
    
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"⚠ Skip: {split_dir} not found")
            continue
        
        # Create labels directory alongside images
        labels_dir = data_dir / f"{split}_labels"
        
        print(f"\n{'='*50}")
        print(f"Processing: {split}/")
        print(f"{'='*50}")
        
        for class_folder, class_id in CLASS_MAP.items():
            class_dir = split_dir / class_folder
            if not class_dir.exists():
                print(f"  ⚠ Skip: {class_folder}/ not found")
                continue
            
            # Create output dir mirroring class structure
            out_dir = labels_dir / class_folder
            out_dir.mkdir(parents=True, exist_ok=True)
            
            images = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in IMAGE_EXTENSIONS]
            
            class_name = "healthy_feces" if class_id == 0 else "suspicious_feces"
            label_content = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
            
            count = 0
            for img_path in images:
                label_path = out_dir / f"{img_path.stem}.txt"
                
                # Faster check: skip if exists
                if label_path.exists():
                    count += 1
                    continue

                try:
                    # Direct open is slightly faster for 100k+ repetitions
                    with open(label_path, 'w', encoding='utf-8') as f:
                        f.write(label_content)
                    count += 1
                    stats[class_name] += 1
                except Exception as e:
                    print(f"    ✗ Error writing {label_path.name}: {e}")
                    continue
                
                # Frequent progress update for large counts
                if count % 10000 == 0:
                    print(f"    ... {count}/{len(images)} labels written")
            
            print(f"  ✓ {class_folder:>25s} → {class_name:<20s} ({count} labels total)")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total labels generated: {total_labels}")
    print(f"  healthy_feces (class 0): {stats['healthy_feces']}")
    print(f"  suspicious_feces (class 1): {stats['suspicious_feces']}")
    print(f"\nLabel format: <class_id> {cx} {cy} {w:.2f} {h:.2f}")
    print(f"Bbox margin: {margin*100:.0f}%")
    print(f"\nOutput directories:")
    for split in splits:
        labels_dir = data_dir / f"{split}_labels"
        if labels_dir.exists():
            print(f"  {labels_dir}")
    print(f"\n✓ Done! Ready for YOLO training.")


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO labels from classification folders")
    parser.add_argument("--data-dir", type=str, default="../archive/data",
                        help="Path to data directory with train/val/test splits")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Bbox margin from image edge (0.05 = 5%%)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return
    
    print(f"Data directory: {data_dir}")
    print(f"Classes: {list(CLASS_MAP.keys())}")
    print(f"Mapping: Healthy → 0, All diseases → 1")
    
    generate_labels(data_dir, args.margin)


if __name__ == "__main__":
    main()
