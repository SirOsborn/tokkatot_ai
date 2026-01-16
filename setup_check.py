"""
Quick start script to verify installation and setup.
"""

import sys

def check_imports():
    """Check if all required packages are installed."""
    required = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'matplotlib',
        'seaborn',
        'sklearn',
        'pandas',
        'tqdm'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -e .")
        return False
    else:
        print("\n✓ All packages installed!")
        return True


def check_data():
    """Check if dataset is present."""
    from pathlib import Path
    
    data_dir = Path('archive/data')
    
    if not data_dir.exists():
        print(f"\n⚠️  Data directory not found: {data_dir}")
        print("Please ensure your dataset is in the correct location.")
        return False
    
    splits = ['train', 'val', 'test']
    classes = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
    
    all_good = True
    for split in splits:
        split_dir = data_dir / split
        if split_dir.exists():
            print(f"✓ {split}/")
            for cls in classes:
                cls_dir = split_dir / cls
                if cls_dir.exists():
                    count = len(list(cls_dir.glob('*.[jp][pn]g'))) + len(list(cls_dir.glob('*.jpeg')))
                    print(f"  └─ {cls}: {count} images")
                else:
                    print(f"  └─ {cls}: NOT FOUND")
                    all_good = False
        else:
            print(f"✗ {split}/ - NOT FOUND")
            all_good = False
    
    return all_good


def main():
    print("="*60)
    print("TOKKATOT AI - SETUP VERIFICATION")
    print("="*60)
    
    print("\n1. Checking Python packages...")
    print("-"*60)
    packages_ok = check_imports()
    
    print("\n2. Checking dataset...")
    print("-"*60)
    data_ok = check_data()
    
    print("\n" + "="*60)
    if packages_ok and data_ok:
        print("✓ SETUP COMPLETE - Ready to train!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Start training: python main.py train")
        print("  2. Monitor with TensorBoard: tensorboard --logdir outputs/logs")
        print("  3. Test inference: python main.py test <image_path>")
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("="*60)
        if not packages_ok:
            print("  → Install packages: pip install -e .")
        if not data_ok:
            print("  → Verify dataset structure in archive/data/")


if __name__ == '__main__':
    main()
