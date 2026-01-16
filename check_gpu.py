"""
GPU availability check for Tokkatot AI training.
Run this before starting training to verify CUDA setup.
"""

import torch
import sys


def check_gpu():
    """Check GPU availability and specifications."""
    print("="*60)
    print("GPU AVAILABILITY CHECK")
    print("="*60)
    
    # Check PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        # Get GPU details
        for i in range(num_gpus):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Test GPU computation
        print("\n" + "-"*60)
        print("Testing GPU Computation...")
        print("-"*60)
        
        try:
            # Create test tensor
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(test_tensor, test_tensor)
            
            print("✓ GPU computation test PASSED")
            print(f"  Test tensor shape: {result.shape}")
            print(f"  Device: {result.device}")
            
            # Test mixed precision (important for training)
            print("\nTesting Mixed Precision (AMP)...")
            from torch.cuda.amp import autocast
            
            with autocast():
                test_fp16 = torch.randn(1000, 1000).cuda()
                result_fp16 = torch.matmul(test_fp16, test_fp16)
            
            print("✓ Mixed precision (AMP) SUPPORTED")
            print("  This will speed up training significantly!")
            
        except Exception as e:
            print(f"✗ GPU computation test FAILED: {e}")
            return False
        
        # Memory test
        print("\n" + "-"*60)
        print("GPU Memory Status:")
        print("-"*60)
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Available: {total - allocated:.2f} GB")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS FOR TRAINING")
        print("="*60)
        
        if total < 4:
            print("⚠️  Low GPU memory detected (<4GB)")
            print("   → Recommended batch_size: 8-16")
            print("   → Consider using CPU for EfficientNetB0 only")
        elif total < 8:
            print("✓ Moderate GPU memory (4-8GB)")
            print("   → Recommended batch_size: 16-32")
            print("   → Should handle both models fine")
        else:
            print("✓ High GPU memory (>8GB)")
            print("   → Recommended batch_size: 32-64")
            print("   → Excellent for ensemble training")
        
        print("\n✓ GPU is ready for training!")
        print("  Run: python main.py train")
        
        return True
        
    else:
        print("\n⚠️  No CUDA-capable GPU detected")
        print("\nTraining will use CPU (much slower)")
        print("\nOptions:")
        print("  1. Install CUDA-enabled PyTorch:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("\n  2. Continue with CPU (not recommended for full training)")
        print("     → Use smaller batch_size (4-8)")
        print("     → Training will take much longer")
        
        # Test CPU computation
        print("\nTesting CPU computation...")
        try:
            test_tensor = torch.randn(100, 100)
            result = torch.matmul(test_tensor, test_tensor)
            print("✓ CPU computation works")
            print(f"  Device: {result.device}")
        except Exception as e:
            print(f"✗ CPU computation failed: {e}")
            return False
        
        return False


def main():
    """Main function."""
    success = check_gpu()
    
    print("\n" + "="*60)
    if success:
        print("STATUS: ✓ READY TO TRAIN WITH GPU")
        sys.exit(0)
    else:
        print("STATUS: ⚠️  GPU NOT AVAILABLE (CPU mode)")
        sys.exit(1)


if __name__ == '__main__':
    main()
