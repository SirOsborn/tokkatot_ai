"""
Main entry point for Tokkatot AI - Chicken Disease Detection System
Safety-First Ensemble AI using EfficientNetB0 + DenseNet121

Usage:
    python main.py train           # Train the ensemble models
    python main.py train --resume  # Resume training from checkpoint
    python main.py test <image>    # Test on a single image
    python main.py eval            # Evaluate on test set
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands:")
        print("  train                      - Train the ensemble models from scratch")
        print("  train --resume             - Resume training from last checkpoint")
        print("  train --resume --skip-efficientnet - Skip EfficientNetB0, train only DenseNet121")
        print("  train --resume --skip-densenet     - Skip DenseNet121, train only EfficientNetB0")
        print("  test <image>               - Test inference on an image")
        print("  eval                       - Evaluate ensemble on test set")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'train':
        from train import main as train_main
        train_main()
    
    elif command == 'test':
        from inference import ChickenDiseaseDetector
        
        if len(sys.argv) < 3:
            print("Usage: python main.py test <image_path>")
            return
        
        image_path = sys.argv[2]
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            return
        
        # Load detector
        detector = ChickenDiseaseDetector(
            model_path='outputs/ensemble_model.pth',
            healthy_threshold=0.80,
            uncertainty_threshold=0.50
        )
        
        # Predict with details
        result = detector.predict(image_path, return_details=True)
        
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Classification: {result['classification']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Action: {result['action']}")
        
        print(f"\nEfficientNetB0:")
        print(f"  Prediction: {result['models']['efficientnet']['prediction']}")
        print(f"  Confidence: {result['models']['efficientnet']['confidence']:.2%}")
        print(f"  Healthy Confidence: {result['models']['efficientnet']['healthy_confidence']:.2%}")
        
        print(f"\nDenseNet121:")
        print(f"  Prediction: {result['models']['densenet']['prediction']}")
        print(f"  Confidence: {result['models']['densenet']['confidence']:.2%}")
        print(f"  Healthy Confidence: {result['models']['densenet']['healthy_confidence']:.2%}")
        
        print(f"\nEnsemble:")
        print(f"  Models Agree: {result['ensemble']['agreement']}")
        print(f"  Average Confidence: {result['ensemble']['avg_confidence']:.2%}")
        
        # Display appropriate message based on classification
        if result['classification'] == "Healthy" and not result['should_isolate']:
            print("\n✓ Clear - Chicken can remain with flock")
        else:
            print(f"\n⚠️  ISOLATION REQUIRED - {result['action']}")
    
    elif command == 'eval':
        from evaluate import evaluate_ensemble
        
        # Check if ensemble model exists
        model_path = Path('outputs/ensemble_model.pth')
        if not model_path.exists():
            print("Error: Ensemble model not found!")
            print("Please train the model first: python main.py train")
            return
        
        # Run evaluation
        print("Evaluating ensemble on test set...")
        print("This may take several minutes...\n")
        
        evaluate_ensemble(
            model_path='outputs/ensemble_model.pth',
            data_dir='archive/data',
            save_dir='outputs/evaluation',
            healthy_threshold=0.80,
            uncertainty_threshold=0.50
        )
        
        print("\n" + "="*60)
        print("Evaluation complete!")
        print("Results saved to: outputs/evaluation/")
        print("="*60)
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
