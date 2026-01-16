"""
Evaluation script for analyzing model performance with detailed metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from pathlib import Path
from tqdm import tqdm

from models import create_ensemble
from data_utils import create_dataloaders, CLASS_NAMES


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def evaluate_ensemble(
    model_path='outputs/ensemble_model.pth',
    data_dir='archive/data',
    save_dir='outputs/evaluation',
    healthy_threshold=0.80,
    uncertainty_threshold=0.50
):
    """
    Comprehensive evaluation of ensemble model.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading ensemble model...")
    ensemble = create_ensemble(num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location=device)
    ensemble.efficientnet.load_state_dict(checkpoint['efficientnet_state_dict'])
    ensemble.densenet.load_state_dict(checkpoint['densenet_state_dict'])
    ensemble.to(device)
    ensemble.eval()
    print("✓ Model loaded\n")
    
    # Load test data
    print("Loading test dataset...")
    _, _, test_loader, _ = create_dataloaders(data_dir, batch_size=32, num_workers=4)
    print(f"✓ Test set loaded: {len(test_loader.dataset)} images\n")
    
    # Evaluation
    print("="*60)
    print("EVALUATING ENSEMBLE MODEL")
    print("="*60)
    
    all_targets = []
    ensemble_preds = []
    efficient_preds = []
    dense_preds = []
    
    ensemble_probs = []
    efficient_probs = []
    dense_probs = []
    
    isolation_flags = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            
            # Ensemble prediction
            results = ensemble.predict_with_safety_vote(
                inputs,
                healthy_threshold=healthy_threshold,
                uncertainty_threshold=uncertainty_threshold
            )
            
            # Get individual model predictions
            efficient_logits, dense_logits = ensemble(inputs)
            efficient_prob = torch.softmax(efficient_logits, dim=1)
            dense_prob = torch.softmax(dense_logits, dim=1)
            
            _, eff_pred = torch.max(efficient_prob, dim=1)
            _, den_pred = torch.max(dense_prob, dim=1)
            
            # Store results
            all_targets.extend(targets.cpu().numpy())
            ensemble_preds.extend(results['prediction'].cpu().numpy())
            efficient_preds.extend(eff_pred.cpu().numpy())
            dense_preds.extend(den_pred.cpu().numpy())
            
            ensemble_probs.append(results['efficientnet_probs'].cpu().numpy())
            efficient_probs.append(efficient_prob.cpu().numpy())
            dense_probs.append(dense_prob.cpu().numpy())
            
            isolation_flags.extend(results['should_isolate'].cpu().numpy())
    
    # Convert to arrays
    all_targets = np.array(all_targets)
    ensemble_preds = np.array(ensemble_preds)
    efficient_preds = np.array(efficient_preds)
    dense_preds = np.array(dense_preds)
    isolation_flags = np.array(isolation_flags)
    
    # Combine probability arrays
    ensemble_probs = np.vstack(ensemble_probs)
    efficient_probs = np.vstack(efficient_probs)
    dense_probs = np.vstack(dense_probs)
    
    print("\n" + "="*60)
    print("ISOLATION STATISTICS")
    print("="*60)
    total_samples = len(all_targets)
    isolated_count = isolation_flags.sum()
    print(f"Total samples: {total_samples}")
    print(f"Isolated: {isolated_count} ({100*isolated_count/total_samples:.2f}%)")
    print(f"Classified: {total_samples - isolated_count} ({100*(total_samples-isolated_count)/total_samples:.2f}%)")
    
    # Filter out isolated samples for classification metrics
    valid_mask = ensemble_preds != -1
    valid_ensemble_preds = ensemble_preds[valid_mask]
    valid_efficient_preds = efficient_preds[valid_mask]
    valid_dense_preds = dense_preds[valid_mask]
    valid_targets = all_targets[valid_mask]
    
    # Individual model performance
    print("\n" + "="*60)
    print("EFFICIENTNET-B0 PERFORMANCE")
    print("="*60)
    print(classification_report(all_targets, efficient_preds, target_names=CLASS_NAMES, zero_division=0))
    
    cm_efficient = confusion_matrix(all_targets, efficient_preds)
    plot_confusion_matrix(
        cm_efficient,
        CLASS_NAMES,
        save_dir / 'confusion_matrix_efficientnet.png',
        'EfficientNetB0 Confusion Matrix'
    )
    
    print("\n" + "="*60)
    print("DENSENET-121 PERFORMANCE")
    print("="*60)
    print(classification_report(all_targets, dense_preds, target_names=CLASS_NAMES, zero_division=0))
    
    cm_dense = confusion_matrix(all_targets, dense_preds)
    plot_confusion_matrix(
        cm_dense,
        CLASS_NAMES,
        save_dir / 'confusion_matrix_densenet.png',
        'DenseNet121 Confusion Matrix'
    )
    
    # Ensemble performance (excluding isolated)
    if len(valid_ensemble_preds) > 0:
        print("\n" + "="*60)
        print("ENSEMBLE PERFORMANCE (Classified Samples)")
        print("="*60)
        print(classification_report(valid_targets, valid_ensemble_preds, target_names=CLASS_NAMES, zero_division=0))
        
        cm_ensemble = confusion_matrix(valid_targets, valid_ensemble_preds)
        plot_confusion_matrix(
            cm_ensemble,
            CLASS_NAMES,
            save_dir / 'confusion_matrix_ensemble.png',
            'Ensemble Confusion Matrix (Classified Only)'
        )
    
    # Model agreement analysis
    print("\n" + "="*60)
    print("MODEL AGREEMENT ANALYSIS")
    print("="*60)
    agreement_mask = efficient_preds == dense_preds
    agreement_rate = agreement_mask.sum() / len(agreement_mask)
    print(f"Models agree: {agreement_mask.sum()}/{len(agreement_mask)} ({agreement_rate*100:.2f}%)")
    
    # Agreement when both correct
    both_correct = (efficient_preds == all_targets) & (dense_preds == all_targets)
    print(f"Both correct: {both_correct.sum()}/{len(both_correct)} ({both_correct.sum()/len(both_correct)*100:.2f}%)")
    
    # False negative analysis (critical metric)
    print("\n" + "="*60)
    print("FALSE NEGATIVE ANALYSIS (Disease → Healthy)")
    print("="*60)
    healthy_idx = 1
    
    # EfficientNet false negatives
    eff_fn = (efficient_preds == healthy_idx) & (all_targets != healthy_idx)
    print(f"EfficientNet FN: {eff_fn.sum()}/{(all_targets != healthy_idx).sum()}")
    
    # DenseNet false negatives
    dense_fn = (dense_preds == healthy_idx) & (all_targets != healthy_idx)
    print(f"DenseNet FN: {dense_fn.sum()}/{(all_targets != healthy_idx).sum()}")
    
    # Ensemble false negatives (on classified samples)
    if len(valid_ensemble_preds) > 0:
        ensemble_fn = (valid_ensemble_preds == healthy_idx) & (valid_targets != healthy_idx)
        print(f"Ensemble FN (classified): {ensemble_fn.sum()}/{(valid_targets != healthy_idx).sum()}")
    
    # Check if ensemble caught the false negatives
    caught_fn = eff_fn | dense_fn  # Either model made FN
    isolated_fn = caught_fn & isolation_flags  # FN that were isolated
    print(f"\nFalse negatives caught by isolation: {isolated_fn.sum()}/{caught_fn.sum()}")
    
    # Save detailed report
    report_path = save_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ENSEMBLE MODEL EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Healthy Threshold: {healthy_threshold}\n")
        f.write(f"Uncertainty Threshold: {uncertainty_threshold}\n\n")
        
        f.write("ISOLATION STATISTICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Isolated: {isolated_count} ({100*isolated_count/total_samples:.2f}%)\n")
        f.write(f"Classified: {total_samples - isolated_count} ({100*(total_samples-isolated_count)/total_samples:.2f}%)\n\n")
        
        f.write("EFFICIENTNET-B0\n")
        f.write("-"*60 + "\n")
        f.write(classification_report(all_targets, efficient_preds, target_names=CLASS_NAMES, zero_division=0))
        f.write("\n")
        
        f.write("DENSENET-121\n")
        f.write("-"*60 + "\n")
        f.write(classification_report(all_targets, dense_preds, target_names=CLASS_NAMES, zero_division=0))
        f.write("\n")
        
        if len(valid_ensemble_preds) > 0:
            f.write("ENSEMBLE (Classified Only)\n")
            f.write("-"*60 + "\n")
            f.write(classification_report(valid_targets, valid_ensemble_preds, target_names=CLASS_NAMES, zero_division=0))
    
    print(f"\n✓ Evaluation complete! Results saved to {save_dir}")


if __name__ == '__main__':
    evaluate_ensemble()
