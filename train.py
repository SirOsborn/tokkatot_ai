"""
Training script for ensemble chicken disease detection.
Trains EfficientNetB0 and DenseNet121 independently with focus on recall.
"""

import os
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    recall_score,
    precision_score,
    f1_score
)

from models import EfficientNetB0Classifier, DenseNet121Classifier, create_ensemble
from data_utils import create_dataloaders, CLASS_NAMES, IDX_TO_CLASS


class FocalLoss(nn.Module):
    """
    Focal Loss to focus training on hard examples and rare classes.
    Helps improve recall by paying more attention to disease cases.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class RecallFocusedLoss(nn.Module):
    """
    Custom loss that heavily penalizes false negatives (diseased → healthy).
    Combines CrossEntropy with asymmetric penalty.
    """
    
    def __init__(self, class_weights=None, false_negative_penalty=5.0):
        super(RecallFocusedLoss, self).__init__()
        self.class_weights = class_weights
        self.fn_penalty = false_negative_penalty
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    
    def forward(self, inputs, targets):
        base_loss = self.ce_loss(inputs, targets)
        
        # Get predictions
        _, preds = torch.max(inputs, 1)
        
        # Find false negatives: predicted healthy (1) but actually diseased
        healthy_idx = 1
        false_negatives = (preds == healthy_idx) & (targets != healthy_idx)
        
        # Apply extra penalty to false negatives
        loss = torch.where(
            false_negatives,
            base_loss * self.fn_penalty,
            base_loss
        )
        
        return loss.mean()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Calculate recall (most important metric)
    epoch_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_recall


def validate(model, dataloader, criterion, device, epoch, writer=None, phase='Val'):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [{phase}]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Calculate comprehensive metrics
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Per-class recall (critical for disease detection)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'recall_per_class': recall_per_class,
        'predictions': all_preds,
        'targets': all_targets
    }


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    num_epochs,
    device,
    class_weights,
    save_dir,
    learning_rate=1e-4,
    use_focal_loss=True,
    use_recall_focus=True,
    resume_from=None
):
    """
    Train a single model with focus on maximizing recall.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    
    # Setup loss function with recall focus
    if use_recall_focus:
        criterion = RecallFocusedLoss(
            class_weights=class_weights.to(device),
            false_negative_penalty=5.0
        )
        print("Using RecallFocusedLoss (5x penalty for false negatives)")
    elif use_focal_loss:
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
        print("Using FocalLoss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using CrossEntropyLoss")
    
    # Optimizer with different learning rates for backbone and classifier
    optimizer = optim.AdamW([
        {'params': model.backbone.features.parameters() if hasattr(model.backbone, 'features') else model.backbone.parameters(), 
         'lr': learning_rate * 0.1},  # Lower LR for pretrained backbone
        {'params': model.backbone.classifier.parameters(), 'lr': learning_rate}
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_recall = 0.0
    best_epoch = 0
    patience_counter = 0
    
    if resume_from is not None:
        print(f"\n📂 Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_recall = checkpoint['recall']
        best_epoch = checkpoint['epoch']
        print(f"✓ Resumed from epoch {checkpoint['epoch']}")
        print(f"✓ Best recall so far: {best_recall:.4f}")
    
    # Tensorboard
    log_dir = save_dir / 'logs' / model_name / datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir)
    
    # Training loop
    patience = 15
    
    for epoch in range(start_epoch, num_epochs + 1):
        # Train
        train_loss, train_acc, train_recall = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer, 'Val')
        
        # Logging
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Recall: {train_recall:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
              f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        print("\nPer-class Recall (Val):")
        for idx, recall_val in enumerate(val_metrics['recall_per_class']):
            print(f"  {IDX_TO_CLASS[idx]}: {recall_val:.4f}")
        
        # Tensorboard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        
        for idx, recall_val in enumerate(val_metrics['recall_per_class']):
            writer.add_scalar(f'Recall_PerClass/{IDX_TO_CLASS[idx]}', recall_val, epoch)
        
        # Learning rate scheduling based on recall
        scheduler.step(val_metrics['recall'])
        
        # Save best model based on recall
        if val_metrics['recall'] > best_recall:
            best_recall = val_metrics['recall']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'recall': val_metrics['recall'],
                'accuracy': val_metrics['accuracy'],
                'f1': val_metrics['f1'],
                'recall_per_class': val_metrics['recall_per_class']
            }
            
            checkpoint_path = save_dir / 'checkpoints' / f'{model_name}_best.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model (Recall: {best_recall:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
        
        print(f"Best Recall: {best_recall:.4f} (Epoch {best_epoch})")
    
    writer.close()
    
    # Load best model for return
    best_checkpoint = torch.load(save_dir / 'checkpoints' / f'{model_name}_best.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print(f"\n✓ {model_name} training complete!")
    print(f"Best Recall: {best_recall:.4f} at epoch {best_epoch}")
    
    return model, best_recall


def main():
    """Main training pipeline."""
    import sys
    
    # Check for resume flag
    resume_training = '--resume' in sys.argv
    
    # Configuration
    DATA_DIR = 'archive/data'
    SAVE_DIR = Path('outputs')
    SAVE_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    IMG_SIZE = 224
    NUM_WORKERS = 4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE
    )
    
    # Check for existing checkpoints
    efficientnet_checkpoint = SAVE_DIR / 'checkpoints' / 'EfficientNetB0_best.pth'
    densenet_checkpoint = SAVE_DIR / 'checkpoints' / 'DenseNet121_best.pth'
    
    resume_efficientnet = efficientnet_checkpoint if resume_training and efficientnet_checkpoint.exists() else None
    resume_densenet = densenet_checkpoint if resume_training and densenet_checkpoint.exists() else None
    
    if resume_training:
        print("\n" + "="*60)
        print("RESUME MODE ENABLED")
        print("="*60)
        if resume_efficientnet:
            print(f"✓ Found EfficientNetB0 checkpoint: {efficientnet_checkpoint}")
        else:
            print(f"⚠️  No EfficientNetB0 checkpoint found - will train from scratch")
        if resume_densenet:
            print(f"✓ Found DenseNet121 checkpoint: {densenet_checkpoint}")
        else:
            print(f"⚠️  No DenseNet121 checkpoint found - will train from scratch")
        print("="*60)
    
    # Train EfficientNetB0
    efficientnet = EfficientNetB0Classifier(num_classes=len(CLASS_NAMES))
    efficientnet, efficientnet_recall = train_model(
        model=efficientnet,
        model_name='EfficientNetB0',
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
        class_weights=class_weights,
        save_dir=SAVE_DIR,
        learning_rate=LEARNING_RATE,
        use_recall_focus=True,
        resume_from=resume_efficientnet
    )
    
    # Train DenseNet121
    densenet = DenseNet121Classifier(num_classes=len(CLASS_NAMES))
    densenet, densenet_recall = train_model(
        model=densenet,
        model_name='DenseNet121',
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
        class_weights=class_weights,
        save_dir=SAVE_DIR,
        learning_rate=LEARNING_RATE,
        use_recall_focus=True,
        resume_from=resume_densenet
    )
    
    # Create and save ensemble
    print("\n" + "="*60)
    print("Creating Ensemble Model")
    print("="*60)
    
    ensemble = create_ensemble(num_classes=len(CLASS_NAMES))
    ensemble.efficientnet.load_state_dict(efficientnet.state_dict())
    ensemble.densenet.load_state_dict(densenet.state_dict())
    
    # Save ensemble
    ensemble_path = SAVE_DIR / 'ensemble_model.pth'
    torch.save({
        'efficientnet_state_dict': efficientnet.state_dict(),
        'densenet_state_dict': densenet.state_dict(),
        'efficientnet_recall': efficientnet_recall,
        'densenet_recall': densenet_recall
    }, ensemble_path)
    
    print(f"\n✓ Ensemble model saved to {ensemble_path}")
    print(f"  EfficientNetB0 Recall: {efficientnet_recall:.4f}")
    print(f"  DenseNet121 Recall: {densenet_recall:.4f}")
    
    # Evaluate ensemble on test set
    print("\n" + "="*60)
    print("Evaluating Ensemble on Test Set")
    print("="*60)
    
    ensemble.to(device)
    ensemble.eval()
    
    all_preds = []
    all_targets = []
    isolation_count = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            results = ensemble.predict_with_safety_vote(
                inputs,
                healthy_threshold=0.80,
                uncertainty_threshold=0.50
            )
            
            preds = results['prediction'].cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            isolation_count += results['should_isolate'].sum().item()
    
    # Filter out isolated samples for metrics
    valid_mask = np.array(all_preds) != -1
    valid_preds = np.array(all_preds)[valid_mask]
    valid_targets = np.array(all_targets)[valid_mask]
    
    if len(valid_preds) > 0:
        print("\nTest Set Metrics (excluding isolated samples):")
        print(classification_report(valid_targets, valid_preds, target_names=CLASS_NAMES, zero_division=0))
        
        print(f"\nIsolation Statistics:")
        print(f"  Total samples: {len(all_preds)}")
        print(f"  Isolated: {isolation_count} ({100*isolation_count/len(all_preds):.2f}%)")
        print(f"  Classified: {len(valid_preds)} ({100*len(valid_preds)/len(all_preds):.2f}%)")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
