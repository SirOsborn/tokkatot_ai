import torch

ckpt = torch.load('outputs/checkpoints/EfficientNetB0_best.pth', map_location='cpu')
print('✓ EfficientNetB0 checkpoint loaded successfully!\n')
print('Checkpoint contents:')
for k, v in ckpt.items():
    if k == 'recall_per_class':
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: {type(v).__name__}')

print(f'\n📊 Training Progress:')
print(f'  Epoch completed: {ckpt["epoch"]}')
print(f'  Best Recall: {ckpt["recall"]:.4f}')
print(f'  Accuracy: {ckpt["accuracy"]:.4f}')
print(f'  F1 Score: {ckpt["f1"]:.4f}')
print(f'\n✓ Model is ready to resume or use for inference!')
