"""
Create evaluation visualizations for the technical report.
Generates comparison charts, per-class metrics, and isolation statistics.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for academic papers
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Output directory
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

# Data from evaluation report
classes = ['Coccidiosis', 'Healthy', 'New Castle\nDisease', 'Salmonella']
classes_short = ['Coccidiosis', 'Healthy', 'NCD', 'Salmonella']

# Individual model metrics (on full test set)
efficientnet_metrics = {
    'precision': [0.98, 0.98, 0.97, 0.98],
    'recall': [0.99, 0.96, 0.98, 0.98],
    'f1': [0.99, 0.97, 0.98, 0.98],
    'accuracy': 0.98
}

densenet_metrics = {
    'precision': [0.97, 0.98, 0.94, 0.97],
    'recall': [0.98, 0.93, 0.98, 0.97],
    'f1': [0.98, 0.96, 0.96, 0.97],
    'accuracy': 0.97
}

# Ensemble metrics (on classified samples only)
ensemble_metrics = {
    'precision': [0.99, 1.00, 0.99, 0.99],
    'recall': [1.00, 0.98, 1.00, 1.00],
    'f1': [0.99, 0.99, 0.99, 1.00],
    'accuracy': 0.99
}

# Isolation statistics
total_samples = 70677
isolated = 3540
classified = 67137

# ==================== FIGURE 1: Model Performance Comparison ====================
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

models = ['EfficientNetB0', 'DenseNet121', 'Ensemble\n(Classified)']
accuracies = [efficientnet_metrics['accuracy'], 
              densenet_metrics['accuracy'], 
              ensemble_metrics['accuracy']]
precisions = [np.mean(efficientnet_metrics['precision']),
              np.mean(densenet_metrics['precision']),
              np.mean(ensemble_metrics['precision'])]
recalls = [np.mean(efficientnet_metrics['recall']),
           np.mean(densenet_metrics['recall']),
           np.mean(ensemble_metrics['recall'])]
f1_scores = [np.mean(efficientnet_metrics['f1']),
             np.mean(densenet_metrics['f1']),
             np.mean(ensemble_metrics['f1'])]

x = np.arange(len(models))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#2E86AB')
bars2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#A23B72')
bars3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#F18F01')
bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#C73E1D')

ax.set_xlabel('Model', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Overall Model Performance Comparison', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='lower right')
ax.set_ylim([0.92, 1.01])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(output_dir / 'model_performance_comparison.png', bbox_inches='tight')
plt.savefig(output_dir / 'model_performance_comparison.svg', bbox_inches='tight')
print("✓ Created: model_performance_comparison.png")
plt.close()

# ==================== FIGURE 2: Per-Class Recall Comparison ====================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

x = np.arange(len(classes))
width = 0.25

bars1 = ax.bar(x - width, efficientnet_metrics['recall'], width, 
               label='EfficientNetB0', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x, densenet_metrics['recall'], width, 
               label='DenseNet121', color='#A23B72', alpha=0.8)
bars3 = ax.bar(x + width, ensemble_metrics['recall'], width, 
               label='Ensemble', color='#F18F01', alpha=0.8)

ax.set_xlabel('Disease Class', fontweight='bold')
ax.set_ylabel('Recall', fontweight='bold')
ax.set_title('Per-Class Recall Comparison (Critical Metric for Disease Detection)', 
             fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(loc='lower right')
ax.set_ylim([0.90, 1.02])
ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5, label='95% Threshold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'per_class_recall_comparison.png', bbox_inches='tight')
plt.savefig(output_dir / 'per_class_recall_comparison.svg', bbox_inches='tight')
print("✓ Created: per_class_recall_comparison.png")
plt.close()

# ==================== FIGURE 3: Isolation Statistics ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
colors = ['#2E86AB', '#F18F01']
explode = (0, 0.05)
sizes = [classified, isolated]
labels = [f'Classified\n{classified:,}\n({classified/total_samples*100:.2f}%)',
          f'Isolated\n{isolated:,}\n({isolated/total_samples*100:.2f}%)']

wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                     autopct='', startangle=90, textprops={'fontsize': 11})
ax1.set_title('Sample Classification Distribution', fontweight='bold', pad=15)

# Bar chart with details
categories = ['Total\nSamples', 'Classified\n(High Confidence)', 'Isolated\n(Uncertainty)']
values = [total_samples, classified, isolated]
colors_bar = ['#6C757D', '#2E86AB', '#F18F01']

bars = ax2.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1)
ax2.set_ylabel('Number of Samples', fontweight='bold')
ax2.set_title('Ensemble Safety Routing Statistics', fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'isolation_statistics.png', bbox_inches='tight')
plt.savefig(output_dir / 'isolation_statistics.svg', bbox_inches='tight')
print("✓ Created: isolation_statistics.png")
plt.close()

# ==================== FIGURE 4: Comprehensive Metrics Heatmap ====================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create a cleaner version with proper rows
rows = ['EfficientNet - Precision', 'EfficientNet - Recall', 'EfficientNet - F1',
        'DenseNet - Precision', 'DenseNet - Recall', 'DenseNet - F1',
        'Ensemble - Precision', 'Ensemble - Recall', 'Ensemble - F1']
cols = classes_short + ['Average']

data_clean = [
    efficientnet_metrics['precision'] + [np.mean(efficientnet_metrics['precision'])],
    efficientnet_metrics['recall'] + [np.mean(efficientnet_metrics['recall'])],
    efficientnet_metrics['f1'] + [np.mean(efficientnet_metrics['f1'])],
    densenet_metrics['precision'] + [np.mean(densenet_metrics['precision'])],
    densenet_metrics['recall'] + [np.mean(densenet_metrics['recall'])],
    densenet_metrics['f1'] + [np.mean(densenet_metrics['f1'])],
    ensemble_metrics['precision'] + [np.mean(ensemble_metrics['precision'])],
    ensemble_metrics['recall'] + [np.mean(ensemble_metrics['recall'])],
    ensemble_metrics['f1'] + [np.mean(ensemble_metrics['f1'])],
]

im = ax.imshow(data_clean, cmap='RdYlGn', aspect='auto', vmin=0.92, vmax=1.0)

# Set ticks and labels
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(rows)))
ax.set_xticklabels(cols)
ax.set_yticklabels(rows)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(rows)):
    for j in range(len(cols)):
        text = ax.text(j, i, f'{data_clean[i][j]:.3f}',
                      ha="center", va="center", color="black", fontsize=8, fontweight='bold')

ax.set_title('Comprehensive Performance Metrics Across All Models', 
             fontweight='bold', pad=15, fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score', rotation=270, labelpad=15, fontweight='bold')

# Add horizontal lines to separate models
ax.axhline(y=2.5, color='black', linewidth=2)
ax.axhline(y=5.5, color='black', linewidth=2)

plt.tight_layout()
plt.savefig(output_dir / 'metrics_heatmap.png', bbox_inches='tight')
plt.savefig(output_dir / 'metrics_heatmap.svg', bbox_inches='tight')
print("✓ Created: metrics_heatmap.png")
plt.close()

# ==================== FIGURE 5: Recall Improvement Visualization ====================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Show recall improvements
improvement_data = []
for i, class_name in enumerate(classes_short):
    base_recall = min(efficientnet_metrics['recall'][i], densenet_metrics['recall'][i])
    ensemble_recall = ensemble_metrics['recall'][i]
    improvement = (ensemble_recall - base_recall) * 100
    improvement_data.append({
        'class': class_name,
        'base': base_recall,
        'ensemble': ensemble_recall,
        'improvement': improvement
    })

x = np.arange(len(classes_short))
width = 0.35

base_recalls = [d['base'] for d in improvement_data]
ensemble_recalls = [d['ensemble'] for d in improvement_data]

bars1 = ax.bar(x - width/2, base_recalls, width, label='Best Individual Model', 
               color='#6C757D', alpha=0.7)
bars2 = ax.bar(x + width/2, ensemble_recalls, width, label='Ensemble', 
               color='#2E86AB', alpha=0.9)

ax.set_ylabel('Recall Score', fontweight='bold')
ax.set_xlabel('Disease Class', fontweight='bold')
ax.set_title('Recall Improvement: Individual Models vs. Ensemble', 
             fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes_short)
ax.legend()
ax.set_ylim([0.92, 1.02])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels and improvement arrows
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    h1 = bar1.get_height()
    h2 = bar2.get_height()
    
    ax.text(bar1.get_x() + bar1.get_width()/2., h1 + 0.003,
            f'{h1:.3f}', ha='center', va='bottom', fontsize=8)
    ax.text(bar2.get_x() + bar2.get_width()/2., h2 + 0.003,
            f'{h2:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Draw improvement arrow if there's improvement
    if h2 > h1:
        ax.annotate('', xy=(i + width/2, h2 - 0.003), xytext=(i - width/2, h1 + 0.003),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

plt.tight_layout()
plt.savefig(output_dir / 'recall_improvement.png', bbox_inches='tight')
plt.savefig(output_dir / 'recall_improvement.svg', bbox_inches='tight')
print("✓ Created: recall_improvement.png")
plt.close()

print("\n" + "="*60)
print("All evaluation visualizations created successfully!")
print("="*60)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated files:")
print("  • model_performance_comparison.png/svg")
print("  • per_class_recall_comparison.png/svg")
print("  • isolation_statistics.png/svg")
print("  • metrics_heatmap.png/svg")
print("  • recall_improvement.png/svg")
print("\nExisting confusion matrices:")
print("  • confusion_matrix_efficientnet.png")
print("  • confusion_matrix_densenet.png")
print("  • confusion_matrix_ensemble.png")
