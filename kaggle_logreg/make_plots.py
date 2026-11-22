import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'output'

with open(f'{OUT_DIR}/results.json') as f:
    results = json.load(f)

# 1. Validation metrics bar chart
metrics = {
    'Accuracy': results['accuracy'],
    'Macro-F1': results['macro_f1'],
    'Weighted-F1': results['weighted_f1'],
}
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(metrics.keys(), metrics.values(), color=['#4C72B0', '#55A868', '#C44E52'])
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Logistic Regression Validation Metrics (GUIDE dataset)')
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/validation_metrics.png', dpi=150)
plt.close(fig)
print('Saved validation_metrics.png')

# 2. Per-class F1 bar chart (useful here since macro/weighted F1 diverge a lot)
classes = results['target_classes_order']
per_class = results['per_class_report']
f1_scores = [per_class[c]['f1-score'] for c in classes]
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(classes, f1_scores, color='#8172B2')
ax.set_ylim(0, 1)
ax.set_ylabel('F1-score')
ax.set_title('Logistic Regression Per-Class F1 (validation)')
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/per_class_f1.png', dpi=150)
plt.close(fig)
print('Saved per_class_f1.png')
