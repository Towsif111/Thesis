import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

OUT_DIR = 'output'

with open(f'{OUT_DIR}/results.json') as f:
    results = json.load(f)
with open(f'{OUT_DIR}/preprocessing_info.json') as f:
    prep = json.load(f)

model = CatBoostClassifier()
model.load_model(f'{OUT_DIR}/catboost_model.cbm')

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
ax.set_title('CatBoost Validation Metrics (GUIDE dataset)')
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/validation_metrics.png', dpi=150)
plt.close(fig)
print('Saved validation_metrics.png')

# 2. Top-20 feature importance (PredictionValuesChange, CatBoost's default)
feature_names = prep['features']
importances = model.get_feature_importance()
idx = np.argsort(importances)[::-1][:20]
top_names = [feature_names[i] for i in idx]
top_vals = [importances[i] for i in idx]

fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(top_names[::-1], top_vals[::-1], color='#55A868')
ax.set_xlabel('Importance (PredictionValuesChange)')
ax.set_title('CatBoost — Top 20 Feature Importance')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/feature_importance_top20.png', dpi=150)
plt.close(fig)
print('Saved feature_importance_top20.png')

print('\nTop 10 features by importance:')
for n, v in list(zip(top_names, top_vals))[:10]:
    print(f'  {n}: {v:.2f}')
