import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'output'

with open(f'{OUT_DIR}/results.json') as f:
    results = json.load(f)

model = joblib.load(f'{OUT_DIR}/lightgbm_model.pkl')
booster = model.booster_

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
ax.set_title('LightGBM Validation Metrics (GUIDE dataset)')
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/validation_metrics.png', dpi=150)
plt.close(fig)
print('Saved validation_metrics.png')

# 2. Top-20 feature importance (gain and split), side by side
feature_names = booster.feature_name()
gain_importance = booster.feature_importance(importance_type='gain')
split_importance = booster.feature_importance(importance_type='split')

def top_n(names, values, n=20):
    idx = np.argsort(values)[::-1][:n]
    return [names[i] for i in idx], [values[i] for i in idx]

gain_names, gain_vals = top_n(feature_names, gain_importance)
split_names, split_vals = top_n(feature_names, split_importance)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].barh(gain_names[::-1], gain_vals[::-1], color='#4C72B0')
axes[0].set_title('Top 20 Features — Gain Importance\n(total loss reduction contributed)')
axes[0].set_xlabel('Total gain')

axes[1].barh(split_names[::-1], split_vals[::-1], color='#DD8452')
axes[1].set_title('Top 20 Features — Split Importance\n(number of times used to split)')
axes[1].set_xlabel('Number of splits')

fig.suptitle('LightGBM Feature Importance (GUIDE dataset, validation-tuned model)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/feature_importance_top20.png', dpi=150)
plt.close(fig)
print('Saved feature_importance_top20.png')

print('\nTop 10 by gain:')
for n, v in list(zip(gain_names, gain_vals))[:10]:
    print(f'  {n}: {v:.1f}')

print('\nTop 10 by split count:')
for n, v in list(zip(split_names, split_vals))[:10]:
    print(f'  {n}: {v:.0f}')
