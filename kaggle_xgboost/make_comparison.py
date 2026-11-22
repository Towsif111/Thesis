import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    ('LightGBM', 'E:/Dataset/kaggle_lightgbm/output/results.json', False),
    ('Logistic Regression', 'E:/Dataset/kaggle_logreg/output/results.json', True),
    ('CatBoost', 'E:/Dataset/kaggle_catboost/output/results.json', False),
    ('XGBoost', 'E:/Dataset/kaggle_xgboost/output/xgboost/results.json', False),
]

names = []
accs, macros, weighteds = [], [], []
for name, path, check_converged in MODELS:
    with open(path) as f:
        r = json.load(f)
    label = name
    if check_converged and r.get('converged') is False:
        label = f'{name}\n(non-converged)'
    names.append(label)
    accs.append(r['accuracy'])
    macros.append(r['macro_f1'])
    weighteds.append(r['weighted_f1'])
    print(f"{name}: accuracy={r['accuracy']}, macro_f1={r['macro_f1']}, weighted_f1={r['weighted_f1']}")

x = np.arange(len(names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width, accs, width, label='Accuracy', color='#4C72B0')
b2 = ax.bar(x, macros, width, label='Macro-F1', color='#55A868')
b3 = ax.bar(x + width, weighteds, width, label='Weighted-F1', color='#C44E52')

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f'{h:.4f}',
                 ha='center', va='bottom', fontsize=8, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Model Comparison on GUIDE Validation Set (actual saved results)')
ax.legend(loc='lower right')
plt.tight_layout()
OUT = 'E:/Dataset/kaggle_xgboost/output/xgboost/model_comparison.png'
plt.savefig(OUT, dpi=150)
print('Saved', OUT)
