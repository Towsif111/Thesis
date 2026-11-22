# Microsoft GUIDE Security Incident Classification — Thesis Project

Multiclass classification of security alerts into `IncidentGrade` (`TruePositive` / `BenignPositive` / `FalsePositive`) on the Microsoft GUIDE dataset, comparing four baseline models and a soft-voting hybrid ensemble.

## Dataset

[Microsoft Security Incident Prediction (GUIDE)](https://www.kaggle.com/datasets/microsoft/microsoft-security-incident-prediction) on Kaggle.

Raw CSV/XLSX files are **not included in this repository** (multi-GB; excluded via `.gitignore`). Download them from the link above and place `GUIDE_Train.csv` / `GUIDE_Test.csv` in the repository root to reproduce the notebooks.

## Project structure

Each model has its own self-contained folder: a notebook, Kaggle kernel metadata, a `push.sh` script that pushes the notebook to Kaggle / runs it / pulls results back, and an `output/` folder with the small results artifacts (JSON/CSV metrics, confusion matrices, charts). Large model binaries and per-row prediction CSVs are excluded from version control (see `.gitignore`) but are produced locally/on Kaggle when a notebook is run.

- `kaggle_notebook/` — exploratory data analysis and preprocessing audit (`guide_eda.ipynb`): leakage review, feature bucketing, column cleaning decisions used by every model below.
- `kaggle_lightgbm/` — LightGBM baseline (`guide_lightgbm.ipynb`). Builds the shared cleaned dataset and the group-aware `(OrgId, IncidentId)` train/validation split (`split_assignment.csv`) that every other notebook reuses, native categorical handling.
- `kaggle_logreg/` — Logistic Regression baseline (`guide_logreg.ipynb`). Reuses the LightGBM split; sparse one-hot encoding + scaling; `saga` solver.
- `kaggle_catboost/` — CatBoost baseline (`guide_catboost.ipynb`). Reuses the LightGBM split; native categorical handling (`cat_features`); `max_ctr_complexity=1` to avoid a categorical-combination blowup.
- `kaggle_xgboost/` — XGBoost baseline (`guide_xgboost.ipynb`). Reuses the LightGBM split; native pandas-categorical support (`enable_categorical=True`), version-adaptive early-stopping API.
- `kaggle_hybrid_ensemble/` — Equal-weight soft-voting ensemble of LightGBM + CatBoost + XGBoost (`guide_hybrid_ensemble.ipynb`). No model is retrained; reuses each model's saved validation probabilities (regenerating LightGBM's, since it hadn't been saved).
- `kaggle_modeling/` — Earlier, superseded combined-training notebook; kept for history, not part of the final comparison.

Every notebook documents its own leakage review, feature set, and limitations inline (markdown cells); treat the notebooks themselves as the primary source of methodology detail.

## How to run

Each folder follows the same pattern:

```bash
cd kaggle_<model>
./push.sh   # pushes the notebook to Kaggle, waits for it to finish, pulls output/ back
```

This requires a configured Kaggle API token (`~/.kaggle/kaggle.json`, **not** included in this repo — see [Kaggle API docs](https://www.kaggle.com/docs/api)) and `kaggle-api` installed (`pip install -r requirements.txt`). Models that depend on another model's output (every one except LightGBM) declare that dependency via `kernel_sources` in their `kernel-metadata.json`, so Kaggle mounts the upstream kernel's saved files automatically — push LightGBM first.

To regenerate a model's own charts locally from its already-pulled `output/results.json` (no retraining), run that folder's `make_plots.py`.

## Results (validation split)

All metrics below are on the **held-out validation split** (`split_assignment.csv`, group-aware on `(OrgId, IncidentId)`, built by the LightGBM notebook and reused by every other model). **None of these numbers come from the official test set**, which was never loaded by any training/ensembling notebook.

| Model | Accuracy | Macro-F1 | Weighted-F1 | TruePositive Recall |
|---|---|---|---|---|
| Logistic Regression *(did not converge — `max_iter` reached)* | 0.7364 | 0.6964 | 0.7227 | 0.6702 |
| LightGBM | 0.8060 | 0.7879 | 0.8019 | 0.7748 |
| XGBoost | 0.8062 | 0.7885 | 0.8014 | 0.7707 |
| CatBoost | 0.8110 | 0.7946 | 0.8074 | 0.7748 |
| **Hybrid (equal-weight soft voting)** | **0.8173** | **0.8013** | **0.8134** | **0.7803** |

The hybrid averages the three tree-model probabilities with fixed equal weights (1/3 each) — no weight tuning or stacking was performed. In this run it outperformed every individual model on every reported metric; that was not assumed going in, and equal weighting is not guaranteed to be optimal (see `kaggle_hybrid_ensemble/output/results.json` for full detail and `comparison.csv` for the per-model breakdown).

Logistic Regression's `max_iter=100` cap was reached before convergence (`saga` solver on ~7.5M training rows) — its numbers are included for reference but are not a fully-converged result.

## Notes on methodology

- All four individual models and the hybrid ensemble were trained/evaluated on the **exact same cleaned rows and the exact same train/validation split**, verified in every notebook via zero `(OrgId, IncidentId)` group overlap between train and validation.
- `SuspicionLevel`/`LastVerdict` are automated, pre-triage detection-system outputs kept under a documented (not independently re-verified) assumption that they are available prior to the final human `IncidentGrade` label — flagged as an open limitation in every notebook that uses them.
- `ActionGrouped`/`ActionGranular` were excluded from all models as remediation-action leakage (they describe what an analyst did after/during triage).
