# -----------------------------
# run_xgboost.py
# Train XGBoost model to predict response label
# Calculate ROC AUC curve
# -----------------------------

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# SET PATHS + VARIABLES
# -----------------------------
INPUT_H5AD = "xgboost_input_data/melanoma_tcells_labeled.h5ad"
OUTPUT_DIR = "xgboost_output/"
ROC_PLOT = os.path.join(OUTPUT_DIR, "roc_auc_curve.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA + CELL EMBEDDINGS
# -----------------------------
print("Loading data...")
adata = ad.read_h5ad(INPUT_H5AD)
print(f"AnnData shape : {adata.shape}  (cells x embedding dims)")
print(f"Samples       : {adata.obs['sample'].nunique()}")
print(f"Response      : {adata.obs['response'].value_counts().to_dict()}")

X = adata.X
y = (adata.obs["response"] == "Responder").astype(int).values
samples = adata.obs["sample"].values
cell_ids = adata.obs_names.tolist()
unique_samples = adata.obs["sample"].unique()

# -----------------------------
# XGBOOST LOO CV (TRAIN)
# -----------------------------
print("Training XGBoost with leave-one-sample-out CV...")

cell_results = []
 
for i, held_out in enumerate(unique_samples):
    print(f"  [{i+1}/{len(unique_samples)}] {held_out}      ", end="\r")
 
    train = samples != held_out
    test  = samples == held_out
 
    n_pos = y[train].sum()
    n_neg = (1 - y[train]).sum()
 
    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.01,
        n_estimators=25,
        objective="binary:logistic",
        scale_pos_weight=n_neg / n_pos,
        eval_metric="logloss",
        verbosity=0
    )

    model.fit(X[train], y[train])
 
    probs        = model.predict_proba(X[test])[:, 1]
    true_label   = y[test][0]
    test_cell_ids = [c for c, m in zip(cell_ids, test) if m]
 
    for cell_id, prob in zip(test_cell_ids, probs):
        cell_results.append({
            "cell_id":        cell_id,
            "sample":         held_out,
            "true_response":  true_label,
            "prob_responder": prob,
        })
 
print(f"\nDone. {len(cell_results)} cells predicted.")
 
# -----------------------------
# SAVING CELL-LEVEL PREDICTIONS
# -----------------------------
print("Saving cell-level predictions...")

cell_df = pd.DataFrame(cell_results).set_index("cell_id")
cell_df.to_csv(os.path.join(OUTPUT_DIR, "cell_predictions.csv"))
print("Cell predictions saved.")
 
# -----------------------------
# CALCULATING SAMPLE-LEVEL SCORES --- FIX DOUBLE CHECK THIS!!!!
# -----------------------------
print("Aggregating to sample-level scores")
 
sample_df = cell_df.groupby("sample").agg(
    true_response=("true_response", "first"),
    sample_score=("prob_responder", "mean"),
    n_cells=("prob_responder", "count"),
).reset_index()
sample_df.to_csv(os.path.join(OUTPUT_DIR, "sample_predictions.csv"), index=False)
print("\nSample-level predictions:")
print(sample_df.to_string(index=False))

# -----------------------------
# ROC AUC CURVE
# -----------------------------
print("calculating ROC AUC curve...")
 
fpr, tpr, _ = roc_curve(sample_df["true_response"], sample_df["sample_score"])
roc_auc = auc(fpr, tpr)
print(f"\nAUC: {roc_auc:.4f}")
 
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC-AUC — Geneformer + XGBoost\nMelanoma ICI Response Prediction", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_auc_curve.png"), dpi=300)
plt.show()
print("ROC curve saved.")