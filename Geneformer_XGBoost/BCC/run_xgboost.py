# -----------------------------
# run_xgboost.py
# Run + Validate XGBoost model trained on melanoma 
# Calculate ROC AUC curve
# Basal cell carcinoma dataset - GSE123813
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
from scipy import stats

# -----------------------------
# SET PATHS + VARIABLES
# -----------------------------
INPUT_H5AD = "xgboost_input_data/bcc_tcells_labeled.h5ad"
MODEL_PATH = "../fully_trained_xgboost/results/melanoma_xgb_model.json"
OUTPUT_DIR = "xgboost_output/"
ROC_PLOT = os.path.join(OUTPUT_DIR, "roc_auc_curve.png")
CELL_PREDS = os.path.join(OUTPUT_DIR, "bcc_cell_predictions.csv")
SAMPLE_PREDS = os.path.join(OUTPUT_DIR, "bcc_sample_predictions.csv")
ROC_PLOT = os.path.join(OUTPUT_DIR, "bcc_roc_auc_curve.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA + CELL EMBEDDINGS
# -----------------------------
print("Loading bcc data...")
adata = ad.read_h5ad(INPUT_H5AD)
print(f"AnnData shape : {adata.shape}  (cells x embedding dims)")
print(f"Samples       : {adata.obs['sample'].nunique()}")
print(f"Response      : {adata.obs['response'].value_counts().to_dict()}")

X = adata.X
y = (adata.obs["response"] == "Responder").astype(int).values
samples = adata.obs["sample"].values
cell_ids = adata.obs_names.tolist()
#unique_samples = adata.obs["sample"].unique()

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
print("Loading trained XGBoost model...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print("Model loaded.")

# -----------------------------
# XGBOOST (PREDICT)
# -----------------------------
print("Predicting with trained XGBoost model...")

probs = model.predict_proba(X)[:, 1]  # P(Responder) per cell
print(f"Predictions complete. {len(probs)} cells predicted.")

# -----------------------------
# SAVE PREDICTIONS
# -----------------------------
print("Saving response predictions...")

cell_df = pd.DataFrame({
    "cell_id":        cell_ids,
    "sample":         samples,
    "true_response":  y,
    "prob_responder": probs,
}).set_index("cell_id")
 
cell_df.to_csv(CELL_PREDS)
print(f"Cell predictions saved: {CELL_PREDS}")

# -----------------------------
# CREATE SAMPLE LEVEL SCORE
# -----------------------------
sample_df = cell_df.groupby("sample").agg(
    true_response=("true_response", "first"),
    sample_score=("prob_responder", "mean"),
    n_cells=("prob_responder", "count"),
).reset_index()
 
sample_df.to_csv(SAMPLE_PREDS, index=False)
print("\nSample-level predictions:")
print(sample_df.to_string(index=False))

# -----------------------------
# ROC AUC CURVE
# -----------------------------
print("calculating ROC AUC curve...")
 
fpr, tpr, _ = roc_curve(sample_df["true_response"], sample_df["sample_score"])
roc_auc     = auc(fpr, tpr)
 
_, p = stats.mannwhitneyu(
    sample_df[sample_df["true_response"] == 1]["sample_score"],
    sample_df[sample_df["true_response"] == 0]["sample_score"],
    alternative="two-sided"
)
 
print(f"\nAUC: {roc_auc:.4f}")
print(f"p-value (Mann-Whitney): {p:.4e}")
 
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC-AUC — Geneformer + XGBoost\nBCC ICI Response Prediction", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROC_PLOT, dpi=300)
plt.show()
print(f"ROC curve saved: {ROC_PLOT}")