# -----------------------------
# train_final_model.py
# Train final XGBoost model on all melanoma cells using tuned hyperparameters
# -----------------------------

import os
import numpy as np
import pandas as pd
import anndata as ad
import xgboost as xgb

LABELED_H5AD  = "../xgboost_input_data/melanoma_tcells_labeled.h5ad"
OUTPUT_DIR    = "results/"
MODEL_PATH    = os.path.join(OUTPUT_DIR, "melanoma_xgb_model.json")
INFO_PATH     = os.path.join(OUTPUT_DIR, "melanoma_training_info.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading labeled embeddings...")
adata = ad.read_h5ad(LABELED_H5AD)
print(f"Shape    : {adata.shape}  (cells x embedding dims)")
print(f"Samples  : {adata.obs['sample'].nunique()}")
print(f"Response : {adata.obs['response'].value_counts().to_dict()}")

X = adata.X
y = (adata.obs["response"] == "Responder").astype(int).values

# -----------------------------
# CALCULATE scale_pos_weight
# -----------------------------
n_pos = y.sum()
n_neg = (1 - y).sum()
scale_pos_weight = n_neg / n_pos
print(f"\nn_pos (Responder)    : {n_pos}")
print(f"n_neg (Non-responder): {n_neg}")
print(f"scale_pos_weight     : {scale_pos_weight:.3f}")

# -----------------------------
# TRAIN
# -----------------------------
print("\nTraining final XGBoost model on all melanoma cells...")
model = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.01,
    n_estimators=25,
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    verbosity=0,
)
model.fit(X, y)
print("Training complete.")

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save_model(MODEL_PATH)
print(f"Model saved: {MODEL_PATH}")

# -----------------------------
# SAVE TRAINING INFO
# -----------------------------
info = f"""Melanoma XGBoost Final Model — Training Summary
================================================
Input              : {LABELED_H5AD}
Cells              : {adata.shape[0]}
Embedding dims     : {adata.shape[1]}
Samples            : {adata.obs['sample'].nunique()}
Responder cells    : {n_pos}
Non-responder cells: {n_neg}
scale_pos_weight   : {scale_pos_weight:.4f}

Hyperparameters (tuned via LOO CV on melanoma)
----------------------------------------------
max_depth          : 3
learning_rate      : 0.01
n_estimators       : 25
objective          : binary:logistic

LOO CV AUC (melanoma internal validation): 0.901

Model saved to: {MODEL_PATH}
"""

with open(INFO_PATH, "w") as f:
    f.write(info)
print(f"Training info saved: {INFO_PATH}")
print("\nDone. Use melanoma_xgb_model.json for external validation.")