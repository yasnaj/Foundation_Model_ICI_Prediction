# -----------------------------
# label_geneformer.py
# Attach response + sample labels
# Basal cell carcinoma dataset - GSE123813
# -----------------------------

import pandas as pd
import anndata as ad

EMBEDDINGS_H5AD = "xgboost_input_data/bcc_tcells_embeddings.h5ad"
#METADATA_FILE = "raw_data/NIHMS1531727-supplement-2.xlsx"
OUTPUT_H5AD = "xgboost_input_data/bcc_tcells_labeled.h5ad"

# Response labels from Yost et al. 2019 supplementary
BCC_RESPONSE = {
    "su001": "Responder",
    "su002": "Responder",
    "su003": "Responder",
    "su004": "Responder",
    "su005": "Non-responder",
    "su006": "Non-responder",
    "su007": "Non-responder",
    "su008": "Non-responder",
    "su009": "Responder",
    "su010": "Non-responder",
    "su012": "Responder",
}

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading data...")

adata = ad.read_h5ad(EMBEDDINGS_H5AD)
print(f"AnnData shape : {adata.shape}  (cells x embedding dims)")
print(f"Sample cell IDs  : {adata.obs_names[:3].tolist()}")

# -----------------------------
# EXTRACT SAMPLE ID FROM CELL ID
# -----------------------------
# Format: bcc.su001.pre.tcell_AAACCTGC -> su001
print("\nExtracting sample IDs from cell barcodes...")

adata.obs["sample"] = [c.split(".")[1] for c in adata.obs_names]
print(f"Unique samples: {sorted(adata.obs['sample'].unique())}")

# -----------------------------
# ATTACH RESPONSE LABELS
# -----------------------------
print("Attaching response labels...")

adata.obs["response"] = adata.obs["sample"].map(BCC_RESPONSE)
n_missing = adata.obs["response"].isna().sum()
if n_missing > 0:
    print(f"WARNING: {n_missing} cells have no response label")
    print(f"  Samples missing: {adata.obs[adata.obs['response'].isna()]['sample'].unique()}")
 
print(f"Response distribution:\n{adata.obs['response'].value_counts()}")
print(f"\nSample obs preview:")
print(adata.obs.head())

# -----------------------------
# SAVE
# -----------------------------
adata.write_h5ad(OUTPUT_H5AD)
print(f"\nSaved: {OUTPUT_H5AD}")