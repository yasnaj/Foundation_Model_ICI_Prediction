# -----------------------------
# label_geneformer.py
# Attach response + sample labels
# -----------------------------

import pandas as pd
import anndata as ad

EMBEDDINGS_H5AD = "xgboost_input_data/melanoma_tcells_embeddings.h5ad"
METADATA_FILE = "raw_data/GSE120575_patient_ID_single_cells.txt"
OUTPUT_H5AD = "xgboost_input_data/melanoma_tcells_labeled.h5ad"

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading data...")
adata = ad.read_h5ad(EMBEDDINGS_H5AD)
print(f"AnnData shape : {adata.shape}  (cells x embedding dims)")
print(f"Sample cell IDs  : {adata.obs_names[:3].tolist()}")

# -----------------------------
# LOAD METADATA
# -----------------------------
meta_raw = pd.read_csv(METADATA_FILE, sep="\t", skiprows=19, encoding="latin1", header=0)
meta = meta_raw.iloc[:, [1, 4, 5]].copy()
meta.columns = ["cell_id", "sample", "response"]
meta = meta[meta["response"].isin(["Responder", "Non-responder"])].set_index("cell_id")
print(f"Metadata rows    : {len(meta)}")

# -----------------------------
# JOIN ON CELL ID + SAVE
# -----------------------------
print("Attaching labels...")
adata.obs["sample"]   = meta.loc[adata.obs_names, "sample"]
adata.obs["response"] = meta.loc[adata.obs_names, "response"]

assert adata.obs["sample"].isna().sum() == 0, "Some cells missing sample label"
assert adata.obs["response"].isna().sum() == 0, "Some cells missing response label"

print(f"\nobs columns      : {adata.obs.columns.tolist()}")
print(f"Unique samples   : {adata.obs['sample'].nunique()}")
print(f"Response values  : {adata.obs['response'].value_counts().to_dict()}")
print(f"\nSample obs:")
print(adata.obs.head())
 
adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
print(f"\nSaved: {OUTPUT_H5AD}")