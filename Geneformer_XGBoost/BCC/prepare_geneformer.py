# -----------------------------
# prepare_geneformer.py
# Convert preprocess scRNA-seq data for Geneformer input
# Basal cell carcinoma dataset - GSE123813
# -----------------------------

import pandas as pd
import numpy as np
import anndata as ad
import mygene # for gene ID conversion (HGNC short names -> Ensembl ID)
# Drops genes with no Ensembl mapping + duplicate mappings (keeps first)

# -----------------------------
# SET PATHS + VARIABLES
# -----------------------------
mg = mygene.MyGeneInfo()
INPUT_FILE = "cleaned_data/cleaned_GSE123813.txt"
OUTPUT_FILE = "geneformer_input_data/prepared_GSE123813.h5ad"

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading data...")
df = pd.read_csv(INPUT_FILE, sep="\t", index_col=0)
print(f"Data shape: {df.shape} (genes x cells) <- expect 9541 genes x 33106 cells")
print(f"Sample cell IDs   : {df.columns[:3].tolist()}")
print(f"Sample gene names : {df.index[:3].tolist()}")

# -----------------------------
# EMSEMBL ID CONVERSION
# -----------------------------
print("Converting gene names to Ensembl IDs...")
gene_symbols = df.index.tolist()
 
results = mg.querymany(
    gene_symbols,
    scopes="symbol",
    fields="ensembl.gene",
    species="human",
    returnall=False,
    verbose=False,
)
 
# Build symbol → Ensembl ID dict (first hit per symbol wins)
symbol_to_ensembl = {}
for hit in results:
    sym = hit.get("query")
    if "ensembl" not in hit:
        continue
    ensembl_field = hit["ensembl"]
    if isinstance(ensembl_field, list):
        eid = ensembl_field[0]["gene"]
    else:
        eid = ensembl_field["gene"]
    if sym not in symbol_to_ensembl:
        symbol_to_ensembl[sym] = eid
 
n_mapped   = len(symbol_to_ensembl)
n_unmapped = len(gene_symbols) - n_mapped
print(f"Mapped successfully : {n_mapped}")
print(f"No Ensembl match   : {n_unmapped}  (dropped — non-coding / pseudogenes)")
 
# Filter and rename
mapped_symbols  = [g for g in gene_symbols if g in symbol_to_ensembl]
df_mapped       = df.loc[mapped_symbols].copy()
df_mapped.index = [symbol_to_ensembl[g] for g in mapped_symbols]
 
# Drop duplicate Ensembl IDs (keep first)
n_before = len(df_mapped)
df_mapped = df_mapped[~df_mapped.index.duplicated(keep="first")]
n_dropped = n_before - len(df_mapped)
if n_dropped:
    print(f"Dropped {n_dropped} duplicate Ensembl IDs")
 
print(f"Genes after mapping : {df_mapped.shape[0]}")
print(f"Sample cell IDs   : {df_mapped.columns[:3].tolist()}")
print(f"Sample gene names : {df_mapped.index[:3].tolist()}")

# -----------------------------
# COMPUTE N COUNTS
# -----------------------------
# Geneformer needs adat.obs["n_counts"] to exist
print("Computing n_counts per cell...")
n_counts = df_mapped.sum(axis=0)   # sum across genes for each cell
print(f"n_counts range: {n_counts.min():.1f} – {n_counts.max():.1f}")

# -----------------------------
# DROP CELLS WITH ZERO EXPRESSION
# -----------------------------
print("Dropping cells with zero total expression...")
zero_mask = n_counts == 0
n_zero = zero_mask.sum()
if n_zero > 0:
    print(f"Dropping {n_zero} cell(s) with zero expression")
    df_mapped = df_mapped.loc[:, ~zero_mask]
    n_counts  = n_counts[~zero_mask]

# -----------------------------
# CONVERT TO ANNDATA
# -----------------------------
# Transpose to cells x genes
expr_matrix = df_mapped.T
 
adata = ad.AnnData(
    X   = expr_matrix.values.astype(np.float32),
    obs = pd.DataFrame(index=expr_matrix.index),    # cell IDs preserved exactly
    var = pd.DataFrame(index=expr_matrix.columns),  # Ensembl IDs
)
 
# Attach n_counts to obs — required by Geneformer tokenizer
adata.obs["n_counts"] = n_counts.loc[adata.obs_names].values
adata.var["ensembl_id"]  = adata.var_names 

# -----------------------------
# SAVE OUTPUT
# -----------------------------
print(f"AnnData shape : {adata.shape}  (cells x genes)")
print(f"obs columns   : {adata.obs.columns.tolist()}")
print(f"var columns   : {adata.var.columns.tolist()}")
adata.write_h5ad(OUTPUT_FILE)

print("Done.")