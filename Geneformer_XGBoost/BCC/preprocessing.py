# -----------------------------
# preprocessing.py
# Filter genes, expression level, and T-cell 
# Basal cell carcinoma dataset - GSE123813
# -----------------------------

import pandas as pd
import numpy as np

# -----------------------------
# SET PATHS + VARIABLES
# -----------------------------
INPUT_FILE = "raw_data/GSE123813_bcc_scRNA_counts.txt"
OUTPUT_FILE = "cleaned_data/cleaned_GSE123813.txt"
TCELL_META = "raw_data/GSE123813_bcc_tcell_metadata.txt"
GENE_PREFIXES_TO_REMOVE = ("MT-", "RPS", "RPL", "MRP", "MTRNR")
MIN_CELL_PERCENT = 0.03

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath, sep="\t", index_col=0)
    print(f"Data shape: {df.shape}  (genes x cells)")
    return df

# -----------------------------
# FILTER GENE TYPES:
# 1. NON-CODING GENES
# 2. MITOCHONDRIAL GENES
# 3. RIBOSOMAL GENES
# -----------------------------
def remove_ribosomal_genes(df, exclude_prefixes):
    print("Removing ribosomal genes...")
    mask = ~df.index.str.startswith(exclude_prefixes)
    filtered_df = df[mask]
    print(f"Remaining number of genes: {filtered_df.shape[0]}")
    return filtered_df

# -----------------------------
# FILTER LOW EXPRESSED GENES
# -----------------------------
def remove_genes_not_expressed(df, min_percent):
    print("Filtering genes not expressed in at least 3 percent of cells...")
    num_cells = df.shape[1]
    min_cells = int(np.ceil(min_percent * num_cells))
    print(f"Gene must be expressed in >= {min_cells} cells (out of {num_cells})")
    expressed_cells_per_gene = (df > 0).sum(axis=1)
    keep_genes = expressed_cells_per_gene >= min_cells
    filtered_df = df.loc[keep_genes]
    print(f"Remaining number of genes: {filtered_df.shape[0]}")
    return filtered_df

# -----------------------------
# FILTER FOR T-CELLS
# -----------------------------
# Done AFTER gene filtering (so gene % threshold is computed on full cell set first)
def filter_cells_by_cluster(df, tcell_meta_file):
    print("Filtering T-cell clusters...")
 
    meta = pd.read_csv(tcell_meta_file, sep="\t")
    tcell_ids = set(meta["cell.id"].astype(str).str.strip().tolist())
    print(f"T cell IDs in metadata: {len(tcell_ids)}")
 
    expr_cols = df.columns.astype(str).str.strip().tolist()
    df.columns = expr_cols
 
    keep = [c for c in expr_cols if c in tcell_ids]
    print(f"T cells matched in count matrix: {len(keep)}")
    return df[keep]

# -----------------------------
# SAVE OUTPUT
# -----------------------------
def save_data(df, filepath):
    print("Saving cleaned data...")
    df.to_csv(filepath, sep="\t")
    print("Done.")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    # Load: genes x cells
    df = load_data(INPUT_FILE)
    print(f"After loading                     : {df.shape}  (genes x cells)")

    # Filter genes
    df = remove_ribosomal_genes(df, GENE_PREFIXES_TO_REMOVE)
    print(f"After removing mitochondrial genes: {df.shape}  (genes x cells)")
 
    # Filter low-expressed genes (computed on full cell set, before cell filtering)
    df = remove_genes_not_expressed(df, MIN_CELL_PERCENT)
    print(f"After removing low-expressed genes: {df.shape}  (genes x cells)")
 
    # Filter to T-cell clusters
    df = filter_cells_by_cluster(df, TCELL_META)
    print(f"After T-cell filtering.           : {df.shape}  (genes x cells)")

    # Output: genes x cells (downstream loaders transpose when building AnnData)
    # Uncomment the next line if you need cells x genes on disk:
    # df = df.T

    save_data(df, OUTPUT_FILE)
    print(f"Final shape: {df.shape} (genes x cells)")


if __name__ == "__main__":
    main()

