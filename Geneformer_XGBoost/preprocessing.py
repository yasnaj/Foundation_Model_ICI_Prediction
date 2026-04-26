import pandas as pd
import numpy as np

# -----------------------------
# SET PATHS + VARIABLES
# -----------------------------
INPUT_FILE = "raw_data/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt"
OUTPUT_FILE = "cleaned_data/cleaned_GSE120575.txt"
CLUSTER_FILE = "raw_data/NIHMS1510803-supplement-10.xlsx"
#METADATA_FILE = "path/to/your/metadata_file.csv"
T_CELL_CLUSTERS = [5, 6, 7, 8, 9, 10, 11]
GENE_PREFIXES_TO_REMOVE = ("MT-", "RPS", "RPL", "MRP", "MTRNR")
MIN_CELL_PERCENT = 0.03
# END GOAL: 10,082 GENES X 16,291 CELLS -> then T-cell filtering

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(filepath):
    print("Loading data...")
    #df = pd.read_csv(filepath, sep="\t", index_col=0, header=0) # setting as 0 excludes from df -> for dropping
    df = pd.read_csv(filepath, sep="\t", header=0, index_col=0, skiprows=[1])
    # Row 1: The actual cell IDs (A10_P3_M11, A11_P1_M11, etc.) — this is the real column header
    # Row 2: Patient labels (Pre_P1, Pre_P1, Post_P1, etc.) — this is metadata, not a header
    # df = df.drop(index=df.index[0])

    print(f"Data shape: {df.shape} (genes x cells) <- expect ~(55737? genes x 16291 cells)")
    return df

# -----------------------------
# FILTER GENE TYPES:
# 1. NON-CODING GENES
# 2. MITOCHONDRIAL GENES
# 3. RIBOSOMAL GENES
# -----------------------------
def remove_ribosomal_genes(df, exclude_prefixes):
    print("Removing ribosomal genes...")
    # create a boolean array/mask fo T/F to identify genes that do NOT start with the specified prefixes
    mask = ~df.index.str.startswith(exclude_prefixes) 
    filter_df = df[mask] #keep only rows where mask == True
    print(f"Remaining number of genes: {filter_df.shape[0]}") #prints number of genes remaining
    return filter_df

# -----------------------------
# FILTER LOW EXPRESSED GENES
# -----------------------------
def remove_genes_not_expressed(df, min_percent):
    print("Filtering genes not expressed in at least 3 percent of cells...")
    num_cells = df.shape[1]
    min_cells = int(np.ceil(min_percent * num_cells))
    print(f"Gene must be expressed in ≥ {min_cells} cells (out of {num_cells})")

    expressed_cells_per_gene = (df > 0).sum(axis=1)
    keep_genes = expressed_cells_per_gene >= min_cells
    filtered_df = df.loc[keep_genes]
    print(f"Remaining number of genes: {filtered_df.shape[0]}")
    return filtered_df
    # both functions work with df and return filtered_df, but they are local variables.

# -----------------------------
# FILTER FOR T-CELLS
# -----------------------------
# Done AFTER gene filtering (so gene % threshold is computed on full cell set first)
def filter_cells_by_cluster(df, cluster_file, clusters_to_keep):
    print("Filtering T-cell clusters...")
 
    clusters = pd.read_excel(cluster_file, sheet_name='Cluster annotation-Fig1B-C')
    clusters = clusters.dropna(how="all")
    clusters.columns = clusters.columns.str.strip()
 
    valid_cells = (
        clusters[clusters["Cluster number"].isin(clusters_to_keep)]["Cell Name"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    print(f"Cells to match from cluster file: {len(valid_cells)}")
 
    # Normalise expression column names
    expr_cols = df.columns.astype(str).str.strip().tolist()
    df.columns = expr_cols
 
    print(f"Sample cluster IDs : {valid_cells[:3]}")
    print(f"Sample expr cols   : {expr_cols[:3]}")
 
    valid_set = set(valid_cells)
    expr_set  = set(expr_cols)
 
    # 1. Exact match
    exact = list(expr_set & valid_set)
    print(f"Exact matches: {len(exact)}")
 
    # 2. Partial: expr col starts with cluster ID (cluster ID is a prefix)
    partial_a = []
    for col in expr_cols:
        if col in valid_set:
            continue
        for cell_id in valid_set:
            if col.startswith(cell_id):
                partial_a.append(col)
                break
    print(f"Partial matches (expr col starts with cluster ID): {len(partial_a)}")
 
    # 3. Partial: cluster ID starts with expr col (expr col is a prefix)
    partial_b = []
    for col in expr_cols:
        if col in valid_set:
            continue
        for cell_id in valid_set:
            if cell_id.startswith(col):
                partial_b.append(col)
                break
    print(f"Partial matches (cluster ID starts with expr col): {len(partial_b)}")
 
    selected = list(set(exact + partial_a + partial_b))
    print(f"Total unique cells kept: {len(selected)}")
 
    if len(selected) == 0:
        print("\n  *** WARNING: 0 cells matched! ***")
        print(f"First 5 cluster IDs : {valid_cells[:5]}")
        print(f"First 5 expr cols   : {expr_cols[:5]}")
 
    return df[selected]

# -----------------------------
# ADD LABELS
# -----------------------------
# Not done, as will do after extracting Geneformer/GenePT embeddings!

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
    df = filter_cells_by_cluster(df, CLUSTER_FILE, T_CELL_CLUSTERS)
    print(f"After T-cell filtering.           : {df.shape}  (genes x cells)")

    # Output: genes x cells (downstream loaders transpose when building AnnData)
    # Uncomment the next line if you need cells x genes on disk:
    # df = df.T

    save_data(df, OUTPUT_FILE)
    print(f"Final shape: {df.shape} (genes x cells)")
    print("Target: ~10,082 genes x number of T-cells")


if __name__ == "__main__":
    main()

