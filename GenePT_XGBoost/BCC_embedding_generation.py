import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('cleaned_GSE123813.txt', sep = '\t') # 500 seconds

df.set_index('Unnamed: 0', inplace = True)
df.index.name = 'gene'
#df.shape # 9.5k x 33k
gene_vectors = pickle.load(open('GenePT_gene_embedding_ada_text.pickle', 'rb'))

# convert dictionary of vectors to a Matrix for faster computation
valid_genes = [g for g in df.index if g in gene_vectors] # list of genes 
gene_matrix = np.stack([gene_vectors[g] for g in valid_genes]) # rows of genes stacked - ordered by valid genes
gene_index = pd.Index(valid_genes)
df_filtered = df.loc[valid_genes].astype(float)  # ordered by valid genes, currently genes as rows 

# matrix multiplication: (Cells x Genes) @ (Genes x Embedding_Dims)
def compute_all_embeddings(expr_df, gene_mat, gene_idx):
    # expr_df= Genes(rows) x Cells(columns)
    # gene_mat= Genes(rows) x Embed_Dims(columns)
    # Transpose to get Cells x Genes
    expr_matrix = expr_df.T.values 
    numerator = expr_matrix @ gene_mat
    
    non_zero_counts = (expr_matrix > 0).sum(axis=1, keepdims=True)
    non_zero_counts[non_zero_counts == 0] = 1
    
    embeddings = numerator / non_zero_counts
    
    return embeddings

all_embeddings = compute_all_embeddings(df_filtered, gene_matrix, gene_index)

embedding_df = pd.DataFrame(
    all_embeddings, 
    index=df.columns, 
    columns=[f'dim_{i}' for i in range(all_embeddings.shape[1])]
)

# testing
print(f"Computed embeddings for {embedding_df.shape[0]} cells")
print(f"Embedding matrix shape: {embedding_df.shape}")
print(f"Any NaN values: {embedding_df.isna().any().any()}")
print(f"Any zero-weight cells: {(df_filtered.T.values.sum(axis=1) == 0).sum()}")
print(f"Mean embedding norm: {np.linalg.norm(all_embeddings, axis=1).mean():.4f}")

cell_names = embedding_df.index.values
embedding_df.index.name = 'cell'

patients = [s.split('.')[1] for s in cell_names]
embedding_df['sample'] = patients

responses = [1 if p in ['su001', 'su002', 'su003', 'su004', 'su009', 'su012'] else 0 for p in patients]
embedding_df['response'] = responses
print(f"Cells identified as responders: {embedding_df['response'].sum()}")

print(f"Total cells: {len(embedding_df)}")
print(f"Responders (1): {(embedding_df['response'] == 1).sum()}")
print(f"Non-Responders (0): {(embedding_df['response'] == 0).sum()}")
print(f"Unique Patients: {embedding_df['sample'].nunique()}")

# saving embeddings
embedding_df.to_csv('BCC_embeddings.csv', index=True)
