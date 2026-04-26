import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('cleaned_GSE120575.txt', sep = '\t')
df = df.set_index('Unnamed: 0')
df.index.name = 'gene'
df.drop('H9_P5_M67_L001_T_enriched', axis = 1, inplace = True)
gene_vectors = pickle.load(open('GenePT_gene_embedding_ada_text.pickle', 'rb'))


valid_genes = [g for g in df.index if g in gene_vectors]
len(valid_genes)

# convert dictionary of vectors to a Matrix for faster computation
valid_genes = [g for g in df.index if g in gene_vectors]
gene_matrix = np.stack([gene_vectors[g] for g in valid_genes]) #rows of genes stacked - ordered by valid genes
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
# mean norm should be close to what you saw for the single cell (~0.91)

# adding response labels
patients_responses = pd.read_csv('GSE120575_patient_ID_single_cells.txt', sep = '\t', header = [0])

metadata = pd.DataFrame({'Patients': patients_responses.iloc[:,3],
                    'Response': patients_responses.iloc[:,4],
                    'Cell': patients_responses.iloc[:,0]})

metadata = metadata.set_index('Cell')
#metadata_dict = metadata.to_dict(orient = 'index')
#embedding_df.drop(['Response'], axis = 1, inplace = True)
response = metadata['Response'].map ({'Responder': 1, 'Non-responder': 0})
embedding_df['response'] = embedding_df.index.map(response)
embedding_df['sample'] = embedding_df.index.map(metadata['Patients'])

# 4. Clean up any cells that don't have metadata
embedding_df = embedding_df.dropna(subset=['response', 'sample'])


# testing
print(f"Cells identified as responders: {embedding_df['response'].sum()}")
print(embedding_df[['response']].head())
embedding_df.columns
embedding_df.index.name = 'cell'

print(f"Total cells: {len(embedding_df)}")
print(f"Responders (1): {(embedding_df['response'] == 1).sum()}")
print(f"Non-Responders (0): {(embedding_df['response'] == 0).sum()}")
print(f"Unique Patients: {embedding_df['sample'].nunique()}")

# saving embeddings
embedding_df.to_csv('melanoma_embeddings.csv', index=True)
