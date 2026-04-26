from sklearn.model_selection import ParameterGrid
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

embedding_df = pd.read_csv('melanoma_embeddings.csv')
embedding_df.set_index('cell', inplace=True)

feature_cols = [c for c in embedding_df.columns if c not in ['response', 'sample', 'index']]
X = embedding_df[feature_cols].values
y = embedding_df['response'].values
sample_ids = embedding_df['sample'].values

# Get unique samples to iterate over
samples = np.unique(sample_ids)

binary_scores = {}
prob_scores   = {}

# initial training based on PRECISE's XGBoost hyperparameters
param_grid = {
    'max_depth': [3, 5, 6],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [25, 50]
}

# further training
param_grid = {
    'max_depth': [2, 3, 4],              
    'learning_rate': [0.01],    
    'n_estimators': [25, 50],       
    'subsample': [0.5, 0.7],            
    'colsample_bytree': [0.1, 0.3],     
    'min_child_weight': [3, 5]         
}

# final parameter choices
param_grid = {
    'max_depth': [3],               
    'learning_rate': [0.01],     
    'n_estimators': [25]
}

best_auc = 0
sample_true_labels = {
    s: embedding_df.loc[embedding_df['sample'] == s, 'response'].iloc[0] 
    for s in samples
}
best_params = None
total_combos = len(ParameterGrid(param_grid))

for i, params in enumerate(ParameterGrid(param_grid)):
    fold_probs = {}
    print(f"Testing Combo {i+1}/{total_combos}: {params}")
    
    for held_out in samples:
        train_mask = sample_ids != held_out
        test_mask = sample_ids == held_out
        
        model = xgb.XGBClassifier(
            **params,
            objective='binary:logistic',
            scale_pos_weight=(y[train_mask] == 0).sum() / (y[train_mask] == 1).sum(),
            eval_metric='logloss',
            n_jobs=-1
        )
        model.fit(X[train_mask], y[train_mask])

        fold_probs[held_out] = model.predict_proba(X[test_mask])[:, 1].mean()
    
    # AUC calculation
    current_true = [sample_true_labels[s] for s in samples]
    current_probs = [fold_probs[s] for s in samples]
    current_auc = roc_auc_score(current_true, current_probs)
    
    if current_auc > best_auc:
        best_auc = current_auc
        best_params = params
        print(f"  *** New Best! ***")
    if current_auc > best_auc:
        best_auc = current_auc
        best_params = params

print(f"Best AUC: {best_auc:.3f}")
print(f"Best Params: {best_params}")

fpr, tpr, _ = roc_curve(current_true, current_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {best_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC - GenePT + XGBoost Melanoma ICI Response Prediction ')
plt.legend()
plt.savefig('Melanoma_ROC_AUC.png', dpi=300, bbox_inches='tight')
plt.show()
