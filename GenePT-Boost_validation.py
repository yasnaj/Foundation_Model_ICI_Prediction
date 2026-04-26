import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt

bcc_data = pd.read_csv('BCC_embeddings.csv') # 500s
embedding_df = pd.read_csv('melanoma_embeddings.csv')
embedding_df.set_index('cell', inplace = True)
bcc_data.set_index('cell', inplace = True)


feature_cols = [c for c in embedding_df.columns if c not in ['response', 'sample', 'index']]
X = embedding_df[feature_cols].values
y = embedding_df['response'].values
sample_ids = embedding_df['sample'].values

# samples to iterate over
samples = np.unique(sample_ids)

binary_scores = {}
prob_scores   = {}

final_model = xgb.XGBClassifier(
    learning_rate = 0.01,
    max_depth = 3,
    n_estimators = 25,
    objective='binary:logistic',
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42
)

final_model.fit(X, y)

bcc_feature_cols = [c for c in bcc_data.columns if c not in ['response', 'sample']]
X_bcc = bcc_data[bcc_feature_cols].values.astype('float32')
y_bcc = bcc_data['response'].values.astype(int)
bcc_samples = bcc_data['sample'].values

unique_bcc_patients = np.unique(bcc_samples)
bcc_patient_probs = {}
bcc_patient_true = {}

for patient in unique_bcc_patients:
    mask = bcc_samples == patient
    
    probs = final_model.predict_proba(X_bcc[mask])[:, 1]
    bcc_patient_probs[patient] = np.mean(probs)
   
    bcc_patient_true[patient] = y_bcc[mask][0]

actual = [bcc_patient_true[p] for p in unique_bcc_patients]
predicted = [bcc_patient_probs[p] for p in unique_bcc_patients]
transfer_auc = roc_auc_score(actual, predicted)

print(f"--- Transfer Learning Results ---")
print(f"Melanoma -> BCC AUC: {transfer_auc:.4f}")

fpr, tpr, _ = roc_curve(actual, predicted)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {transfer_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC - GenePT + XGBoost BCC ICI Response Prediction ')
plt.savefig('BCC_ROC_AUC.png', dpi=300, bbox_inches='tight')
plt.legend()
plt.show()
