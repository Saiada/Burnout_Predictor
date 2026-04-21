import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load files
sart_path = os.path.join("real_data_model2", "cleaned_master_sart.npy")
nback_path = os.path.join("real_data_model2", "cleaned_master_nback.npy")

sart_array = np.load(sart_path, allow_pickle=True)
nback_array = np.load(nback_path, allow_pickle=True)

# SART Feature Extraction
sart_filtered = sart_array[sart_array[:, 2] != 'response_action']
sart_type = sart_filtered[:, 3]
sart_rt_str = sart_filtered[:, 4].astype(str)
sart_rt_str[sart_rt_str == 'nan'] = '0'
sart_rt = sart_rt_str.astype(float)
sart_mean_rt = np.mean(sart_rt[sart_rt > 0]) 
commission_errors = np.sum(sart_type == 'incorrect')

# N-Back Feature Extraction
nback_filtered = nback_array[nback_array[:, 2] != 'response_action']
nback_type = nback_filtered[:, 4]
nback_rt_str = nback_filtered[:, 5].astype(str)
nback_rt_str[nback_rt_str == 'nan'] = '0'
nback_rt = nback_rt_str.astype(float)
nback_mean_rt = np.mean(nback_rt[nback_rt > 0])
targets_mask = nback_filtered[:, 3] == 'target'
hits = np.sum((nback_type == 'hit') & targets_mask)
total_targets = np.sum(targets_mask)
accuracy = (hits / total_targets) * 100 if total_targets > 0 else 0

print(f"Extracted Aggregate Features:")
print(f"SART Mean RT: {sart_mean_rt:.2f}ms | Commission Errors: {commission_errors}")
print(f"N-Back Mean RT: {nback_mean_rt:.2f}ms | Accuracy: {accuracy:.2f}%\n")

# --- 2. Constructing Feature Vectors ---
num_subjects = 500
X_subject_base = np.array([sart_mean_rt, commission_errors, nback_mean_rt, accuracy])
X_synthetic = np.repeat(X_subject_base.reshape(1, -1), num_subjects, axis=0)

np.random.seed(42)
X_synthetic[:, 0] += np.random.normal(0, 30, num_subjects) 
X_synthetic[:, 1] += np.random.randint(-5, 6, num_subjects) 
X_synthetic[:, 1] = np.clip(X_synthetic[:, 1], 0, None)
X_synthetic[:, 2] += np.random.normal(0, 40, num_subjects) 
X_synthetic[:, 3] -= np.random.normal(0, 8, num_subjects) 
X_synthetic[:, 3] = np.clip(X_synthetic[:, 3], 0, 100) 

# --- THE UPGRADE: Rule-Based Labeling WITH NOISE ---
y_synthetic = []

for i in range(num_subjects):
    score = 0
    # Added a small random Gaussian drift to the averages so the boundaries blur
    if X_synthetic[i, 0] > (sart_mean_rt + np.random.normal(0, 5)): score += 1
    if X_synthetic[i, 1] > ((commission_errors/24) + np.random.normal(0, 0.5)): score += 1
    if X_synthetic[i, 2] > (nback_mean_rt + np.random.normal(0, 10)): score += 1
    if X_synthetic[i, 3] < (accuracy + np.random.normal(0, 2)): score += 1
    X_synthetic[:, 3] = np.clip(X_synthetic[:, 3], 0, 100)
    
    label = 1 if score >= 2 else 0
    
    # Human Noise: 15% chance to flip the result randomly
    if np.random.random() < 0.15:
        label = 1 - label
        
    y_synthetic.append(label)

y_synthetic = np.array(y_synthetic)

# --- 3. Split and Train ---
X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("--- Model Evaluation Results (With Human Noise) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(rf_model, "sart_2back_rf_model.pkl")
print("Model saved as sart_2back_rf_model.pkl")