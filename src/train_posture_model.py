import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score

print("=== Starting Posture Monitoring Model (Lying vs Standing) Training ===")
start_time = time.time()

# 1. Load 15-day posture monitoring dataset
data_path = r"data\processed\hardware_compatible\posture_monitoring_15days_1min.csv"
df_posture = pd.read_csv(data_path)
print(f"Loaded posture dataset: {len(df_posture):,} records, cols: {df_posture.columns.tolist()}")

# Feature columns
POSTURE_FEATURES = ['collar_temp_C', 'barn_temp_C', 'barn_humidity_per', 'THI']
# Add engineered thermal feature
df_posture['temp_diff'] = df_posture['collar_temp_C'] - df_posture['barn_temp_C']
POSTURE_FEATURES.append('temp_diff')

# Extract hour of day to capture diurnal rest patterns
df_posture['dt_parsed'] = pd.to_datetime(df_posture['datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
df_posture['hour'] = df_posture['dt_parsed'].dt.hour
df_posture['hour_sin'] = np.sin(2 * np.pi * df_posture['hour'] / 24.0)
df_posture['hour_cos'] = np.cos(2 * np.pi * df_posture['hour'] / 24.0)
POSTURE_FEATURES.extend(['hour_sin', 'hour_cos'])

print(f"Features: {POSTURE_FEATURES}")

# Split into Train/Val (C01-C08) and Test (C09, C10)
test_cows = ['C09', 'C10']
test_mask = df_posture['cow_id'].isin(test_cows)

df_tr_val = df_posture[~test_mask].copy()
df_te = df_posture[test_mask].copy()

X_tr_val = df_tr_val[POSTURE_FEATURES].values
y_tr_val = df_tr_val['lying'].values

X_te = df_te[POSTURE_FEATURES].values
y_te = df_te['lying'].values

print(f"Train/Val set (8 cows C01-C08): {len(df_tr_val):,} samples")
print(f"Test set (2 unseen cows C09, C10): {len(df_te):,} samples")

# GroupKFold splits
fold_pairs = [
    (['C01', 'C02'], ['C03', 'C04', 'C05', 'C06', 'C07', 'C08']),
    (['C03', 'C04'], ['C01', 'C02', 'C05', 'C06', 'C07', 'C08']),
    (['C05', 'C06'], ['C01', 'C02', 'C03', 'C04', 'C07', 'C08']),
    (['C07', 'C08'], ['C01', 'C02', 'C03', 'C04', 'C05', 'C06'])
]

oof_probs = np.zeros(len(df_tr_val), dtype=np.float32)
cv_results = []

print("\n--- Running 4-Fold GroupKFold Cross-Validation for Posture ---")
for fold_idx, (val_cows, trn_cows) in enumerate(fold_pairs, 1):
    val_idx = np.where(df_tr_val['cow_id'].isin(val_cows))[0]
    trn_idx = np.where(df_tr_val['cow_id'].isin(trn_cows))[0]
    
    X_tr, y_tr = X_tr_val[trn_idx], y_tr_val[trn_idx]
    X_va, y_va = X_tr_val[val_idx], y_tr_val[val_idx]
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method='hist',
        device='cuda',
        eval_metric='logloss',
        random_state=42 + fold_idx
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    val_p = model.predict_proba(X_va)[:, 1]
    oof_probs[val_idx] = val_p
    val_pred = (val_p >= 0.5).astype(int)
    
    acc = accuracy_score(y_va, val_pred)
    f1 = f1_score(y_va, val_pred)
    auc = roc_auc_score(y_va, val_p)
    cv_results.append({
        'fold': fold_idx,
        'val_cows': val_cows,
        'accuracy': acc,
        'f1': f1,
        'roc_auc': auc
    })
    print(f"Fold {fold_idx} (Val cows: {val_cows}) -> Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

oof_preds = (oof_probs >= 0.5).astype(int)
overall_acc = accuracy_score(y_tr_val, oof_preds)
overall_f1 = f1_score(y_tr_val, oof_preds)
overall_auc = roc_auc_score(y_tr_val, oof_probs)
print(f"\nOverall Out-Of-Fold Posture CV -> Acc: {overall_acc:.4f}, F1: {overall_f1:.4f}, AUC: {overall_auc:.4f}")

# Train final posture model on all 8 cows
print("\n--- Training Final Posture Model on All 8 Training Cows ---")
final_posture_model = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method='hist',
    device='cuda',
    eval_metric='logloss',
    random_state=42
)
final_posture_model.fit(X_tr_val, y_tr_val, verbose=False)

# Evaluate on Unseen Test Cows (C09, C10)
print("\n--- Evaluating Posture Model on Unseen Test Cows (C09, C10) ---")
test_p = final_posture_model.predict_proba(X_te)[:, 1]
test_pred = (test_p >= 0.5).astype(int)

test_acc = accuracy_score(y_te, test_pred)
test_f1 = f1_score(y_te, test_pred)
test_auc = roc_auc_score(y_te, test_p)
sens = recall_score(y_te, test_pred) # Sensitivity / Lying Recall
spec = recall_score(y_te == 0, test_pred == 0) # Specificity / Standing Recall

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Lying Sensitivity (Recall): {sens:.4f}")
print(f"Standing Specificity: {spec:.4f}")

cm = confusion_matrix(y_te, test_pred)
print("Confusion Matrix (0=Standing, 1=Lying):\n", cm)

# Save posture model
posture_model_path = "models/checkpoints/posture_monitoring_model.json"
final_posture_model.save_model(posture_model_path)
print(f"Saved posture model to {posture_model_path}")

# Feature importance
imp = pd.DataFrame({
    'feature': POSTURE_FEATURES,
    'importance': final_posture_model.feature_importances_
}).sort_values('importance', ascending=False)
imp.to_csv("models/checkpoints/posture_feature_importance.csv", index=False)
print("Posture Feature Importance:\n", imp.to_string(index=False))

# Save metrics
metrics = {
    'model_name': 'Longitudinal Posture Monitoring (XGBoost)',
    'features': POSTURE_FEATURES,
    'cv_folds': cv_results,
    'oof_accuracy': float(overall_acc),
    'oof_f1': float(overall_f1),
    'oof_roc_auc': float(overall_auc),
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_roc_auc': float(test_auc),
    'test_sensitivity': float(sens),
    'test_specificity': float(spec),
    'test_confusion_matrix': cm.tolist(),
    'elapsed_time_sec': float(time.time() - start_time)
}

with open("models/checkpoints/posture_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)

print(f"\nPosture monitoring pipeline completed in {time.time() - start_time:.2f} seconds!")
