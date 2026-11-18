import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from features import prepare_data, FEATURE_COLS, CLASS_NAMES

print("=== Starting XGBoost Training with GroupKFold Validation ===")
start_time = time.time()

# 1. Load and prepare feature dataset
df = prepare_data()
print(f"Loaded dataset: {df.shape[0]:,} rows, {len(FEATURE_COLS)} features")

# 2. Split into Train/Val (C01-C08) and Test (C09, C10)
test_cows = ['C09', 'C10']
test_mask = df['cow_id'].isin(test_cows)

df_train_val = df[~test_mask].copy()
df_test = df[test_mask].copy()

X_train_val = df_train_val[FEATURE_COLS].values
y_train_val = df_train_val['target'].values
groups_train_val = df_train_val['cow_id'].values

X_test = df_test[FEATURE_COLS].values
y_test = df_test['target'].values
groups_test = df_test['cow_id'].values

print(f"Train/Val set (8 cows C01-C08): {len(df_train_val):,} samples")
print(f"Test set (2 unseen cows C09, C10): {len(df_test):,} samples")

# 3. Define 4-fold GroupKFold splits across the 8 cows
# Fold 1: Val=[C01, C02], Train=[C03..C08]
# Fold 2: Val=[C03, C04], Train=[C01, C02, C05..C08]
# Fold 3: Val=[C05, C06], Train=[C01..C04, C07, C08]
# Fold 4: Val=[C07, C08], Train=[C01..C06]
fold_pairs = [
    (['C01', 'C02'], ['C03', 'C04', 'C05', 'C06', 'C07', 'C08']),
    (['C03', 'C04'], ['C01', 'C02', 'C05', 'C06', 'C07', 'C08']),
    (['C05', 'C06'], ['C01', 'C02', 'C03', 'C04', 'C07', 'C08']),
    (['C07', 'C08'], ['C01', 'C02', 'C03', 'C04', 'C05', 'C06'])
]

oof_preds = np.zeros((len(df_train_val), 7), dtype=np.float32)
cv_scores = []

print("\n--- Running 4-Fold GroupKFold Cross-Validation ---")
for fold_idx, (val_cows, trn_cows) in enumerate(fold_pairs, 1):
    val_idx = np.where(df_train_val['cow_id'].isin(val_cows))[0]
    trn_idx = np.where(df_train_val['cow_id'].isin(trn_cows))[0]
    
    X_tr, y_tr = X_train_val[trn_idx], y_train_val[trn_idx]
    X_va, y_va = X_train_val[val_idx], y_train_val[val_idx]
    
    # Compute inverse class frequency weights
    classes, counts = np.unique(y_tr, return_counts=True)
    total = len(y_tr)
    class_weights = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights_tr = np.array([class_weights[y] for y in y_tr])
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method='hist',
        device='cuda',
        eval_metric='mlogloss',
        random_state=42 + fold_idx
    )
    
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weights_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    fit_time = time.time() - t0
    
    val_probs = model.predict_proba(X_va)
    oof_preds[val_idx] = val_probs
    val_preds = np.argmax(val_probs, axis=1)
    
    acc = accuracy_score(y_va, val_preds)
    f1_macro = f1_score(y_va, val_preds, average='macro')
    f1_weighted = f1_score(y_va, val_preds, average='weighted')
    
    cv_scores.append({
        'fold': fold_idx,
        'val_cows': val_cows,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'fit_time_sec': fit_time
    })
    print(f"Fold {fold_idx} (Val cows: {val_cows}) -> Acc: {acc:.4f}, Macro F1: {f1_macro:.4f}, Weighted F1: {f1_weighted:.4f} ({fit_time:.1f}s)")

oof_pred_labels = np.argmax(oof_preds, axis=1)
overall_oof_acc = accuracy_score(y_train_val, oof_pred_labels)
overall_oof_macro = f1_score(y_train_val, oof_pred_labels, average='macro')
overall_oof_weighted = f1_score(y_train_val, oof_pred_labels, average='weighted')

print(f"\nOverall Out-Of-Fold CV -> Acc: {overall_oof_acc:.4f}, Macro F1: {overall_oof_macro:.4f}, Weighted F1: {overall_oof_weighted:.4f}")

# 4. Train final model on all 8 training cows
print("\n--- Training Final XGBoost Model on all 8 Training Cows (C01-C08) ---")
classes, counts = np.unique(y_train_val, return_counts=True)
total = len(y_train_val)
class_weights = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
sample_weights_all = np.array([class_weights[y] for y in y_train_val])

final_xgb = xgb.XGBClassifier(
    n_estimators=350,
    learning_rate=0.08,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method='hist',
    device='cuda',
    eval_metric='mlogloss',
    random_state=42
)

final_xgb.fit(X_train_val, y_train_val, sample_weight=sample_weights_all, verbose=False)

# 5. Evaluate on Unseen Test Cows (C09, C10)
print("\n--- Evaluating Final XGBoost Model on Unseen Test Cows (C09, C10) ---")
test_probs = final_xgb.predict_proba(X_test)
test_preds = np.argmax(test_probs, axis=1)

test_acc = accuracy_score(y_test, test_preds)
test_f1_macro = f1_score(y_test, test_preds, average='macro')
test_f1_weighted = f1_score(y_test, test_preds, average='weighted')

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro F1: {test_f1_macro:.4f}")
print(f"Test Weighted F1: {test_f1_weighted:.4f}")

report_dict = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True)
report_str = classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4)
print("\nTest Classification Report:\n", report_str)

cm = confusion_matrix(y_test, test_preds)
print("Confusion Matrix:\n", cm)

# 6. Save Model Artifacts
os.makedirs("models/checkpoints", exist_ok=True)
model_path = "models/checkpoints/xgboost_cattle_model.json"
final_xgb.save_model(model_path)
print(f"Saved model to {model_path}")

# Save test predictions for ensembling
np.save("models/checkpoints/xgb_test_probs.npy", test_probs)
np.save("models/checkpoints/xgb_oof_probs.npy", oof_preds)
np.save("models/checkpoints/y_test.npy", y_test)
np.save("models/checkpoints/y_train_val.npy", y_train_val)

# Feature importance
importances = final_xgb.feature_importances_
feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': importances
}).sort_values('importance', ascending=False)
feat_imp.to_csv("models/checkpoints/xgb_feature_importance.csv", index=False)
print("Top 10 Most Important Features:\n", feat_imp.head(10).to_string(index=False))

# Save metrics JSON
metrics = {
    'model_name': 'XGBoost',
    'cv_folds': cv_scores,
    'oof_accuracy': float(overall_oof_acc),
    'oof_macro_f1': float(overall_oof_macro),
    'oof_weighted_f1': float(overall_oof_weighted),
    'test_accuracy': float(test_acc),
    'test_macro_f1': float(test_f1_macro),
    'test_weighted_f1': float(test_f1_weighted),
    'test_classification_report': report_dict,
    'test_confusion_matrix': cm.tolist(),
    'elapsed_time_sec': float(time.time() - start_time)
}

with open("models/checkpoints/xgboost_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)

print(f"\nXGBoost pipeline completed in {time.time() - start_time:.2f} seconds!")
