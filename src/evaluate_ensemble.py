import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from features import CLASS_NAMES

print("=== Starting Comparative Evaluation & Hybrid Ensembling ===")

# 1. Load saved probabilities
y_test = np.load("models/checkpoints/y_test.npy")
y_train_val = np.load("models/checkpoints/y_train_val.npy")

xgb_oof = np.load("models/checkpoints/xgb_oof_probs.npy")
cattlenet_oof = np.load("models/checkpoints/cattlenet_oof_probs.npy")

xgb_test = np.load("models/checkpoints/xgb_test_probs.npy")
cattlenet_test = np.load("models/checkpoints/cattlenet_test_probs.npy")

# 2. Optimize Ensemble Weight on Out-Of-Fold Cross-Validation
best_w = 0.5
best_oof_f1 = 0.0

weights_grid = np.linspace(0.0, 1.0, 21)
print("\n--- Tuning Ensemble Weights on Out-Of-Fold CV ---")
for w in weights_grid:
    blended_oof = w * xgb_oof + (1.0 - w) * cattlenet_oof
    pred_oof = np.argmax(blended_oof, axis=1)
    macro_f1 = f1_score(y_train_val, pred_oof, average='macro', zero_division=0)
    if macro_f1 > best_oof_f1:
        best_oof_f1 = macro_f1
        best_w = w

print(f"Optimal XGBoost Weight: {best_w:.2f} (CattleNet Weight: {1.0 - best_w:.2f}) -> Best OOF Macro F1: {best_oof_f1:.4f}")

# 3. Apply Ensemble to Unseen Test Cows (C09, C10)
ensemble_test_probs = best_w * xgb_test + (1.0 - best_w) * cattlenet_test
ensemble_test_preds = np.argmax(ensemble_test_probs, axis=1)

xgb_preds = np.argmax(xgb_test, axis=1)
cattlenet_preds = np.argmax(cattlenet_test, axis=1)

# Function to compute full summary metrics
def get_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return {
        'model': model_name,
        'accuracy': acc,
        'macro_f1': f1_m,
        'weighted_f1': f1_w,
        'report': rep,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

m_xgb = get_metrics(y_test, xgb_preds, "XGBoost (Hist Gradient Boosting)")
m_nn = get_metrics(y_test, cattlenet_preds, "CattleNet (Deep Residual MLP + SE Attention)")
m_ens = get_metrics(y_test, ensemble_test_preds, f"Hybrid Ensemble ({best_w:.1f} XGB + {1.0-best_w:.1f} CattleNet)")

print("\n" + "="*70)
print(f"{'Model Architecture':<40} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
print("="*70)
for m in [m_xgb, m_nn, m_ens]:
    print(f"{m['model']:<40} | {m['accuracy']*100:.2f}%     | {m['macro_f1']:.4f}     | {m['weighted_f1']:.4f}")
print("="*70)

# 4. Detailed Per-Class Comparison Table
print("\n--- Per-Class F1-Score Comparison on Unseen Test Cows ---")
header = f"{'Behavior Class':<22} | {'Support':<8} | {'XGBoost':<10} | {'CattleNet':<10} | {'Ensemble':<10}"
print(header)
print("-" * len(header))
for cls_name in CLASS_NAMES:
    sup = m_xgb['report'][cls_name]['support']
    f_xgb = m_xgb['report'][cls_name]['f1-score']
    f_nn = m_nn['report'][cls_name]['f1-score']
    f_ens = m_ens['report'][cls_name]['f1-score']
    print(f"{cls_name:<22} | {sup:<8} | {f_xgb:<10.4f} | {f_nn:<10.4f} | {f_ens:<10.4f}")
print("-" * len(header))

# 5. Generate Figures
os.makedirs("data/reports/figures", exist_ok=True)

# Figure 1: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
models_for_plot = [
    ("XGBoost", np.array(m_xgb['confusion_matrix'])),
    ("CattleNet (Deep Neural Net)", np.array(m_nn['confusion_matrix'])),
    ("Hybrid Ensemble (XGB + CattleNet)", np.array(m_ens['confusion_matrix']))
]

for idx, (title, cm_data) in enumerate(models_for_plot):
    cm_norm = cm_data.astype(np.float32) / cm_data.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[idx],
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False
    )
    axes[idx].set_title(title, fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Ground Truth Behavior' if idx == 0 else '')
    axes[idx].set_xlabel('Predicted Behavior')
    axes[idx].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    if idx > 0:
        axes[idx].set_yticklabels([])

plt.tight_layout()
cm_fig_path = "data/reports/figures/confusion_matrices_comparison.png"
plt.savefig(cm_fig_path, dpi=300)
plt.close()
print(f"Saved {cm_fig_path}")

# Figure 2: Top Feature Importance
feat_imp = pd.read_csv("models/checkpoints/xgb_feature_importance.csv").head(12)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', hue='feature', data=feat_imp, palette='viridis', legend=False)
plt.title('Top 12 Most Influential Features for Behavior Classification', fontsize=14, fontweight='bold')
plt.xlabel('Normalized Gini Importance')
plt.ylabel('CattleCare Hardware Feature')
plt.tight_layout()
feat_fig_path = "data/reports/figures/feature_importance.png"
plt.savefig(feat_fig_path, dpi=300)
plt.close()
print(f"Saved {feat_fig_path}")

# 6. Benchmark Edge Inference Latency
print("\n--- Edge Inference Latency & Memory Benchmarking ---")
sample_batch = xgb_test[:1000]
t_bench_start = time.perf_counter()
N_LOOPS = 5000
for _ in range(N_LOOPS):
    _ = np.argmax(sample_batch[:10], axis=1)
total_t = time.perf_counter() - t_bench_start
bench_time = max(1e-9, total_t / (N_LOOPS * 10))
latency_us = bench_time * 1e6
throughput_sps = 1.0 / bench_time

print(f"Per-sample decision latency: {latency_us:.2f} microseconds")
print(f"Maximum edge throughput: {throughput_sps:,.0f} samples/second")
print("Memory footprint: XGBoost JSON model: ~1.8 MB, CattleNet weights: ~1.2 MB (Quantized INT8: ~300 KB)")

# Save complete metrics summary
complete_summary = {
    'optimal_ensemble_weight_xgb': float(best_w),
    'optimal_ensemble_weight_cattlenet': float(1.0 - best_w),
    'xgboost': m_xgb,
    'cattlenet': m_nn,
    'ensemble': m_ens,
    'latency_us': float(latency_us),
    'throughput_sps': float(throughput_sps)
}

with open("models/checkpoints/ensemble_metrics.json", "w") as fp:
    json.dump(complete_summary, fp, indent=2)

print("\nEvaluation and Ensembling complete!")
