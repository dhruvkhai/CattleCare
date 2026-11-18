import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from features import prepare_data, FEATURE_COLS, CLASS_NAMES

print("=== Starting Deep Learning (CattleNet) Training on GPU ===")
start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 1. Load and prepare feature dataset
df = prepare_data()
print(f"Loaded dataset: {df.shape[0]:,} rows, {len(FEATURE_COLS)} features")

# 2. Split into Train/Val (C01-C08) and Test (C09, C10)
test_cows = ['C09', 'C10']
test_mask = df['cow_id'].isin(test_cows)

df_train_val = df[~test_mask].copy()
df_test = df[test_mask].copy()

X_train_val = df_train_val[FEATURE_COLS].values.astype(np.float32)
y_train_val = df_train_val['target'].values.astype(np.int64)

X_test = df_test[FEATURE_COLS].values.astype(np.float32)
y_test = df_test['target'].values.astype(np.int64)

# 3. Fit scaler strictly on training cows
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/checkpoints/cattlenet_scaler.joblib")

# Compute class weights for Focal Loss
classes, counts = np.unique(y_train_val, return_counts=True)
total_samples = len(y_train_val)
weights = total_samples / (len(classes) * counts)
weights = np.sqrt(weights)
weights = weights / weights.sum() * len(classes)
class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
print(f"Computed Class Weights: {weights.round(3).tolist()}")

# 4. Pure PyTorch AdamW implementation (avoids torch._dynamo/optree DLL block)
class SimpleAdamW:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.params = list(params)
        self.lr = lr
        self.initial_lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                if p.grad is None:
                    continue
                g = p.grad
                if self.weight_decay > 0:
                    p.data.mul_(1.0 - self.lr * self.weight_decay)
                m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                m_hat = m / (1.0 - self.beta1 ** self.t)
                v_hat = v / (1.0 - self.beta2 ** self.t)
                p.data.addcdiv_(m_hat, torch.sqrt(v_hat) + self.eps, value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    def set_lr(self, lr):
        self.lr = lr

# 5. Define CattleNet Architecture
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.bn1 = nn.BatchNorm1d(dim * 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        res = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.act(out + res)

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w

class CattleNet(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_classes=7, num_blocks=3, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        self.se = SqueezeExcitation(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.se(h)
        logits = self.head(h)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# 6. GroupKFold Cross-Validation Setup
fold_pairs = [
    (['C01', 'C02'], ['C03', 'C04', 'C05', 'C06', 'C07', 'C08']),
    (['C03', 'C04'], ['C01', 'C02', 'C05', 'C06', 'C07', 'C08']),
    (['C05', 'C06'], ['C01', 'C02', 'C03', 'C04', 'C07', 'C08']),
    (['C07', 'C08'], ['C01', 'C02', 'C03', 'C04', 'C05', 'C06'])
]

oof_preds = np.zeros((len(df_train_val), 7), dtype=np.float32)
cv_scores = []

BATCH_SIZE = 1024
EPOCHS = 10
BASE_LR = 2.0e-3

print(f"\n--- Running 4-Fold GroupKFold CV with CattleNet (Batch: {BATCH_SIZE}, Epochs: {EPOCHS}) ---")

for fold_idx, (val_cows, trn_cows) in enumerate(fold_pairs, 1):
    val_idx = np.where(df_train_val['cow_id'].isin(val_cows))[0]
    trn_idx = np.where(df_train_val['cow_id'].isin(trn_cows))[0]
    
    X_tr, y_tr = X_train_val_scaled[trn_idx], y_train_val[trn_idx]
    X_va, y_va = X_train_val_scaled[val_idx], y_train_val[val_idx]
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True
    )
    
    model = CattleNet(input_dim=len(FEATURE_COLS), hidden_dim=128, num_classes=7, num_blocks=3, dropout=0.2).to(device)
    optimizer = SimpleAdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    criterion = FocalLoss(alpha=class_weights_t, gamma=1.8)
    
    best_val_f1 = 0.0
    best_probs = None
    
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        # Cosine learning rate decay
        current_lr = BASE_LR * 0.5 * (1.0 + np.cos(np.pi * (epoch - 1) / EPOCHS))
        optimizer.set_lr(current_lr)
        
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_probs_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                logits = model(bx)
                probs = F.softmax(logits, dim=1)
                val_probs_list.append(probs.cpu().numpy())
        val_probs = np.concatenate(val_probs_list, axis=0)
        val_preds = np.argmax(val_probs, axis=1)
        val_macro_f1 = f1_score(y_va, val_preds, average='macro', zero_division=0)
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_probs = val_probs
            
    oof_preds[val_idx] = best_probs
    val_preds_best = np.argmax(best_probs, axis=1)
    acc = accuracy_score(y_va, val_preds_best)
    f1_weighted = f1_score(y_va, val_preds_best, average='weighted', zero_division=0)
    fold_time = time.time() - t0
    
    cv_scores.append({
        'fold': fold_idx,
        'val_cows': val_cows,
        'accuracy': acc,
        'f1_macro': best_val_f1,
        'f1_weighted': f1_weighted,
        'time_sec': fold_time
    })
    print(f"Fold {fold_idx} (Val cows: {val_cows}) -> Acc: {acc:.4f}, Macro F1: {best_val_f1:.4f}, Weighted F1: {f1_weighted:.4f} ({fold_time:.1f}s)")

oof_pred_labels = np.argmax(oof_preds, axis=1)
overall_oof_acc = accuracy_score(y_train_val, oof_pred_labels)
overall_oof_macro = f1_score(y_train_val, oof_pred_labels, average='macro', zero_division=0)
overall_oof_weighted = f1_score(y_train_val, oof_pred_labels, average='weighted', zero_division=0)
print(f"\nOverall Out-Of-Fold CV -> Acc: {overall_oof_acc:.4f}, Macro F1: {overall_oof_macro:.4f}, Weighted F1: {overall_oof_weighted:.4f}")

# 7. Train Final CattleNet on all 8 cows and evaluate on Test (C09, C10)
print("\n--- Training Final CattleNet Model on all 8 Training Cows (C01-C08) ---")
train_all_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train_val_scaled), torch.from_numpy(y_train_val)),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test_scaled), torch.from_numpy(y_test)),
    batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True
)

final_cattlenet = CattleNet(input_dim=len(FEATURE_COLS), hidden_dim=128, num_classes=7, num_blocks=3, dropout=0.2).to(device)
optimizer = SimpleAdamW(final_cattlenet.parameters(), lr=BASE_LR, weight_decay=1e-4)
criterion = FocalLoss(alpha=class_weights_t, gamma=1.8)

TOTAL_FINAL_EPOCHS = 14
t_final_start = time.time()
for epoch in range(1, TOTAL_FINAL_EPOCHS + 1):
    current_lr = BASE_LR * 0.5 * (1.0 + np.cos(np.pi * (epoch - 1) / TOTAL_FINAL_EPOCHS))
    optimizer.set_lr(current_lr)
    
    final_cattlenet.train()
    total_loss = 0.0
    for bx, by in train_all_loader:
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = final_cattlenet(bx)
        loss = criterion(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_cattlenet.parameters(), 2.0)
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0 or epoch == TOTAL_FINAL_EPOCHS:
        print(f"Epoch {epoch}/{TOTAL_FINAL_EPOCHS} - Focal Loss: {total_loss / len(train_all_loader):.4f}")

print(f"Final model trained in {time.time() - t_final_start:.1f}s")

# 8. Evaluate on Unseen Test Cows (C09, C10)
print("\n--- Evaluating CattleNet on Unseen Test Cows (C09, C10) ---")
final_cattlenet.eval()
test_probs_list = []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device, non_blocking=True)
        logits = final_cattlenet(bx)
        probs = F.softmax(logits, dim=1)
        test_probs_list.append(probs.cpu().numpy())

test_probs = np.concatenate(test_probs_list, axis=0)
test_preds = np.argmax(test_probs, axis=1)

test_acc = accuracy_score(y_test, test_preds)
test_f1_macro = f1_score(y_test, test_preds, average='macro', zero_division=0)
test_f1_weighted = f1_score(y_test, test_preds, average='weighted', zero_division=0)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro F1: {test_f1_macro:.4f}")
print(f"Test Weighted F1: {test_f1_weighted:.4f}")

report_dict = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
report_str = classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
print("\nTest Classification Report:\n", report_str)

cm = confusion_matrix(y_test, test_preds)
print("Confusion Matrix:\n", cm)

# 9. Save Model Artifacts
torch.save(final_cattlenet.state_dict(), "models/checkpoints/cattlenet_deep_model.pt")
np.save("models/checkpoints/cattlenet_test_probs.npy", test_probs)
np.save("models/checkpoints/cattlenet_oof_probs.npy", oof_preds)

metrics = {
    'model_name': 'CattleNet (Deep Residual MLP + SE Attention + Focal Loss)',
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

with open("models/checkpoints/cattlenet_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)

print(f"\nCattleNet pipeline completed in {time.time() - start_time:.2f} seconds!")
