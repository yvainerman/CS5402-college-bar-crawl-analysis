import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

from preprocess import RESULTS_DIR

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

X = np.load("project/processed/seq/X.npy")
y = np.load("project/processed/seq/y.npy")
pids = np.load("project/processed/seq/pids.npy")

print(f"Total windows : {len(y)}")
print(f"Window shape  : {X.shape[1:]}")
print(f"Drunk windows : {int(np.sum(y == 1))}  ({100 * np.mean(y):.1f}%)")
print(f"Sober windows : {int(np.sum(y == 0))}")

# patient-level split.
# pick the two patient whos drunk rate is closest to 50 for test/val
all_pids = sorted(set(pids))
drunk_rate = {pid: float(np.mean(y[pids == pid])) for pid in all_pids}

balance_rank = sorted(all_pids, key=lambda p: abs(drunk_rate[p] - 0.5))
test_patient = balance_rank[0]
val_patient = balance_rank[1]
train_patients = [p for p in all_pids if p not in (test_patient, val_patient)]

print(f"\nTrain patients ({len(train_patients)}): {train_patients}")
print(f"Test patient : {test_patient}  (drunk rate {drunk_rate[test_patient]:.1%})")
print(f"Val patient : {val_patient}   (drunk rate {drunk_rate[val_patient]:.1%})")

train_mask = np.isin(pids, train_patients)
val_mask = pids == val_patient
test_mask = pids == test_patient

X_train = X[train_mask].astype(np.float32)
y_train = y[train_mask].astype(np.float32)
X_val = X[val_mask].astype(np.float32)
y_val = y[val_mask].astype(np.float32)
X_test = X[test_mask].astype(np.float32)
y_test = y[test_mask].astype(np.float32)

print(f"\nTraining: {len(y_train):,}  ({100 * y_train.mean():.1f}% drunk)")
print(f"Validation: {len(y_val):,}  ({100 * y_val.mean():.1f}% drunk)")
print(f"Test: {len(y_test):,}  ({100 * y_test.mean():.1f}% drunk)")

# per window standardization (per channel)
# each window is z-scored using only its OWN mean/std
# different patients hold their phone in different orientations, shifts axes
# diff phones/positions in pockets also shift axes.
# subtracting each windows own mean kills the patient-level baseline
def per_window_zscore(X):
    # X: (N, 200, 3) -> z-score across axis=1 (time), per channel
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)

X_train = per_window_zscore(X_train)
X_val = per_window_zscore(X_val)
X_test = per_window_zscore(X_test)


# pytorch tensors/loaders
def to_loader(X, y, batch_size, shuffle):
    X_t = torch.from_numpy(X).permute(0, 2, 1)   # (N, 3, 200)
    y_t = torch.from_numpy(y).float()
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
    )

train_loader = to_loader(X_train, y_train, batch_size=64,  shuffle=True)
val_loader = to_loader(X_val,   y_val,   batch_size=256, shuffle=False)
test_loader = to_loader(X_test,  y_test,  batch_size=256, shuffle=False)

# two conv blocks, then global avg pool and a single linear layer.
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3,  out_channels=16,
                               kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1) # global avg pool
        self.dropout = nn.Dropout(0.3) # regularize head
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, 3, 200)
        x = self.pool(self.relu(self.conv1(x))) # (B, 16, 100)
        x = self.pool(self.relu(self.conv2(x))) # (B, 32,  50)
        x = self.gap(x).squeeze(-1) # (B, 32)
        x = self.dropout(x) # only active in train mode
        return self.fc(x).squeeze(-1) # (B,)


model = SimpleCNN()
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: SimpleCNN  ({n_params:,} parameters)")


# train
n_pos = float((y_train == 1).sum())
n_neg = float((y_train == 0).sum())

pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

N_EPOCHS = 15

# axis augment. randomly flipping signs or permuting we force
# model to be invariant to orientation choices
def augment_axes(x):
    # x shape: (B, 3, 200)  -- channel dim is 1
    signs = (torch.randint(0, 2, (3,), device=x.device).float() * 2 - 1)
    x = x * signs.view(1, 3, 1)
    perm = torch.randperm(3, device=x.device)
    x = x[:, perm, :]
    return x


@torch.no_grad()
def predict_probs(loader):
    model.eval()
    probs = []
    for xb, _ in loader:
        xb = xb.to(device)

        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


print("\nTraining CNN ...")

# track best val AUC and model weights at that epoch.
best_val_auc = -1.0
best_state = None

train_loss_history = []
val_auc_history = []

for epoch in range(1, N_EPOCHS + 1):

    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        xb = augment_axes(xb)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)

    train_loss = total_loss / len(train_loader.dataset)
    train_loss_history.append(train_loss)

    val_probs_e = predict_probs(val_loader)
    if len(np.unique(y_val)) > 1:
        fpr_v, tpr_v, _ = roc_curve(y_val, val_probs_e)
        val_auc = auc(fpr_v, tpr_v)
        val_auc_history.append(val_auc)
        marker = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            marker = "  *"
        print(f"  epoch {epoch:2d}/{N_EPOCHS}  "
              f"train loss = {train_loss:.4f}   val AUC = {val_auc:.3f}{marker}")
    else:
        val_auc_history.append(np.nan)
        print(f"  epoch {epoch:2d}/{N_EPOCHS}  train loss = {train_loss:.4f}")

# restore best AUC epoch weights before any test-set evaluation
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Restored best weights (val AUC = {best_val_auc:.3f}).")

print("Training complete.")

# predict
val_probs  = predict_probs(val_loader)
test_probs = predict_probs(test_loader)


# choose threshold with YDJ on val set
thresholds = np.arange(0.05, 0.95, 0.01)
sens_curve = np.full_like(thresholds, np.nan, dtype=float)
spec_curve = np.full_like(thresholds, np.nan, dtype=float)
j_curve = np.full_like(thresholds, np.nan, dtype=float)

best_threshold = 0.5
best_score = -1.0

if len(np.unique(y_val)) > 1:
    for i, t in enumerate(thresholds):
        preds = (val_probs >= t).astype(int)
        tp = int(((preds == 1) & (y_val == 1)).sum())
        tn = int(((preds == 0) & (y_val == 0)).sum())
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fn = int(((preds == 0) & (y_val == 1)).sum())
        if (tp + fn) == 0 or (tn + fp) == 0:
            continue
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        j = sens + spec - 1
        sens_curve[i] = sens
        spec_curve[i] = spec
        j_curve[i] = j
        if j > best_score:
            best_score = j
            best_threshold = t

print(f"\nThreshold (chosen on val, Youden's J): {best_threshold:.2f}  "
      f"(val Youden J = {best_score:.3f})")


# test
y_pred = (test_probs >= best_threshold).astype(int)

tp = int(((y_pred == 1) & (y_test == 1)).sum())
tn = int(((y_pred == 0) & (y_test == 0)).sum())
fp = int(((y_pred == 1) & (y_test == 0)).sum())
fn = int(((y_pred == 0) & (y_test == 1)).sum())
sens = tp / (tp + fn) if (tp + fn) else 0
spec = tn / (tn + fp) if (tn + fp) else 0

print(f"\n--- Test set ---")
print(f"  Sensitivity : {sens:.3f}")
print(f"  Specificity : {spec:.3f}")
print("\nClassification Report:\n")
print(classification_report(
    y_test.astype(int), y_pred,
    target_names=["Sober", "Intoxicated"],
    zero_division=0,
))


# plots
fpr, tpr, _ = roc_curve(y_test, test_probs)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="darkorange", lw=2,
        label=f"CNN (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve -- CNN (patient split)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cnn_roc.png"), dpi=150, bbox_inches="tight")
plt.close()

cm = confusion_matrix(y_test.astype(int), y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=["Sober", "Intoxicated"]).plot(
    cmap=plt.cm.Blues, ax=ax,
)
plt.title("Confusion Matrix -- CNN (patient split)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cnn_cm.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved plots to {RESULTS_DIR}")
print(f"FINAL: AUC = {roc_auc:.4f}")