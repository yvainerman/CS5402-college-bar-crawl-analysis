import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
)

from preprocess import RESULTS_DIR

X = np.load("project/processed/tab/X.npy")
y = np.load("project/processed/tab/y.npy")
pids = np.load("project/processed/tab/pids.npy")

if X.ndim > 2:
    X = X.reshape(X.shape[0], -1)

print(f"Total windows : {len(y)}")
print(f"Total features: {X.shape[1]}")
print(f"Drunk windows : {int(np.sum(y == 1))}  ({100 * np.mean(y):.1f}%)")
print(f"Sober windows : {int(np.sum(y == 0))}")

# stratified patient-level split
all_pids = sorted(set(pids))
drunk_rate = {pid: float(np.mean(y[pids == pid])) for pid in all_pids}

# rank patients by closeness-to-50%, pick top two
balance_rank = sorted(all_pids, key=lambda p: abs(drunk_rate[p] - 0.5))
test_patient = balance_rank[0]
val_patient = balance_rank[1]
train_patients = [p for p in all_pids if p not in (test_patient, val_patient)]

print(f"\nTrain patients ({len(train_patients)}): {train_patients}")
print(f"Test  patient : {test_patient}  (drunk rate {drunk_rate[test_patient]:.1%})")
print(f"Val   patient : {val_patient}   (drunk rate {drunk_rate[val_patient]:.1%})")

train_mask = np.isin(pids, train_patients)
val_mask = pids == val_patient
test_mask = pids == test_patient

X_train = X[train_mask].astype(np.float64)
y_train = y[train_mask]
X_val = X[val_mask].astype(np.float64)
y_val = y[val_mask]
X_test = X[test_mask].astype(np.float64)
y_test = y[test_mask]

print(f"\nTraining   : {len(y_train):,}  ({100 * y_train.mean():.1f}% drunk)")
print(f"Validation : {len(y_val):,}  ({100 * y_val.mean():.1f}% drunk)")
print(f"Test       : {len(y_test):,}  ({100 * y_test.mean():.1f}% drunk)")

# scale with train stats
means = X_train.mean(axis=0)
stds = X_train.std(axis=0)
stds[stds == 0] = 1.0

X_train = (X_train - means) / stds
X_val   = (X_val   - means) / stds
X_test  = (X_test  - means) / stds

# train
print("\nTraining Logistic Regression ...")

model = LogisticRegression(
    class_weight = "balanced",
    C = 0.05,
    max_iter = 2000,
    random_state = 42,
)
model.fit(X_train, y_train)
print("Training complete.")

# YDJ
val_probs = model.predict_proba(X_val)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]

best_threshold = 0.5
best_score = -1.0

if len(np.unique(y_val)) > 1:
    for t in np.arange(0.05, 0.95, 0.01):
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
    y_test, y_pred,
    target_names=["Sober", "Intoxicated"],
    zero_division=0,
))


# plots
fpr, tpr, _ = roc_curve(y_test, test_probs)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="steelblue", lw=2,
        label=f"Logistic Regression (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve -- Logistic Regression (patient split)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "logreg_roc.png"), dpi=150, bbox_inches="tight")
plt.close()

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=["Sober", "Intoxicated"]).plot(
    cmap=plt.cm.Blues, ax=ax,
)
plt.title("Confusion Matrix -- Logistic Regression (patient split)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "logreg_cm.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved plots to {RESULTS_DIR}")
print(f"FINAL: AUC = {roc_auc:.4f}")