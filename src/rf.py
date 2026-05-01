import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Loading data!
X = np.load('processed/X.npy')
y = np.load('processed/y.npy')

# threshold for classifying as drunk
threshold = 0.5

# Making the data 2D to play nicer with scikit
if X.ndim > 2:
    X = X.reshape(X.shape[0], -1)

# Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Statistics stuff
probs = model.predict_proba(X_test)[:, 1]

def plot_roc_curve(y_test, probs):
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='magenta', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Random Forest')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
plot_roc_curve(y_test=y_test, probs=probs)

def find_best_threshold(y_test, probs):
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    # Youden's J statistic: maximizes (sensitivity + specificity - 1)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"  TPR (Sensitivity): {tpr[best_idx]:.4f}")
    print(f"  FPR (1 - Specificity): {fpr[best_idx]:.4f}")

    return best_threshold
best_threshold = find_best_threshold(y_test, probs)
y_pred = (probs >= best_threshold).astype(int)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Visualize the model's outcome
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap=plt.cm.magma, ax=ax)
plt.title('Random Forest Confusion Matrix (Baseline)')
plt.show()
