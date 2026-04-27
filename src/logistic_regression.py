import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler

# Loading data!
X = np.load('project/processed/X.npy') 
y = np.load('project/processed/y.npy')

# threshold for classifying as drunk
threshold = 0.65

# Making the data 2D to play nicer with scikit
if X.ndim > 2:
    X = X.reshape(X.shape[0], -1)

# Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model, of course
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Statistics stuff
probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs >= threshold).astype(int)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Visualize the model's outcome
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap=plt.cm.magma, ax=ax)
plt.title('Logistic Regression Confusion Matrix (Baseline)')
plt.show()

# This model yields about 76% accuracy, due to logistic regression just not playing nice with 
# time series data. Not only that, but time series data just also doesn't really do well with
# logistic regression, since the model will fail to take subsequent/previous data into account.