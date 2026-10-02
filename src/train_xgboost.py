import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# --- Load the SELECTED (25-feature) preprocessed data ---
X_train = np.load("data/processed/X_train_selected.npy")
X_val = np.load("data/processed/X_val_selected.npy")
y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# --- Train XGBoost ---
print("\nTraining XGBoost... (this may take a few minutes)")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete.")

# --- Evaluate on validation set ---
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

print("\n--- Validation Results (XGBoost) ---")
print("Accuracy: ", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:   ", recall_score(y_val, y_pred))
print("F1-score: ", f1_score(y_val, y_pred))
print("ROC-AUC:  ", roc_auc_score(y_val, y_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nFull Classification Report:")
print(classification_report(y_val, y_pred, target_names=["BENIGN", "ATTACK"]))

# --- Save the trained model ---
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost.pkl")
print("\nSaved trained model to models/xgboost.pkl")