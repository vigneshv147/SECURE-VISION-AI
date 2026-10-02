import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Load the cleaned (raw, unscaled) dataset from Step 1 ---
df = pd.read_csv("data/processed/cleaned.csv")
print("Loaded shape:", df.shape)

# --- Load the 25 feature names we selected earlier ---
selected_features = joblib.load("data/processed/selected_features.pkl")
print("\nUsing these 25 features:")
print(selected_features)

# --- Convert Label to binary (same as before: BENIGN=0, everything else=1) ---
df["Label"] = df["Label"].apply(lambda x: 0 if x.strip() == "BENIGN" else 1)

# --- Keep ONLY the 25 selected raw feature columns + label ---
X = df[selected_features]
y = df["Label"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Fit a NEW scaler using ONLY these 25 raw features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train a fresh XGBoost model on this 25-feature scaled data ---
print("\nTraining real-time XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("Training complete.")

# --- Quick check that accuracy is still good ---
y_pred = model.predict(X_test_scaled)
print("\n--- Real-time model test results ---")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1-score: ", f1_score(y_test, y_pred))

# --- Save the model, scaler, and feature list together ---
joblib.dump(model, "models/realtime_model.pkl")
joblib.dump(scaler, "models/realtime_scaler.pkl")
joblib.dump(selected_features, "models/realtime_features.pkl")

print("\nSaved models/realtime_model.pkl, realtime_scaler.pkl, realtime_features.pkl")
print("Ready to build the live packet capture script!")