import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Load the cleaned combined data from step 1
df = pd.read_csv("data/processed/cleaned.csv")
print("Loaded shape:", df.shape)

# --- Convert Label to binary: BENIGN = 0, everything else = 1 (ATTACK) ---
df["Label"] = df["Label"].apply(lambda x: 0 if x.strip() == "BENIGN" else 1)

print("\nBinary label counts:")
print(df["Label"].value_counts())

# --- Separate features (X) and target (y) ---
X = df.drop("Label", axis=1)
y = df["Label"]

# Keep only numeric columns (some CICIDS files have stray non-numeric columns)
X = X.select_dtypes(include=[np.number])

print("\nNumber of feature columns used:", X.shape[1])

# --- Split into train (70%), validation (15%), test (15%), stratified ---
# Step 1: split off 70% train, 30% temp (which becomes val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Step 2: split the 30% temp into two equal halves -> 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nTrain shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# --- Scale features (fit scaler on training data only, then apply to val and test) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- Save everything for the next step (model training) ---
os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X_train.npy", X_train_scaled)
np.save("data/processed/X_val.npy", X_val_scaled)
np.save("data/processed/X_test.npy", X_test_scaled)
np.save("data/processed/y_train.npy", y_train.values)
np.save("data/processed/y_val.npy", y_val.values)
np.save("data/processed/y_test.npy", y_test.values)

joblib.dump(scaler, "data/processed/scaler.pkl")
joblib.dump(list(X.columns), "data/processed/feature_columns.pkl")

print("\nSaved X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy, scaler.pkl, feature_columns.pkl")
print("Ready for model training!")