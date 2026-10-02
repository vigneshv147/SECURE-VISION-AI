"""
model_utils.py
----------------
Backend logic: loading models/data and computing predictions/metrics.
dashboard.py imports these functions and only handles displaying results.
"""

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def load_models():
    """Load the trained Random Forest, XGBoost, and Logistic Regression models."""
    rf_model = joblib.load("models/random_forest.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")
    lr_model = joblib.load("models/logistic_regression.pkl")
    return rf_model, xgb_model, lr_model


def load_validation_data():
    """Load validation data for both models (RF uses 78 features, XGBoost uses 25)."""
    X_val_rf = np.load("data/processed/X_val.npy")
    X_val_xgb = np.load("data/processed/X_val_selected.npy")
    y_val = np.load("data/processed/y_val.npy")
    return X_val_rf, X_val_xgb, y_val


def load_test_data():
    """Load test data (25 selected features) for the live detection demo."""
    X_test = np.load("data/processed/X_test_selected.npy")
    y_test = np.load("data/processed/y_test.npy")
    return X_test, y_test


def load_feature_names():
    """Load the list of 25 selected feature names."""
    return joblib.load("data/processed/selected_features.pkl")


def get_metrics(y_true, y_pred, y_proba):
    """Compute the standard 5 classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def get_confusion_matrix(y_true, y_pred):
    """Compute the confusion matrix as a 2x2 array."""
    return confusion_matrix(y_true, y_pred)


def evaluate_model(model, X, y_true):
    """Run a model on data and return predictions, probabilities, and metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = get_metrics(y_true, y_pred, y_proba)
    cm = get_confusion_matrix(y_true, y_pred)
    return y_pred, y_proba, metrics, cm


def predict_flow(model, feature_vector):
    """
    Predict a single network flow.
    feature_vector should be a 2D array of shape (1, n_features).
    Returns: (predicted_label, attack_probability)
    """
    prediction = model.predict(feature_vector)[0]
    probability = model.predict_proba(feature_vector)[0][1]
    label = "ATTACK" if prediction == 1 else "BENIGN"
    return label, probability


def load_attack_type_distribution():
    """Read the original (pre-binarization) attack type labels from cleaned.csv.
    Only reads the Label column to keep this fast on a 2.5M row file."""
    import pandas as pd
    df = pd.read_csv("data/processed/cleaned.csv", usecols=["Label"])
    counts = df["Label"].value_counts()
    return counts


BLOCKED_IPS_FILE = "data/blocked_ips.csv"


def load_realtime_pipeline():
    """Load the model+scaler+feature-list trained on RAW (unscaled) selected
    features. This is the correct pipeline to use for predicting on new,
    user-uploaded CSV data, since it expects raw values, not pre-scaled ones."""
    model = joblib.load("models/realtime_model.pkl")
    scaler = joblib.load("models/realtime_scaler.pkl")
    features = joblib.load("models/realtime_features.pkl")
    return model, scaler, features


def predict_csv(df):
    """
    Run batch predictions on an uploaded DataFrame.
    Expects df to contain (at least) the 25 required feature columns.
    Returns: (results_df, missing_columns)
    results_df has 'Prediction' and 'Confidence' columns appended.
    """
    import numpy as np

    model, scaler, features = load_realtime_pipeline()

    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = [f for f in features if f not in df.columns]
    if missing:
        return None, missing

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    results_df = df.copy()
    results_df["Prediction"] = ["ATTACK" if p == 1 else "BENIGN" for p in predictions]
    results_df["Confidence"] = probabilities

    return results_df, []


def load_full_log():
    """Load the entire detection log (no row limit) for the Reports page."""
    import pandas as pd
    import os
    log_path = "logs/detection_log.csv"
    if not os.path.exists(log_path):
        return None
    return pd.read_csv(log_path)


def filter_log(df, label=None, event_type=None, date_from=None, date_to=None, ip=None):
    """
    Apply optional filters to the detection log.
    label: 'ATTACK' or 'BENIGN' (or None/'' for all)
    event_type: 'FLOW', 'PORT_SCAN', 'BLOCKED_IP' (or None/'' for all)
    date_from, date_to: 'YYYY-MM-DD' strings (or None/'' to skip)
    ip: substring to search for in source or destination
    """
    import pandas as pd

    filtered = df.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], errors="coerce")

    if label:
        filtered = filtered[filtered["label"] == label]
    if event_type:
        filtered = filtered[filtered["type"] == event_type]
    if date_from:
        filtered = filtered[filtered["timestamp"] >= pd.to_datetime(date_from)]
    if date_to:
        # include the whole end day
        end = pd.to_datetime(date_to) + pd.Timedelta(days=1)
        filtered = filtered[filtered["timestamp"] < end]
    if ip:
        mask = filtered["source"].astype(str).str.contains(ip, case=False, na=False) | \
               filtered["destination"].astype(str).str.contains(ip, case=False, na=False)
        filtered = filtered[mask]

    return filtered.sort_values("timestamp", ascending=False)


def load_blocked_ips():
    """Return the current blocklist as a DataFrame (ip, reason, blocked_at)."""
    import pandas as pd
    import os
    if not os.path.exists(BLOCKED_IPS_FILE):
        return pd.DataFrame(columns=["ip", "reason", "blocked_at"])
    return pd.read_csv(BLOCKED_IPS_FILE)


def add_blocked_ip(ip, reason="Manual block"):
    """Add an IP to the blocklist. Does nothing if it's already blocked."""
    import pandas as pd
    import os
    from datetime import datetime

    os.makedirs("data", exist_ok=True)
    df = load_blocked_ips()

    if ip in df["ip"].values:
        return False  # already blocked

    new_row = pd.DataFrame([{
        "ip": ip, "reason": reason,
        "blocked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(BLOCKED_IPS_FILE, index=False)
    return True


def remove_blocked_ip(ip):
    """Remove an IP from the blocklist."""
    df = load_blocked_ips()
    df = df[df["ip"] != ip]
    df.to_csv(BLOCKED_IPS_FILE, index=False)


def load_detection_log(n_rows=50):
    """Read the most recent detection events written by realtime_detection.py."""
    import pandas as pd
    import os
    log_path = "logs/detection_log.csv"
    if not os.path.exists(log_path):
        return None
    df = pd.read_csv(log_path)
    return df.tail(n_rows).iloc[::-1]  # most recent first


def get_random_test_sample(X_test, y_test):
    """Pick a random row from the test set for the live demo."""
    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx].reshape(1, -1)
    true_label = "ATTACK" if y_test[idx] == 1 else "BENIGN"
    return sample, true_label
