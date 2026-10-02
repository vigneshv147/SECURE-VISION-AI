import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# --- Dark theme so these charts match the dashboard's dark UI ---
plt.style.use("dark_background")
DARK_BG = "#0e1117"

# --- Load the trained XGBoost model and selected features ---
model = joblib.load("models/xgboost.pkl")
selected_features = joblib.load("data/processed/selected_features.pkl")

# --- Load validation data (25 selected features) ---
X_val = np.load("data/processed/X_val_selected.npy")

print("Validation data shape:", X_val.shape)

# SHAP can be slow on huge datasets, so we take a random sample of 2000 rows
# This is a common, accepted practice - the summary plot patterns look the
# same whether you use 2000 rows or 200,000 rows, it just takes way less time
np.random.seed(42)
sample_indices = np.random.choice(X_val.shape[0], size=2000, replace=False)
X_sample = X_val[sample_indices]

print("Using a sample of", X_sample.shape[0], "rows for SHAP analysis")

# --- Create the SHAP explainer for the XGBoost (tree-based) model ---
print("\nCreating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

print("Calculating SHAP values... (this may take a minute)")
shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated. Shape:", shap_values.shape)

# --- Make sure the reports folder exists ---
os.makedirs("reports", exist_ok=True)

# --- 1. Summary plot (bar) - shows overall importance of each feature ---
plt.figure(facecolor=DARK_BG)
shap.summary_plot(
    shap_values, X_sample,
    feature_names=selected_features,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("reports/shap_summary_bar.png", facecolor=DARK_BG, dpi=150)
plt.close()
print("Saved reports/shap_summary_bar.png")

# --- 2. Summary plot (dot/beeswarm) - shows importance AND direction of effect ---
plt.figure(facecolor=DARK_BG)
shap.summary_plot(
    shap_values, X_sample,
    feature_names=selected_features,
    show=False
)
plt.tight_layout()
plt.savefig("reports/shap_summary_dot.png", facecolor=DARK_BG, dpi=150)
plt.close()
print("Saved reports/shap_summary_dot.png")

# --- 3. Waterfall plot for ONE single prediction (explains one specific flow) ---
# This shows exactly why row 0 in our sample was classified the way it was
single_explanation = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_sample[0],
    feature_names=selected_features
)

plt.figure(facecolor=DARK_BG)
shap.plots.waterfall(single_explanation, show=False)
plt.tight_layout()
plt.savefig("reports/shap_waterfall_example.png", facecolor=DARK_BG, dpi=150)
plt.close()
print("Saved reports/shap_waterfall_example.png")

print("\nSHAP explainability complete! Check the 'reports' folder for the 3 PNG files.")