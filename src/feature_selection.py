import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# --- Use a dark theme for all plots so they match the dashboard's dark UI ---
plt.style.use("dark_background")
DARK_BG = "#0e1117"
ACCENT = "#58a6ff"

# --- Load data and the trained Random Forest model ---
X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
X_test = np.load("data/processed/X_test.npy")

feature_names = joblib.load("data/processed/feature_columns.pkl")
model = joblib.load("models/random_forest.pkl")

print("Total features before selection:", len(feature_names))

# Put X_train into a DataFrame so we can use column names for correlation analysis
X_train_df = pd.DataFrame(X_train, columns=feature_names)

# --- Step 1: Get feature importance from the trained Random Forest ---
importances = pd.Series(model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

print("\nTop 20 most important features:")
print(importances.head(20))

# Save a bar chart of top 20 features
os.makedirs("reports", exist_ok=True)
fig = plt.figure(figsize=(10, 8), facecolor=DARK_BG)
ax = plt.gca()
ax.set_facecolor(DARK_BG)
importances.head(20).sort_values().plot(kind="barh", color=ACCENT, ax=ax)
plt.title("Top 20 Feature Importances (Random Forest)", color="white")
plt.xlabel("Importance", color="white")
ax.tick_params(colors="white")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", facecolor=DARK_BG, dpi=150)
print("\nSaved chart to reports/feature_importance.png")

# --- Step 2: Correlation analysis - drop highly correlated redundant features ---
# Compute correlation matrix on the training data
corr_matrix = X_train_df.corr().abs()

# Only look at upper triangle to avoid checking each pair twice
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Find features that are highly correlated (> 0.9) with another feature
to_drop = set()
for column in upper_triangle.columns:
    correlated_features = upper_triangle.index[upper_triangle[column] > 0.9].tolist()
    for other_feature in correlated_features:
        # Between the two correlated features, drop the one with LOWER importance
        if importances[column] >= importances[other_feature]:
            to_drop.add(other_feature)
        else:
            to_drop.add(column)

print(f"\nDropping {len(to_drop)} highly correlated (redundant) features:")
print(to_drop)

# --- Step 3: Build final feature list ---
# Start from importance ranking, remove the correlated/redundant ones
remaining_features = [f for f in importances.index if f not in to_drop]

# Keep the top 25 of what's left
TOP_N = 25
selected_features = remaining_features[:TOP_N]

print(f"\nFinal selected features ({len(selected_features)}):")
print(selected_features)

# --- Step 4: Save reduced datasets using only selected features ---
# Get the column indices of the selected features in the original feature list
selected_indices = [feature_names.index(f) for f in selected_features]

X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

np.save("data/processed/X_train_selected.npy", X_train_selected)
np.save("data/processed/X_val_selected.npy", X_val_selected)
np.save("data/processed/X_test_selected.npy", X_test_selected)
joblib.dump(selected_features, "data/processed/selected_features.pkl")

print("\nSaved X_train_selected.npy, X_val_selected.npy, X_test_selected.npy, selected_features.pkl")
print("Feature selection complete!")