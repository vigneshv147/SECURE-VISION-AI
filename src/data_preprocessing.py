import os
import glob
import pandas as pd
import numpy as np

# Folder where the 8 CICIDS2017 CSV files are kept
raw_folder = "data/raw"

# Folder where we will save the cleaned output
processed_folder = "data/processed"
os.makedirs(processed_folder, exist_ok=True)

# Get list of all CSV files in the raw folder
csv_files = glob.glob(raw_folder + "/*.csv")
print("Found files:", csv_files)

# Load and combine all CSV files into one big dataframe
all_data = []
for file in csv_files:
    print("Reading:", file)
    data = pd.read_csv(file, encoding="latin1", low_memory=False)
    all_data.append(data)

df = pd.concat(all_data, ignore_index=True)
print("Combined shape:", df.shape)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove duplicate rows
df = df.drop_duplicates()

# Replace infinity values with NaN, then drop rows with NaN
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("Shape after cleaning:", df.shape)

# Show how many rows belong to each label (BENIGN, DDoS, PortScan, etc.)
print(df["Label"].value_counts())

# Save the cleaned data
df.to_csv(processed_folder + "/cleaned.csv", index=False)
print("Saved cleaned file to data/processed/cleaned.csv")