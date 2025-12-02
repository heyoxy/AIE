import kagglehub
import pandas as pd
import numpy as np
from scipy import stats
import os

path = kagglehub.dataset_download("julianbloise/winners-formula-1-1950-to-2025")
print("Path to dataset files:", path)

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in dataset path.")

dataset_file = os.path.join(path, csv_files[0])
print("Using dataset file:", dataset_file)

df = pd.read_csv(dataset_file)

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print("Numeric Columns:", numeric_cols.tolist())
print("Categorical Columns:", categorical_cols.tolist())

desc = pd.DataFrame(index=numeric_cols)
desc['Mean'] = df[numeric_cols].mean()
desc['Median'] = df[numeric_cols].median()
desc['Mode'] = df[numeric_cols].mode().iloc[0]

print("--- Mean, Median, Mode (Numeric Features) ---")
print(desc)

print("--- Balance Check (Categorical Features) ---")

for col in categorical_cols:
    counts = df[col].value_counts(normalize=True) * 100
    print(f"\n{col} Distribution (%):")
    print(counts.head(10))  # Avoid spam output

    # Simple imbalance check
    if counts.iloc[0] > 50:
        print(f"{col} is IMBALANCED (Top category = {counts.index[0]}: {counts.iloc[0]:.2f}%)")
    else:
        print(f"{col} is relatively BALANCED")
