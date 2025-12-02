!pip download kagglehub
!pip download pandas
!pip download numpy
!pip download matplotlib
!pip download seaborn
!pip download scipy

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

print("Dataset Shape:", df.shape)
print("Dataset Info:")
print(df.info())

numeric_cols = df.select_dtypes(include=[np.number]).columns
print("Numeric Columns:", numeric_cols.tolist())

desc = pd.DataFrame(index=numeric_cols)
desc['Mean'] = df[numeric_cols].mean()
desc['Median'] = df[numeric_cols].median()
desc['Mode'] = df[numeric_cols].mode().iloc[0]

desc['Std Dev'] = df[numeric_cols].std()
desc['Skewness'] = df[numeric_cols].skew()
desc['Kurtosis'] = df[numeric_cols].kurt()

print("--- Statistical Summary ---")
print(desc)

print("--- Normality Tests (Shapiro-Wilk) ---")
for col in numeric_cols:
    try:
        stat, p = stats.shapiro(df[col].dropna())
        print(f"{col}: W={stat:.3f}, p={p:.3f} --> {'Normal' if p>0.05 else 'Not Normal'}")
    except Exception as e:
        print(f"{col}: Could not run Shapiro-Wilk ({e})")

for col in numeric_cols:
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f"Histogram of {col}")

    plt.subplot(1,2,2)
    sns.boxplot(x=df[col].dropna())
    plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    df[col].value_counts().head(20).plot(kind='bar')  # Top 20 to avoid clutter
    plt.title(f"Class Distribution: {col}")
    plt.ylabel("Count")
    plt.show()

if len(numeric_cols) > 1:
    plt.figure(figsize=(10,6))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap (Numeric Features)", fontsize=14)
    plt.show()
else:
    print("Not enough numeric columns for correlation heatmap.")