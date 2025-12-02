!pip install -q kagglehub pandas numpy matplotlib seaborn scikit-learn

import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

path = kagglehub.dataset_download("julianbloise/winners-formula-1-1950-to-2025")
print("Path to dataset files:", path)

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in dataset path.")

dataset_file = os.path.join(path, csv_files[0])
print("Using dataset file:", dataset_file)

df = pd.read_csv(dataset_file)

df['favorable'] = (df['team'] == 'Ferrari').astype(int)

sensitive = 'continent'
outcome = 'favorable'

def fairness_metrics(data, s=sensitive, y=outcome):
    g1 = data[data[s] == data[s].unique()[0]][y]
    g0 = data[data[s] != data[s].unique()[0]][y]
    return {
        "Demographic Parity Diff": abs(g1.mean() - g0.mean()),
        "Disparate Impact Ratio": g0.mean() / g1.mean() if g1.mean() > 0 else 0,
        "Equal Opportunity Diff": abs(g1.mean() - g0.mean()),
    }

print("📊 Before Mitigation:", fairness_metrics(df))
sns.barplot(x=df[sensitive], y=df[outcome])
plt.title(f"Ferrari Win Rate by {sensitive} (Before)")
plt.show()

majority_grp = df[sensitive].value_counts().idxmax()
df_majority = df[df[sensitive] == majority_grp]
df_minority = df[df[sensitive] != majority_grp]
df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)
df_bal = pd.concat([df_majority, df_minority_upsampled])

X_train, X_test, y_train, y_test = train_test_split(
    df_bal.drop(columns=[outcome]), df_bal[outcome], test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train.select_dtypes(include=[np.number]), y_train)

y_pred = model.predict(X_test.select_dtypes(include=[np.number]))
acc = accuracy_score(y_test, y_pred)

print("\nModel Accuracy After Mitigation:", acc)
print("\nAfter Mitigation:", fairness_metrics(df_bal))

sns.barplot(x=df_bal[sensitive], y=df_bal[outcome])
plt.title(f"Ferrari Win Rate by {sensitive} (After)")
plt.show()

comparison = pd.DataFrame([fairness_metrics(df), fairness_metrics(df_bal)], index=["Before", "After"])
print("\nComparison:\n", comparison)

