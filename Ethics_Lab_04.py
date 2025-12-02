!pip install fairlearn
!pip install lime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate
)

# --- Step 1: Load dataset ---
df = pd.read_csv("StudentsPerformance.csv")

# --- Step 2: Basic preprocessing ---
# Encode categorical features
label_enc = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = label_enc.fit_transform(df[col])

# Create target variable: 1 if passed math (>50), else 0
df['passed_math'] = (df['math score'] >= 50).astype(int)

# Sensitive attribute (you can change to another column like 'race/ethnicity')
sensitive_feature = 'gender'

# Define features and target
X = df.drop(columns=['math score', 'passed_math'])
y = df['passed_math']

# --- Step 3: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Step 4: Standardize features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Step 5: Train model ---
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Step 6: Model performance ---
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.3f}")

# --- Step 7: Fairlearn MetricFrame ---
mf = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'selection_rate': selection_rate,
        'true_positive_rate': true_positive_rate,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df.loc[y_test.index, sensitive_feature]
)

print("\n=== Fairness Metrics by Sensitive Group ===") # Fixed indentation here
print(mf.by_group)

# --- Step 8: Equal Opportunity Difference ---
eo_diff = mf.difference(method='between_groups')['true_positive_rate']
print(f"\nEqual Opportunity Difference: {eo_diff:.3f}") # Fixed indentation here

# --- Step 9: Disparate Impact ---
# Ratio of minimum to maximum selection rate across groups
sr = mf.by_group['selection_rate']
disparate_impact = sr.min() / sr.max()
print(f"Disparate Impact (min/max selection rate): {disparate_impact:.3f}")

# --- Step 10: Predictive Parity ---
# Precision parity = ratio or difference of precision between groups
precision = {}
for g in df[sensitive_feature].unique():
    y_true_g = y_test[df.loc[y_test.index, sensitive_feature] == g]
    y_pred_g = y_pred[df.loc[y_test.index, sensitive_feature] == g]
    tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g).ravel()
    precision[g] = tp / (tp + fp + 1e-9)
print(f"\nPredictive Parity by Group: {precision}")

# --- Step 11: Threshold Limit Analysis ---
print("\n=== Threshold Fairness Check ===")
y_prob = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 1.0, 0.1)
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tpr = true_positive_rate(y_test, y_pred_t)
    print(f"Threshold {t:.1f}: True Positive Rate = {tpr:.3f}")

# -------------------------------------------------------------------------------------------------------
!pip install fairlearn
!pip install lime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate
)

# --- Step 1: Load dataset ---
df = pd.read_csv("StudentsPerformance.csv")

# --- Step 2: Basic preprocessing ---
# Encode categorical features
label_enc = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = label_enc.fit_transform(df[col])

# Create target variable: 1 if passed math (>50), else 0
df['passed_math'] = (df['math score'] >= 50).astype(int)

# Sensitive attribute (you can change to another column like 'race/ethnicity')
sensitive_feature = 'gender'

# Define features and target
X = df.drop(columns=['math score', 'passed_math'])
y = df['passed_math']

# --- Step 3: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Step 4: Standardize features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Step 5: Train model ---
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Step 6: Model performance ---
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.3f}")

# --- Step 7: Fairlearn MetricFrame ---
mf = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'selection_rate': selection_rate,
        'true_positive_rate': true_positive_rate,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df.loc[y_test.index, sensitive_feature]
)

print("\n=== Fairness Metrics by Sensitive Group ===")
print(mf.by_group)

# --- Step 8: Equal Opportunity Difference ---
eo_diff = mf.difference(method='between_groups')['true_positive_rate']
print(f"\nEqual Opportunity Difference: {eo_diff:.3f}")

# --- Step 9: Disparate Impact ---
# Ratio of minimum to maximum selection rate across groups
sr = mf.by_group['selection_rate']
disparate_impact = sr.min() / sr.max()
print(f"Disparate Impact (min/max selection rate): {disparate_impact:.3f}")

# --- Step 10: Predictive Parity ---
# Precision parity = ratio or difference of precision between groups
precision = {}
for g in df[sensitive_feature].unique():
    y_true_g = y_test[df.loc[y_test.index, sensitive_feature] == g]
    y_pred_g = y_pred[df.loc[y_test.index, sensitive_feature] == g]
    tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g).ravel()
    precision[g] = tp / (tp + fp + 1e-9)
print(f"\nPredictive Parity by Group: {precision}")

# --- Step 11: Threshold Limit Analysis ---
print("\n=== Threshold Fairness Check ===")
y_prob = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 1.0, 0.1)
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tpr = true_positive_rate(y_test, y_pred_t)
    print(f"Threshold {t:.1f}: True Positive Rate = {tpr:.3f}")
#-----------------------------------------------------------------------------
# LIME Explanation - Affairs Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from lime.lime_tabular import LimeTabularExplainer

# Step 1: Load dataset

df = pd.read_csv("Affairs.csv")

# Encode categorical variables if any
label_enc = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = label_enc.fit_transform(df[col])

# Target variable (assuming 'affair' column = 1 if affair, else 0)
target = 'affairs' if 'affairs' in df.columns else df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

# Step 2: Split and scale

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train model

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 4: LIME Explainer

explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    training_labels=y_train,
    feature_names=X.columns,
    class_names=['No Affair', 'Affair'],
    mode='classification'
)

# Step 5: Pick a sample to explain

i = 3  # any sample index
exp = explainer.explain_instance(
    data_row=X_test_scaled[i],
    predict_fn=model.predict_proba
)

# Step 6: Display results

print("\nLIME Explanation for Individual Record:")
print(f"Actual: {'Affair' if y_test.iloc[i]==1 else 'No Affair'}")
print(f"Predicted Probabilities: {model.predict_proba([X_test_scaled[i]])}")

exp.show_in_notebook(show_table=True)
# Or save explanation
exp.save_to_file('lime_affair.html')
print("LIME explanation saved as lime_affair.html")
