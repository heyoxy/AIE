# LIME Explanation - StudentsPerformance Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from lime.lime_tabular import LimeTabularExplainer

# Step 1: Load and preprocess

df = pd.read_csv("StudentsPerformance.csv")

# Encode categorical features
label_enc = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = label_enc.fit_transform(df[col])

# Create target: passed_math = 1 if math score >= 50
df['passed_math'] = (df['math score'] >= 50).astype(int)

X = df.drop(columns=['math score', 'passed_math'])
y = df['passed_math']

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

# Step 4: Create LIME Explainer

explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    training_labels=y_train,
    feature_names=X.columns,
    class_names=['Fail', 'Pass'],
    mode='classification'
)

# Step 5: Pick a sample to explain

i = 5  # You can change this index
exp = explainer.explain_instance(
    data_row=X_test_scaled[i],
    predict_fn=model.predict_proba
)

# Step 6: Display explanation

print("\nLIME Explanation for Student Sample:")
print(f"Actual Label: {'Pass' if y_test.iloc[i]==1 else 'Fail'}")
print(f"Predicted Probabilities: {model.predict_proba([X_test_scaled[i]])}")

exp.show_in_notebook(show_table=True)
# Or if not in Jupyter:
exp.save_to_file('lime_student.html')
print("LIME explanation saved as lime_student.html")