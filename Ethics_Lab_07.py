!pip install pandas numpy matplotlib seaborn scipy openml

!pip install scikit-learn shap lime


import openml, shap, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------- Simple config ----------
SENSITIVE = ['sex', 'age', 'embarked']   # features to check
EPS = 0.05               # relative-share threshold for flagging
TOPK = 5                 # rank threshold to flag
SHAP_SAMPLE = 300        # how many test rows to explain (max)
# -----------------------------------

# 1) Load and keep a few columns
ds = openml.datasets.get_dataset(40945)   # Titanic
X_raw, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format='dataframe')
df = X_raw.copy()
df['survived'] = y
cols = ['pclass','sex','age','sibsp','parch','fare','embarked','survived']
df = df[cols].dropna().reset_index(drop=True)

# 2) Prepare features (one-hot for categorical)
y = df['survived'].astype(int)
X = df.drop(columns=['survived'])
cat_cols = ['sex','embarked','pclass']
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
feature_names = X_enc.columns.tolist()

# 3) Split & train
X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = train_test_split(
    X_enc, y, X, test_size=0.25, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# 4) SHAP explanations (robust handling)
sample = X_test.sample(n=min(SHAP_SAMPLE, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_out = explainer.shap_values(sample)

# Normalize shap_out -> 2D array shap_vals (n_samples x n_features)
if isinstance(shap_out, list):
    shap_vals = shap_out[1] if len(shap_out) > 1 else shap_out[0]
elif isinstance(shap_out, np.ndarray):
    if shap_out.ndim == 3:            # (n, features, n_classes)
        idx = 1 if shap_out.shape[2] > 1 else 0
        shap_vals = shap_out[:, :, idx]
    elif shap_out.ndim == 2:
        shap_vals = shap_out
    else:
        raise ValueError("Unhandled SHAP shape: " + str(shap_out.shape))
else:
    raise ValueError("Unhandled SHAP return type")

shap_abs = pd.DataFrame(np.abs(shap_vals), columns=sample.columns, index=sample.index)

# 5) Aggregate: mean, std, rank, relative share
agg = pd.DataFrame({
    'feature': shap_abs.columns,
    'mean_abs': shap_abs.mean(axis=0),
    'std_abs': shap_abs.std(axis=0)
}).sort_values('mean_abs', ascending=False).reset_index(drop=True)
agg['rank'] = np.arange(1, len(agg)+1)
agg['relative_share'] = agg['mean_abs'] / agg['mean_abs'].sum()

print("\nTop features by mean |SHAP|:")
print(agg.head(10))

# 6) Simple flagging for sensitive features
flags = []
for s in SENSITIVE:
    related = [f for f in agg['feature'] if f.startswith(s + '_') or f == s]
    if not related and s in agg['feature'].values:
        related = [s]
    if not related:
        print(f" - Sensitive feature '{s}' not found (skipping).")
        continue
    mean_sum = agg.set_index('feature').loc[related, 'mean_abs'].sum()
    rel = mean_sum / agg['mean_abs'].sum()
    ranks = agg.set_index('feature').loc[related, 'rank']
    min_rank = int(ranks.min())
    flagged = (rel > EPS) or (min_rank <= TOPK)
    flags.append({'sensitive': s, 'related_cols': related, 'relative_share': rel, 'min_rank': min_rank, 'flagged': flagged})

print("\nSensitive feature checks:")
print(pd.DataFrame(flags))

# ---------------------------
# Minimal-fix perturbation function (handles pandas CategoricalDtype)
# ---------------------------
def perturb_rows(orig_df, feature):
    dfp = orig_df.copy()
    if feature not in dfp.columns:
        return dfp
    # Use pandas dtype check (safe for categorical dtypes)
    if pd.api.types.is_numeric_dtype(dfp[feature].dtype):
        dfp[feature] = dfp[feature].median()
    else:
        vals = dfp[feature].dropna().unique().tolist()
        if len(vals) > 1:
            mode = dfp[feature].mode().iloc[0]
            alt = next((v for v in vals if v != mode), vals[0])
            dfp[feature] = alt
    return dfp

# 7) Simple perturbation: numeric->median, categorical->swap to other value
perturb_orig = X_test_orig.sample(n=min(200, len(X_test_orig)), random_state=42)
perturb_enc = X_test.loc[perturb_orig.index]
if hasattr(model, "predict_proba"):
    orig_probs = model.predict_proba(perturb_enc)[:, 1]
else:
    orig_probs = model.predict(perturb_enc)

perturb_stats = []
for s in SENSITIVE:
    if s not in X.columns:
        print(f"Skipping perturbation for '{s}' (not in original X columns).")
        continue
    pert = perturb_rows(perturb_orig, s)
    pert_enc = pd.get_dummies(pert, columns=cat_cols, drop_first=False).reindex(columns=feature_names, fill_value=0)
    if hasattr(model, "predict_proba"):
        pert_probs = model.predict_proba(pert_enc)[:, 1]
    else:
        pert_probs = model.predict(pert_enc)
    delta = np.abs(orig_probs - pert_probs)
    perturb_stats.append({'sensitive': s, 'mean_abs_delta': float(delta.mean())})

print("\nPerturbation results (mean absolute change in predicted prob):")
print(pd.DataFrame(perturb_stats))

# 8) Plot top features
plt.figure(figsize=(7,6))
topn = min(15, len(agg))
sns.barplot(x='mean_abs', y='feature', data=agg.head(topn))
plt.title('Top features by mean |SHAP|')
plt.xlabel('Mean |SHAP|')
plt.tight_layout()
plt.show()
