
# Improved membership-inference demo (threshold attack)
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

np.random.seed(0)

# Load & scale data (scaling avoids convergence warnings)
X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into target train (members) and target test (non-members)
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y
)

# Train target model (increase max_iter)
target = LogisticRegression(max_iter=2000, solver='lbfgs')
target.fit(X_target_train, y_target_train)
print("Target model test accuracy:", f"{target.score(X_target_test, y_target_test):.4f}")

# Compute confidence for true label
def true_label_conf(model, X, y):
    probs = model.predict_proba(X)
    return probs[np.arange(len(y)), y]

s_train = true_label_conf(target, X_target_train, y_target_train)   # members
s_test  = true_label_conf(target, X_target_test,  y_target_test)    # non-members

# Prepare attack evaluation set
s_attack = np.hstack([s_train, s_test])
y_attack_true = np.hstack([np.ones(len(s_train)), np.zeros(len(s_test))])  # 1=member

# 1) midpoint threshold (mid between mean train & mean test)
thr_mid = 0.5 * (s_train.mean() + s_test.mean())

# 2) optimal threshold from ROC (Youden's J = tpr - fpr)
fpr, tpr, ths = roc_curve(y_attack_true, s_attack)
youden_idx = np.argmax(tpr - fpr)
thr_youden = ths[youden_idx]

def eval_attack(threshold):
    y_pred = (s_attack > threshold).astype(int)
    acc = accuracy_score(y_attack_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_attack_true, y_pred, average='binary', zero_division=0)
    return acc, prec, recall, f1

acc_mid, prec_mid, recall_mid, f1_mid = eval_attack(thr_mid)
acc_y, prec_y, recall_y, f1_y = eval_attack(thr_youden)
auc = roc_auc_score(y_attack_true, s_attack)

print("\nAttack AUC (confidence as score):", f"{auc:.4f}")
print("\nMidpoint threshold results:")
print(f"  threshold={thr_mid:.4f}  -> acc={acc_mid:.4f}, prec={prec_mid:.4f}, recall={recall_mid:.4f}, F1={f1_mid:.4f}")

print("\nYouden (ROC-optimal) threshold results:")
print(f"  threshold={thr_youden:.4f}  -> acc={acc_y:.4f}, prec={prec_y:.4f}, recall={recall_y:.4f}, F1={f1_y:.4f}")
