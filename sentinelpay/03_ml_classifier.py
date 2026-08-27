"""
SentinelPay AI - ML Classification Layer
===========================================
Supervised model trained on KNOWN fraud patterns only.
Deliberately excludes 'zero_day_drain' from the TRAINING labels
(it's still in the test set for evaluation) so you can show, in your
demo, that this layer alone misses the novel pattern - motivating the
Isolation Forest layer.

Uses XGBoost if available; falls back to sklearn's
HistGradientBoostingClassifier (same gradient-boosted-trees family,
built into sklearn, no install needed) so this script runs anywhere.
On your own machine just `pip install xgboost` and it'll use that.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                              classification_report, average_precision_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

df = pd.read_csv("features.csv", parse_dates=["timestamp"])

FEATURES = [
    "amount", "hour", "is_odd_hour", "is_new_device", "is_new_merchant",
    "amount_zscore", "amount_vs_user_avg", "seconds_since_last_txn",
    "txns_last_1h", "device_shared_users", "rule_score",
]

# --- Simulate "we've never seen a zero-day sample" at TRAIN time ---------
train_pool = df[df["fraud_type"] != "zero_day_drain"].copy()
zero_day = df[df["fraud_type"] == "zero_day_drain"].copy()

X = train_pool[FEATURES]
y = train_pool["is_fraud"]
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, train_pool.index, test_size=0.25, random_state=3, stratify=y
)

# add the withheld zero-day transactions into the TEST set only
X_test = pd.concat([X_test, zero_day[FEATURES]])
y_test = pd.concat([y_test, zero_day["is_fraud"]])
idx_test = idx_test.append(zero_day.index) if hasattr(idx_test, "append") else np.concatenate([idx_test, zero_day.index])

if HAS_XGB:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=3,
    )
else:
    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.1, random_state=3,
        class_weight="balanced",
    )

model.fit(X_train, y_train)

proba_test = model.predict_proba(X_test)[:, 1]
print(f"Model: {'XGBoost' if HAS_XGB else 'sklearn HistGradientBoostingClassifier (xgboost not installed)'}")
print(f"ROC-AUC (overall test set incl. zero-day): {roc_auc_score(y_test, proba_test):.3f}")
print(f"Average Precision:                          {average_precision_score(y_test, proba_test):.3f}")
print(classification_report(y_test, (proba_test >= 0.5).astype(int), digits=3))

# --- The point of this script: show recall SPECIFICALLY on zero-day ------
zd_proba = model.predict_proba(zero_day[FEATURES])[:, 1]
print(f"\nZero-day-only recall @0.5 threshold: {(zd_proba >= 0.5).mean():.1%}  "
      f"(this is what Isolation Forest layer needs to cover)")

# --- Feature importance (doubles as input to Explainability layer) -------
if HAS_XGB:
    importances = model.feature_importances_
else:
    # HistGradientBoostingClassifier has no feature_importances_; use permutation importance
    from sklearn.inspection import permutation_importance
    importances = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=3).importances_mean

imp_df = pd.DataFrame({"feature": FEATURES, "importance": importances}).sort_values("importance", ascending=False)
print("\nFeature importance:")
print(imp_df.to_string(index=False))

# --- Score ALL transactions (for downstream Risk Fusion) -----------------
df["ml_score"] = model.predict_proba(df[FEATURES])[:, 1] * 100
df.to_csv("features_with_ml.csv", index=False)
imp_df.to_csv("feature_importance.csv", index=False)
print("\nSaved features_with_ml.csv (adds ml_score column, 0-100)")

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(FEATURES, "models/feature_columns.pkl")

print("\nModel saved successfully!")
