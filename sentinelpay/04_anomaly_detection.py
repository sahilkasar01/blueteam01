"""
SentinelPay AI - Anomaly Detection Layer (Isolation Forest)
===============================================================
Unsupervised - no fraud labels used at all. Learns what "normal"
looks like and flags deviations. This is the layer that should catch
the zero_day_drain pattern that 03_ml_classifier.py missed entirely.

Also runs Local Outlier Factor (LOF) as a secondary check, per your plan.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

df = pd.read_csv("features_with_ml.csv", parse_dates=["timestamp"])

ANOMALY_FEATURES = [
    "amount", "amount_zscore", "amount_vs_user_avg", "is_new_device",
    "is_new_merchant", "is_odd_hour", "seconds_since_last_txn",
    "txns_last_1h", "device_shared_users",
]

X = df[ANOMALY_FEATURES].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

# contamination ~= expected fraud rate; set slightly above known rate
# since we're deliberately trying to catch UNKNOWN patterns too
iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=3)
iso.fit(X_scaled)
# decision_function: higher = more normal. Flip + rescale to 0-100 "risk"
raw = -iso.decision_function(X_scaled)
df["anomaly_score"] = 100 * (raw - raw.min()) / (raw.max() - raw.min())

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
lof_labels = lof.fit_predict(X_scaled)  # -1 = outlier, 1 = inlier
lof_raw = -lof.negative_outlier_factor_
df["lof_score"] = 100 * (lof_raw - lof_raw.min()) / (lof_raw.max() - lof_raw.min())
df["lof_flag"] = (lof_labels == -1).astype(int)

print("Isolation Forest - overall ROC-AUC vs ground truth:",
      round(roc_auc_score(df["is_fraud"], df["anomaly_score"]), 3))

print("\nMean anomaly_score by fraud_type (this is the key result):")
print(df.groupby("fraud_type")["anomaly_score"].mean().sort_values(ascending=False))

zd = df[df["fraud_type"] == "zero_day_drain"]
threshold = df["anomaly_score"].quantile(0.95)  # top 5% flagged
print(f"\nZero-day recall @top-5% anomaly threshold ({threshold:.1f}): "
      f"{(zd['anomaly_score'] >= threshold).mean():.1%}")
print(f"(Compare to 0% recall from the ML classifier layer alone)")

df.to_csv("features_with_anomaly.csv", index=False)
print("\nSaved features_with_anomaly.csv (adds anomaly_score, lof_score, lof_flag)")

X = df[ANOMALY_FEATURES].fillna(0) 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
os.makedirs("models", exist_ok=True)

joblib.dump(iso, "models/anomaly_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(ANOMALY_FEATURES, "models/anomaly_feature_columns.pkl")

print("Anomaly model saved successfully!")