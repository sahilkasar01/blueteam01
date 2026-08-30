"""
SentinelPay AI - Feature Engineering + Rule/Velocity Engine
=============================================================
1. Builds per-transaction features that every downstream layer
   (XGBoost, Isolation Forest, Risk Fusion) will reuse.
2. Implements a deterministic Rule/Velocity engine -> rule_score (0-100).

Output: features.csv
"""

import numpy as np
import pandas as pd
 
df = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
 
# ---------------------------------------------------------------------
# 1. Per-user historical baselines (computed causally: only using data
#    UP TO but not including the current transaction, to avoid leakage)
# ---------------------------------------------------------------------
df["hour"] = df["timestamp"].dt.hour
df["is_odd_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
 
user_known_devices = {}     # user_id -> set of devices seen so far
user_known_merchants = {}   # user_id -> set of merchants seen so far
user_amounts = {}           # user_id -> list of past amounts
user_last_txn_time = {}     # user_id -> last timestamp
user_recent_times = {}      # user_id -> list of recent timestamps (for velocity)
 
is_new_device, is_new_merchant, amount_zscore = [], [], []
txns_last_1h, seconds_since_last_txn, amount_vs_user_avg, user_txn_count = [], [], [], []
 
for _, row in df.iterrows():
    uid = row["user_id"]
    devices = user_known_devices.setdefault(uid, set())
    merchants = user_known_merchants.setdefault(uid, set())
    amounts = user_amounts.setdefault(uid, [])
    recent = user_recent_times.setdefault(uid, [])
 
    is_new_device.append(int(row["device_id"] not in devices))
    is_new_merchant.append(int(row["merchant_id"] not in merchants))
 
    if len(amounts) >= 3:
        mu, sigma = np.mean(amounts), np.std(amounts) + 1e-6
        amount_zscore.append((row["amount"] - mu) / sigma)
        amount_vs_user_avg.append(row["amount"] / (mu + 1e-6))
    else:
        amount_zscore.append(0.0)
        amount_vs_user_avg.append(1.0)
    user_txn_count.append(len(amounts))  # history size BEFORE this transaction - the cold-start signal
 
    last_t = user_last_txn_time.get(uid)
    seconds_since_last_txn.append((row["timestamp"] - last_t).total_seconds() if last_t is not None else 1e9)
 
    # velocity: txns by this user in the last 60 minutes (excluding current)
    recent = [t for t in recent if (row["timestamp"] - t).total_seconds() <= 3600]
    txns_last_1h.append(len(recent))
 
    # update state AFTER computing features (causal)
    devices.add(row["device_id"])
    merchants.add(row["merchant_id"])
    amounts.append(row["amount"])
    user_last_txn_time[uid] = row["timestamp"]
    recent.append(row["timestamp"])
    user_recent_times[uid] = recent
 
df["is_new_device"] = is_new_device
df["is_new_merchant"] = is_new_merchant
df["amount_zscore"] = amount_zscore
df["amount_vs_user_avg"] = amount_vs_user_avg
df["seconds_since_last_txn"] = seconds_since_last_txn
df["txns_last_1h"] = txns_last_1h
df["user_txn_count"] = user_txn_count
 
# device shared across DIFFERENT users -> precursor for graph layer,
# but also a cheap rule signal on its own
device_user_counts = df.groupby("device_id")["user_id"].nunique()
df["device_shared_users"] = df["device_id"].map(device_user_counts)
 
# ---------------------------------------------------------------------
# 2. Rule / Velocity engine (deterministic, interpretable, weights are
#    hand-tuned - this is meant to be your fast, explainable baseline
#    signal, NOT the final decision)
# ---------------------------------------------------------------------
def rule_score(row):
    score = 0
    reasons = []
    if row["txns_last_1h"] >= 5:
        score += 30
        reasons.append(f"{row['txns_last_1h']} txns in last 1h (velocity)")
    if row["is_new_device"] and row["amount_vs_user_avg"] > 3:
        score += 25
        reasons.append("new device + amount >3x user avg")
    if row["is_odd_hour"] and row["is_new_device"]:
        score += 20
        reasons.append("odd hour + new device")
    if row["device_shared_users"] >= 3:
        score += 20
        reasons.append(f"device shared by {row['device_shared_users']} users")
    if row["amount_zscore"] > 3:
        score += 15
        reasons.append(f"amount z-score {row['amount_zscore']:.1f}")
    if row["seconds_since_last_txn"] < 60 and row["txns_last_1h"] >= 3:
        score += 10
        reasons.append("rapid repeat transactions")
 
    # --- Cold-start / absolute-amount safety net -----------------------
    # A brand-new user has no history, so amount_zscore/amount_vs_user_avg
    # default to "looks normal" - that's a blind spot the relative checks
    # above can never catch. Add an ABSOLUTE ceiling independent of any
    # user's personal average, plus extra suspicion when a large amount
    # comes from a user/device with no established history at all.
    ABSOLUTE_HIGH_AMOUNT = 100_000       # tune to your real transaction scale
    ABSOLUTE_EXTREME_AMOUNT = 1_000_000  # tune to your real transaction scale
    has_no_baseline = row.get("user_txn_count", 3) < 3  # see note below on wiring this in
 
    if row["amount"] >= ABSOLUTE_EXTREME_AMOUNT:
        score += 50
        reasons.append(f"amount {row['amount']:.0f} exceeds extreme absolute threshold")
    elif row["amount"] >= ABSOLUTE_HIGH_AMOUNT:
        score += 30
        reasons.append(f"amount {row['amount']:.0f} exceeds high absolute threshold")
 
    if has_no_baseline and row["amount"] >= ABSOLUTE_HIGH_AMOUNT:
        score += 25
        reasons.append("large first-time amount from a user/device with no transaction history")
 
    return min(score, 100), "; ".join(reasons)
 
results = df.apply(rule_score, axis=1)
df["rule_score"] = results.apply(lambda x: x[0])
df["rule_reasons"] = results.apply(lambda x: x[1])
 
df.to_csv("features.csv", index=False)
 
print(f"Feature engineering complete: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nRule engine performance (sanity check vs ground truth):")
print(pd.crosstab(df["is_fraud"], df["rule_score"] >= 30, rownames=["is_fraud"], colnames=["rule_score>=30"]))
print("\nRule score by fraud_type (mean):")
print(df.groupby("fraud_type")["rule_score"].mean().sort_values(ascending=False))