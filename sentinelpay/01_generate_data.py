"""
SentinelPay AI - Synthetic Transaction Data Generator
=======================================================
Generates a realistic-ish synthetic transaction dataset with:
  - normal user behavior (home device/merchant/location patterns)
  - 3 "known" fraud patterns (labeled, used to TRAIN XGBoost + rules)
  - 1 "novel / zero-day" fraud pattern that is DELIBERATELY withheld
    from the XGBoost training labels, so you can demo that the
    Isolation Forest / anomaly layer catches what the supervised
    model misses. This is the whole point of your layered design.

Output: transactions.csv  (feeds into 02_rules_engine.py, 03_ml_classifier.py, etc.)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N_USERS = 250
N_DAYS = 30
MERCHANTS = [f"M{str(i).zfill(3)}" for i in range(1, 61)]
CITIES = ["Mumbai", "Pune", "Delhi", "Bengaluru", "Chennai", "Hyderabad",
          "Kolkata", "Jaipur", "Ahmedabad", "Lucknow"]
DEVICE_POOL = [f"D{str(i).zfill(4)}" for i in range(1, 900)]  # shared pool -> enables device-ring fraud

START = datetime(2026, 7, 1)

# --- Build user profiles -----------------------------------------------
users = []
for uid in range(1, N_USERS + 1):
    home_city = rng.choice(CITIES)
    n_devices = rng.integers(1, 3)  # each user normally has 1-2 devices
    devices = list(rng.choice(DEVICE_POOL, size=n_devices, replace=False))
    fav_merchants = list(rng.choice(MERCHANTS, size=rng.integers(3, 8), replace=False))
    avg_amount = rng.uniform(300, 4000)  # INR-ish baseline spend
    users.append({
        "user_id": f"U{str(uid).zfill(4)}",
        "home_city": home_city,
        "devices": devices,
        "fav_merchants": fav_merchants,
        "avg_amount": avg_amount,
    })
users_df = pd.DataFrame(users)

rows = []
txn_counter = 1


def new_txn(user, ts, amount, device, merchant, city, label, fraud_type):
    global txn_counter
    rows.append({
        "transaction_id": f"T{str(txn_counter).zfill(6)}",
        "user_id": user["user_id"],
        "timestamp": ts,
        "amount": round(max(amount, 10), 2),
        "device_id": device,
        "merchant_id": merchant,
        "city": city,
        "is_fraud": label,          # ground truth (for evaluation)
        "fraud_type": fraud_type,   # "" for legit, else pattern name
    })
    txn_counter += 1


# --- 1. Normal transactions ---------------------------------------------
for _, user in users_df.iterrows():
    n_txns = rng.integers(15, 45)  # per user over the month
    for _ in range(n_txns):
        day_offset = rng.integers(0, N_DAYS)
        hour = int(np.clip(rng.normal(14, 4), 0, 23))  # daytime-biased
        ts = START + timedelta(days=int(day_offset), hours=hour, minutes=int(rng.integers(0, 60)))
        amount = max(rng.normal(user["avg_amount"], user["avg_amount"] * 0.35), 20)
        device = rng.choice(user["devices"])
        merchant = rng.choice(user["fav_merchants"])
        new_txn(user, ts, amount, device, merchant, user["home_city"], 0, "")

# --- 2. FRAUD PATTERN A: Card testing / velocity burst -------------------
# Many small transactions in a short window, same device, different merchants.
# -> This is exactly what the Rule/Velocity engine + XGBoost should catch.
n_velocity_attacks = 25
for _ in range(n_velocity_attacks):
    user = users_df.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
    day_offset = rng.integers(0, N_DAYS)
    base_ts = START + timedelta(days=int(day_offset), hours=int(rng.integers(0, 23)))
    device = rng.choice(user["devices"])
    burst_len = rng.integers(5, 12)
    for i in range(burst_len):
        ts = base_ts + timedelta(minutes=int(i * rng.integers(1, 3)))
        amount = rng.uniform(10, 150)  # small "testing" amounts
        merchant = rng.choice(MERCHANTS)
        new_txn(user, ts, amount, device, merchant, user["home_city"], 1, "velocity_burst")

# --- 3. FRAUD PATTERN B: Account takeover ---------------------------------
# New device, new city (far from home), high amount, odd hour.
n_ato_attacks = 25
for _ in range(n_ato_attacks):
    user = users_df.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
    day_offset = rng.integers(0, N_DAYS)
    hour = int(rng.choice([1, 2, 3, 4, 23]))  # odd hours
    ts = START + timedelta(days=int(day_offset), hours=hour, minutes=int(rng.integers(0, 60)))
    new_device = rng.choice(DEVICE_POOL)  # not in user's normal device list
    new_city = rng.choice([c for c in CITIES if c != user["home_city"]])
    amount = user["avg_amount"] * rng.uniform(4, 12)  # unusually large
    merchant = rng.choice(MERCHANTS)
    new_txn(user, ts, amount, new_device, merchant, new_city, 1, "account_takeover")

# --- 4. FRAUD PATTERN C: Shared-device mule ring --------------------------
# Several DIFFERENT users all transact from the SAME device within a short
# window -> caught by graph analysis (device-sharing centrality), not by
# per-transaction features alone.
n_rings = 6
for _ in range(n_rings):
    ring_users = users_df.sample(rng.integers(3, 6))
    shared_device = rng.choice(DEVICE_POOL)
    day_offset = rng.integers(0, N_DAYS)
    base_ts = START + timedelta(days=int(day_offset), hours=int(rng.integers(9, 22)))
    merchant = rng.choice(MERCHANTS)  # ring often cashes out via same merchant
    for i, (_, user) in enumerate(ring_users.iterrows()):
        ts = base_ts + timedelta(minutes=int(i * rng.integers(2, 8)))
        amount = rng.uniform(1000, 6000)
        new_txn(user, ts, amount, shared_device, merchant, user["home_city"], 1, "device_ring")

# --- 5. ZERO-DAY / NOVEL PATTERN (withheld from ML training labels) ------
# "Low-and-slow" drain: attacker makes a SMALL number of MODERATE txns
# spread across many days, from the user's OWN usual device (stolen
# session/cookie, not a new device) but to brand-new merchant categories
# never seen for that user, and slightly above their normal amount.
# Nothing about a single transaction looks extreme -> rules/ML (trained
# on patterns A/B/C) largely miss it. Isolation Forest should flag it as
# anomalous relative to the user's own baseline behavior.
n_zero_day = 15
for _ in range(n_zero_day):
    user = users_df.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
    device = rng.choice(user["devices"])  # SAME device - looks "trusted"
    unseen_merchants = [m for m in MERCHANTS if m not in user["fav_merchants"]]
    n_drain_txns = rng.integers(3, 6)
    days = sorted(rng.choice(range(N_DAYS), size=n_drain_txns, replace=False))
    for d in days:
        hour = int(np.clip(rng.normal(14, 4), 0, 23))
        ts = START + timedelta(days=int(d), hours=hour, minutes=int(rng.integers(0, 60)))
        amount = user["avg_amount"] * rng.uniform(1.5, 2.5)  # elevated but not crazy
        merchant = rng.choice(unseen_merchants)
        new_txn(user, ts, amount, device, merchant, user["home_city"], 1, "zero_day_drain")

# --- Assemble --------------------------------------------------------------
df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
df.to_csv("transactions.csv", index=False)
users_df.drop(columns=["devices", "fav_merchants"]).to_csv("users.csv", index=False)

print(f"Generated {len(df)} transactions for {N_USERS} users over {N_DAYS} days")
print(df["fraud_type"].value_counts(dropna=False))
print(f"\nOverall fraud rate: {df['is_fraud'].mean():.2%}")
