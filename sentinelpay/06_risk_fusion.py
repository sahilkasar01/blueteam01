"""
SentinelPay AI - Risk Fusion + Uncertainty-Aware Decisioning
=================================================================
Combines (rule_score, ml_score, anomaly_score, graph_score) into one
final_risk (0-100), then applies your 3-band decision logic:

    ALLOW  : final_risk < LOW_THRESHOLD
    HOLD   : LOW_THRESHOLD <= final_risk < HIGH_THRESHOLD  (uncertain -> step-up verification)
    BLOCK  : final_risk >= HIGH_THRESHOLD

This is the piece that specifically demonstrates your "don't blindly
allow medium-risk transactions" design goal - the zero-day pattern
should mostly fall in the HOLD band rather than being missed entirely.

Weighted-sum version is used here (fast, transparent, good for a demo).
A logistic-regression meta-model version is also included, commented,
if you want the "slightly more real" variant later.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("features_with_graph.csv", parse_dates=["timestamp"])

# --- Weighted sum fusion ---------------------------------------------------
# Weights: ML gets the most trust (best overall AUC), rules/graph are
# high-precision-but-narrow signals, anomaly covers the gap.
WEIGHTS = {
    "rule_score": 0.15,
    "ml_score": 0.40,
    "anomaly_score": 0.30,
    "graph_score": 0.15,
}
df["final_risk"] = sum(df[col] * w for col, w in WEIGHTS.items())

LOW_THRESHOLD, HIGH_THRESHOLD = 25, 60  # tune these against your demo data

# --- Escalation override -----------------------------------------------
# A weighted AVERAGE can dilute one layer being very confident (e.g. the
# zero-day pattern scores high on anomaly_score but ~0 on ml_score, so the
# blended average looks tame). Real fraud stacks handle this with an "OR"
# override: any single layer crossing its own high-confidence bar escalates
# the decision by itself, regardless of the composite. Thresholds below
# are picked from this dataset's score distributions - re-tune on yours.
ESCALATE_TO_HOLD = {"anomaly_score": 35, "graph_score": 20, "rule_score": 30}
ESCALATE_TO_BLOCK = {"ml_score": 70, "graph_score": 40, "rule_score": 55}


def decide(row):
    if row["final_risk"] >= HIGH_THRESHOLD or any(row[c] >= t for c, t in ESCALATE_TO_BLOCK.items()):
        return "BLOCK"
    if row["final_risk"] >= LOW_THRESHOLD or any(row[c] >= t for c, t in ESCALATE_TO_HOLD.items()):
        return "HOLD"
    return "ALLOW"


df["decision"] = df.apply(decide, axis=1)

print("Overall final_risk ROC-AUC:", round(roc_auc_score(df["is_fraud"], df["final_risk"]), 3))

print("\nDecision breakdown by fraud_type (this is your headline demo table):")
print(pd.crosstab(df["fraud_type"], df["decision"]))

print("\n--- Zero-day pattern specifically (the layer this whole design justifies) ---")
zd = df[df["fraud_type"] == "zero_day_drain"]
print(zd["decision"].value_counts())
print(f"Caught (HOLD or BLOCK): {(zd['decision'] != 'ALLOW').mean():.1%}  "
      f"vs 0% recall from ML classifier alone (see 03_ml_classifier.py output)")

print("\n--- Legit traffic - check false-positive burden on HOLD queue ---")
legit = df[df["is_fraud"] == 0]
print(legit["decision"].value_counts(normalize=True).round(4))

# ---------------------------------------------------------------------
# OPTIONAL: logistic-regression meta-model version (uncomment to use)
# ---------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
meta_X = df[["rule_score", "ml_score", "anomaly_score", "graph_score"]]
meta_y = df["is_fraud"]
meta_model = LogisticRegression(class_weight="balanced")
meta_model.fit(meta_X, meta_y)
df["final_risk_meta"] = meta_model.predict_proba(meta_X)[:, 1] * 100
print("Meta-model coefficients:", dict(zip(meta_X.columns, meta_model.coef_[0])))
"""
SentinelPay AI - Risk Fusion + Uncertainty-Aware Decisioning
=================================================================
Combines (rule_score, ml_score, anomaly_score, graph_score) into one
final_risk (0-100), then applies your 3-band decision logic:

    ALLOW  : final_risk < LOW_THRESHOLD
    HOLD   : LOW_THRESHOLD <= final_risk < HIGH_THRESHOLD  (uncertain -> step-up verification)
    BLOCK  : final_risk >= HIGH_THRESHOLD

This is the piece that specifically demonstrates your "don't blindly
allow medium-risk transactions" design goal - the zero-day pattern
should mostly fall in the HOLD band rather than being missed entirely.

Weighted-sum version is used here (fast, transparent, good for a demo).
A logistic-regression meta-model version is also included, commented,
if you want the "slightly more real" variant later.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("features_with_graph.csv", parse_dates=["timestamp"])

# --- Weighted sum fusion ---------------------------------------------------
# Weights: ML gets the most trust (best overall AUC), rules/graph are
# high-precision-but-narrow signals, anomaly covers the gap.
WEIGHTS = {
    "rule_score": 0.15,
    "ml_score": 0.40,
    "anomaly_score": 0.30,
    "graph_score": 0.15,
}
df["final_risk"] = sum(df[col] * w for col, w in WEIGHTS.items())

# --- Signal disagreement score ---------------------------------------------
# An adversary optimizing to fool ONE specific layer (e.g. crafting a
# transaction that scores low on the ML classifier) will often still look
# inconsistent ACROSS layers, even while looking clean to the layer it
# targeted. High variance between rule/ml/anomaly/graph scores is itself
# a red flag for exactly this kind of targeted evasion - independent of
# whether the average score looks tame. This is the piece that specifically
# helps against a GenAI adversary probing your decision boundary rather
# than just replaying known fraud patterns.
SCORE_COLS = ["rule_score", "ml_score", "anomaly_score", "graph_score"]
df["disagreement_score"] = df[SCORE_COLS].std(axis=1)
DISAGREEMENT_THRESHOLD = df["disagreement_score"].quantile(0.90)  # top 10% most inconsistent

LOW_THRESHOLD, HIGH_THRESHOLD = 25, 60  # tune these against your demo data

# --- Escalation override -----------------------------------------------
# A weighted AVERAGE can dilute one layer being very confident (e.g. the
# zero-day pattern scores high on anomaly_score but ~0 on ml_score, so the
# blended average looks tame). Real fraud stacks handle this with an "OR"
# override: any single layer crossing its own high-confidence bar escalates
# the decision by itself, regardless of the composite. Thresholds below
# are picked from this dataset's score distributions - re-tune on yours.
ESCALATE_TO_HOLD = {"anomaly_score": 35, "graph_score": 20, "rule_score": 30}
ESCALATE_TO_BLOCK = {"ml_score": 70, "graph_score": 40, "rule_score": 55}


def decide(row):
    if row["final_risk"] >= HIGH_THRESHOLD or any(row[c] >= t for c, t in ESCALATE_TO_BLOCK.items()):
        return "BLOCK"
    if (row["final_risk"] >= LOW_THRESHOLD
            or any(row[c] >= t for c, t in ESCALATE_TO_HOLD.items())
            or row["disagreement_score"] >= DISAGREEMENT_THRESHOLD):
        return "HOLD"
    return "ALLOW"


df["decision"] = df.apply(decide, axis=1)

print("Overall final_risk ROC-AUC:", round(roc_auc_score(df["is_fraud"], df["final_risk"]), 3))

print("\nDecision breakdown by fraud_type (this is your headline demo table):")
print(pd.crosstab(df["fraud_type"], df["decision"]))

print("\n--- Zero-day pattern specifically (the layer this whole design justifies) ---")
zd = df[df["fraud_type"] == "zero_day_drain"]
print(zd["decision"].value_counts())
print(f"Caught (HOLD or BLOCK): {(zd['decision'] != 'ALLOW').mean():.1%}  "
      f"vs 0% recall from ML classifier alone (see 03_ml_classifier.py output)")

print("\n--- Legit traffic - check false-positive burden on HOLD queue ---")
legit = df[df["is_fraud"] == 0]
print(legit["decision"].value_counts(normalize=True).round(4))

print("\n--- Disagreement score: what it catches on its own ---")
would_allow_without_disagreement = (
    (df["final_risk"] < LOW_THRESHOLD)
    & ~df[list(ESCALATE_TO_HOLD)].apply(lambda r: any(r[c] >= t for c, t in ESCALATE_TO_HOLD.items()), axis=1)
    & ~df[list(ESCALATE_TO_BLOCK)].apply(lambda r: any(r[c] >= t for c, t in ESCALATE_TO_BLOCK.items()), axis=1)
)
caught_only_by_disagreement = would_allow_without_disagreement & (df["decision"] == "HOLD") & (df["is_fraud"] == 1)
print(f"Fraud cases HELD *only* because of signal disagreement "
      f"(would've been ALLOWed by every other rule): {caught_only_by_disagreement.sum()}")
if caught_only_by_disagreement.sum() > 0:
    print(df.loc[caught_only_by_disagreement, ["transaction_id", "fraud_type"] + SCORE_COLS + ["disagreement_score"]])

# ---------------------------------------------------------------------
# OPTIONAL: logistic-regression meta-model version (uncomment to use)
# ---------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
meta_X = df[["rule_score", "ml_score", "anomaly_score", "graph_score"]]
meta_y = df["is_fraud"]
meta_model = LogisticRegression(class_weight="balanced")
meta_model.fit(meta_X, meta_y)
df["final_risk_meta"] = meta_model.predict_proba(meta_X)[:, 1] * 100
print("Meta-model coefficients:", dict(zip(meta_X.columns, meta_model.coef_[0])))

df.to_csv("final_scored_transactions.csv", index=False)
print("\nSaved final_scored_transactions.csv - this is your full pipeline output.")
