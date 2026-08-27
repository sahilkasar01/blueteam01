"""
SentinelPay AI - Explanation Layer (RAG-lite + Template Generation)
========================================================================
No API key required. Generates analyst-readable explanations by
composing sentences from the actual triggered signals - deterministic,
free, and works offline, which matters for a live demo you don't want
depending on network/API uptime.

An LLM call (Claude) is still wired in as an OPTIONAL swap: see
get_explanation_llm() at the bottom. Point EXPLAIN_FN at it instead of
the template function if/when you have an API key - the rest of the
pipeline (retrieval, prompt building, output) is unchanged either way.
This is a legitimate architecture choice to describe in your pitch:
"rule-based explanation generation with a pluggable LLM interface."
"""

import pandas as pd
import numpy as np
import json

df = pd.read_csv("final_scored_transactions.csv", parse_dates=["timestamp"])

SIM_FEATURES = ["amount", "amount_zscore", "amount_vs_user_avg", "is_new_device",
                 "is_new_merchant", "txns_last_1h", "device_shared_users"]


def find_similar_past_cases(row, df, k=3):
    """Cheap 'retrieval': nearest past flagged (HOLD/BLOCK) transactions
    by Euclidean distance in feature space, excluding the current row."""
    past = df[(df["decision"] != "ALLOW") & (df["timestamp"] < row["timestamp"])]
    if past.empty:
        return []
    diffs = past[SIM_FEATURES].astype(float).values - row[SIM_FEATURES].astype(float).values
    dists = np.linalg.norm(diffs, axis=1)
    top_idx = np.argsort(dists)[:k]
    return past.iloc[top_idx][["transaction_id", "fraud_type", "decision"] + SIM_FEATURES].to_dict("records")


def build_prompt(row, similar_cases):
    """Kept for the optional LLM path - same structured context either way."""
    return f"""You are a fraud analyst assistant. Given the signals below, write a 2-3 sentence
explanation of WHY this transaction was flagged, in plain language an analyst can act on
immediately. Cite the specific signals that drove the decision. Do not invent facts not given.

Transaction: {row['transaction_id']} | User: {row['user_id']} | Amount: {row['amount']:.2f}
Decision: {row['decision']} | Final risk score: {row['final_risk']:.1f}/100

Layer signals:
- Rule engine ({row['rule_score']}/100): {row['rule_reasons'] or 'none triggered'}
- ML classifier score: {row['ml_score']:.1f}/100
- Anomaly score (Isolation Forest): {row['anomaly_score']:.1f}/100
- Graph score: {row['graph_score']}/100: {row['graph_reasons'] or 'none triggered'}

Similar past flagged cases for context:
{json.dumps(similar_cases, indent=2, default=str)}
"""


# ---------------------------------------------------------------------
# TEMPLATE-BASED EXPLANATION (default - no API key needed)
# ---------------------------------------------------------------------
DECISION_OPENERS = {
    "BLOCK": "This transaction was BLOCKED",
    "HOLD": "This transaction was HELD for step-up verification",
    "ALLOW": "This transaction was allowed",
}


def _describe_signal(name, score, reasons):
    if score >= 60:
        strength = "strongly"
    elif score >= 30:
        strength = "moderately"
    elif score > 0:
        strength = "mildly"
    else:
        return None
    if reasons:
        return f"{name} flagged it {strength} ({score:.0f}/100) - {reasons}"
    return f"{name} scored it {strength} elevated ({score:.0f}/100)"


def generate_explanation_template(row, similar_cases):
    signals = [
        _describe_signal("the rule engine", row["rule_score"], row.get("rule_reasons", "")),
        _describe_signal("the ML classifier", row["ml_score"], ""),
        _describe_signal("anomaly detection", row["anomaly_score"], ""),
        _describe_signal("graph analysis", row["graph_score"], row.get("graph_reasons", "")),
    ]
    signals = [s for s in signals if s]
    # rank by strength: put "strongly" signals first, then "moderately"
    signals.sort(key=lambda s: ("strongly" not in s, "moderately" not in s))

    opener = DECISION_OPENERS.get(row["decision"], "This transaction was scored")
    sentence1 = f"{opener} with a final risk score of {row['final_risk']:.0f}/100 for user {row['user_id']}."

    if signals:
        sentence2 = "Key drivers: " + "; ".join(signals[:3]) + "."
    else:
        sentence2 = "No individual layer strongly triggered - the decision reflects the combined weighted score."

    if similar_cases:
        types = {(c.get("fraud_type") if isinstance(c.get("fraud_type"), str) and c.get("fraud_type") else "unlabeled")
                 for c in similar_cases}
        sentence3 = (f"This matches the pattern of {len(similar_cases)} similar recent case(s) "
                     f"previously flagged as: {', '.join(sorted(types))}.")
    else:
        sentence3 = "No closely matching prior flagged cases were found for this user."

    return " ".join([sentence1, sentence2, sentence3])


# ---------------------------------------------------------------------
# OPTIONAL: real Claude call (needs ANTHROPIC_API_KEY + `pip install anthropic`)
# Swap EXPLAIN_FN below to use this once you have a key.
# ---------------------------------------------------------------------
def get_explanation_llm(row, similar_cases, model="claude-haiku-4-5-20251001"):
    import anthropic
    client = anthropic.Anthropic()
    prompt = build_prompt(row, similar_cases)
    resp = client.messages.create(
        model=model, max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


EXPLAIN_FN = generate_explanation_template  # <- swap to get_explanation_llm when you have a key


if __name__ == "__main__":
    # Demo: explain the highest-risk BLOCK and the highest-risk HOLD
    for decision in ["BLOCK", "HOLD"]:
        subset = df[df["decision"] == decision].sort_values("final_risk", ascending=False)
        if subset.empty:
            continue
        row = subset.iloc[0]
        similar = find_similar_past_cases(row, df)
        print(f"\n{'='*70}\n{decision} example - {row['transaction_id']} ({row['fraud_type'] or 'unlabeled'})\n{'='*70}")
        print(EXPLAIN_FN(row, similar))
