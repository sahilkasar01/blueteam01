"""
SentinelPay AI - End-to-End Pipeline Orchestrator
======================================================
Runs all layers in order and prints a final summary. Each stage is a
standalone script (so you can also run/debug them individually) - this
just chains them and stops early with a clear error if any stage fails.

Usage:
    python3 main.py

Requires: numpy, pandas, scikit-learn, networkx  (all pip-installable)
Optional: xgboost (falls back to sklearn's HistGradientBoostingClassifier
          automatically if not installed - see 03_ml_classifier.py)
"""

import subprocess
import sys
import time
from pathlib import Path

STAGES = [
    ("01_generate_data.py",      "Generating synthetic transaction data"),
    ("02_features_and_rules.py", "Building features + running rule/velocity engine"),
    ("03_ml_classifier.py",      "Training ML classifier (XGBoost/HistGBM)"),
    ("04_anomaly_detection.py",  "Running Isolation Forest + LOF anomaly detection"),
    ("05_graph_analysis.py",     "Running graph analysis (device/merchant rings)"),
    ("06_risk_fusion.py",        "Fusing signals into final risk + decision"),
]

BASE_DIR = Path(__file__).resolve().parent


def run_stage(script, description):
    print(f"\n{'='*72}\n>> {description}\n   ({script})\n{'='*72}", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=str(BASE_DIR),
    )  # no capture_output - stdout/stderr stream straight to your terminal live
    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode}. Stopping pipeline.")
        sys.exit(result.returncode)
    print(f"[done in {time.time() - t0:.1f}s]", flush=True)


def print_final_summary():
    import pandas as pd
    df = pd.read_csv(BASE_DIR / "final_scored_transactions.csv")

    print(f"\n{'='*72}\nPIPELINE SUMMARY\n{'='*72}")
    print(f"Total transactions scored: {len(df)}")
    print(f"\nDecisions overall:\n{df['decision'].value_counts()}")
    print(f"\nDecisions by fraud_type:\n{pd.crosstab(df['fraud_type'].fillna('(legit)'), df['decision'])}")

    zd = df[df["fraud_type"] == "zero_day_drain"]
    if not zd.empty:
        caught = (zd["decision"] != "ALLOW").mean()
        print(f"\nZero-day pattern caught (HOLD or BLOCK): {caught:.1%}")

    legit = df[df["is_fraud"] == 0]
    print(f"\nFalse-positive burden on legit traffic:\n{legit['decision'].value_counts(normalize=True).round(4)}")
    print(f"\nFull output: {BASE_DIR / 'final_scored_transactions.csv'}")


if __name__ == "__main__":
    pipeline_start = time.time()
    for script, description in STAGES:
        run_stage(script, description)
    print_final_summary()
    print(f"\nTotal pipeline runtime: {time.time() - pipeline_start:.1f}s")
    print("\nNext: run 07_explanation_layer.py separately (needs your Anthropic API key)")
    print("      to generate analyst-readable explanations for flagged transactions.")