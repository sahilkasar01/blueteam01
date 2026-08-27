"""
SentinelPay AI - Red Team vs Blue Team Adaptive Feedback Loop
=============================================================

RED TEAM:
- Starts with a zero-day-style fraud transaction.
- Mutates the attack to reduce the current Blue Team ML score.
- Simulates an adaptive attacker.

BLUE TEAM:
- Uses the current ML model + rule engine.
- BLOCK / HOLD / ALLOW decision deta hai.
- If attack is MISSED (ALLOW), analyst confirms fraud.
- Confirmed attack + nearby variants are added to training data.
- Model is retrained immediately.

DASHBOARD CONNECTION:
- Every round is exported to:
      red_blue_history.csv
      red_blue_live.json

These files can be read by your dashboard/API.

Run:
    python 08_redteam_blueteam_loop.py
"""

import json
import os
import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False


# ============================================================
# CONFIG
# ============================================================

rng = np.random.default_rng(7)

FEATURES = [
    "amount",
    "hour",
    "is_odd_hour",
    "is_new_device",
    "is_new_merchant",
    "amount_zscore",
    "amount_vs_user_avg",
    "seconds_since_last_txn",
    "txns_last_1h",
    "device_shared_users",
    "rule_score",
]

RULE_THRESHOLD_FOR_HOLD = 30
ML_THRESHOLD_FOR_HOLD = 40

N_ROUNDS = 8

HISTORY_FILE = "red_blue_history.csv"
LIVE_FILE = "red_blue_live.json"


# ============================================================
# DASHBOARD STATE
# ============================================================

def create_dashboard_state():
    """
    Initial dashboard state.
    """

    return {
        "status": "INITIALIZING",
        "current_round": 0,
        "total_rounds": N_ROUNDS,

        "red_team": {
            "attack_type": "zero_day_drain",
            "ml_score": 0,
            "mutation_attempts": 0,
            "status": "READY",
        },

        "blue_team": {
            "ml_score": 0,
            "rule_score": 0,
            "decision": "WAITING",
            "model_status": "INITIALIZING",
        },

        "feedback_loop": {
            "missed": 0,
            "caught": 0,
            "retrained": 0,
        },

        "history": []
    }


dashboard_state = create_dashboard_state()


def update_dashboard_file():
    """
    Writes current state so dashboard can read it.
    """

    try:
        with open(LIVE_FILE, "w") as f:
            json.dump(dashboard_state, f, indent=2)
    except Exception as e:
        print(f"[Dashboard] Could not update live JSON: {e}")


def save_history_row(row):
    """
    Append one round to CSV.
    """

    try:
        row_df = pd.DataFrame([row])

        if os.path.exists(HISTORY_FILE):
            row_df.to_csv(
                HISTORY_FILE,
                mode="a",
                header=False,
                index=False
            )
        else:
            row_df.to_csv(
                HISTORY_FILE,
                mode="w",
                header=True,
                index=False
            )

    except Exception as e:
        print(f"[Dashboard] Could not save history: {e}")


# ============================================================
# BLUE TEAM - MODEL TRAINING
# ============================================================

def train_model(train_df):

    X = train_df[FEATURES]
    y = train_df["is_fraud"]

    if HAS_XGB:

        spw = (y == 0).sum() / max((y == 1).sum(), 1)

        model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=3
        )

    else:

        model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.1,
            random_state=3
        )

    model.fit(X, y)

    return model


# ============================================================
# BLUE TEAM - SCORING
# ============================================================

def blue_team_score(model, txn_features):

    x = pd.DataFrame([txn_features])[FEATURES]

    ml_score = model.predict_proba(x)[:, 1][0] * 100

    rule_score = float(txn_features["rule_score"])

    if ml_score >= 70:

        decision = "BLOCK"

    elif (
        ml_score >= ML_THRESHOLD_FOR_HOLD
        or rule_score >= RULE_THRESHOLD_FOR_HOLD
    ):

        decision = "HOLD"

    else:

        decision = "ALLOW"

    return ml_score, decision


# ============================================================
# RED TEAM - ADAPTIVE MUTATION
# ============================================================

def red_team_mutate(base_txn, model, n_tries=25):

    """
    Red Team tries to reduce the Blue Team ML score.

    This is a safe simulation of adversarial model testing.
    """

    best = dict(base_txn)

    best_score, _ = blue_team_score(
        model,
        best
    )

    for _ in range(n_tries):

        candidate = dict(best)

        # Reduce amount anomaly
        candidate["amount_vs_user_avg"] = max(
            1.05,
            candidate["amount_vs_user_avg"]
            - rng.uniform(0.02, 0.12)
        )

        # Reduce amount z-score
        candidate["amount_zscore"] = max(
            0.0,
            candidate["amount_zscore"]
            - rng.uniform(0.05, 0.30)
        )

        # Reduce velocity
        candidate["txns_last_1h"] = max(
            0,
            candidate["txns_last_1h"]
            - rng.integers(0, 2)
        )

        # Increase time between transactions
        candidate["seconds_since_last_txn"] = (
            candidate["seconds_since_last_txn"]
            + rng.uniform(0, 3600)
        )

        # Sometimes reuse known merchant
        if rng.random() < 0.3:
            candidate["is_new_merchant"] = 0

        # Keep amount internally consistent
        old_avg_ratio = max(
            best["amount_vs_user_avg"],
            1e-6
        )

        candidate["amount"] = (
            candidate["amount"]
            * (
                candidate["amount_vs_user_avg"]
                / old_avg_ratio
            )
        )

        cand_score, _ = blue_team_score(
            model,
            candidate
        )

        if cand_score < best_score:

            best = candidate
            best_score = cand_score

    return best, best_score


# ============================================================
# DATA AUGMENTATION AFTER ANALYST CONFIRMATION
# ============================================================

def augment_confirmed_case(txn, n_neighbors=6):

    rows = [
        dict(txn)
    ]

    for _ in range(n_neighbors):

        neighbor = dict(txn)

        neighbor["amount_vs_user_avg"] = max(
            1.0,
            neighbor["amount_vs_user_avg"]
            * rng.uniform(0.9, 1.1)
        )

        neighbor["amount_zscore"] = max(
            0.0,
            neighbor["amount_zscore"]
            + rng.uniform(-0.3, 0.3)
        )

        neighbor["txns_last_1h"] = max(
            0,
            neighbor["txns_last_1h"]
            + rng.integers(-1, 2)
        )

        neighbor["seconds_since_last_txn"] = max(
            0.0,
            neighbor["seconds_since_last_txn"]
            + rng.uniform(-1800, 1800)
        )

        rows.append(neighbor)

    return rows


# ============================================================
# DASHBOARD UPDATE - ROUND
# ============================================================

def update_round_dashboard(
    round_num,
    attack,
    ml_score,
    decision,
    red_score_before_retrain
):

    if decision == "ALLOW":
        outcome = "MISSED"
    else:
        outcome = "CAUGHT"

    dashboard_state["current_round"] = round_num
    dashboard_state["status"] = "RUNNING"

    dashboard_state["red_team"] = {
        "attack_type": "zero_day_drain",
        "ml_score_before_blue": round(red_score_before_retrain, 2),
        "ml_score": round(ml_score, 2),
        "mutation_attempts": 25,
        "status": "ATTACKED",

        "features": {
            "amount": round(float(attack["amount"]), 2),
            "amount_vs_user_avg": round(
                float(attack["amount_vs_user_avg"]),
                2
            ),
            "amount_zscore": round(
                float(attack["amount_zscore"]),
                2
            ),
            "txns_last_1h": int(
                attack["txns_last_1h"]
            )
        }
    }

    dashboard_state["blue_team"] = {
        "ml_score": round(ml_score, 2),
        "rule_score": round(
            float(attack["rule_score"]),
            2
        ),
        "decision": decision,
        "model_status": "ACTIVE"
    }

    dashboard_state["feedback_loop"]["caught"] = int(
        sum(
            1
            for x in dashboard_state["history"]
            if x["decision"] != "ALLOW"
        )
    )

    dashboard_state["feedback_loop"]["missed"] = int(
        sum(
            1
            for x in dashboard_state["history"]
            if x["decision"] == "ALLOW"
        )
    )

    dashboard_state["history"].append({
        "round": round_num,
        "ml_score": round(float(ml_score), 2),
        "rule_score": round(
            float(attack["rule_score"]),
            2
        ),
        "decision": decision,
        "outcome": outcome
    })

    update_dashboard_file()


# ============================================================
# MAIN RED TEAM vs BLUE TEAM LOOP
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(" SENTINELPAY AI")
    print(" RED TEAM vs BLUE TEAM ADAPTIVE FEEDBACK LOOP")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD FEATURES
    # --------------------------------------------------------

    if not os.path.exists("features.csv"):

        print("\nERROR: features.csv not found.")

        print(
            "Run your feature-generation pipeline first."
        )

        raise SystemExit(1)

    df = pd.read_csv(
        "features.csv",
        parse_dates=["timestamp"]
    )

    print(
        f"\nLoaded {len(df)} transactions from features.csv"
    )

    # --------------------------------------------------------
    # ZERO DAY DATA
    # --------------------------------------------------------

    train_df = df[
        df["fraud_type"] != "zero_day_drain"
    ].copy()

    zero_day_examples = df[
        df["fraud_type"] == "zero_day_drain"
    ]

    if len(zero_day_examples) == 0:

        print(
            "\nERROR: No zero_day_drain transactions found."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # INITIAL MODEL
    # --------------------------------------------------------

    print(
        "\n[BLUE TEAM] Training initial model..."
    )

    model = train_model(train_df)

    dashboard_state["blue_team"]["model_status"] = "TRAINED"

    update_dashboard_file()

    print(
        "[BLUE TEAM] Initial model ready."
    )

    # --------------------------------------------------------
    # RED TEAM SEED
    # --------------------------------------------------------

    seed_txn = (
        zero_day_examples.iloc[0][FEATURES]
        .to_dict()
    )

    attack = dict(seed_txn)

    print(
        "\n[RED TEAM] Zero-day attack selected."
    )

    print(
        f"Amount: ₹{attack['amount']:.2f}"
    )

    # --------------------------------------------------------
    # HISTORY RESET
    # --------------------------------------------------------

    history = []

    dashboard_state["history"] = []
    dashboard_state["feedback_loop"] = {
        "missed": 0,
        "caught": 0,
        "retrained": 0
    }

    update_dashboard_file()

    # --------------------------------------------------------
    # ADAPTIVE ROUNDS
    # --------------------------------------------------------

    for round_num in range(
        1,
        N_ROUNDS + 1
    ):

        print("\n" + "-" * 70)

        print(
            f"ROUND {round_num}/{N_ROUNDS}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # RED TEAM
        # ----------------------------------------------------

        attack, red_team_score_before_retrain = (
            red_team_mutate(
                attack,
                model
            )
        )

        # ----------------------------------------------------
        # BLUE TEAM
        # ----------------------------------------------------

        ml_score, decision = blue_team_score(
            model,
            attack
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if decision == "ALLOW":

            outcome = "MISSED (ALLOW)"

        else:

            outcome = f"CAUGHT ({decision})"

        print(
            "\n[RED TEAM]"
        )

        print(
            f"Amount vs User Avg : "
            f"{attack['amount_vs_user_avg']:.2f}"
        )

        print(
            f"Amount Z-Score     : "
            f"{attack['amount_zscore']:.2f}"
        )

        print(
            f"Transactions / 1h  : "
            f"{attack['txns_last_1h']}"
        )

        print(
            "\n[BLUE TEAM]"
        )

        print(
            f"ML Score   : {ml_score:.1f}"
        )

        print(
            f"Rule Score : "
            f"{attack['rule_score']:.0f}"
        )

        print(
            f"Decision   : {decision}"
        )

        print(
            f"Outcome    : {outcome}"
        )

        # ----------------------------------------------------
        # SAVE ROUND HISTORY
        # ----------------------------------------------------

        history_row = {
            "round": round_num,
            "ml_score": round(
                float(ml_score),
                2
            ),
            "rule_score": round(
                float(attack["rule_score"]),
                2
            ),
            "decision": decision,
            "outcome": outcome,
            "amount": round(
                float(attack["amount"]),
                2
            ),
            "amount_vs_user_avg": round(
                float(attack["amount_vs_user_avg"]),
                2
            ),
            "amount_zscore": round(
                float(attack["amount_zscore"]),
                2
            ),
            "txns_last_1h": int(
                attack["txns_last_1h"]
            )
        }

        history.append(history_row)

        save_history_row(
            history_row
        )

        # ----------------------------------------------------
        # UPDATE DASHBOARD
        # ----------------------------------------------------

        update_round_dashboard(
            round_num,
            attack,
            ml_score,
            decision,
            red_team_score_before_retrain
        )

        # ----------------------------------------------------
        # FEEDBACK LOOP
        # ----------------------------------------------------

        if decision == "ALLOW":

            print(
                "\n[FEEDBACK LOOP]"
            )

            print(
                "Attack MISSED."
            )

            print(
                "Analyst confirms transaction as FRAUD."
            )

            print(
                "Adding confirmed attack + nearby variants..."
            )

            new_rows = []

            for r in augment_confirmed_case(
                attack
            ):

                r["is_fraud"] = 1

                new_rows.append(r)

            train_df = pd.concat(
                [
                    train_df,
                    pd.DataFrame(new_rows)
                ],
                ignore_index=True
            )

            print(
                f"Added {len(new_rows)} fraud examples."
            )

            # ------------------------------------------------
            # RETRAIN
            # ------------------------------------------------

            print(
                "[BLUE TEAM] Retraining model..."
            )

            model = train_model(
                train_df
            )

            dashboard_state[
                "feedback_loop"
            ]["retrained"] += 1

            dashboard_state[
                "blue_team"
            ]["model_status"] = "RETRAINED"

            update_dashboard_file()

            # ------------------------------------------------
            # RECHECK SAME ATTACK
            # ------------------------------------------------

            ml_score_after, decision_after = (
                blue_team_score(
                    model,
                    attack
                )
            )

            print(
                "\n[BLUE TEAM] After retraining:"
            )

            print(
                f"ML Score : "
                f"{ml_score_after:.1f}"
            )

            print(
                f"Decision : "
                f"{decision_after}"
            )

            dashboard_state[
                "blue_team"
            ]["ml_score_after_retrain"] = round(
                float(ml_score_after),
                2
            )

            dashboard_state[
                "blue_team"
            ]["decision_after_retrain"] = (
                decision_after
            )

            update_dashboard_file()

        else:

            print(
                "\n[FEEDBACK LOOP]"
            )

            print(
                "Attack was successfully detected."
            )

            print(
                "No retraining required this round."
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    hist_df = pd.DataFrame(
        history
    )

    caught = (
        hist_df["decision"] != "ALLOW"
    ).sum()

    missed = (
        hist_df["decision"] == "ALLOW"
    ).sum()

    catch_rate = (
        caught / N_ROUNDS
    ) * 100

    dashboard_state["status"] = "COMPLETED"

    dashboard_state["summary"] = {
        "total_rounds": N_ROUNDS,
        "caught": int(caught),
        "missed": int(missed),
        "catch_rate": round(
            float(catch_rate),
            2
        ),
        "retrained": dashboard_state[
            "feedback_loop"
        ]["retrained"]
    }

    update_dashboard_file()

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        hist_df[
            [
                "round",
                "ml_score",
                "rule_score",
                "decision",
                "outcome"
            ]
        ]
    )

    print(
        f"\nCaught : {caught}/{N_ROUNDS}"
    )

    print(
        f"Missed : {missed}/{N_ROUNDS}"
    )

    print(
        f"Catch Rate : {catch_rate:.1f}%"
    )

    print(
        "\nDashboard files updated:"
    )

    print(
        f"  -> {HISTORY_FILE}"
    )

    print(
        f"  -> {LIVE_FILE}"
    )

    print(
        "\nAdaptive feedback loop completed."
    )