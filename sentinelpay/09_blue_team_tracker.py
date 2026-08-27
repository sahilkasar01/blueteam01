import os
import time
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "red_team_transactions.csv"
OUTPUT_FILE = "blue_team_results.csv"

MODEL_FILE = "models/fraud_model.pkl"
FEATURE_FILE = "models/feature_columns.pkl"


# ============================================================
# LOAD TRAINED BLUE TEAM MODEL
# ============================================================

print("=" * 70)
print("       SENTINELPAY BLUE TEAM TRANSACTION TRACKER")
print("=" * 70)

if not os.path.exists(MODEL_FILE):
    print("\nERROR: Trained model not found!")
    print("Please run 03_ml_classifier.py first.")
    exit()

model = joblib.load(MODEL_FILE)
FEATURES = joblib.load(FEATURE_FILE)

print("\nBlue Team ML model loaded successfully.")
print("Features expected by model:")

for feature in FEATURES:
    print(" -", feature)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

if not os.path.exists("features.csv"):
    print("\nERROR: features.csv not found!")
    exit()

historical_data = pd.read_csv(
    "features.csv",
    parse_dates=["timestamp"]
)

print("\nHistorical Blue Team data loaded.")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(transaction, history):

    txn = transaction.copy()

    # Convert timestamp
    txn["timestamp"] = pd.to_datetime(txn["timestamp"])

    user_id = txn["user_id"]

    # --------------------------------------------------------
    # Get historical transactions of same user
    # --------------------------------------------------------

    if "user_id" in history.columns:

        user_history = history[
            history["user_id"] == user_id
        ].copy()

    else:

        user_history = pd.DataFrame()

    # --------------------------------------------------------
    # BASIC FEATURES
    # --------------------------------------------------------

    amount = float(txn["amount"])

    hour = txn["timestamp"].hour

    is_odd_hour = 1 if hour <= 5 else 0


    # --------------------------------------------------------
    # USER AMOUNT BEHAVIOUR
    # --------------------------------------------------------

    if len(user_history) > 0:

        if "amount" in user_history.columns:

            user_avg_amount = user_history["amount"].mean()

            user_std_amount = user_history["amount"].std()

            if pd.isna(user_std_amount) or user_std_amount == 0:
                user_std_amount = 1

        else:

            user_avg_amount = amount
            user_std_amount = 1

    else:

        # New user fallback

        user_avg_amount = historical_data["amount"].mean()

        user_std_amount = historical_data["amount"].std()

        if user_std_amount == 0:
            user_std_amount = 1


    amount_vs_user_avg = amount / max(
        user_avg_amount,
        1
    )

    amount_zscore = (
        amount - user_avg_amount
    ) / user_std_amount


    # --------------------------------------------------------
    # NEW DEVICE
    # --------------------------------------------------------

    if len(user_history) > 0 and "device_id" in user_history.columns:

        known_devices = set(
            user_history["device_id"].astype(str)
        )

        is_new_device = int(
            str(txn["device_id"])
            not in known_devices
        )

    else:

        is_new_device = 1


    # --------------------------------------------------------
    # NEW MERCHANT
    # --------------------------------------------------------

    if len(user_history) > 0 and "merchant_id" in user_history.columns:

        known_merchants = set(
            user_history["merchant_id"].astype(str)
        )

        is_new_merchant = int(
            str(txn["merchant_id"])
            not in known_merchants
        )

    else:

        is_new_merchant = 1


    # --------------------------------------------------------
    # TIME SINCE LAST TRANSACTION
    # --------------------------------------------------------

    seconds_since_last_txn = 999999

    if len(user_history) > 0:

        if "timestamp" in user_history.columns:

            user_history["timestamp"] = pd.to_datetime(
                user_history["timestamp"]
            )

            previous_transactions = user_history[
                user_history["timestamp"] < txn["timestamp"]
            ]

            if len(previous_transactions) > 0:

                last_time = previous_transactions[
                    "timestamp"
                ].max()

                seconds_since_last_txn = (
                    txn["timestamp"] - last_time
                ).total_seconds()


    # --------------------------------------------------------
    # TRANSACTIONS IN LAST 1 HOUR
    # --------------------------------------------------------

    txns_last_1h = 0

    if len(user_history) > 0:

        start_time = (
            txn["timestamp"]
            - pd.Timedelta(hours=1)
        )

        recent_txns = user_history[
            (
                user_history["timestamp"] >= start_time
            )
            &
            (
                user_history["timestamp"]
                < txn["timestamp"]
            )
        ]

        txns_last_1h = len(recent_txns)


    # --------------------------------------------------------
    # DEVICE SHARED USERS
    # --------------------------------------------------------

    device_shared_users = 1

    if "device_id" in history.columns:

        device_users = history[
            history["device_id"].astype(str)
            == str(txn["device_id"])
        ]["user_id"].nunique()

        device_shared_users = max(
            device_users,
            1
        )


    # ========================================================
    # RULE ENGINE
    # ========================================================

    rule_score = 0


    # High amount anomaly

    if amount_vs_user_avg >= 5:

        rule_score += 35

    elif amount_vs_user_avg >= 3:

        rule_score += 20


    # New device

    if is_new_device == 1:

        rule_score += 20


    # New merchant

    if is_new_merchant == 1:

        rule_score += 10


    # High transaction velocity

    if txns_last_1h >= 10:

        rule_score += 30

    elif txns_last_1h >= 5:

        rule_score += 15


    # Odd hour

    if is_odd_hour == 1:

        rule_score += 10


    # Shared device

    if device_shared_users >= 5:

        rule_score += 25


    rule_score = min(
        rule_score,
        100
    )


    # ========================================================
    # RETURN FEATURES
    # ========================================================

    features = {

        "amount": amount,

        "hour": hour,

        "is_odd_hour": is_odd_hour,

        "is_new_device": is_new_device,

        "is_new_merchant": is_new_merchant,

        "amount_zscore": amount_zscore,

        "amount_vs_user_avg": amount_vs_user_avg,

        "seconds_since_last_txn": seconds_since_last_txn,

        "txns_last_1h": txns_last_1h,

        "device_shared_users": device_shared_users,

        "rule_score": rule_score

    }

    return features


# ============================================================
# ML DETECTION
# ============================================================

def get_ml_score(features):

    X = pd.DataFrame(
        [features]
    )

    X = X[FEATURES]

    probability = model.predict_proba(X)[0][1]

    return probability * 100


# ============================================================
# ANOMALY SCORE
# Simple baseline version
# ============================================================

def get_anomaly_score(features):

    score = 0


    # Large deviation from normal user spending

    if features["amount_zscore"] > 3:

        score += 30

    elif features["amount_zscore"] > 2:

        score += 20


    # New device

    if features["is_new_device"]:

        score += 20


    # High velocity

    if features["txns_last_1h"] >= 10:

        score += 30

    elif features["txns_last_1h"] >= 5:

        score += 15


    # Shared device

    if features["device_shared_users"] >= 5:

        score += 20


    return min(score, 100)


# ============================================================
# GRAPH SCORE
# ============================================================

def get_graph_score(transaction, history):

    score = 0

    device = str(
        transaction["device_id"]
    )

    if "device_id" in history.columns:

        shared_users = history[
            history["device_id"].astype(str)
            == device
        ]["user_id"].nunique()

        if shared_users >= 10:

            score += 50

        elif shared_users >= 5:

            score += 30

        elif shared_users >= 3:

            score += 15


    return min(
        score,
        100
    )


# ============================================================
# RISK FUSION
# ============================================================

def calculate_final_risk(
    ml_score,
    rule_score,
    anomaly_score,
    graph_score
):

    final_score = (

        0.40 * ml_score +

        0.25 * rule_score +

        0.20 * anomaly_score +

        0.15 * graph_score

    )

    return final_score


# ============================================================
# FINAL DECISION
# ============================================================

def make_decision(final_score):

    if final_score >= 70:

        return "BLOCK"

    elif final_score >= 40:

        return "HOLD"

    else:

        return "ALLOW"


# ============================================================
# PROCESS ONE RED TEAM TRANSACTION
# ============================================================

def process_transaction(
    transaction,
    history
):

    # Feature engineering

    features = create_features(
        transaction,
        history
    )


    # ML

    ml_score = get_ml_score(
        features
    )


    # Rules

    rule_score = features[
        "rule_score"
    ]


    # Anomaly

    anomaly_score = get_anomaly_score(
        features
    )


    # Graph

    graph_score = get_graph_score(
        transaction,
        history
    )


    # Risk fusion

    final_score = calculate_final_risk(

        ml_score,

        rule_score,

        anomaly_score,

        graph_score

    )


    # Decision

    decision = make_decision(
        final_score
    )


    result = {

        **transaction.to_dict(),

        **features,

        "ml_score": round(
            ml_score,
            2
        ),

        "anomaly_score": round(
            anomaly_score,
            2
        ),

        "graph_score": round(
            graph_score,
            2
        ),

        "final_risk_score": round(
            final_score,
            2
        ),

        "blue_team_decision": decision

    }


    return result


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    if not os.path.exists(INPUT_FILE):

        print("\nWaiting for Red Team data...")

        print(
            f"Expected file: {INPUT_FILE}"
        )

        exit()


    print(
        "\nRed Team transaction file detected!"
    )


    red_team_data = pd.read_csv(
        INPUT_FILE
    )


    print(
        f"\nTotal transactions received: "
        f"{len(red_team_data)}"
    )


    results = []


    # ========================================================
    # PROCESS EACH TRANSACTION
    # ========================================================

    for index, transaction in red_team_data.iterrows():

        print("\n" + "=" * 70)

        print(
            f"Processing transaction "
            f"{index + 1}/{len(red_team_data)}"
        )

        print(
            "Transaction ID:",
            transaction["transaction_id"]
        )

        print(
            "User ID:",
            transaction["user_id"]
        )

        print(
            "Amount: ₹",
            transaction["amount"]
        )


        # Blue Team detection

        result = process_transaction(
            transaction,
            historical_data
        )


        # Display result

        print("\nBLUE TEAM ANALYSIS")

        print(
            f"ML Score: "
            f"{result['ml_score']:.2f}"
        )

        print(
            f"Rule Score: "
            f"{result['rule_score']:.2f}"
        )

        print(
            f"Anomaly Score: "
            f"{result['anomaly_score']:.2f}"
        )

        print(
            f"Graph Score: "
            f"{result['graph_score']:.2f}"
        )

        print(
            f"\nFINAL RISK SCORE: "
            f"{result['final_risk_score']:.2f}"
        )

        print(
            f"BLUE TEAM DECISION: "
            f"{result['blue_team_decision']}"
        )


        if result["blue_team_decision"] == "BLOCK":

            print(
                "\n🚨 FRAUD ALERT!"
            )

        elif result["blue_team_decision"] == "HOLD":

            print(
                "\n⚠️ TRANSACTION UNDER REVIEW"
            )

        else:

            print(
                "\n✓ TRANSACTION ALLOWED"
            )


        results.append(
            result
        )


        # Add transaction to history

        historical_data = pd.concat(

            [
                historical_data,
                pd.DataFrame([result])
            ],

            ignore_index=True

        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("\n" + "=" * 70)

    print("BLUE TEAM MONITORING COMPLETE")

    print("=" * 70)

    print(
        f"\nResults saved to:"
        f" {OUTPUT_FILE}"
    )


    print("\nSUMMARY")

    print(
        results_df[
            "blue_team_decision"
        ].value_counts()
    )