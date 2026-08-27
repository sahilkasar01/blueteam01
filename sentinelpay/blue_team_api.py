import os
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# BLUE TEAM API
# ============================================================

app = FastAPI(
    title="SentinelPay Blue Team API",
    description="SentinelPay AI fraud detection and adaptive defense API",
    version="2.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "models/fraud_model.pkl"
FEATURE_FILE = "models/feature_columns.pkl"

HISTORY_FILE = "features.csv"
LOG_FILE = "blue_team_api_log.csv"


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} not found. Run ML training first."
    )

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        f"{FEATURE_FILE} not found."
    )

if not os.path.exists(HISTORY_FILE):
    raise FileNotFoundError(
        f"{HISTORY_FILE} not found."
    )


model = joblib.load(MODEL_FILE)
FEATURES = joblib.load(FEATURE_FILE)

print("✓ Blue Team ML model loaded")
print("✓ Features loaded")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

history = pd.read_csv(
    HISTORY_FILE,
    parse_dates=["timestamp"]
)

print(f"✓ Loaded {len(history)} historical transactions")


# ============================================================
# TRANSACTION FORMAT
# ============================================================

class Transaction(BaseModel):

    transaction_id: str
    user_id: str
    timestamp: str
    amount: float
    device_id: str
    merchant_id: str
    location: str


# ============================================================
# FEATURE FORMAT
# Used by Red Team adaptive attack loop
# ============================================================

class FeatureTransaction(BaseModel):

    transaction_id: str = "RED_ATTACK"
    user_id: str = "unknown"

    amount: float
    hour: int
    is_odd_hour: int
    is_new_device: int
    is_new_merchant: int

    amount_zscore: float
    amount_vs_user_avg: float

    seconds_since_last_txn: float
    txns_last_1h: int
    device_shared_users: int

    rule_score: float


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(txn):

    global history

    timestamp = pd.to_datetime(txn.timestamp)

    user_history = history[
        history["user_id"].astype(str)
        == str(txn.user_id)
    ].copy()

    amount = txn.amount


    # --------------------------------------------------------
    # HOUR
    # --------------------------------------------------------

    hour = timestamp.hour

    is_odd_hour = int(hour <= 5)


    # --------------------------------------------------------
    # USER AMOUNT BEHAVIOUR
    # --------------------------------------------------------

    if len(user_history) > 0:

        user_avg = user_history["amount"].mean()
        user_std = user_history["amount"].std()

        if pd.isna(user_std) or user_std == 0:
            user_std = 1

    else:

        user_avg = history["amount"].mean()
        user_std = history["amount"].std()

        if pd.isna(user_std) or user_std == 0:
            user_std = 1


    amount_vs_user_avg = (
        amount / max(user_avg, 1)
    )

    amount_zscore = (
        (amount - user_avg) / user_std
    )


    # --------------------------------------------------------
    # NEW DEVICE
    # --------------------------------------------------------

    known_devices = set(
        user_history["device_id"].astype(str)
    )

    is_new_device = int(
        str(txn.device_id) not in known_devices
    )


    # --------------------------------------------------------
    # NEW MERCHANT
    # --------------------------------------------------------

    known_merchants = set(
        user_history["merchant_id"].astype(str)
    )

    is_new_merchant = int(
        str(txn.merchant_id) not in known_merchants
    )


    # --------------------------------------------------------
    # TIME SINCE LAST TRANSACTION
    # --------------------------------------------------------

    seconds_since_last_txn = 999999

    if len(user_history) > 0:

        previous = user_history[
            pd.to_datetime(
                user_history["timestamp"]
            ) < timestamp
        ]

        if len(previous) > 0:

            last_time = pd.to_datetime(
                previous["timestamp"]
            ).max()

            seconds_since_last_txn = (
                timestamp - last_time
            ).total_seconds()


    # --------------------------------------------------------
    # TRANSACTION VELOCITY
    # --------------------------------------------------------

    start_time = (
        timestamp - pd.Timedelta(hours=1)
    )

    txns_last_1h = len(
        user_history[
            (
                pd.to_datetime(
                    user_history["timestamp"]
                ) >= start_time
            )
            &
            (
                pd.to_datetime(
                    user_history["timestamp"]
                ) < timestamp
            )
        ]
    )


    # --------------------------------------------------------
    # SHARED DEVICE
    # --------------------------------------------------------

    device_shared_users = history[
        history["device_id"].astype(str)
        == str(txn.device_id)
    ]["user_id"].nunique()

    device_shared_users = max(
        device_shared_users,
        1
    )


    # ========================================================
    # RULE ENGINE
    # ========================================================

    rule_score = 0


    if amount_vs_user_avg >= 5:
        rule_score += 35

    elif amount_vs_user_avg >= 3:
        rule_score += 20


    if is_new_device:
        rule_score += 20


    if is_new_merchant:
        rule_score += 10


    if txns_last_1h >= 10:
        rule_score += 30

    elif txns_last_1h >= 5:
        rule_score += 15


    if is_odd_hour:
        rule_score += 10


    if device_shared_users >= 5:
        rule_score += 25


    rule_score = min(
        rule_score,
        100
    )


    return {

        "amount": amount,

        "hour": hour,

        "is_odd_hour": is_odd_hour,

        "is_new_device": is_new_device,

        "is_new_merchant": is_new_merchant,

        "amount_zscore": amount_zscore,

        "amount_vs_user_avg": amount_vs_user_avg,

        "seconds_since_last_txn":
            seconds_since_last_txn,

        "txns_last_1h":
            txns_last_1h,

        "device_shared_users":
            device_shared_users,

        "rule_score":
            rule_score
    }


# ============================================================
# ML SCORE
# ============================================================

def get_ml_score(features):

    X = pd.DataFrame(
        [features]
    )

    X = X[FEATURES]

    probability = (
        model.predict_proba(X)[0][1]
    )

    return float(
        probability * 100
    )


# ============================================================
# ANOMALY SCORE
# ============================================================

def get_anomaly_score(features):

    score = 0


    if features["amount_zscore"] > 3:
        score += 30

    elif features["amount_zscore"] > 2:
        score += 20


    if features["is_new_device"]:
        score += 20


    if features["txns_last_1h"] >= 10:
        score += 30

    elif features["txns_last_1h"] >= 5:
        score += 15


    if features["device_shared_users"] >= 5:
        score += 20


    return min(
        score,
        100
    )


# ============================================================
# GRAPH SCORE
# ============================================================

def get_graph_score(txn):

    shared_users = history[
        history["device_id"].astype(str)
        == str(txn.device_id)
    ]["user_id"].nunique()


    if shared_users >= 10:
        return 100

    elif shared_users >= 5:
        return 60

    elif shared_users >= 3:
        return 30

    return 0


# ============================================================
# RISK FUSION
# ============================================================

def calculate_final_risk(
    ml_score,
    rule_score,
    anomaly_score,
    graph_score
):

    return (

        0.40 * ml_score

        + 0.25 * rule_score

        + 0.20 * anomaly_score

        + 0.15 * graph_score

    )


# ============================================================
# DECISION
# ============================================================

def make_decision(final_score):

    if final_score >= 70:
        return "BLOCK"

    elif final_score >= 40:
        return "HOLD"

    return "ALLOW"


# ============================================================
# COMMON BLUE TEAM SCORING
# ============================================================

def run_blue_team(features, graph_score=0):

    ml_score = get_ml_score(features)

    rule_score = features["rule_score"]

    anomaly_score = get_anomaly_score(features)

    final_score = calculate_final_risk(
        ml_score,
        rule_score,
        anomaly_score,
        graph_score
    )

    decision = make_decision(
        final_score
    )

    return {

        "ml_score": round(
            ml_score,
            2
        ),

        "rule_score": round(
            rule_score,
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

        "decision": decision
    }


# ============================================================
# NORMAL TRANSACTION API
# Red Team / frontend can send real transaction
# ============================================================

@app.post("/transaction")
def analyze_transaction(txn: Transaction):

    global history


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    features = create_features(txn)


    # --------------------------------------------------------
    # GRAPH SCORE
    # --------------------------------------------------------

    graph_score = get_graph_score(txn)


    # --------------------------------------------------------
    # BLUE TEAM DECISION
    # --------------------------------------------------------

    result = run_blue_team(
        features,
        graph_score
    )


    result = {

        "transaction_id":
            txn.transaction_id,

        "user_id":
            txn.user_id,

        "amount":
            txn.amount,

        **result

    }


    # --------------------------------------------------------
    # LIVE HISTORY
    # --------------------------------------------------------

    new_row = {

        "transaction_id":
            txn.transaction_id,

        "user_id":
            txn.user_id,

        "timestamp":
            txn.timestamp,

        "amount":
            txn.amount,

        "device_id":
            txn.device_id,

        "merchant_id":
            txn.merchant_id,

        "location":
            txn.location,

        **features

    }


    history = pd.concat(
        [
            history,
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # SAVE LOG
    # --------------------------------------------------------

    save_log(
        new_row,
        result
    )


    return result


# ============================================================
# RED TEAM ADAPTIVE FEATURE API
#
# This is important for 08_redteam_blueteam_loop.py
# because Red Team directly mutates features.
# ============================================================

@app.post("/score-features")
def score_features(txn: FeatureTransaction):

    features = {

        "amount":
            txn.amount,

        "hour":
            txn.hour,

        "is_odd_hour":
            txn.is_odd_hour,

        "is_new_device":
            txn.is_new_device,

        "is_new_merchant":
            txn.is_new_merchant,

        "amount_zscore":
            txn.amount_zscore,

        "amount_vs_user_avg":
            txn.amount_vs_user_avg,

        "seconds_since_last_txn":
            txn.seconds_since_last_txn,

        "txns_last_1h":
            txn.txns_last_1h,

        "device_shared_users":
            txn.device_shared_users,

        "rule_score":
            txn.rule_score
    }


    result = run_blue_team(
        features,
        graph_score=0
    )


    result["transaction_id"] = (
        txn.transaction_id
    )

    result["user_id"] = (
        txn.user_id
    )

    result["amount"] = (
        txn.amount
    )


    save_log(
        features,
        result
    )


    return result


# ============================================================
# SAVE LOG
# ============================================================

def save_log(
    transaction_data,
    result
):

    row = {

        **transaction_data,

        **result

    }


    log_df = pd.DataFrame(
        [row]
    )


    if os.path.exists(LOG_FILE):

        log_df.to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        log_df.to_csv(
            LOG_FILE,
            index=False
        )


# ============================================================
# DASHBOARD - ALL TRANSACTIONS
# ============================================================

@app.get("/dashboard/transactions")
def dashboard_transactions():

    if not os.path.exists(
        LOG_FILE
    ):

        return []


    df = pd.read_csv(
        LOG_FILE
    )


    df = df.fillna("")


    return df.to_dict(
        orient="records"
    )


# ============================================================
# DASHBOARD - SUMMARY
# ============================================================

@app.get("/dashboard/summary")
def dashboard_summary():

    if not os.path.exists(
        LOG_FILE
    ):

        return {

            "total_transactions": 0,

            "blocked": 0,

            "held": 0,

            "allowed": 0,

            "average_risk": 0,

            "high_risk": 0
        }


    df = pd.read_csv(
        LOG_FILE
    )


    total = len(df)


    blocked = int(
        (
            df["decision"]
            == "BLOCK"
        ).sum()
    )


    held = int(
        (
            df["decision"]
            == "HOLD"
        ).sum()
    )


    allowed = int(
        (
            df["decision"]
            == "ALLOW"
        ).sum()
    )


    average_risk = round(
        df["final_risk_score"]
        .astype(float)
        .mean(),
        2
    )


    high_risk = int(
        (
            df["final_risk_score"]
            >= 70
        ).sum()
    )


    return {

        "total_transactions":
            total,

        "blocked":
            blocked,

        "held":
            held,

        "allowed":
            allowed,

        "average_risk":
            average_risk,

        "high_risk":
            high_risk
    }


# ============================================================
# DASHBOARD - LATEST TRANSACTION
# ============================================================

@app.get("/dashboard/latest")
def dashboard_latest():

    if not os.path.exists(
        LOG_FILE
    ):

        return {
            "message":
                "No transactions yet"
        }


    df = pd.read_csv(
        LOG_FILE
    )


    latest = (
        df.tail(1)
        .fillna("")
        .to_dict(
            orient="records"
        )[0]
    )


    return latest


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {

        "status":
            "ONLINE",

        "team":
            "BLUE TEAM",

        "version":
            "2.0",

        "message":
            "SentinelPay Blue Team API is running"
    }


# ============================================================
# RUN
# ============================================================

# uvicorn blue_team_api:app --reload