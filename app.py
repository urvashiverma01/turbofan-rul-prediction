"""
Flask app: RUL prediction from the last 5 cycles of engine sensor readings.

WHY 5 rows of input instead of 1 (unlike your SMS-spam / Titanic apps):
The model was trained on rolling mean/std/slope over a 5-cycle window, so
at inference time it needs that same 5-cycle window to compute identical
features - a single snapshot reading has no "trend" information in it.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Native XGBoost format (model_xgb.json) instead of pickle - avoids any
# pickle-protocol / sklearn-wrapper incompatibility across machines/OSes.
model = xgb.Booster()
model.load_model("model_xgb.json")
with open("feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

WINDOW = 5
USEFUL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]
# Raw column order as it appears in the original C-MAPSS txt files
ALL_COLS = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def slope(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return ((x - x_mean) * (y - y.mean())).sum() / denom


def build_features(rows):
    """
    rows: list of dicts or list of lists, each representing one cycle's
    reading in ALL_COLS order, oldest cycle first, most recent cycle last.
    Must contain exactly WINDOW rows.
    Returns a single-row DataFrame matching FEATURE_COLS exactly.
    """
    df = pd.DataFrame(rows, columns=ALL_COLS)
    current = df.iloc[-1]

    feat = {"setting_1": current["setting_1"], "setting_2": current["setting_2"]}
    for s in USEFUL_SENSORS:
        feat[s] = current[s]
        feat[f"{s}_rmean{WINDOW}"] = df[s].mean()
        feat[f"{s}_rstd{WINDOW}"] = df[s].std(ddof=1) if len(df) > 1 else 0.0
        feat[f"{s}_slope{WINDOW}"] = slope(df[s].values)

    return pd.DataFrame([feat])[FEATURE_COLS]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    rows = data.get("cycles")

    if not rows or len(rows) != WINDOW:
        return jsonify({"error": f"Provide exactly {WINDOW} cycles of readings, oldest first."}), 400

    try:
        X = build_features(rows)
    except Exception as e:
        return jsonify({"error": f"Could not parse input rows: {e}"}), 400

    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    pred = float(model.predict(dmatrix)[0])
    pred = max(0.0, pred)  # RUL can't be negative

    return jsonify({
        "predicted_RUL": round(pred, 1),
        "note": "Capped at 125 during training - values near 125 mean 'healthy, plenty of margin', not a precise countdown."
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
