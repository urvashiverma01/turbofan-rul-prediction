# Turbofan Engine RUL Prediction

Predicts the Remaining Useful Life (RUL) — operational cycles left before failure — of a turbofan engine from its recent sensor readings, using NASA's C-MAPSS (FD001) turbofan degradation dataset.

**Live demo:** _add your Render URL here once deployed_
**Dataset:** [NASA C-MAPSS Turbofan Engine Degradation Simulation](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

## Overview

Each engine in the dataset runs from healthy to failure across ~130–360 cycles, recorded through 3 operational settings and 21 sensors. The goal: given a short window of recent readings, predict how many cycles remain before failure — the kind of signal a predictive maintenance system would use to schedule servicing before something breaks in the field.

## Pipeline

1. **Exploration** — confirmed FD001 is a single-operating-condition, single-fault-mode subset; identified 7 sensors with zero/near-zero variance and dropped them.
2. **Labeling** — computed RUL per row (`max_cycle - current_cycle`), then **capped at 125 cycles**. Early in an engine's life there's no sensor evidence to distinguish "300 cycles left" from "280 cycles left" — capping prevents the model from being penalized for that unobservable precision and better reflects the real flat-then-falling shape of degradation.
3. **Feature engineering** — for each of the 14 informative sensors: raw value, 5-cycle rolling mean, rolling std, and rolling slope (trend/velocity of degradation), computed per-engine to avoid leakage across units.
4. **Modeling** — Random Forest and XGBoost regression baselines. XGBoost was evaluated as the primary model; an LSTM was considered but judged unnecessary for FD001 specifically, since its single operating condition/fault mode means rolling-window tabular features already capture most of the useful signal — LSTMs pay off more on the harder FD002/FD004 subsets.
5. **Evaluation** — RMSE plus the **NASA/PHM08 asymmetric scoring function**, which penalizes *late* predictions (overestimating remaining life — a real safety risk) more heavily than *early* ones (underestimating — just costs unnecessary maintenance).
6. **Deployment** — Flask app with a custom instrument-panel UI: paste the last 5 cycles of sensor readings, get a live RUL gauge reading.

## Results (FD001 test set, 100 engines)

| Model | RMSE | NASA Score (lower = better) |
|---|---|---|
| Random Forest | 18.83 | 1025.0 |
| **XGBoost** | **18.28** | **865.4** |

XGBoost's larger relative improvement on NASA score vs. RMSE indicates it specifically makes fewer *late* (dangerous) prediction errors — the failure mode this metric is designed to catch.

## Tech stack

Python · pandas · scikit-learn · XGBoost · Flask · vanilla JS/SVG (gauge UI)

## Running locally

```bash
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:5000`.

## Project structure

```
├── app.py               # Flask app: feature engineering + inference
├── model_xgb.json        # trained XGBoost model (native format)
├── feature_cols.json     # exact feature column order expected by the model
├── requirements.txt
└── templates/
    └── index.html         # instrument-panel UI
```
