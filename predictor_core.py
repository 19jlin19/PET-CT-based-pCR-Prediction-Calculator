from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple
import json

import joblib
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "final_randomforest_pipeline.joblib"
METADATA_PATH = APP_DIR / "metadata.json"

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURES = METADATA["features"]
THRESHOLD = float(METADATA["threshold"])


def load_model():
    """Load the serialized Random Forest sklearn pipeline."""
    return joblib.load(MODEL_PATH)


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def calculate_suv_ratio(suvmax1: float, suvmax2: float) -> float:
    """Calculate metabolic response ratio: (SUVmax1 - SUVmax2) / SUVmax1."""
    suvmax1 = float(suvmax1)
    suvmax2 = float(suvmax2)
    if suvmax1 <= 0:
        raise ValueError("SUVmax1 must be greater than 0 to calculate SUV ratio.")
    return (suvmax1 - suvmax2) / suvmax1


def make_input_dataframe(
    suvmax1: float,
    suvmax2: float,
    suv_ratio: float,
    smoking: int,
    recist_response: str,
    histological: str,
    cn_status: str,
) -> pd.DataFrame:
    """Create a one-row dataframe with the exact feature order expected by the model."""
    row = {
        "SUV_ratio": float(suv_ratio),
        "SUVmax2": float(suvmax2),
        "SUVmax1": float(suvmax1),
        "Smoking": int(smoking),
        "RECIST_response": str(recist_response),
        "Histological": str(histological),
        "cN": str(cn_status),
    }
    return pd.DataFrame([row], columns=FEATURES)


def predict_probability(df: pd.DataFrame) -> np.ndarray:
    """Predict pCR probabilities for a dataframe containing the required model features."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    model = get_model()
    probs = model.predict_proba(df[FEATURES])[:, 1]
    return probs


def classify_probability(probability: float, threshold: float = THRESHOLD) -> str:
    return "Predicted pCR" if float(probability) >= threshold else "Predicted non-pCR"


def probability_band(probability: float) -> str:
    """Simple descriptive probability bands for display only."""
    p = float(probability)
    if p < 0.25:
        return "Low probability"
    if p < THRESHOLD:
        return "Intermediate-low probability"
    if p < 0.70:
        return "Intermediate-high probability"
    return "High probability"


def predict_single(
    suvmax1: float,
    suvmax2: float,
    suv_ratio: float,
    smoking: int,
    recist_response: str,
    histological: str,
    cn_status: str,
) -> Dict[str, object]:
    df = make_input_dataframe(
        suvmax1=suvmax1,
        suvmax2=suvmax2,
        suv_ratio=suv_ratio,
        smoking=smoking,
        recist_response=recist_response,
        histological=histological,
        cn_status=cn_status,
    )
    prob = float(predict_probability(df)[0])
    return {
        "pcr_probability": prob,
        "pcr_probability_percent": prob * 100,
        "threshold": THRESHOLD,
        "classification": classify_probability(prob),
        "probability_band": probability_band(prob),
        "input": df.iloc[0].to_dict(),
    }


def validate_batch_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a batch input dataframe."""
    data = df.copy()
    if "SUV_ratio" not in data.columns and {"SUVmax1", "SUVmax2"}.issubset(data.columns):
        data["SUV_ratio"] = (data["SUVmax1"].astype(float) - data["SUVmax2"].astype(float)) / data["SUVmax1"].astype(float)
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data["Smoking"] = data["Smoking"].astype(int)
    for c in ["RECIST_response", "Histological", "cN"]:
        data[c] = data[c].astype(str)
    for c in ["SUVmax1", "SUVmax2", "SUV_ratio"]:
        data[c] = data[c].astype(float)
    return data[FEATURES]


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    x = validate_batch_dataframe(df)
    probs = predict_probability(x)
    out = x.copy()
    out["pCR_probability"] = probs
    out["pCR_probability_percent"] = probs * 100
    out["threshold"] = THRESHOLD
    out["prediction"] = [classify_probability(p) for p in probs]
    out["probability_band"] = [probability_band(p) for p in probs]
    return out
