from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

base_dir = Path(__file__).resolve().parent
model_path = base_dir / ".." / "models" / "market_response_model.pkl"
scaler_path = base_dir / ".." / "models" / "market_response_scaler.pkl"
encoders_path = base_dir / ".." / "models" / "market_response_encoders.pkl"

try:
    market_model = joblib.load(model_path)
    market_scaler = joblib.load(scaler_path)
    market_encoders = joblib.load(encoders_path)
except FileNotFoundError as exc:
    raise RuntimeError(
        "Market response model artifacts not found. "
        "Run backend/py_files/market_responce_prediction.py first."
    ) from exc


class MarketResponseRequest(BaseModel):
    age: int
    annual_income: float
    credit_score: int
    no_of_children: int
    gender: Union[int, str]
    employed: Union[int, str]
    marital_status: Union[int, str]


def encode_market_value(column: str, value: Union[int, str]) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    encoder = market_encoders.get(column)
    if encoder is None:
        return int(value)
    try:
        return int(encoder.transform([str(value)])[0])
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value for {column}: {exc}",
        ) from exc


def predict_market_response(request: MarketResponseRequest) -> Dict[str, Any]:
    features = np.array(
        [
            [
                request.age,
                request.annual_income,
                request.credit_score,
                request.no_of_children,
                encode_market_value("gender", request.gender),
                encode_market_value("employed", request.employed),
                encode_market_value("marital_status", request.marital_status),
            ]
        ]
    )

    scaled_features = market_scaler.transform(features)
    prediction = market_model.predict(scaled_features)[0]
    result = "Responded" if int(prediction) == 1 else "Not Responded"
    return {"prediction": result, "prediction_numeric": int(prediction)}
