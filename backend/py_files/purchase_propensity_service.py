from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel


class PredictRequest(BaseModel):
    data: Dict[str, Any]


base_dir = Path(__file__).resolve().parent
model_path = base_dir / ".." / "models" / "purchase_propensity_model.pkl"
scaler_path = base_dir / ".." / "models" / "purchase_propensity_scaler.pkl"
encoders_path = base_dir / ".." / "models" / "purchase_propensity_encoders.pkl"
data_path = (
    base_dir
    / ".."
    / "processed_datasets"
    / "Customer_Attributes_and_Purchase_Propensity_processed.csv"
)

try:
    purchase_model = joblib.load(model_path)
    purchase_scaler = joblib.load(scaler_path)
    purchase_encoders = joblib.load(encoders_path)
except FileNotFoundError as exc:
    raise RuntimeError(
        "Purchase propensity artifacts not found. "
        "Run backend/py_files/purchase_propensity.py first."
    ) from exc

try:
    training_data = pd.read_csv(data_path)
except FileNotFoundError as exc:
    raise RuntimeError(f"Training data not found at {data_path}") from exc

if "Purchased" not in training_data.columns:
    raise RuntimeError("Purchased column not found in dataset")

expected_columns = training_data.drop(columns=["Purchased"]).columns.tolist()


def prepare_input(data: Dict[str, Any]) -> pd.DataFrame:
    input_df = pd.DataFrame([data])

    for column, encoder in purchase_encoders.items():
        if column not in input_df.columns:
            continue
        try:
            input_df[column] = encoder.transform(input_df[column].astype(str))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for {column}: {exc}",
            ) from exc

    return input_df.reindex(columns=expected_columns, fill_value=0)


def predict_purchase_propensity(request: PredictRequest) -> Dict[str, Any]:
    if not request.data:
        raise HTTPException(status_code=400, detail="Input data is required")

    input_df = prepare_input(request.data)
    scaled_input = purchase_scaler.transform(input_df)
    prediction = purchase_model.predict(scaled_input)[0]

    response: Dict[str, Any] = {"prediction": int(prediction)}
    if hasattr(purchase_model, "predict_proba"):
        proba = purchase_model.predict_proba(scaled_input)[0]
        response["probability"] = float(proba[1])

    return response
