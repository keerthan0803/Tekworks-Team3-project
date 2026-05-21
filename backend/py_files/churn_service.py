from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pickle
from fastapi import HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder


class PredictRequest(BaseModel):
    data: Dict[str, Any]


base_dir = Path(__file__).resolve().parent


def load_bundle() -> Dict[str, Any]:
    model_path = base_dir / ".." / "models" / "churn_model_ada.pkl"
    data_path = base_dir / ".." / "processed_datasets" / "churn_prediction_data.csv"

    try:
        with model_path.open("rb") as file_obj:
            model = pickle.load(file_obj)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Model file not found at {model_path}") from exc

    try:
        training_data = pd.read_csv(data_path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Training data not found at {data_path}") from exc

    categorical_cols = training_data.select_dtypes(include=["object", "category"]).columns.tolist()
    if "Churn Label" in categorical_cols:
        categorical_cols.remove("Churn Label")

    encoded_training = pd.get_dummies(training_data, columns=categorical_cols)
    expected_columns = encoded_training.drop("Churn Label", axis=1).columns.tolist()

    return {
        "model": model,
        "categorical_cols": categorical_cols,
        "expected_columns": expected_columns,
    }


def prepare_input(
    data: Dict[str, Any],
    categorical_cols: List[str],
    expected_columns: List[str],
) -> pd.DataFrame:
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    return input_df.reindex(columns=expected_columns, fill_value=0)


bundle = load_bundle()


def predict_churn(request: PredictRequest) -> Dict[str, Any]:
    if not request.data:
        raise HTTPException(status_code=400, detail="Input data is required")

    input_df = prepare_input(
        request.data,
        bundle["categorical_cols"],
        bundle["expected_columns"],
    )

    prediction = bundle["model"].predict(input_df)[0]
    response: Dict[str, Any]
    if isinstance(prediction, (int, float)):
        response = {"prediction": int(prediction)}
    else:
        prediction_str = str(prediction)
        response = {"prediction": prediction_str}
        if prediction_str.lower() in {"yes", "no"}:
            response["prediction_numeric"] = 1 if prediction_str.lower() == "yes" else 0

    if hasattr(bundle["model"], "predict_proba"):
        proba = bundle["model"].predict_proba(input_df)[0]
        response["probability"] = float(proba[1])

    return response
