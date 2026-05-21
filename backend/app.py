from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
app = FastAPI()


class PredictRequest(BaseModel):
	data: Dict[str, Any]


base_dir = Path(__file__).resolve().parent


def load_bundle(
	model_name: str,
	data_name: str,
	target_col: str,
	encoding: str,
) -> Dict[str, Any]:
	model_path = base_dir / "models" / model_name
	data_path = base_dir / "processed_datasets" / data_name

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
	if target_col in categorical_cols:
		categorical_cols.remove(target_col)

	label_encoders: Dict[str, LabelEncoder] = {}
	if encoding == "label":
		for column in categorical_cols:
			encoder = LabelEncoder()
			training_data[column] = encoder.fit_transform(training_data[column])
			label_encoders[column] = encoder
		expected_columns = training_data.drop(target_col, axis=1).columns.tolist()
	else:
		encoded_training = pd.get_dummies(training_data, columns=categorical_cols)
		expected_columns = encoded_training.drop(target_col, axis=1).columns.tolist()

	return {
		"model": model,
		"target_col": target_col,
		"encoding": encoding,
		"categorical_cols": categorical_cols,
		"expected_columns": expected_columns,
		"label_encoders": label_encoders,
	}


def prepare_input(
	data: Dict[str, Any],
	encoding: str,
	categorical_cols: List[str],
	expected_columns: List[str],
	label_encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> pd.DataFrame:
	input_df = pd.DataFrame([data])
	if encoding == "label":
		label_encoders = label_encoders or {}
		for column in categorical_cols:
			if column not in input_df.columns:
				continue
			encoder = label_encoders.get(column)
			if encoder is None:
				continue
			try:
				input_df[column] = encoder.transform(input_df[column])
			except ValueError as exc:
				raise HTTPException(
					status_code=400,
					detail=f"Invalid value for {column}: {exc}",
				) from exc
		return input_df.reindex(columns=expected_columns, fill_value=0)

	input_df = pd.get_dummies(input_df, columns=categorical_cols)
	return input_df.reindex(columns=expected_columns, fill_value=0)


churn_bundle = load_bundle(
	"churn_model_ada.pkl",
	"churn_prediction_data.csv",
	"Churn Label",
	encoding="onehot",
)
subscription_bundle = load_bundle(
	"subscription_renewal_model.pkl",
	"subscription_renewal_data.csv",
	"Churn",
	encoding="label",
)


@app.get("/")
def health_check() -> Dict[str, str]:
	return {"status": "ok"}



def predict_with_bundle(request: PredictRequest, bundle: Dict[str, Any]) -> Dict[str, Any]:
	if not request.data:
		raise HTTPException(status_code=400, detail="Input data is required")

	input_df = prepare_input(
		request.data,
		bundle["encoding"],
		bundle["categorical_cols"],
		bundle["expected_columns"],
		bundle.get("label_encoders"),
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


@app.post("/predict/churn")
def predict_churn(request: PredictRequest) -> Dict[str, Any]:
	return predict_with_bundle(request, churn_bundle)


@app.post("/predict/subscription")
def predict_subscription(request: PredictRequest) -> Dict[str, Any]:
	return predict_with_bundle(request, subscription_bundle)
