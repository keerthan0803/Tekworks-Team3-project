from pathlib import Path
import traceback

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_and_save() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / ".." / "raw_datasets" / "campaign_responses.csv"
    model_path = base_dir / ".." / "models" / "market_response_model.pkl"
    scaler_path = base_dir / ".." / "models" / "market_response_scaler.pkl"
    encoders_path = base_dir / ".." / "models" / "market_response_encoders.pkl"

    data = pd.read_csv(data_path)
    if "customer_id" in data.columns:
        data = data.drop("customer_id", axis=1)

    categorical_cols = ["gender", "employed", "marital_status"]
    label_encoders = {}
    for column in categorical_cols:
        if column in data.columns:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column].astype(str))
            label_encoders[column] = encoder

    if data["responded"].dtype == "object":
        data["responded"] = data["responded"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0}
        )

    features = [
        "age",
        "annual_income",
        "credit_score",
        "no_of_children",
        "gender",
        "employed",
        "marital_status",
    ]
    X = data[features]
    y = data["responded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, encoders_path)

    print("Market response model, scaler, and encoders saved successfully")


if __name__ == "__main__":
    try:
        train_and_save()
    except Exception as exc:
        print(f"Training failed: {exc}")
        traceback.print_exc()