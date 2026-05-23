cls
from pathlib import Path
import traceback

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_and_save() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = (
        base_dir
        / ".."
        / "processed_datasets"
        / "Customer_Attributes_and_Purchase_Propensity_processed.csv"
    )
    model_path = base_dir / ".." / "models" / "purchase_propensity_model.pkl"
    scaler_path = base_dir / ".." / "models" / "purchase_propensity_scaler.pkl"
    encoders_path = base_dir / ".." / "models" / "purchase_propensity_encoders.pkl"

    model_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    if "Purchased" not in data.columns:
        raise ValueError("Purchased column not found in dataset")

    categorical_cols = ["City", "Marital_Status"]
    label_encoders = {}
    for column in categorical_cols:
        if column in data.columns:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column].astype(str))
            label_encoders[column] = encoder

    features = data.drop(columns=["Purchased"])
    target = data["Purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, encoders_path)

    score = model.score(X_test_scaled, y_test)
    print(f"Purchase propensity model saved. Accuracy: {score:.4f}")


if __name__ == "__main__":
    try:
        train_and_save()
    except Exception as exc:
        print(f"Training failed: {exc}")
        traceback.print_exc()
