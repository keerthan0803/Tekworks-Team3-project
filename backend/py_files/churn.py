import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import pickle
import traceback


try:
    data = pd.read_csv('../processed_datasets/churn_prediction_data.csv')

    # Encode all categorical columns (object/category) except the target
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Churn Label' in categorical_cols:
        categorical_cols.remove('Churn Label')
    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols)

    x = data.drop('Churn Label', axis=1)
    y = data['Churn Label']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = AdaBoostClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # save the model in models folder
    with open('../models/churn_model_ada.pkl', 'wb') as f:
        pickle.dump(model, f)
except Exception as exc:
    print(f"Error: {exc}")
    traceback.print_exc()