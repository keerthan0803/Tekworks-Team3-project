import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import traceback

try:
    data = pd.read_csv('../processed_datasets/cleaned_churn_data.csv')

    x = data.drop('Churn Label', axis=1)
    y = data['Churn Label']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)

    # save the model in models folder
    with open('../models/churn_model_lr.pkl', 'wb') as f:
        pickle.dump(model, f)
except Exception as exc:
    print(f"Error: {exc}")
    traceback.print_exc()