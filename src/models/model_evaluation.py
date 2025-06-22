import os

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_data():
    X_test = pd.read_csv('data/processed/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').to_numpy()

    if y_test.ndim > 1:
        y_test = y_test.ravel()

    return X_test, y_test


def main():
    X_test, y_test = load_data()
    model = joblib.load("models/model/trained_model.pkl")
    y_pred = model.predict(X_test)

    scores = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

    # Export des scores
    os.makedirs("metrics", exist_ok=True)
    pd.DataFrame.from_dict(scores, orient='index', columns=["value"]).to_json('metrics/scores.json')

    # Export des prédictions
    df_preds = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    df_preds.to_csv("data/predictions.csv", index=False)


if __name__ == "__main__":
    main()
