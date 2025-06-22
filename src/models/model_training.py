import logging
import os

import joblib
import pandas as pd
from model_registry import MODEL_REGISTRY


def load_data():
    X_train = pd.read_csv('data/processed/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').to_numpy()

    if y_train.ndim > 1:
        y_train = y_train.ravel()

    return X_train, y_train


def model_init():
    # Chargement des paramètres
    gridsearch_dict = joblib.load("models/gridsearch/gridsearch.pkl")

    # Récupération de la classe et instanciation
    model_class = type(gridsearch_dict["model"])
    clean_params = {
        k.replace("model__", ""): v
        for k, v in gridsearch_dict.items()
        if k.startswith("model__")
    }
    model = model_class(**clean_params)
    return model


def main():
    X_train, y_train = load_data()
    model = model_init()
    model.fit(X_train, y_train)
    os.makedirs("models/model", exist_ok=True)
    joblib.dump(model, "models/model/trained_model.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
