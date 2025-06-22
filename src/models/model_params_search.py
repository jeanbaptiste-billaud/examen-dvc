import logging
import os
from pprint import pprint

import joblib
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from model_registry import MODEL_REGISTRY


def run_lazy_regression(X_train, X_test, y_train, y_test):
    lazy = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = lazy.fit(X_train, X_test, y_train, y_test)
    return models.sort_values(by="R-Squared", ascending=False)


def gridsearch_pipeline(model_classe):
    """
    création d'un pipeline pour GridSearchCV afin de tester plusieurs modèles
    :param model_classe:
    :return: pipe, param_grid
    """
    pipe = Pipeline([
        ("model", model_classe[0]())  # valeur par défaut, remplacée dans le grid
    ])

    param_grid = []
    for model in model_classe:
        base = {"model": [model()]}

        if model.__name__ == "LGBMRegressor":
            base["model__force_col_wise"] = [True]

        if hasattr(model(), "n_estimators"):
            base["model__n_estimators"] = [50, 100, 200]
        if hasattr(model(), "max_depth"):
            base["model__max_depth"] = [None, 3, 10, 20]
        if hasattr(model(), "min_samples_split"):
            base["model__min_samples_split"] = [2, 5]
        if hasattr(model(), "learning_rate"):
            base["model__learning_rate"] = [0.05, 0.1]
        if hasattr(model(), "subsample"):
            base["model__subsample"] = [0.8, 1.0]
        if hasattr(model(), "samples_leaf"):
            base['min_samples_leaf'] = [1, 2]

        param_grid.append(base)
    pprint(param_grid)

    return pipe, param_grid


def run_grid_search(X_train, y_train, model_classe):
    if y_train.ndim > 1:
        y_train = y_train.ravel()

    pipe, param_grid = gridsearch_pipeline(model_classe)
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results["model_name"] = cv_results["param_model"].apply(lambda m: type(m).__name__)
    cv_results = cv_results.sort_values('rank_test_score', ascending=True)

    columns_to_keep = ["model_name", "mean_test_score", "std_test_score", "rank_test_score", "params"]
    filtered = cv_results[columns_to_keep]
    filtered.to_json("metrics/gridsearch_results_named.json", orient="records", indent=4)
    cv_results.to_json('models/gridsearch/GridSearch_results.json', orient='records')

    best_estimator = type(grid_search.best_params_['model']).__name__

    return best_estimator, grid_search.best_params_


def main():
    logger = logging.getLogger(__name__)
    logger.info('Recherche de modèle de regression avec lazypredict')
    # step 1: import
    X_train_scale = pd.read_csv('data/processed/X_train_scaled.csv')
    X_test_scale = pd.read_csv('data/processed/X_test_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').to_numpy()
    y_test = pd.read_csv('data/processed/y_test.csv').to_numpy()

    # step 2 : identification du modèle par "lazyregression"
    lazy_results = run_lazy_regression(X_train_scale, X_test_scale, y_train, y_test)
    os.makedirs("models/gridsearch", exist_ok=True)
    lazy_results.to_json('models/gridsearch/lazy_results.json')
    top_model_names = list(lazy_results[:5].index)
    logger.info(f"Top models : {top_model_names}")

    # step 3 : instanciation de la classe
    model_class = list()
    model_map = MODEL_REGISTRY
    for name in top_model_names:
        if name in model_map:
            model_class.append(model_map[name])
        else:
            print(f"⚠️ Le modèle {name} n’est pas dans model_map.")

    # step 4 : recherche de paramètres avec GridSearch
    logger.info('Recherche des paramètres optimaux avec GridSearchCV')
    best_model, best_params = run_grid_search(X_train_scale, y_train, model_class)
    joblib.dump(best_params, 'models/gridsearch/gridsearch.pkl')
    logger.info(f"Modèle retenu par GridSaearch : {best_model}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
