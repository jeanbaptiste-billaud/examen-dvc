import pickle

from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyRegressor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def run_lazy_regression(X_train, X_test, y_train, y_test):
    lazy = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = lazy.fit(X_train, X_test, y_train, y_test)
    return models.sort_values(by="R-Squared", ascending=False)


def creat_model_map():
    """
    instancie un dictionnaire liant les modèles sklearn à leurs noms
    """
    MODEL_MAP = {
        "RandomForestRegressor": RandomForestRegressor,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "LinearRegression": LinearRegression,
        "SVR": SVR,
        "KNeighborsRegressor": KNeighborsRegressor
    }
    return MODEL_MAP


def run_grid_search(X_train, y_train, model_classe):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(model_classe(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score', ascending=True).to_json('metrics/GridSearch_results.json', orient='records')

    return grid_search.best_estimator_, grid_search.best_params_


def main():
    # step 1: import
    X_train_scale = pd.read_csv('data/preprocessed/X_train_scale.csv')
    X_test_scale = pd.read_csv('data/preprocessed/X_test_scale.csv')
    y_train = pd.read_csv('data/preprocessed/y_train.csv')
    y_test = pd.read_csv('data/preprocessed/y_test.csv')

    # step 2 : identification du modèle par "lazyregression"
    lazy_results = run_lazy_regression(X_train_scale, X_test_scale, y_train, y_test)
    lazy_results.to_json('metrics/lazy_results.json')
    top_model_name = lazy_results.index[0]
    print(f"Modèle recommandé par LazyRegressor : {top_model_name}")

    # step 3 : instanciation de la classe
    model_class = None
    model_map = creat_model_map()
    if top_model_name in model_map:
        model_class = model_map[top_model_name]
        print(f"{model_class} pret à être instancié :")
    else:
        print(f"⚠️ Le modèle {top_model_name} n’est pas dans model_map.")

    # step 4 : recherche de paramètres avec GridSearch
    best_model, best_params = run_grid_search(X_train_scale, y_train, model_class)
    gs_dict = {"model":top_model_name, "params":best_params}
    with open('./models/gridsearch.pkl', 'wb') as handle:
        pickle.dump(gs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()