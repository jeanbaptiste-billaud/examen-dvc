from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor

MODEL_REGISTRY = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "LGBMRegressor": LGBMRegressor,
    "HistGradientBoostingRegressor" : HistGradientBoostingRegressor,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "LinearRegression": LinearRegression,
    "NuSVR" : NuSVR,
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "MLPRegressor" : MLPRegressor,
    "BayesianRidge" : BayesianRidge,
}
