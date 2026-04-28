import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate(self, train_arr, test_arr):
        try:
            logger.info("Splitting arrays")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "K-Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
            }
            params = {
                "Linear Regression": {},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.01, 0.1, 1.0]},
                "K-Neighbors": {"n_neighbors": [3, 5, 7]},
                "Decision Tree": {"criterion": ["squared_error", "friedman_mse"]},
                "Random Forest": {"n_estimators": [50, 100]},
                "Gradient Boosting": {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200]},
                "AdaBoost": {"learning_rate": [0.01, 0.1], "n_estimators": [50, 100]},
                "XGBoost": {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200]},
            }
            report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            best_name = max(report, key=report.get)
            best_score = report[best_name]
            logger.info(f"Best model: {best_name} R2={best_score:.4f}")
            best_model = models[best_name]
            save_object(self.config.model_path, best_model)
            y_pred = best_model.predict(X_test)
            return r2_score(y_test, y_pred)
        except Exception as e:
            raise CustomException(e, sys)
