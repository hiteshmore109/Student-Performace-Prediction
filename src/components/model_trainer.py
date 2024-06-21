import os
import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train, test):
        try:
            logging.info("Initiating model training")
            X_train, y_train, X_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1],
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "CatBoost": CatBoostRegressor(logging_level="Silent"),
            }

            params = {
                "Linear Regression": {},
                "Ridge": {"alpha": [0.1, 1, 10]},
                "Lasso": {"alpha": [0.1, 1, 10]},
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, 15],
                },
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "max_depth": [5, 10, 15],
                },
                "CatBoost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
            }

            best_model_name, best_model_score, best_model = evaluate_model(
                X_train, y_train, X_test, y_test, models, params
            )
            logging.info(
                f"Best model: {best_model_name}, R2 Score: {best_model_score}, Best Parameters: {best_model.get_params()}"
            )
            save_object(self.model_trainer_config.trained_model_path, best_model)
            logging.info("Best Model training completed and saved successfully")

        except Exception as e:
            raise CustomException(e, sys)
