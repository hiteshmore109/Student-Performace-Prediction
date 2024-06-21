import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(object, file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        model_names = list(models.keys())
        report = {}

        for m in model_names:
            model = models[m]
            model_params = params[m]
            gs = GridSearchCV(model, model_params, cv=5, n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report[m] = [r2_score(y_test, y_pred), gs.best_estimator_]

        best_model = max(report.items(), key=lambda x: x[1][0])
        return best_model[0], best_model[1][0], best_model[1][1]
    except Exception as e:
        raise CustomException(e, sys)
