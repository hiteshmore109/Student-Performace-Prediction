import os
import sys
import pickle

from src.exception import CustomException


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(object, file)

    except Exception as e:
        raise CustomException(e, sys)
