import sys
import os
import pandas as pd

from src.logger import logging
from src.utils import load_object
from src.exception import CustomException


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_dataframe(self):
        try:
            data_dict = {
                "gender": self.gender,
                "race_ethnicity": self.race_ethnicity,
                "parental_level_of_education": self.parental_level_of_education,
                "lunch": self.lunch,
                "test_preparation_course": self.test_preparation_course,
                "reading_score": self.reading_score,
                "writing_score": self.writing_score,
            }

            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info("Preprocessor and Model loaded successfully")

            features_preprocessed = preprocessor.fit_transform(features)
            prediction = model.predict(features_preprocessed)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
