import os
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        """Build and return the ColumnTransformer preprocessing pipeline."""
        try:
            numerical_cols = ["writing score", "reading score"]
            categorical_cols = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler",  StandardScaler(with_mean=False)),
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols),
            ])

            logger.info(f"Numerical cols: {numerical_cols}")
            logger.info(f"Categorical cols: {categorical_cols}")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logger.info("Train/test data loaded for transformation")

            target = "math score"
            preprocessor = self.get_preprocessor()

            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]
            X_test  = test_df.drop(columns=[target])
            y_test  = test_df[target]

            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t  = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_t, np.array(y_train)]
            test_arr  = np.c_[X_test_t,  np.array(y_test)]

            save_object(self.config.preprocessor_path, preprocessor)
            logger.info("Preprocessor saved")

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
