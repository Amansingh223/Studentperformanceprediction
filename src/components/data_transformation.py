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
        try:
            numerical_cols = [
                "attendance_pct",
                "internal_marks",
                "assignment_score",
                "quiz_score",
                "study_hours_per_week",
                "previous_cgpa",
            ]
            categorical_cols = [
                "branch",
                "semester",
                "activity_participation",
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols),
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Train/test data loaded")

            # Target column
            target = "final_percentage"

            # Columns to drop (not features)
            drop_cols = ["student_name", "roll_number", "pass_fail", "grade", target]

            preprocessor = self.get_preprocessor()

            # Ensure semester is treated as string (categorical)
            train_df["semester"] = train_df["semester"].astype(str)
            test_df["semester"] = test_df["semester"].astype(str)

            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df[target]
            X_test = test_df.drop(columns=drop_cols)
            y_test = test_df[target]

            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_t, np.array(y_train)]
            test_arr = np.c_[X_test_t, np.array(y_test)]

            save_object(self.config.preprocessor_path, preprocessor)
            logger.info("Preprocessor saved")

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
