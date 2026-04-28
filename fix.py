# data_transformation.py
open('src/components/data_transformation.py', 'w', encoding='utf-8').write('''import os
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
            target = "math score"
            preprocessor = self.get_preprocessor()
            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]
            X_test = test_df.drop(columns=[target])
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
''')
print("data_transformation.py fixed")

# model_trainer.py
open('src/components/model_trainer.py', 'w', encoding='utf-8').write('''import os
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
''')
print("model_trainer.py fixed")

# predict_pipeline.py
open('src/pipeline/predict_pipeline.py', 'w', encoding='utf-8').write('''import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            data = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
''')
print("predict_pipeline.py fixed")

print("\nAll done! Now run: python src/components/data_ingestion.py")