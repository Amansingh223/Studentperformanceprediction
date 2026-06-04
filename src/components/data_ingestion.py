import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    raw_data_path:   str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path:  str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate(self):
        logger.info("Starting data ingestion")
        try:
            df = pd.read_csv(
                os.path.join("notebook", "data", "student_performance_data.csv")
            )
            logger.info(f"Dataset loaded: {df.shape}")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.config.train_data_path, index=False)
            test.to_csv(self.config.test_data_path, index=False)

            logger.info(f"Train: {train.shape} | Test: {test.shape}")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate()
    print("Ingestion done")

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate(train_path, test_path)
    print("Transformation done")

    trainer = ModelTrainer()
    r2 = trainer.initiate(train_arr, test_arr)
    print(f"Best model R2 score: {r2:.4f}")