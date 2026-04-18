import pandas as pd
import os
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.datascience import logger
from src.datascience.entity.config_entity import ModelTrainerConfig
from dotenv import load_dotenv

load_dotenv()


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        # 🔹 MLflow setup (DagsHub)
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("End-to-End-MLops-Workflow")

        with mlflow.start_run():

            logger.info("Loading training and test data")

            # 🔹 Load data
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            train_x = train_data.drop([self.config.target_column], axis=1)
            test_x = test_data.drop([self.config.target_column], axis=1)
            train_y = train_data[[self.config.target_column]]
            test_y = test_data[[self.config.target_column]]

            # 🔹 Log parameters
            mlflow.log_param("alpha", self.config.alpha)
            mlflow.log_param("l1_ratio", self.config.l1_ratio)

            logger.info("Training model")

            # 🔹 Train model
            model = ElasticNet(
                alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42
            )
            model.fit(train_x, train_y)

            # 🔹 Predictions
            y_pred = model.predict(test_x)

            # 🔹 Metrics

            mse = mean_squared_error(test_y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_y, y_pred)
            r2 = r2_score(test_y, y_pred)

            # 🔹 Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            logger.info(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

            # 🔹 Save model locally
            os.makedirs(self.config.root_dir, exist_ok=True)
            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            joblib.dump(model, model_path)

            # 🔹 Log model to MLflow (NO registry)
            mlflow.sklearn.log_model(model, "model")

            # 🔹 Log model file as artifact
            mlflow.log_artifact(model_path)

            # 🔹 (Optional) log datasets
            mlflow.log_artifact(self.config.train_data_path)
            mlflow.log_artifact(self.config.test_data_path)

            logger.info("Model training and MLflow logging completed successfully")
