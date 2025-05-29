import os
import sys
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from dotenv import load_dotenv
from dataclasses import dataclass
load_dotenv()

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        acc = accuracy_score(actual, pred)
        prec = precision_score(actual, pred)
        rec = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return acc, prec, rec, f1

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train = train_array[:, :-1]
            print("Xtrain" , X_train)
            y_train = train_array[:, -1]
            print("Ytrain" , y_train)
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]


            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(eval_metric='logloss'),
                "CatBoost": CatBoostClassifier(verbose=0),
                "KNeighbors": KNeighborsClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000)
            }

            params = {
                "Decision Tree": {
                    # 'criterion': ['gini', 'entropy']
                },
                "Random Forest": {
                    # 'n_estimators': [10, 40 , 45 , 50],
                    # 'max_depth' : [2, 10, 20],
                    # 'criterion' : ['gini', 'entropy']
                    
                },
                "Gradient Boosting": {
                    # 'n_estimators': [10, 50, 100 , 300],
                    # 'learning_rate': [0.01, 0.1, 0.2]
                },
                "XGBoost": {
                    # 'learning_rate': [0.01, 0.1, 0.2],
                    # 'n_estimators': [10, 50, 250]
                },
                "CatBoost": {
                    # 'learning_rate': [0.01, 0.1, 0.2],
                    # 'iterations': [10, 50, 200]
                },
                "AdaBoost": {
                    # 'n_estimators': [10, 50, 100 , 250],
                    # 'learning_rate': [0.01, 0.1, 0.2]
                },
                "KNeighbors": {
                    # 'n_neighbors': [3, 5, 7 , 9]
                },
                "Logistic Regression": {}
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, params, classification=True)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print("This is the best model name", best_model_name)

            best_param = params.get(best_model_name, {})

            # MLflow tracking
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                best_model.fit(X_train, y_train)
                predicted = best_model.predict(X_test)

                acc, prec, rec, f1 = self.eval_metrics(y_test, predicted)

                mlflow.log_params(best_param)
                mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
               })


                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
