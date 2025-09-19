import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import mlflow.sklearn
from sklearn.metrics import accuracy_score

def train_models(X_train,Y_train,X_test,Y_test,C=1.0):
    mlflow.set_experiment("Heart Disease Prediction")

    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=200, C=C)
        model.fit(X_train,Y_train)
        mlflow.log_param("C", C)
        

        train_acc = accuracy_score(Y_train, model.predict(X_train))
        mlflow.log_metric("train_accuracy", float(train_acc))
        test_acc = accuracy_score(Y_test, model.predict(X_test))
        mlflow.log_metric("test_accuracy", float(test_acc))

        mlflow.sklearn.log_model(model, name = "logistic_model")
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/logistic_model",
            name="HeartDiseaseModel"
        )

        joblib.dump(model, "C:/Users/Dalbir/Downloads/Machine Learning(2025)/Classification/Logistic_Regression/Heart_Disease/model.pkl")
    return model

def evaluate_model(model,X,Y):
    pred = model.predict(X)
    model_acc = accuracy_score(Y,pred)
    return model_acc