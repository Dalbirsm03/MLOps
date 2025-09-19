import os
import pandas as pd
from sklearn.model_selection import train_test_split
from Classification.Logistic_Regression.Heart_Disease.src.data_loader import data_loader
from Classification.Logistic_Regression.Heart_Disease.src.model_training import train_models
from sklearn.linear_model import LogisticRegression
    
def test_data_upload():
    file = "C:/Users/Dalbir/Downloads/Machine Learning(2025)/Datasets/HeartDisease.csv"
    df = pd.read_csv("C:/Users/Dalbir/Downloads/Machine Learning(2025)/Datasets/HeartDisease.csv")
    X_train, X_test, Y_train, Y_test = data_loader(file)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert Y_train.shape[0] > 0
    assert Y_test.shape[0] > 0

    assert 'target' not in X_train.columns

def test_model_type():
    df = pd.DataFrame({
        "age": [29, 45, 61, 50, 38, 55, 62, 41],
        "cholesterol": [200, 250, 240, 180, 210, 220, 260, 190],
        "target": [0, 1, 1, 0, 0, 1, 1, 0]
    })

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test = X.iloc[:6], X.iloc[6:]
    y_train, y_test = y.iloc[:6], y.iloc[6:]

    model = train_models(X_train, y_train, X_test, y_test, C=1.0)

    assert isinstance(model, LogisticRegression)