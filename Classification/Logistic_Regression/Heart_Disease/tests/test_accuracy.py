import pandas as pd
from sklearn.linear_model import LogisticRegression
from Classification.Logistic_Regression.Heart_Disease.src.model_training import evaluate_model
def test_accuracy_between_0_and_1():
    df = pd.DataFrame({
        "age": [29, 45, 61, 50, 38, 55],
        "cholesterol": [200, 250, 240, 180, 210, 220],
        "target": [0, 1, 1, 0, 0, 1]
    })

    X = df.drop(columns=["target"])
    y = df["target"]
    model = LogisticRegression(max_iter=200).fit(X, y)

    acc = evaluate_model(model, X, y)
    assert 0.0 <= acc <= 1.0
