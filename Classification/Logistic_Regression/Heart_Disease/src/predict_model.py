import numpy as np
import joblib

def predict(input_data):
    model = joblib.load("C:/Users/Dalbir/Downloads/Machine Learning(2025)/Classification/Logistic_Regression/Heart_Disease/model.pkl")
    array_input = np.asarray(input_data)
    final_input = array_input.reshape(1, -1)
    prediction = model.predict(final_input)
    return prediction[0]

def print_prediction(pred):
    if pred == 1:
        print("He is a heart patient")
    else:
        print("He is OKAY")
