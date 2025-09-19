from Classification.Logistic_Regression.Heart_Disease.src.data_loader import data_loader
from Classification.Logistic_Regression.Heart_Disease.src.model_training import train_models,evaluate_model
from Classification.Logistic_Regression.Heart_Disease.src.predict_model import predict,print_prediction

X_train, X_test, Y_train, Y_test = data_loader("C:/Users/Dalbir/Downloads/Machine Learning(2025)/Datasets/HeartDisease.csv")
model = train_models(X_train,Y_train,X_test,Y_test)
print("Train Accuracy:", evaluate_model(model, X_train, Y_train))
print("Test Accuracy:", evaluate_model(model, X_test, Y_test))
sample_input = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
pred = predict(sample_input)
print_prediction(pred)