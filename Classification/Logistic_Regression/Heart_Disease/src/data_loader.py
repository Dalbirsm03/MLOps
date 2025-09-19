import pandas as pd
from sklearn.model_selection import train_test_split

def data_loader(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    Y = df['target']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1, stratify=Y, random_state=1)
    return X_train, X_test, Y_train, Y_test