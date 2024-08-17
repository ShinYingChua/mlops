from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == "__main__":
    train()
