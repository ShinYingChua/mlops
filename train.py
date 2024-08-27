from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
import os
from azureml.core import Workspace, Run, Model


def train():
    # Load the dataset
    X, y = load_iris(return_X_y=True)

    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Save the model to the outputs directory
    os.makedirs('outputs', exist_ok=True)
    model_path = 'outputs/model.joblib'
    joblib.dump(model, model_path)

    # Register the model in Azure ML
    ws = Workspace.from_config()  # Load the workspace from the config file
    run = Run.get_context()  # Get the current run

    # Register the model with Azure ML
    Model.register(workspace=ws,
                   model_path=model_path,  # This is the local path where the model is saved
                   model_name='iris-logistic-regression',  # Name of the model in Azure ML
                   tags={'area': 'iris-classification',
                         'type': 'logistic-regression'},
                   description='A logistic regression model trained on the Iris dataset.')


if __name__ == "__main__":
    train()
