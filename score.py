import joblib
import numpy as np
from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path("iris_model")
    model = joblib.load(model_path)


def run(data):
    try:
        data = np.array(data)
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
