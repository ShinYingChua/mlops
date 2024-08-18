import joblib
import numpy as np
from azureml.core.model import Model
import json


def init():
    global model
    model_path = Model.get_model_path("iris_model")
    model = joblib.load(model_path)


def run(data):
    try:
        # Ensure that the raw_data is parsed correctly as a dictionary
        if isinstance(data, str):
            data = json.loads(data)

        # Convert the incoming data to a NumPy array
        data = np.array(data['data'])

        # Check if the array is 1D and reshape it to 2D (1 sample with n features)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Predict using the model
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
