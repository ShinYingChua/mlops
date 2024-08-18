import joblib
import numpy as np
from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path("iris_model")
    model = joblib.load(model_path)


def run(data):
    try:
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
