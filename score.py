import joblib
import numpy as np
from azureml.core.model import Model
import json

# Define a mapping from class labels to Iris flower names
CLASS_LABELS = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}


def init():
    global model
    model_path = Model.get_model_path("iris-logistic-regression")
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
        class_labels = model.predict(data)

        # Map the numeric class labels to their corresponding flower names
        result = [CLASS_LABELS[label] for label in class_labels]

        return result
    except Exception as e:
        error = str(e)
        return error
