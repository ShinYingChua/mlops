from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import InferenceConfig
import os

# Connect to the Azure ML workspace
ws = Workspace.from_config()

# Load the registered model
model = Model(ws, name="iris_model")

# Define the environment (use the same environment used during training)
env = ws.environments['my-azureml-env']

# Define the inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)

# Define the AKS deployment target
aks_target = AksWebservice(ws, "my-aks-cluster")

# Define the deployment configuration
deployment_config = AksWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1)

# Deploy the model
service = Model.deploy(
    ws, "iris-service", [model], inference_config, deployment_config, aks_target)
service.wait_for_deployment(show_output=True)
