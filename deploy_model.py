from azureml.core import Workspace, Environment, Model, ComputeTarget
from azureml.core.compute import AksCompute
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import InferenceConfig
import os

# Fetch Service Principal credentials from environment variables
subscription_id = os.getenv("AZUREML_SUBSCRIPTION_ID")
resource_group = os.getenv("AZUREML_RESOURCE_GROUP")
workspace_name = os.getenv("AZUREML_WORKSPACE_NAME")
tenant_id = os.getenv("AZUREML_TENANT_ID")
client_id = os.getenv("AZUREML_CLIENT_ID")
client_secret = os.getenv("AZUREML_CLIENT_SECRET")

# Set up Service Principal Authentication
sp_auth = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=client_id,
    service_principal_password=client_secret
)


# Connect to Azure ML Workspace
ws = Workspace(subscription_id=subscription_id, resource_group=resource_group,
               workspace_name=workspace_name, auth=sp_auth)

# Attach the existing AKS cluster if not already attached
aks_name = "my-aks-cluster2"
if aks_name not in ws.compute_targets:
    attach_config = AksCompute.attach_configuration(
        resource_group=resource_group, cluster_name=aks_name)
    aks_target = ComputeTarget.attach(ws, aks_name, attach_config)
    aks_target.wait_for_completion(show_output=True)
else:
    aks_target = ws.compute_targets[aks_name]

# Create or retrieve the environment
try:
    env = ws.environments['my-azureml-env']
except KeyError:
    # If the environment doesn't exist, create it from the environment.yml file
    env = Environment.from_conda_specification(
        name='my-azureml-env', file_path='environment.yml')
    env.register(workspace=ws)  # Register the environment in the workspace

# Load the registered model
model = Model(ws, name="iris-logistic-regression")

# Define the environment (use the same environment used during training)
env = ws.environments['my-azureml-env']

# Define the inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)

# Define the deployment configuration
deployment_config = AksWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1)

# Deploy the model
service = Model.deploy(
    ws, "iris-service", [model], inference_config, deployment_config, aks_target)
service.wait_for_deployment(show_output=True)
