from azureml.core.model import Model
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import InferenceConfig

model = Model(ws, name="iris_model")

inference_config = InferenceConfig(entry_script="score.py", environment=env)

aks_target = AksWebservice(ws, "my-aks-cluster")

deployment_config = AksWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1)

service = Model.deploy(
    ws, "iris-service", [model], inference_config, deployment_config, aks_target)
service.wait_for_deployment(show_output=True)
