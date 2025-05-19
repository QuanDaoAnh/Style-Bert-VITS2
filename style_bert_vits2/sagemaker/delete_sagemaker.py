
import boto3
sagemaker_client = boto3.client('sagemaker', region_name='ap-northeast-1')

endpoint_name = 'mlce-tts-triton-endpoint'
endpoint_config_name = 'mlce-tts-triton-endpoint-config'
model_name = 'mlce-tts-triton-models'

sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
sagemaker_client.delete_model(ModelName=model_name)