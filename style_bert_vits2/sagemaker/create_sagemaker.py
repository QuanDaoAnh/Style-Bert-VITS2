import boto3
from typing import Any
from datetime import datetime
import json

def create_role(): # type: ignore
    iam_client = boto3.client('iam')

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%dT%H%M%S")
    role_name = f'AmazonSageMaker-ExecutionRole-{formatted_time}'

    role = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Role for SageMaker to access S3 and other resources"
    )

    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
    )
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
    )

    return role

def create_model(
        model_name: str,
        image: str,
        model_url: str,
        mode: str,
        env: dict[str,str],
        role_arn: str,
    ):
    sagemaker_client = boto3.client('sagemaker', region_name='ap-northeast-1')
    container = {
        'Image': image,
        'ModelDataUrl': model_url,
        'Mode': mode,
        'Environment': env
    }
    sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer=container
    )

def create_endpoint(
    endpoint_name: str,
    endpoint_config_name: str,
    product_variant: list[dict[str, Any]],
):
    sagemaker_client = boto3.client('sagemaker', region_name='ap-northeast-1')

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=product_variant
    )
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

if __name__ == "main":
    model_name = 'mlce-tts-triton-models'
    image = '461014077827.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-tritonserver-custom:latest'
    model_url = 's3://mlce-tts-triton-models/model_repository/'
    mode = 'MultiModel'
    env = {
        'SAGEMAKER_TRITON_DEFAULT_MODEL_NAME': 'pipeline',
    }

    endpoint_name = 'mlce-tts-triton-endpoint'
    endpoint_config_name = 'mlce-tts-triton-endpoint-config'
    product_variant = [{
        'VariantName': 'default-variant-name',
        'ModelName': model_name,
        'InstanceType': 'ml.g4dn.xlarge',
        'InitialInstanceCount': 1,
        'InitialVariantWeight': 1.0
    }]

    role = create_role()

    create_model(
        model_name,
        image,
        model_url,
        mode,
        env,
        role["Role"]["Arn"]
    )
    create_endpoint(
        endpoint_name,
        endpoint_config_name,
        product_variant
    )