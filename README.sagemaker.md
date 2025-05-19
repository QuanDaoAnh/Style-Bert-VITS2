# Requirements
1. Needed role for EC2
- if run server in EC2 - else use key for access to SageMaker
- AmazonSageMakerFullAccess
- IAMFullAccess
- Must to connect EC2 by root
2. Role for SageMaker
- AmazonSageMakerFullAccess
- AmazoneS3FullAccess
- Create role by code in file [create_sagemaker.py](style_bert_vits2/sagemaker/create_sagemaker.py)
3. ECR image
- use custom ECR:
```
461014077827.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-tritonserver-custom:latest
```
- image template
```
FROM <aws_account_id>.dkr.ecr.<region>.amazonaws.com/sagemaker-tritonserver:23.10-py3

RUN pip install style-bert-vits2 protobuf==4.25 accelerate
RUN pip install hf_xet
RUN pip install sentencepiece
```
- **aws_account_id** and **region** based on: [Here](https://docs.aws.amazon.com/en_jp/sagemaker/latest/dg/neo-deployment-hosting-services-container-images.html)
4. S3 bucket structure
```
mlce-tts-triton-models
├── model_repository
│   ├── model_1.tar.gz
```
5. Model structure example
- Link: [fastlabel_male_angry_jp](style_bert_vits2/sagemaker/model_repository/fastlabel_male_angry_jp)
```
fastlabel_male_angry_jp
├──pipeline
│   ├──1 # must have
│   ├──config.pbtxt
├──bert_jp
│   ├──1
│   │   ├──model.py
│   │   ├──triton_python_backend_utils.py
│   ├──config.pbtxt
├──tts_jp
│   ├──1
│   │   ├──model.py
│   │   ├──model.safetensors # weight of model
│   │   ├──triton_python_backend_utils.py
│   ├──config.json
│   ├──config.pbtxt
│   ├──style_vectors.npy # style vectors file of model
```

# Run code
1. Create sagemaker role, model, endpoint config, endpoint
```
python style_bert_vits2/sagemaker/create_sagemaker.py
```
2. Delete sagemaker endpoint
```
python style_bert_vits2/sagemaker/delete_sagemaker.py
```

# Performance
1. Deploying the BERT model and the Style BERT VITS 2 model separately:
- **First-time** TTS execution (including model loading from S3): 25 seconds
- **Subsequent** TTS executions: 0.7 seconds
- **Limitation**: Requires two separate requests to SageMaker (one for BERT and one for Style BERT VITS 2)
2. Combining the BERT and Style BERT VITS 2 models into a single model:
- **First-time** TTS execution (including model loading from S3): 15 seconds
- **Subsequent** TTS executions: 0.3–0.4 seconds
- **Limitation**: The combined model is large, consuming around 3GB of GPU VRAM per instance
3. Running the BERT model locally and deploying Style BERT VITS 2 on SageMaker:
- **First-time** TTS execution (including model loading from S3): 11 seconds
- **Subsequent** TTS executions: 0.5 seconds
- **Limitation**: The output from the BERT model (used as input to SageMaker) is larger than the original text, leading to slower request times to SageMaker. Additionally, running the BERT service locally also requires a GPU.