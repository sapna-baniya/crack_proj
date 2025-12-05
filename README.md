# Crack Detection MLOps Pipeline on AWS

## Project Overview
End-to-end MLOps pipeline for road crack detection using CNN, Vision Transformer (ViT), and Residual Vision Transformer (RvT) with explainability using LIME and Grad-CAM.

## Architecture Components

### 1. Data Ingestion & Catalog
- **S3 Buckets**: Raw and processed data storage
- **AWS Glue Data Catalog**: Metadata management
- **Google Drive Integration**: Download datasets

### 2. Data Curation
- **AWS Lambda**: Serverless data preprocessing
- **Amazon EMR**: Distributed data processing (if needed)
- **Amazon SageMaker Processing**: Data validation and transformation

### 3. Labeling & Versioning
- **SageMaker Ground Truth**: Data labeling (if needed)
- **AWS Elemental Delta**: Data versioning
- **AWS Glue**: ETL jobs

### 4. Feature/Artifact Store
- **Amazon S3**: Model artifacts and features
- **Amazon DynamoDB**: Metadata tracking
- **Amazon ECR**: Container registry

### 5. Model Training & Evaluation
- **Amazon SageMaker Training**: CNN, ViT, RvT models
- **Model Comparison**: Performance metrics
- **Explainability**: LIME and Grad-CAM

### 6. Model Registry & CI/CD
- **SageMaker Model Registry**: Version control
- **AWS CodePipeline**: Automated deployment

### 7. Inference
- **API Gateway**: REST API
- **AWS Lambda**: Inference handler
- **SageMaker Endpoint**: Model serving

### 8. Dashboard & Monitoring
- **AWS Amplify**: Frontend hosting
- **Amazon CloudFront**: CDN
- **CloudWatch**: Monitoring and logging

## Project Structure
```
crack-detection-pipeline/
├── data/
│   ├── raw/                    # Raw data from Google Drive
│   ├── processed/              # Processed data
│   └── scripts/
│       ├── download_data.py    # Download from Google Drive
│       ├── upload_to_s3.py     # Upload to S3
│       └── preprocess.py       # Data preprocessing
├── infrastructure/
│   ├── terraform/              # Infrastructure as Code
│   └── cloudformation/         # Alternative IaC
├── src/
│   ├── models/
│   │   ├── cnn_model.py       # CNN architecture
│   │   ├── vit_model.py       # Vision Transformer
│   │   └── rvt_model.py       # Residual Vision Transformer
│   ├── training/
│   │   ├── train.py           # Training script
│   │   └── evaluate.py        # Evaluation script
│   ├── explainability/
│   │   ├── lime_explain.py    # LIME implementation
│   │   └── gradcam.py         # Grad-CAM implementation
│   └── inference/
│       ├── predict.py         # Prediction script
│       └── lambda_handler.py  # Lambda inference handler
├── pipelines/
│   ├── sagemaker_pipeline.py  # SageMaker Pipeline definition
│   └── data_pipeline.py       # Data processing pipeline
├── deployment/
│   ├── deploy_model.py        # Model deployment script
│   └── api/
│       └── app.py             # API Gateway configuration
├── monitoring/
│   ├── model_monitor.py       # Model monitoring
│   └── data_quality.py        # Data quality checks
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_explainability.ipynb
├── tests/
│   ├── test_models.py
│   └── test_pipeline.py
├── config/
│   ├── config.yaml            # Configuration file
│   └── aws_config.yaml        # AWS resource configuration
├── requirements.txt
├── setup.py
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- AWS Account with student credits
- AWS CLI configured
- Google Drive credentials (for dataset download)

### Installation
```bash
# Clone the repository
git clone <your-repo>
cd crack-detection-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### AWS Configuration
```bash
# Configure AWS credentials
aws configure

# Set your region (e.g., us-east-1)
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
```

## Usage

### 1. Download Data from Google Drive
```bash
python data/scripts/download_data.py --drive-folder-id <your-folder-id>
```

### 2. Upload Data to S3
```bash
python data/scripts/upload_to_s3.py --bucket-name crack-detection-data
```

### 3. Run Data Preprocessing
```bash
python data/scripts/preprocess.py
```

### 4. Deploy Infrastructure
```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

### 5. Run Training Pipeline
```bash
python pipelines/sagemaker_pipeline.py
```

### 6. Deploy Model
```bash
python deployment/deploy_model.py --model-name best-model
```

## Model Performance Tracking
- Training metrics stored in S3
- Model artifacts versioned in SageMaker Model Registry
- Explainability outputs (LIME/Grad-CAM) saved with predictions

## Cost Optimization
- Use S3 Intelligent-Tiering
- Stop SageMaker notebooks when not in use
- Use spot instances for training
- Set CloudWatch alarms for budget monitoring

## Next Steps
1. ✅ Set up project structure
2. ⬜ Download and prepare dataset
3. ⬜ Deploy AWS infrastructure
4. ⬜ Implement data pipeline
5. ⬜ Build and train models
6. ⬜ Add explainability (LIME/Grad-CAM)
7. ⬜ Deploy inference endpoint
8. ⬜ Create monitoring dashboard

## Contact
For questions or issues, please open an issue in the repository.