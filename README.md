# 🏠 Housing ML End-to-End

<p align="center">
  <strong>Production-oriented machine learning pipeline for housing price prediction using XGBoost</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-3.0.4-FF6600?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MLflow-MLOps-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-Cloud-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/putti-vignesh/Regression_ML_EndtoEnd?style=flat-square" />
  <img src="https://img.shields.io/github/languages/top/putti-vignesh/Regression_ML_EndtoEnd?style=flat-square" />
  <img src="https://img.shields.io/github/repo-size/putti-vignesh/Regression_ML_EndtoEnd?style=flat-square" />
</p>

---

## 🎯 Project Overview

**Housing Regression MLE** is an end-to-end machine learning pipeline for predicting housing prices using **XGBoost**.

The project follows ML engineering best practices with modular pipelines, experiment tracking via MLflow, containerization, AWS cloud deployment, and comprehensive testing.

### ✨ Highlights

| Area                        | Implementation                                                                |
| --------------------------- | ----------------------------------------------------------------------------- |
| 🤖 Machine Learning         | XGBoost Regression                                                            |
| 🔄 Pipeline                 | Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference |
| 📈 Experiment Tracking      | MLflow                                                                        |
| ⚡ Hyperparameter Tuning     | Optuna                                                                        |
| 🚀 API                      | FastAPI                                                                       |
| 📊 Dashboard                | Streamlit                                                                     |
| 🐳 Containers               | Docker                                                                        |
| ☁️ Cloud                    | AWS S3, ECR, ECS Fargate                                                      |
| 🧪 Testing                  | Pytest                                                                        |
| 🔁 CI/CD                    | GitHub Actions                                                                |
| 🛡️ Data Leakage Prevention | Time-based splits + training-only encoders                                    |

### 🔄 System Overview

```text
                         🏠 HOUSING ML SYSTEM

     ┌───────────┐
     │  Raw Data │
     └─────┬─────┘
           │
           ▼
   ┌───────────────┐
   │ Data Pipeline │
   │ Load           │
   │ Preprocess     │
   │ Feature Eng.   │
   └───────┬───────┘
           │
           ▼
   ┌────────────────┐
   │ Training       │
   │ XGBoost        │
   │ Optuna         │
   │ MLflow         │
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ Model          │
   │ Evaluation     │
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ Inference      │
   │ Single / Batch │
   └───────┬────────┘
           │
      ┌────┴─────┐
      ▼          ▼
 ┌─────────┐ ┌───────────┐
 │ FastAPI │ │ Streamlit │
 │  :8000  │ │   :8501   │
 └────┬────┘ └─────┬─────┘
      │            │
      └─────┬──────┘
            ▼
       ┌──────────┐
       │  Docker  │
       └────┬─────┘
            ▼
     ┌──────────────┐
     │ AWS          │
     │ S3 / ECR     │
     │ ECS Fargate  │
     └──────────────┘
```

---

## ⚙️ Environment Setup

Install dependencies using `uv`.

```bash
# Install dependencies using uv
uv sync
```

---

## 🧪 Testing

The project includes comprehensive testing for the feature, training, and inference pipelines.

### Run All Tests

```bash
# Run all tests
pytest
```

### Run Specific Test Modules

```bash
# Feature tests
pytest tests/test_features.py

# Training tests
pytest tests/test_training.py

# Inference tests
pytest tests/test_inference.py
```

### Run with Verbose Output

```bash
pytest -v
```

### 🧪 Testing Flow

```text
              TEST SUITE
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
     Features  Training  Inference
        │         │         │
        └─────────┼─────────┘
                  ▼
              Pytest
```

---

## 🔄 Data Pipeline

The data pipeline follows:

```text
Raw Data
   │
   ▼
┌──────────────┐
│ Load & Split │
└──────┬───────┘
       ▼
┌──────────────┐
│ Preprocessing│
└──────┬───────┘
       ▼
┌────────────────────┐
│ Feature Engineering│
└──────────┬─────────┘
           ▼
   Training Dataset
```

### 1. Load and Split Raw Data

```bash
# 1. Load and split raw data
python src/feature_pipeline/load.py
```

The project uses a time-aware split:

```text
Training  → before 2020
Evaluation → 2020–2021
Holdout    → 2022+
```

### 2. Preprocess Splits

```bash
# 2. Preprocess splits
python -m src.feature_pipeline.preprocess
```

Preprocessing includes:

* City normalization
* Deduplication
* Outlier removal
* Data cleaning

### 3. Feature Engineering

```bash
# 3. Feature engineering
python -m src.feature_pipeline.feature_engineering
```

Features include:

* Date features
* Frequency encoding for ZIP codes
* Target encoding for `city_full`

---

## 🧠 Training Pipeline

The training pipeline consists of:

```text
Feature-Engineered Data
          │
          ▼
   ┌─────────────┐
   │ XGBoost     │
   │ Training    │
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │ Optuna      │
   │ Tuning      │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │   MLflow    │
   │ Experiments │
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │ Evaluation  │
   └─────────────┘
```

### Train Baseline Model

```bash
# Train baseline model
python src/training_pipeline/train.py
```

### Hyperparameter Tuning

```bash
# Hyperparameter tuning with MLflow
python src/training_pipeline/tune.py
```

### Model Evaluation

```bash
# Model evaluation
python src/training_pipeline/eval.py
```

---

## 🔮 Inference

The inference pipeline uses the trained model and saved encoders.

### Single Inference

```bash
# Single inference
python src/inference_pipeline/inference.py --input data/raw/holdout.csv --output predictions.csv
```

### Batch Monthly Predictions

```bash
# Batch monthly predictions
python src/batch/run_monthly.py
```

### Inference Flow

```text
                  Trained Model
                       │
                       ▼
              ┌─────────────────┐
              │ Inference       │
              │ Pipeline        │
              └────────┬────────┘
                       │
                ┌──────┴──────┐
                ▼             ▼
          Single Input    Holdout Data
                │             │
                ▼             ▼
           Prediction    Batch Prediction
```

---

## 🚀 API Service

The FastAPI service provides production-style REST API access to the trained model.

### Start FastAPI Server Locally

```bash
# Start FastAPI server locally
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### API

```text
http://localhost:8000
```

### API Capabilities

* Health checks
* Prediction endpoint
* S3 model integration
* Batch processing
* Production inference

---

## 📊 Streamlit Dashboard

The Streamlit application provides an interactive interface for housing price predictions.

### Start Streamlit Dashboard

```bash
# Start Streamlit dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Dashboard

```text
http://localhost:8501
```

### Features

* Real-time predictions through FastAPI
* Filtering by year
* Filtering by month
* Filtering by region
* Prediction vs actual visualization
* MAE
* RMSE
* Percentage Error
* Yearly trend analysis
* Highlighted selected periods

---

## 🐳 Docker

Both the FastAPI backend and Streamlit dashboard can be containerized.

### Build API Container

```bash
# Build API container
docker build -t housing-regression .
```

### Build Streamlit Container

```bash
# Build Streamlit container
docker build -t housing-streamlit -f Dockerfile.streamlit .
```

### Run API Container

```bash
# Run API container
docker run -p 8000:8000 housing-regression
```

### Run Streamlit Container

```bash
# Run Streamlit container
docker run -p 8501:8501 housing-streamlit
```

### Container Architecture

```text
                 Docker
                   │
          ┌────────┴────────┐
          ▼                 ▼
   ┌─────────────┐   ┌─────────────┐
   │ FastAPI     │   │ Streamlit   │
   │ Container   │   │ Container   │
   │    :8000    │   │    :8501    │
   └─────────────┘   └─────────────┘
```

---

## 📈 MLflow Tracking

MLflow is used for experiment tracking and model development.

It tracks:

* Parameters
* Metrics
* Model experiments
* Hyperparameter tuning
* Artifacts

### PowerShell

```powershell
# Start MLflow using the provided PowerShell script
.\scripts\start_mlflow.ps1
```

### Command Prompt

```cmd
# Start MLflow using the provided batch script
.\scripts\start_mlflow.bat 5001
```

### Direct MLflow UI

```bash
# Start MLflow directly using SQLite backend
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5001 --workers 1
```

### MLflow UI

```text
http://localhost:5001
```

---

## 🔧 Key Design Patterns

### 1. Pipeline Modularity

Each pipeline component can be run independently with consistent interfaces.

All modules accept configurable input/output paths for testing isolation.

---

### 2. Cloud-Native Architecture

The project uses AWS services for data storage and containerized deployment.

```text
                 AWS CLOUD
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       S3          ECR         ECS
    Data/Model   Docker      Fargate
                  Images      Services
```

#### S3-First Storage

Models and data automatically sync from S3 buckets.

#### Containerized Services

Both API and dashboard run inside Docker containers.

#### Auto-Scaling Infrastructure

ECS Fargate provides serverless container execution and scaling.

#### Environment-Based Configuration

Separate configurations are supported for local development and production.

---

### 3. Encoder Persistence

Frequency and target encoders are saved as pickle files during training and loaded during inference.

This ensures consistent transformations between training and production inference.

```text
Training
   │
   ▼
Fit Encoder
   │
   ▼
Save Encoder
   │
   ▼
Production Inference
   │
   ▼
Load Same Encoder
```

---

### 4. Configuration Management

Model parameters, file paths, and pipeline settings use sensible defaults but can be overridden through:

* Function parameters
* Configuration files
* Environment variables
* AWS environment variables

---

### 5. Testing Strategy

The project uses:

* Unit tests for individual pipeline components
* Integration tests for end-to-end pipeline flows
* Smoke tests for inference pipeline
* Temporary directories for test isolation

---

## 📦 Dependencies

Key production dependencies from `pyproject.toml`:

### 🤖 ML / Data

* `xgboost==3.0.4`
* `scikit-learn`
* `pandas==2.1.1`
* `numpy==1.26.4`

### 🚀 API

* `fastapi`
* `uvicorn`

### 📊 Dashboard

* `streamlit`
* `plotly`

### ☁️ Cloud

* `boto3`

### 📈 Experimentation

* `mlflow`
* `optuna`

### 🧪 Quality

* `great-expectations`
* `evidently`

---

## 📁 File Structure Notes

```text
Regression_ML_EndtoEnd/
│
├── src/
│   ├── feature_pipeline/
│   │   ├── load.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   │
│   ├── training_pipeline/
│   │   ├── train.py
│   │   ├── tune.py
│   │   └── eval.py
│   │
│   ├── inference_pipeline/
│   │   └── inference.py
│   │
│   ├── batch/
│   │   └── run_monthly.py
│   │
│   └── api/
│       └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
│
├── models/
│
├── mlruns/
│
├── configs/
│
├── notebooks/
│
├── tests/
│
├── scripts/
│
├── app.py
├── Dockerfile
├── Dockerfile.streamlit
├── pyproject.toml
│
├── housing-api-task-def.json
├── streamlit-task-def.json
│
└── .github/
    └── workflows/
        └── ci.yml
```

### 📂 Important Directories

| Directory            | Purpose                             |
| -------------------- | ----------------------------------- |
| `data/`              | Raw, processed, and prediction data |
| `models/`            | Trained models and encoders         |
| `mlruns/`            | MLflow experiment tracking          |
| `configs/`           | YAML configuration files            |
| `notebooks/`         | EDA and experimentation             |
| `tests/`             | Automated tests                     |
| `scripts/`           | Utility and MLflow startup scripts  |
| `.github/workflows/` | CI/CD workflows                     |

### ☁️ AWS Task Definitions

```text
housing-api-task-def.json
streamlit-task-def.json
```

These define the ECS services used for cloud deployment.

### 🔁 CI/CD

```text
.github/
└── workflows/
    └── ci.yml
```

GitHub Actions is used for automated CI/CD workflows.

---

## 🛡️ Data Leakage Prevention

The project implements strict data leakage prevention throughout the pipeline.

### Safeguards

```text
                 RAW DATA
                    │
                    ▼
            Time-Based Split
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    Training     Evaluation    Holdout
       │
       ▼
 Fit Encoders
       │
       ▼
 Train Model
       │
       └──────────────┐
                      ▼
               Saved Encoders
                      │
              ┌───────┴───────┐
              ▼               ▼
         Evaluation       Inference
```

The system prevents leakage through:

* Time-based splits instead of random splitting
* Encoders fitted only on training data
* Leakage-prone columns dropped before training
* Schema alignment enforced between train/evaluation/inference
* Persistent encoders reused during inference

---

## ☁️ Cloud Deployment Architecture

```text
                         AWS CLOUD
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          ┌──────┐       ┌──────┐     ┌──────┐
          │  S3  │       │ ECR  │     │ ECS  │
          │Data/ │       │Docker│     │Fargate│
          │Model │       │Images│     │       │
          └──────┘       └───┬──┘     └───┬───┘
                              │            │
                              │       ┌────┴─────┐
                              │       ▼          ▼
                              │   FastAPI    Streamlit
                              │    :8000       :8501
                              │       │          │
                              └───────┴──────────┘
                                      │
                                      ▼
                              Application Load
                                  Balancer
```

---

## 🔄 Complete Project Flow

```text
┌───────────────────────┐
│    Environment Setup  │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│       Testing         │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│    Data Pipeline      │
│ Load → Preprocess →   │
│ Feature Engineering   │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│   Training Pipeline   │
│ Train → Tune → Eval   │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│      Inference        │
│ Single → Batch        │
└───────────┬───────────┘
            ▼
      ┌─────┴─────┐
      ▼           ▼
┌──────────┐ ┌───────────┐
│ FastAPI  │ │ Streamlit │
└────┬─────┘ └─────┬─────┘
     └──────┬──────┘
            ▼
      ┌───────────┐
      │  Docker   │
      └─────┬─────┘
            ▼
     ┌──────────────┐
     │ AWS S3/ECR/  │
     │ ECS Fargate  │
     └──────────────┘
```

---

<p align="center">
  <strong>From data → model → inference → API → container → cloud.</strong>
</p>

<p align="center">
  Built with Python • XGBoost • MLflow • FastAPI • Docker • AWS
</p>
