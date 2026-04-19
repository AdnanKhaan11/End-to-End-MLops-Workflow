Got it — I’ll keep **DVC explained conceptually (no commands)** and focus on your **actual working pipeline (main.py + MLflow + DagsHub)**. Clean, professional, and honest 👍

---

# 🚀 End-to-End MLOps Workflow (MLflow + DagsHub)

A complete **end-to-end machine learning pipeline** demonstrating modern MLOps practices such as experiment tracking, modular pipeline design, and reproducible workflows.

This project uses:

* **MLflow (via DagsHub)** for experiment tracking
* **Scikit-learn** for model training
* **Custom pipeline architecture** for clean stage-wise execution

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square\&logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=flat-square)
![DagsHub](https://img.shields.io/badge/DagsHub-Remote%20Tracking-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Framework-F7931E?style=flat-square)

---

## 📋 Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Installation & Setup](#installation--setup)
* [Running the Pipeline](#running-the-pipeline)
* [MLflow Tracking (DagsHub)](#mlflow-tracking-dagshub)
* [Pipeline Stages](#pipeline-stages)
* [Use Cases](#use-cases)

---

## 🎯 Overview

This project implements a **complete ML pipeline from data ingestion to model evaluation** using a clean and modular architecture.

### Pipeline Flow:

1. **Data Ingestion** → Load dataset
2. **Data Validation** → Validate structure and schema
3. **Data Transformation** → Prepare and split data
4. **Model Training** → Train ElasticNet model
5. **Model Evaluation** → Evaluate and log results

---

### Why this project matters:

✅ Clean MLOps architecture
✅ Experiment tracking using MLflow
✅ Remote logging using DagsHub
✅ Easy to extend and production-ready structure

---

## ✨ Key Features

### 🔹 1. Experiment Tracking (MLflow + DagsHub)

* Logs **hyperparameters**

  * `alpha`
  * `l1_ratio`

* Logs **metrics**

  * RMSE
  * MAE
  * R² score

* Stores:

  * trained model
  * evaluation results

* All runs are visible in **DagsHub MLflow UI**

---

### 🔹 2. Modular Pipeline Architecture

Each stage is separated into:

* **components** → core logic
* **pipeline files** → execution flow

This makes the project:

* easy to debug
* easy to scale
* easy to maintain

---

### 🔹 3. Artifact Management

All outputs are stored in the `artifacts/` folder:

* Raw data
* Processed data
* Trained model
* Evaluation results

This keeps everything structured and reproducible.

---

### 🔹 4. Data Versioning Concept (DVC)

This project is designed to support **data versioning using DVC concepts**, including:

* Tracking datasets across pipeline stages
* Managing intermediate outputs (like train/test split)
* Maintaining reproducibility of data pipelines

Even without running full DVC pipelines, the structure follows **best practices for data version control**, making it easy to integrate later.

---

## 🛠️ Technology Stack

| Tool         | Purpose                |
| ------------ | ---------------------- |
| Python       | Core programming       |
| Scikit-learn | Machine learning model |
| MLflow       | Experiment tracking    |
| DagsHub      | Remote MLflow server   |
| Pandas       | Data processing        |

---

## 📁 Project Structure

```
End-to-End-MLops-Workflow/
End-to-End-MLops-Workflow/
│
├── 📂 templates/                    # Flask HTML templates
│   ├── index.html                   # Home page (prediction form)
│   ├── train.html                   # Training interface
│   ├── results.html                 # Prediction results
│   └── error.html                   # Error pages
│
├── 📂 static/                       # CSS & static assets
│   ├── style.css                    # Home page styling
│   ├── train.css                    # Training page styling
│   ├── results.css                  # Results page styling
│   └── error.css                    # Error page styling
│
├── 📂 artifacts/                    # Data & model versioning
│   ├── data_ingestion/              # Raw data storage
│   ├── data_validation/             # Validation reports
│   ├── data_transformation/         # Processed data
│   │   ├── train.csv
│   │   └── test.csv
│   ├── model_trainer/               # Trained models
│   └── model_evaluation/            # Evaluation reports
│
├── 📂 src/
│   └── 📂 datascience/
│       ├── 📂 components/           # Core ML logic
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       │
│       ├── 📂 pipeline/             # Pipeline orchestration
│       │   ├── data_ingestion_pipeline.py
│       │   ├── data_validation_pipeline.py
│       │   ├── data_transformation_pipeline.py
│       │   ├── model_trainer_pipeline.py
│       │   └── model_evaluation_pipeline.py
│       │
│       ├── config.py                # Configuration management
│       ├── entity.py                # Data classes
│       ├── logging.py               # Logging setup
│       └── __init__.py
│
├── 📂 config/                       # Configuration files
│
├── 📂 logs/                         # Application logs
│
├── app.py                           # 🔴 Flask application (MAIN WEB SERVER)
├── main.py                          # ML pipeline orchestration
├── params.yaml                      # Pipeline parameters
├── schema.yaml                      # Data schema
├── requirements.txt                 # Dependencies
├── .env                             # Environment variables
├── .gitignore
└── README.md
---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash

git clone https://github.com/AdnanKhaan11/End-to-End-MLops-Workflow.git

cd End-to-End-MLops-Workflow
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup MLflow (DagsHub)

Create `.env` file in root:

```
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=<your-token>
```

---

## ▶️ Running the Pipeline

Run the complete pipeline using:

```bash
python app.py
```

This will execute:

* Data ingestion
* Data validation
* Data transformation
* Model training
* Model evaluation

---

## 📊 MLflow Tracking (DagsHub)

After running the pipeline, MLflow logs will be available on DagsHub.

You can:

* View all runs
* Compare model performance
* Check parameters and metrics
* Download trained models

---

## 🔄 Pipeline Stages

### 📥 Data Ingestion

* Loads dataset
* Stores in artifacts folder

---

### ✅ Data Validation

* Validates dataset structure
* Ensures consistency with schema

---

### 🔄 Data Transformation

* Splits data into training and testing
* Saves `train.csv` and `test.csv`

---

### 🤖 Model Training

* Trains **ElasticNet model**
* Logs:

  * hyperparameters
  * metrics
  * model artifact

---

### 📊 Model Evaluation

* Evaluates model on test data
* Stores results in JSON
* Logs evaluation results to MLflow

---

## 💡 Use Cases

### 👨‍🎓 Students

* Learn real-world MLOps pipeline
* Understand MLflow integration

### 👨‍💻 ML Engineers

* Use as base template for projects
* Extend for production systems

### 🧪 Experimentation

* Compare different hyperparameters
* Track multiple model runs

---

## 🧠 Key Learnings

* How to structure an end-to-end ML pipeline
* How to integrate MLflow with DagsHub
* How to log experiments properly
* How to manage artifacts cleanly

---

## 📞 Contact

**Adnan Khan**
Software Engineer (AI/ML)

* Email: [adnankhaan2244@gmail.com](mailto:adnankhaan2244@gmail.com)
* LinkedIn: *https://www.linkedin.com/in/adnankhaan11/*

---



