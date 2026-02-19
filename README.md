# 🔒 Credit Card Fraud Detection — ML Pipeline & API

<p align="center">
  <b>Real-time fraud detection</b> powered by <b>XGBoost</b> + <b>FastAPI</b><br>
  Data streamed directly from <b>Kaggle</b> — no local storage required
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Docker Deployment](#-docker-deployment)
- [Configuration](#-configuration)
- [Model Performance](#-model-performance)
- [Technologies](#-technologies)

---

## 🎯 Overview

This project is a professional, production-ready **machine learning pipeline** for detecting credit card fraud. It includes:

| Feature | Description |
|---------|-------------|
| **🗃️ Data Streaming** | Downloads Kaggle data on-the-fly without permanent storage |
| **🔧 Feature Engineering** | Age, distance, temporal features, amount transformations |
| **🤖 XGBoost Model** | Tuned for imbalanced classes with optimal threshold search |
| **⚡ FastAPI** | REST API for real-time single & batch predictions |
| **🐳 Docker** | Production-ready containers with docker-compose |
| **🧪 Tests** | Pytest-based API test suite |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Kaggle     │────▶│  Feature          │────▶│   XGBoost    │
│   Dataset    │     │  Engineering      │     │   Training   │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                                     ▼
                                              ┌──────────────┐
┌─────────────┐     ┌──────────────────┐     │  Artifacts   │
│   Client     │────▶│   FastAPI         │────▶│  .joblib     │
│   Request    │     │   /predict        │◀────│  (model,     │
└─────────────┘     └──────────────────┘     │   scaler,    │
                                              │   encoders)  │
                                              └──────────────┘
```

---

## 📁 Project Structure

```
Detection-de-fraud/
├── src/                      # Source package
│   ├── __init__.py           # Package init
│   ├── config.py             # Centralized configuration
│   ├── data.py               # Kaggle data download (no local storage)
│   ├── features.py           # Feature engineering pipeline
│   ├── model.py              # Training, evaluation, serialization
│   ├── schemas.py            # Pydantic request/response models
│   └── api.py                # FastAPI application
├── tests/
│   └── test_api.py           # API test suite
├── artifacts/                # Model artifacts (auto-generated)
├── logs/                     # Training logs (auto-generated)
├── train.py                  # 🚀 Training entry point
├── run_api.py                # 🌐 API server entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container image
├── docker-compose.yml        # Multi-service deployment
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---




## 📊 Model Performance

The model is evaluated with metrics optimized for **imbalanced classification**:

| Metric | Focus |
|--------|-------|
| **PR AUC** | Primary metric — best for rare classes |
| **ROC AUC** | Overall discrimination |
| **F1 Score** | Balance of precision and recall |
| **Optimal threshold** | Automatically found to maximize F1 |

---

## 🛠️ Technologies

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| ML | XGBoost, scikit-learn |
| API | FastAPI, Uvicorn, Pydantic |
| Data | Pandas, NumPy, GeoPy |
| Deployment | Docker, docker-compose |
| Testing | Pytest, HTTPX |
| Logging | Loguru |

---



<p align="center">
  Made by <b>Bertrand</b>
</p>

