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

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Kaggle account** with API credentials

### 2. Setup Kaggle credentials

```bash
# Option A: Environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Option B: .env file
cp .env.example .env
# Edit .env with your credentials
```

> 💡 Get your Kaggle API key at: https://www.kaggle.com/settings → API → Create New Token

### 3. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install packages
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

This will:
1. 📥 Download data from Kaggle (temporary, cleaned up automatically)
2. 🔧 Engineer features (age, distance, temporal, amount)
3. 🤖 Train an XGBoost classifier with imbalance handling
4. 📊 Find optimal decision threshold (max F1)
5. 💾 Save artifacts to `./artifacts/`

### 5. Launch the API

```bash
python run_api.py
```

The API will be available at:
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc**      : http://localhost:8000/redoc
- **Health**     : http://localhost:8000/health

---

## 📡 API Reference

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "trained_at": "2025-02-19T...",
  "feature_count": 14,
  "metrics": {
    "roc_auc": 0.998,
    "f1_score": 0.82
  }
}
```

### `POST /predict` — Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-06-15 14:32:00",
    "cc_num": 4263982640269299,
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "grocery_pos",
    "amt": 1250.00,
    "first": "Jennifer",
    "last": "Banks",
    "gender": "F",
    "street": "561 Perry Cove",
    "city": "Jesup",
    "state": "GA",
    "zip": 31599,
    "lat": 31.5988,
    "long": -81.8826,
    "city_pop": 3495,
    "job": "Psychologist",
    "dob": "1988-03-09",
    "trans_num": "0b242abb623afc578575680df30655b9",
    "unix_time": 1371816865,
    "merch_lat": 36.011293,
    "merch_long": -82.048315
  }'
```

**Response:**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.87432,
  "threshold_used": 0.42,
  "risk_level": "CRITICAL"
}
```

### `POST /predict/batch` — Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'
```

**Response:**
```json
{
  "predictions": [...],
  "total": 50,
  "fraud_count": 3,
  "fraud_rate": 0.06
}
```

---

## 🐳 Docker Deployment

### Build & Train

```bash
# Build the image
docker-compose build

# Train the model (one-time)
docker-compose --profile train run train

# Start the API
docker-compose up api -d
```

### Check logs

```bash
docker-compose logs -f api
```

---

## ⚙️ Configuration

All configuration is managed through environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `KAGGLE_USERNAME` | — | Kaggle API username |
| `KAGGLE_KEY` | — | Kaggle API key |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |
| `MODEL_THRESHOLD` | `0.5` | Fallback decision threshold |

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

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📜 License

MIT License — Free for personal and commercial use.

---

<p align="center">
  Made with ❤️ by <b>Bertrand</b>
</p>

