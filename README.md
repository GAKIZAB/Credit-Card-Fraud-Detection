# Credit Card Fraud Detection – End-to-End ML System

This project implements a **complete real-time credit card fraud detection system**, covering the full pipeline from raw data processing to **production deployment of a Machine Learning model**.

Built on **XGBoost** and **FastAPI**, the architecture transforms financial transactions into **actionable predictions**, enriched with an **interpretable risk score**, suitable for industrial-scale use cases.


## 🎯 Project Objective

The goal is to identify fraudulent transactions within a high-volume stream of banking operations while addressing two major constraints:

* **Extreme class imbalance** (fraud cases are rare)
* **Low-latency requirements** for real-time production usage

The system is designed to be:

* **Robust**
* **Scalable**
* **Production-ready**


## 🧠 Architecture & Methodology

The project is built around **four core technological pillars**:

1. **Advanced Feature Engineering**
   Transforming raw data into meaningful explanatory variables (temporal, geographical, behavioral).

2. **Robust Predictive Modeling**
   Training a classifier optimized for highly imbalanced datasets.

3. **Real-Time Prediction API**
   Exposing the model through a performant and secure REST API.

4. **Industrialization & MLOps**
   Docker-based containerization, automated testing, and a clear separation between training and inference.


## 🚀 Implementation Steps

### 1️⃣ Feature Engineering (`features.py`)

Advanced handling of transactional data.

#### ⏱️ Temporal Analysis

* Extraction of hour, day, and day-of-week features
* Customer age computation
* Capture of spending behavior patterns

#### 🌍 Geospatial Analysis

* Computation of **geodesic distance** between customer and merchant locations (via `geopy`)
* Detection of abnormal geographic behavior

#### 🔢 Numerical Transformations

* Log transformation of transaction amounts (`amt`) to reduce the impact of outliers
* Binning of selected continuous variables

#### 🏷️ Categorical Encoding

* Use of `LabelEncoder` with robust handling of **unseen categories** during inference


### 2️⃣ Model Training (`train.py`, `model.py`)

A structured and reproducible training pipeline:

* **Class imbalance handling**
  Use of `scale_pos_weight` in XGBoost to emphasize the minority (fraud) class

* **Feature normalization**
  `MinMaxScaler` to ensure consistent feature scaling

* **Decision threshold optimization**
  Selection of the threshold maximizing the **F1-Score**, rather than relying on the default 0.5 cutoff

* **Artifact persistence**
  Saving the trained model, scaler, encoders, and metadata using `.joblib`


### 3️⃣ Prediction API (`api.py`)

A production-ready REST API built with **FastAPI**.

#### 🔍 Data Validation

* Strict input schemas defined with **Pydantic** (`schemas.py`)
* Protection against invalid or malformed inputs

#### Real-Time Inference

* `POST /predict`
  → Single-transaction prediction with probability score

#### Batch Prediction

* `POST /predict/batch`
  → Processing of up to **1,000 transactions simultaneously**

#### Interpretable Risk Levels

* Mapping predicted probabilities to business-friendly categories:

  * `LOW`
  * `MEDIUM`
  * `HIGH`
  * `CRITICAL`

#### Monitoring

* `GET /health`
  → API and model health check


### 4️⃣ Industrialization & Testing

* **Docker & Docker Compose**

  * `train` service: model training
  * `api` service: production deployment

* **Automated Testing**

  * Non-regression tests implemented with `pytest` (`test_api.py`)


## 📊 Model Evaluation

Performance is evaluated using metrics suited for imbalanced classification problems:

* **ROC-AUC**
* **PR-AUC**
* **F1-Score** *(primary metric)*
* **Confusion Matrix**
  → Detailed analysis of false positives and false negatives


## 📝 Conclusion

This project demonstrates a **full Machine Learning Engineering workflow**, going far beyond a simple modeling notebook:

* Advanced feature engineering (temporal & geospatial)
* Realistic handling of imbalanced data
* Fine-grained decision threshold optimization
* Scalable, production-ready API
* Modular and maintainable architecture

The solution addresses **real-world credit card fraud detection requirements** while providing a solid foundation for future extensions such as advanced monitoring, automated retraining, or real-time streaming integration.


