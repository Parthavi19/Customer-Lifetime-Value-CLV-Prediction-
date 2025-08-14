# 🛒 Customer Lifetime Value (CLV) Prediction & Segmentation

## 📌 Overview
This project predicts **Customer Lifetime Value (CLV)** over a 12-month horizon and segments customers using **K-Means clustering**.  
It includes:
- XGBoost regression model
- K-Means segmentation
- SHAP explainability
- REST API (FastAPI)
- Interactive dashboard (Streamlit)
- Deployment on Google Cloud Run

---

## 📂 Project Structure
```plaintext
clv_project/
│
├── data/                  # Place your dataset CSV here
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Config & constants
│   ├── data.py            # Data loading & cleaning
│   ├── features.py        # Feature engineering
│   ├── segment.py         # Customer segmentation
│   ├── model.py           # XGBoost training
│   ├── explain.py         # SHAP explainability
│   ├── api.py             # FastAPI endpoints
│   └── train_pipeline.py  # Main training pipeline
│
├── ui/                    # Streamlit dashboard
│   └── app_streamlit.py
│
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker build configuration
├── cloudbuild.yaml        # GCP Cloud Build config
└── README.md              # Project documentation
