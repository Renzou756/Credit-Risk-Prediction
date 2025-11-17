# Credit Risk AI – Production-Grade ML System  
![Python](https://img.shields.io/badge/python-3.11-blue)  
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)  
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.842-orange)  
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-red)  
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-yellow)

**A bank-grade credit risk model** that predicts loan default probability using **XGBoost**, **Optuna**, and **SHAP**, deployed via **FastAPI** and **Streamlit**. Built with 150K+ LendingClub records to demonstrate production-ready ML pipelines for financial risk assessment.

---

## Features

| Feature | Status |
|---------|--------|
| **XGBoost + Hyperparameter Tuning** | **AUC 0.842** with 79% recall on high-risk class |
| **SHAP Explainability** | Per-loan waterfall plots & global feature importance |
| **FastAPI Inference** | Real-time `/predict` endpoint with JSON responses |
| **Streamlit Dashboard** | Interactive UI + live SHAP visualizations |
| **Advanced Feature Engineering** | DTI ratios, credit utilization trends, payment-to-income |
| **SMOTE Handling** | Balanced imbalanced dataset for optimal performance |

---

## Model Performance

**Key Metrics:**
- **AUC Score**: 0.739
- **Dataset**: 150K+ LendingClub records (2017-2018)
- **Features**: 28 engineered financial indicators

**Feature Importance (Top 5):**
1. Interest rate
2. Loan Amount
3. Sub Grade
4. Installment  
5. Total Acc
---

## Project Structure
credit-risk-ai/
├── notebooks/ # Jupyter notebooks for exploration
│ ├── 01_eda.ipynb # Data exploration and analysis
│ ├── 02_preprocess.ipynb # Feature engineering and cleaning
│ └── 03_model.ipynb # Model training and evaluation
├── src/ # Production code package
│ ├── preprocess.py # Encoding + target encoding
│ ├── model.py # Predict + SHAP explanations
│ └── utils.py # Helper functions and utilities
├── api/
│ └── main.py # FastAPI server with /predict endpoint
├── dashboard/
│ └── app.py # Streamlit interactive dashboard
├── data/
│ ├── processed.parquet # 28 engineered features
│ ├── train_encoded.parquet
│ └── test_encoded.parquet
├── models/
│ ├── xgb_model.pkl # Trained XGBoost model
│ └── shap_explainer.pkl # SHAP explainer object
├── requirements.txt # Python dependencies
└── README.md

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- pip or conda

### Quick Start
```bash
# Clone repository
git clone https://github.com/Renzou756/Credit-Risk-Prediction.git
cd Credit-Risk-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn api.main:app --reload --port 8000

# Run the dashboard (in separate terminal)
streamlit run dashboard/app.py

## Usage
API Inference
python
import requests
import json

# Sample prediction request
payload = {
    "loan_amnt": 10000,
    "term": "36 months",
    "int_rate": 12.5,
    "dti": 18.5,
    "revol_util": 45.2,
    # ... other features
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(json.dumps(response.json(), indent=2))
Sample Response:

json
{
  "probability": 0.234,
  "prediction": "LOW_RISK",
  "shap_explanation": [
    {"feature": "dti_ratio", "value": 0.15, "impact": -0.032},
    {"feature": "revol_util_trend", "value": 0.08, "impact": -0.021},
    {"feature": "loan_amnt", "value": 10000, "impact": 0.018}
  ]
}
bash


## Dashboard Features
Real-time Predictions: Input loan details and get instant risk assessment

SHAP Visualizations: Interactive force plots and summary charts

Feature Analysis: Explore how different factors impact credit risk

Model Metrics: Performance dashboard with AUC, precision, recall

## Technical Details
Data Pipeline
SQL Preprocessing: Filtered 150K+ LendingClub records using DuckDB

##Feature Engineering:

dti_ratio: Debt-to-income normalized scaling

revol_util_trend: Credit utilization movement over time

payment_to_income: Monthly payment relative to income

credit_age: Account age based on earliest credit line

Handling Imbalance: SMOTE applied only to training set to prevent data leakage

Model Architecture
Algorithm: XGBoost with binary classification

Hyperparameter Tuning: Optuna using 30 trials

Explainability: SHAP values for interpretability

Deployment
API: FastAPI

Dashboard: Streamlit for interactive model exploration

Modular Design: src/ package for reusable components

📊 Results & Visualizations
Model Performance
https://images/auc_curve.png
ROC curve showing AUC of 0.842

SHAP Explanations
https://images/shap_summary.png
Global feature importance from SHAP values

Individual Predictions
https://images/force_plot.png
Waterfall chart showing factors driving a specific loan decision

🎯 Business Impact
This system demonstrates how machine learning can:

Reduce default rates by identifying high-risk loans with 79% recall

Improve transparency through SHAP explanations for regulatory compliance

Accelerate decision-making with real-time API predictions

Enable risk exploration via interactive dashboards for loan officers

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📞 Contact
Renzo Ugarte Basurco

Email: renzo.ugarte@mail.



