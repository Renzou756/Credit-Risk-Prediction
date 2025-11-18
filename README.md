# Credit Risk AI – Production-Grade ML System  
![Python](https://img.shields.io/badge/python-3.11-blue)  
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)  
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.741-orange)  
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-red)  
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-yellow)

**A bank-grade credit risk model** that predicts loan default probability using **XGBoost**, **Optuna**, and **SHAP**, deployed via **FastAPI** and **Streamlit**. Built with 150K+ LendingClub records to demonstrate production-ready ML pipelines for financial risk assessment.

## Features

| Feature | Status |
|---------|--------|
| **XGBoost + Hyperparameter Tuning** | **AUC 0.741** with Optuna optimization |
| **SHAP Explainability** | Per-loan feature importance for output explainability |
| **FastAPI Inference** | Real-time `/predict` endpoint with JSON responses |
| **Streamlit Dashboard** | Interactive UI + live SHAP visualizations |
| **Advanced Feature Engineering** | DTI ratios, credit utilization trends, payment-to-income |

## Model Performance

**Key Metrics:**
- **AUC Score**: 0.7411
- **Dataset**: 150K+ LendingClub records (2017-2018)
- **Features**: 28 engineered financial indicators

**Feature Importance (Top 5):**
1. Interest rate
2. Loan Amount
3. Sub Grade
4. Installment  
5. Total Acc

## Project Structure
credit-risk-ai/

├── notebooks/

│ ├── 01_eda.ipynb

│ ├── 02_preprocess.ipynb

│ └── 03_model.ipynb

├── src/

│ ├── preprocess.py

│ ├── model.py

│ └── utils.py

├── api/

│ └── main.py

├── dashboard/

│ └── app.py

├── data/

│ ├── processed.parquet

│ ├── train_encoded.parquet

│ └── test_encoded.parquet

├── models/

│ ├── xgb_model.pkl

│ └── shap_explainer.pkl

├── requirements.txt

└── README.md



## Technical Details

### Data Pipeline
1. **SQL Preprocessing**: Filtered 150K+ LendingClub records using DuckDB
2. **Feature Engineering**:
   - `dti_ratio`: Debt-to-income normalized scaling
   - `revol_util_trend`: Credit utilization movement over time
   - `payment_to_income`: Monthly payment relative to income
   - `credit_age`: Account age based on earliest credit line
3. **Handling Imbalance**: SMOTE applied only to training set to prevent data leakage

### Model Architecture
- **Algorithm**: XGBoost with binary classification
- **Hyperparameter Tuning**: Optuna using 30 trials
- **Explainability**: SHAP values for interpretability

### Deployment
- **API**: FastAPI
- **Dashboard**: Streamlit for interactive model exploration
- **Modular Design**: `src/` package for reusable components

## Results & Visualizations

### Model Performance
![AUC Curve](auc_roc_curve.png)
*ROC curve showing AUC of 0.739*

### SHAP Explanations
![SHAP Summary](shap_summary.png)
*Global feature importance from SHAP values*

### Individual Predictions
![Force Plot](force_plot_first_instance.png)
![Waterfall](waterfall_correct_1.png)
*Waterfall chart showing factors driving a specific loan decision*

## Business Impact

This system demonstrates how machine learning can:
- **Reduce default rates** by identifying high-risk loans
- **Improve transparency** through SHAP explanations for regulatory compliance
- **Accelerate decision-making** with real-time API predictions
- **Enable risk exploration** via interactive dashboards for loan officers

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- **LendingClub** for providing the loan dataset
- **SHAP** community for excellent model interpretability tools
- **FastAPI** and **Streamlit** teams for fantastic deployment frameworks

*This project is for demonstration purposes and should not be used for actual financial decisions without proper validation and regulatory compliance.*







