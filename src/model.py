import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from .preprocess import encode_loan_data

_model = joblib.load('models/xgb_model.pkl')
_explainer = joblib.load('models/shap_explainer.pkl')

def predict_loan(loan_dict: dict, target_encoder: dict = None) -> dict:
    """
    Predict default probability & SHAP values for one loan.
    """
    df = pd.DataFrame([loan_dict])
    df_encoded, _ = encode_loan_data(df, is_train=False, target_encoder=target_encoder)
    
    # For Booster object - use .feature_names directly
    expected_features = _model.feature_names
    
    # Add any missing columns with 0
    for feature in expected_features:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Ensure correct column order
    df_encoded = df_encoded[expected_features]
    
    dmatrix = xgb.DMatrix(df_encoded)
    raw_pred = _model.predict(dmatrix)[0]
    prob = 1 / (1 + np.exp(-raw_pred)) 

    shap_values = _explainer.shap_values(df_encoded)
    shap_dict = dict(zip(df_encoded.columns, shap_values[0]))

    return {
        "default_probability": round(float(prob), 4),
        "risk_level": "HIGH" if prob > 0.5 else "LOW",
        "shap_values": {k: float(v) for k, v in shap_dict.items()}
    }