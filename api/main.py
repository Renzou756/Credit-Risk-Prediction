from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from src.model import predict_loan
from src.preprocess import encode_loan_data

app = FastAPI(title="Credit Risk API")

# === LOAD TARGET ENCODER ONCE AT STARTUP ===
raw_train = pd.read_parquet('data/processed.parquet')
raw_train['issue_d'] = pd.to_datetime(raw_train['issue_d'])
train_2017 = raw_train[raw_train['issue_d'].dt.year == 2017].copy()

_, TARGET_ENCODER = encode_loan_data(train_2017, is_train=True)

class LoanInput(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    revol_bal: float
    revol_util: Optional[float] = None
    total_acc: int
    open_acc: int
    pub_rec: int
    inq_last_6mths: int
    delinq_2yrs: int
    emp_length: Optional[str] = None
    verification_status: str  
    home_ownership: str 
    purpose: str         
    grade: str          
    sub_grade: str     
    addr_state: str     
    issue_d: str       
    credit_age: float
    payment_to_income: float
    revol_utilization_trend: Optional[float] = None
    log_annual_inc: float
    log_loan_amnt: float
    log_revol_bal: float
    term_months: int
    fico_avg: float

@app.post("/predict")
def predict(loan: LoanInput):
    loan_dict = loan.dict()
    # Convert issue_d to datetime for consistency
    loan_dict['issue_d'] = pd.to_datetime(loan_dict['issue_d'])
    result = predict_loan(loan_dict, TARGET_ENCODER)
    return result