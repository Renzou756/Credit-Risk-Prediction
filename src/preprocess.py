import pandas as pd
import numpy as np

def encode_loan_data(df: pd.DataFrame, is_train: bool = True, target_encoder: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Applies all preprocessing: one-hot, ordinal, target encoding.
    Returns encoded df and target_encoder (for test data).
    """
    df = df.copy()

    # 1. Impute missing
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(-1)

    # 2. Ordinal: grade, sub_grade
    grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
    df['grade_ord'] = df['grade'].map(grade_map)

    subgrade_map = {f'{g}{i}': (grade_map[g]-1)*5 + i for g in grade_map for i in range(1,6)}
    df['sub_grade_ord'] = df['sub_grade'].map(subgrade_map)

    # 3. One-hot encoding with consistent columns
    categorical_cols = ['home_ownership', 'purpose', 'verification_status']
    
    if is_train:
        # Training mode - create dummies normally
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        # Store the expected columns for inference
        expected_columns = df.columns.tolist()
    else:
        # Inference mode - create dummies and ensure all expected columns exist
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        
        # Add missing columns with 0 values
        expected_columns = [
            'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'revol_bal', 
            'revol_util', 'total_acc', 'open_acc', 'pub_rec', 'inq_last_6mths', 
            'delinq_2yrs', 'emp_length', 'credit_age', 'payment_to_income', 
            'revol_utilization_trend', 'log_annual_inc', 'log_loan_amnt', 'log_revol_bal', 
            'term_months', 'fico_avg', 'grade_ord', 'sub_grade_ord', 'addr_state_te',
            'home_ownership_MORTGAGE', 'home_ownership_NONE', 'home_ownership_OWN', 
            'home_ownership_RENT', 'purpose_credit_card', 'purpose_debt_consolidation', 
            'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 
            'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 
            'purpose_small_business', 'purpose_vacation', 'purpose_wedding', 
            'verification_status_Source Verified', 'verification_status_Verified'
        ]
        
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

    # 4. emp_length
    emp_map = {'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4,
               '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}
    df['emp_length'] = df['emp_length'].map(emp_map).fillna(-1).astype(int)

    # 5. Target encoding: addr_state
    if is_train:
        target_encoder = df.groupby('addr_state')['target'].mean().to_dict()
        df['addr_state_te'] = df['addr_state'].map(target_encoder)
    else:
        global_mean = np.mean(list(target_encoder.values())) if target_encoder else 0.5
        df['addr_state_te'] = df['addr_state'].map(target_encoder).fillna(global_mean)

    # 6. Drop raw cols
    drop_cols = ['grade', 'sub_grade', 'addr_state', 'issue_d']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 7. Ensure correct column order for inference
    if not is_train:
        # Reorder columns to match training
        existing_cols = [col for col in expected_columns if col in df.columns]
        df = df[existing_cols]
        
        # Add any still missing columns
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

    return df, target_encoder