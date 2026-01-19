import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_cols = ['education', 'self_employed', 'property_area']

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            self.label_encoders[col] = le
        
        # Numeric scaling
        numeric_cols = ['income', 'loan_amount', 'credit_score']
        df_clean[numeric_cols] = self.scaler.fit_transform(df_clean[numeric_cols])
        return df_clean

    def transform_single(self, data_dict: Dict) -> np.ndarray:
        """Transform a single application into model-ready format"""
        processed = {}
        
        # Handle categorical features
        for col in self.categorical_cols:
            if col in self.label_encoders:
                processed[col] = self.label_encoders[col].transform([str(data_dict[col])])[0]
            else:
                processed[col] = int(data_dict[col])
        
        # Handle numeric features
        numeric_cols = ['income', 'loan_amount', 'credit_score']
        numeric_data = np.array([[data_dict[col] for col in numeric_cols]])
        scaled_numeric = self.scaler.transform(numeric_data)[0]
        
        for i, col in enumerate(numeric_cols):
            processed[col] = scaled_numeric[i]
        
        # Return in correct order
        feature_order = ['income', 'loan_amount', 'credit_score', 'education', 'self_employed', 'property_area']
        return np.array([[processed[col] for col in feature_order]])
