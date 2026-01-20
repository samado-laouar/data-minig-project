import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    @staticmethod
    def auto_preprocess(df):
        df_copy = df.copy()
        label_encoders = {}
        
        # Handle missing values
        for col in df_copy.columns:
            if df_copy[col].dtype in ['float64', 'int64']:
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            else:
                mode_val = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 'Unknown'
                df_copy[col].fillna(mode_val, inplace=True)
        
        # Encode categorical variables
        for col in df_copy.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            label_encoders[col] = le
        
        return df_copy, label_encoders
    
    @staticmethod
    def handle_missing_values(df, column, method, fill_value=None):
        if method == "Mean":
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == "Median":
            df[column].fillna(df[column].median(), inplace=True)
        elif method == "Mode":
            mode_val = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
            df[column].fillna(mode_val, inplace=True)
        elif method == "Fill with Value" and fill_value:
            df[column].fillna(fill_value, inplace=True)
        return df
    
    @staticmethod
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_mask = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        outliers_count = outlier_mask.sum()
        
        return outliers_count, Q1, Q3, IQR
    
    @staticmethod
    def handle_outliers(df, column, method):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        if method == "Remove":
            df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
        elif method == "Cap":
            df[column] = df[column].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        
        return df
    
    @staticmethod
    def normalize_columns(df, columns):
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df, scaler