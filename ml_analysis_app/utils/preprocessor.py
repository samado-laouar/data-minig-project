import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    @staticmethod
    def auto_preprocess(df):
        df_copy = df.copy()
        label_encoders = {}
        outliers_info = {}   # ← new

        # 1. Replace ? with NaN (you were missing this step!)
        df_copy = df_copy.replace(['?', ' ?', '? ', 'NA', 'na', ''], np.nan)

        # 2. Handle missing values
        for col in df_copy.columns:
            if df_copy[col].dtype in ['float64', 'int64']:
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            else:
                mode_val = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 'Unknown'
                df_copy[col].fillna(mode_val, inplace=True)

        # 3. Outlier detection & Winsorization (for numeric columns)
        numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outlier_mask = (df_copy[col] < lower) | (df_copy[col] > upper)
            count = outlier_mask.sum()
            
            if count > 0:
                outliers_info[col] = {
                    'count': int(count),
                    'percentage': (count / len(df_copy)) * 100,
                    'lower_bound': lower,
                    'upper_bound': upper
                }
                # Winsorization (capping)
                df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)

        # 4. Remove duplicates
        df_copy = df_copy.drop_duplicates()

        # 5. Encode categorical columns
        for col in df_copy.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            label_encoders[col] = le

        return df_copy, label_encoders, outliers_info

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