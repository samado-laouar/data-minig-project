import pandas as pd
import streamlit as st

class DataLoader:
    @staticmethod
    def load_file(uploaded_file):
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            else:
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    @staticmethod
    def get_data_info(df):
        info = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_info': pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Missing': df.isnull().sum().values,
                'Unique': df.nunique().values
            })
        }
        return info