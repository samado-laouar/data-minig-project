import streamlit as st
from utils.data_loader import DataLoader

def render():
    st.header("📁 Step 1: Data Import")
    st.write("Upload your dataset to begin the analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", 
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        df = DataLoader.load_file(uploaded_file)
        
        if df is not None:
            st.session_state.data = df
            info = DataLoader.get_data_info(df)
            
            st.success(f"✅ File uploaded successfully!")
            
            # Display data info in tabs
            tab1, tab2, tab3 = st.tabs(["📋 Preview", "ℹ️ Information", "📊 Statistics"])
            
            with tab1:
                st.subheader("Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
            
            with tab2:
                st.subheader("Dataset Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", info['rows'])
                    st.metric("Total Columns", info['columns'])
                with col2:
                    st.metric("Missing Values", df.isnull().sum().sum())
                    st.metric("Duplicate Rows", df.duplicated().sum())
                
                st.write("**Column Details:**")
                st.dataframe(info['column_info'], use_container_width=True)
            
            with tab3:
                st.subheader("Statistical Summary")
                st.dataframe(df.describe(), use_container_width=True)
            
            st.info("✨ Data loaded successfully! Click 'Next' to proceed to preprocessing.")
    else:
        st.info("👆 Please upload a dataset to get started")