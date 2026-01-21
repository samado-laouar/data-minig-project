import streamlit as st
from utils.preprocessor import Preprocessor
import pandas as pd
import numpy as np

def render():
    st.header("Step 2: Data Preprocessing")
    
    if st.session_state.data is None:
        st.error("No data found. Please go back and upload a dataset first.")
        return
    
    st.write("Choose how you want to preprocess your data")
    
    preprocessing_mode = st.radio(
        "Preprocessing Mode:", 
        ["Automatic (Recommended)", "Manual (Advanced)"],
        horizontal=True
    )
    
    if preprocessing_mode == "Automatic (Recommended)":
        render_automatic_preprocessing()
    else:
        render_manual_preprocessing()

def render_automatic_preprocessing():
    st.subheader("Automatic Preprocessing")
    
    with st.expander("What will be done automatically?", expanded=False):
        st.write("""
        **Automatic preprocessing includes:**
        - Replace '?' symbols with missing values
        - Fill missing numerical values with median
        - Fill missing categorical values with mode
        - **Detect and handle outliers** (using IQR method - Winsorization)
        - Encode all categorical variables to numbers
        - Remove duplicate rows
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("Current dataset shape: " + str(st.session_state.data.shape))
        
        # Check for '?' values
        question_marks = (st.session_state.data == '?').sum().sum()
        if question_marks > 0:
            st.warning(f"Found {question_marks} '?' values that will be treated as missing")
        
        missing_count = st.session_state.data.isnull().sum().sum()
        st.write(f"Missing values (NaN): {missing_count}")
    
    with col2:
        if st.button("Apply Auto Preprocessing", width='stretch', type="primary"):
            with st.spinner("Processing..."):
                df, label_encoders, outliers_info = Preprocessor.auto_preprocess(st.session_state.data)
                
                st.session_state.preprocessed_data = df
                st.session_state.label_encoders = label_encoders
                st.session_state.outliers_info = outliers_info
                
                st.success("Preprocessing completed!")
                st.rerun()
    
    if st.session_state.preprocessed_data is not None:
        st.success("Data preprocessed successfully!")
        
        # Display preprocessing summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("New Shape", str(st.session_state.preprocessed_data.shape))
        with col2:
            st.metric("Missing Values", st.session_state.preprocessed_data.isnull().sum().sum())
        with col3:
            if hasattr(st.session_state, 'outliers_info') and st.session_state.outliers_info:
                total_outliers = sum(info['count'] for info in st.session_state.outliers_info.values())
                st.metric("Outliers Handled", total_outliers)
            else:
                st.metric("Outliers Handled", 0)
        
        # Show outlier details if any were found
        if hasattr(st.session_state, 'outliers_info') and st.session_state.outliers_info:
            with st.expander("Outliers Detection Summary"):
                st.write("**Outliers detected and capped in the following columns:**")
                outlier_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Outliers Count': info['count'],
                        'Percentage': f"{info['percentage']:.2f}%"
                    }
                    for col, info in st.session_state.outliers_info.items()
                ])
                st.dataframe(outlier_df, width='stretch')
                st.info("Outliers were capped using the Winsorization method (IQR × 1.5)")
        
        with st.expander("View Preprocessed Data"):
            st.dataframe(st.session_state.preprocessed_data.head(10), width='stretch')
        
        st.info("Data is ready! Click 'Next' to run algorithms.")

def render_manual_preprocessing():
    st.subheader("Manual Preprocessing")
    df = st.session_state.data.copy()
    
    # Replace '?' with NaN first
    df = df.replace(['?', ' ?', '? '], np.nan)
    
    changes_made = False
    
    # Missing Values Handling
    with st.expander("1. Missing Values Handling", expanded=True):
        # Show '?' detection
        question_marks = (st.session_state.data == '?').sum().sum()
        if question_marks > 0:
            st.warning(f"Found and replaced {question_marks} '?' values with NaN")
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if missing_cols:
            st.warning(f"Found {len(missing_cols)} columns with missing values")
            
            for col in missing_cols:
                st.write(f"**{col}** - Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    method = st.selectbox(
                        f"Method for {col}:", 
                        ["Mean", "Median", "Mode", "Drop Rows", "Fill with Value"],
                        key=f"missing_{col}"
                    )
                
                fill_value = None
                if method == "Fill with Value":
                    with col2:
                        fill_value = st.text_input(f"Value:", key=f"fill_{col}")
                
                if method != "Drop Rows":
                    df = Preprocessor.handle_missing_values(df, col, method, fill_value)
                    changes_made = True
        else:
            st.success("No missing values found!")
    
    # Noise Data Handling - ENHANCED
    with st.expander("2. Outlier Detection & Handling", expanded=True):
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_cols:
            st.write("### Analyze Outliers by Column")
            selected_col = st.selectbox("Select column to analyze:", numeric_cols)
            
            outliers_count, Q1, Q3, IQR = Preprocessor.detect_outliers(df, selected_col)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Outliers Found", outliers_count)
            with col2:
                st.metric("Outlier %", f"{outliers_count/len(df)*100:.1f}%")
            with col3:
                st.metric("Q1", f"{Q1:.2f}")
            with col4:
                st.metric("Q3", f"{Q3:.2f}")
            
            st.write(f"**IQR (Interquartile Range):** {IQR:.2f}")
            st.write(f"**Lower Bound:** {Q1 - 1.5 * IQR:.2f}")
            st.write(f"**Upper Bound:** {Q3 + 1.5 * IQR:.2f}")
            
            if outliers_count > 0:
                st.warning(f"{outliers_count} outliers detected in '{selected_col}'")
                
                noise_method = st.radio(
                    "How to handle outliers:", 
                    ["Keep (No change)", "Remove outliers", "Cap values (Winsorization)"],
                    key="noise",
                    horizontal=True,
                    help="Winsorization caps extreme values at the boundary instead of removing them"
                )
                
                if noise_method == "Remove outliers":
                    df = Preprocessor.handle_outliers(df, selected_col, "Remove")
                    changes_made = True
                    st.success(f"Removed {outliers_count} outliers")
                elif noise_method == "Cap values (Winsorization)":
                    df = Preprocessor.handle_outliers(df, selected_col, "Cap")
                    changes_made = True
                    st.success(f"Capped {outliers_count} outlier values")
            else:
                st.success("No outliers detected!")
            
            st.markdown("---")
            
            # Detect all outliers at once
            st.write("### Detect Outliers in All Numeric Columns")
            if st.button("Scan All Columns", key="scan_outliers"):
                outlier_summary = []
                for col in numeric_cols:
                    count, q1, q3, iqr = Preprocessor.detect_outliers(df, col)
                    if count > 0:
                        outlier_summary.append({
                            'Column': col,
                            'Outliers': count,
                            'Percentage': f"{count/len(df)*100:.1f}%",
                            'Q1': f"{q1:.2f}",
                            'Q3': f"{q3:.2f}",
                            'IQR': f"{iqr:.2f}"
                        })
                
                if outlier_summary:
                    st.warning(f"Found outliers in {len(outlier_summary)} columns:")
                    outlier_df = pd.DataFrame(outlier_summary)
                    st.dataframe(outlier_df, width='stretch')
                    
                    if st.button("Auto-Handle All Outliers (Cap Method)", key="auto_cap"):
                        for col in numeric_cols:
                            count, _, _, _ = Preprocessor.detect_outliers(df, col)
                            if count > 0:
                                df = Preprocessor.handle_outliers(df, col, "Cap")
                        changes_made = True
                        st.success("All outliers have been capped!")
                        st.rerun()
                else:
                    st.success("No outliers found in any numeric column!")
        else:
            st.info("No numeric columns available for outlier detection")
    
    # Data Transformation
    with st.expander("3. Normalization"):
        transform_cols = st.multiselect(
            "Select numeric columns to normalize (StandardScaler):", 
            df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
            help="Normalization scales data to have mean=0 and std=1"
        )
        
        if transform_cols:
            df, scaler = Preprocessor.normalize_columns(df, transform_cols)
            changes_made = True
            st.success(f"Normalized {len(transform_cols)} columns")
    
    # Save button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("Save Preprocessing", width='stretch', type="primary"):
            st.session_state.preprocessed_data = df
            st.success("Manual preprocessing saved!")
            st.balloons()
            st.info("Click 'Next' to proceed to algorithms.")
            
    if st.session_state.preprocessed_data is not None:
        with st.expander("View Current Preprocessed Data"):
            st.dataframe(st.session_state.preprocessed_data.head(), width='stretch')
