import streamlit as st
from utils.preprocessor import Preprocessor

def render():
    st.header("🔧 Step 2: Data Preprocessing")
    
    if st.session_state.data is None:
        st.error("⚠️ No data found. Please go back and upload a dataset first.")
        return
    
    st.write("Choose how you want to preprocess your data")
    
    preprocessing_mode = st.radio(
        "Preprocessing Mode:", 
        ["🤖 Automatic (Recommended)", "✋ Manual (Advanced)"],
        horizontal=True
    )
    
    if preprocessing_mode == "🤖 Automatic (Recommended)":
        render_automatic_preprocessing()
    else:
        render_manual_preprocessing()

def render_automatic_preprocessing():
    st.subheader("Automatic Preprocessing")
    
    with st.expander("ℹ️ What will be done automatically?", expanded=False):
        st.write("""
        **Automatic preprocessing includes:**
        - Fill missing numerical values with median
        - Fill missing categorical values with mode
        - Encode all categorical variables to numbers
        - Remove duplicate rows
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("📊 Current dataset shape: " + str(st.session_state.data.shape))
        st.write(f"Missing values: {st.session_state.data.isnull().sum().sum()}")
    
    with col2:
        if st.button("🚀 Apply Auto Preprocessing", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                df, label_encoders = Preprocessor.auto_preprocess(st.session_state.data)
                
                st.session_state.preprocessed_data = df
                st.session_state.label_encoders = label_encoders
                
                st.success("✅ Preprocessing completed!")
                st.rerun()
    
    if st.session_state.preprocessed_data is not None:
        st.success("✅ Data preprocessed successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("New Shape", str(st.session_state.preprocessed_data.shape))
        with col2:
            st.metric("Missing Values", st.session_state.preprocessed_data.isnull().sum().sum())
        
        with st.expander("👀 View Preprocessed Data"):
            st.dataframe(st.session_state.preprocessed_data.head(10), use_container_width=True)
        
        st.info("✨ Data is ready! Click 'Next' to run algorithms.")

def render_manual_preprocessing():
    st.subheader("Manual Preprocessing")
    df = st.session_state.data.copy()
    
    changes_made = False
    
    # Missing Values Handling
    with st.expander("📌 1. Missing Values Handling", expanded=True):
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
            st.success("✅ No missing values found!")
    
    # Noise Data Handling
    with st.expander("📌 2. Outlier Detection & Handling"):
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select column to analyze:", numeric_cols)
            
            outliers_count, Q1, Q3, IQR = Preprocessor.detect_outliers(df, selected_col)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Outliers Found", outliers_count)
            with col2:
                st.metric("Outlier %", f"{outliers_count/len(df)*100:.1f}%")
            
            if outliers_count > 0:
                noise_method = st.radio(
                    "How to handle outliers:", 
                    ["Keep (No change)", "Remove outliers", "Cap values"],
                    key="noise",
                    horizontal=True
                )
                if noise_method == "Remove outliers":
                    df = Preprocessor.handle_outliers(df, selected_col, "Remove")
                    changes_made = True
                    st.success(f"Removed {outliers_count} outliers")
                elif noise_method == "Cap values":
                    df = Preprocessor.handle_outliers(df, selected_col, "Cap")
                    changes_made = True
                    st.success(f"Capped {outliers_count} outlier values")
            else:
                st.success("✅ No outliers detected!")
        else:
            st.info("No numeric columns available for outlier detection")
    
    # Data Transformation
    with st.expander("📌 3. Normalization"):
        transform_cols = st.multiselect(
            "Select numeric columns to normalize (StandardScaler):", 
            df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
            help="Normalization scales data to have mean=0 and std=1"
        )
        
        if transform_cols:
            df, scaler = Preprocessor.normalize_columns(df, transform_cols)
            changes_made = True
            st.success(f"✅ Normalized {len(transform_cols)} columns")
    
    # Save button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save Preprocessing", use_container_width=True, type="primary"):
            st.session_state.preprocessed_data = df
            st.success("✅ Manual preprocessing saved!")
            st.balloons()
            st.info("✨ Click 'Next' to proceed to algorithms.")
            
    if st.session_state.preprocessed_data is not None:
        with st.expander("👀 View Current Preprocessed Data"):
            st.dataframe(st.session_state.preprocessed_data.head(), use_container_width=True)