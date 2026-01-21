import streamlit as st
import pandas as pd
from models.classification import ClassificationModels
from models.regression import RegressionModels
from utils.visualization import Visualizer
import numpy as np

def render():
    st.header("Step 3: Run Machine Learning Algorithms")
    
    if st.session_state.preprocessed_data is None:
        st.error("No preprocessed data found. Please complete preprocessing first.")
        return
    
    st.write("Select and run different ML algorithms on your data")
    
    df = st.session_state.preprocessed_data.copy()
    
    # Run All Algorithms Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(" RUN ALL ALGORITHMS", width='stretch', type="primary", help="Execute all algorithms automatically"):
            st.session_state.run_all = True
    st.markdown("---")
    
    # Algorithm selection with cards
    st.subheader("Or Select Individual Algorithm")
    
    col1, col2, col3= st.columns(3)
    
    with col1:
        if st.button(" KNN", width='stretch', help="K-Nearest Neighbors Classification"):
            st.session_state.selected_algo = "KNN"
            st.session_state.run_all = False
        if st.button("Decision Tree", width='stretch', help="C4.5 Decision Tree"):
            st.session_state.selected_algo = "Decision Tree C4.5"
            st.session_state.run_all = False
    
    with col2:
        if st.button("Linear Regression", width='stretch', help="Simple Linear Regression"):
            st.session_state.selected_algo = "Linear Regression"
            st.session_state.run_all = False
        if st.button("Neural Network", width='stretch', help="Multi-layer Perceptron"):
            st.session_state.selected_algo = "Neural Network"
            st.session_state.run_all = False
    
    with col3:
        if st.button("Naive Bayes", width='stretch', help="Gaussian Naive Bayes"):
            st.session_state.selected_algo = "Naive Bayes"
            st.session_state.run_all = False
        if st.button("Multiple Regression", width='stretch', help="Multiple Linear Regression"):
            st.session_state.selected_algo = "Multiple Regression"
            st.session_state.run_all = False

    # Handle Run All Algorithms
    if 'run_all' in st.session_state and st.session_state.run_all:
        run_all_algorithms(df)
        st.session_state.run_all = False
        return
    
    if 'selected_algo' not in st.session_state:
        st.info("Click on an algorithm to get started or run all at once")
        return
    
    algorithm = st.session_state.selected_algo
    st.markdown(f"### Running: **{algorithm}**")
    st.markdown("---")
    
    # Target selection
    target_col = st.selectbox(" Select Target Variable:", df.columns.tolist())
    
    if algorithm in ["KNN", "Naive Bayes", "Decision Tree C4.5", "Neural Network"]:
        render_classification_algorithm(df, target_col, algorithm)
    elif algorithm == "Linear Regression":
        render_linear_regression(df)
    elif algorithm == "Multiple Regression":
        render_multiple_regression(df, target_col)
    
    # Show results summary
    if st.session_state.results:
        st.markdown("---")
        st.subheader("Results So Far")
        cols = st.columns(len(st.session_state.results))
        for idx, (algo_name, results) in enumerate(st.session_state.results.items()):
            with cols[idx]:
                if 'accuracy' in results:
                    st.metric(algo_name, f"{results['accuracy']:.3f}", "Accuracy")
                elif 'r2' in results:
                    st.metric(algo_name, f"{results['r2']:.3f}", "R² Score")
        
        st.info("Click 'Next' to view detailed results!")

def render_classification_algorithm(df, target_col, algorithm):
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_size = st.slider("Test Set Size (%):", 10, 50, 20, 5) / 100
    
    with col2:
        st.metric("Training Size", f"{int((1-test_size)*100)}%")
    
    if st.button(f"Run {algorithm}", width='stretch', type="primary"):
        with st.spinner(f"Running {algorithm}..."):
            if algorithm == "KNN":
                results = ClassificationModels.run_knn(X, y, test_size)
                st.session_state.results['KNN'] = results
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best K Value", results['best_k'])
                with col2:
                    st.metric("Best Accuracy", f"{results['accuracy']:.4f}")
                
                fig = Visualizer.plot_knn_accuracy(results['k_accuracies'])
                st.plotly_chart(fig, width='stretch')
                st.success("KNN completed!")
            
            elif algorithm == "Naive Bayes":
                results = ClassificationModels.run_naive_bayes(X, y, test_size)
                st.session_state.results['Naive Bayes'] = results
                
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
                st.success("Naive Bayes completed!")
            
            elif algorithm == "Decision Tree C4.5":
                results = ClassificationModels.run_decision_tree(X, y, test_size)
                st.session_state.results['Decision Tree'] = results
                
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
                
                fig = Visualizer.plot_feature_importance(
                    list(results['feature_importance'].keys()),
                    list(results['feature_importance'].values())
                )
                st.plotly_chart(fig, width='stretch')
                st.success("Decision Tree completed!")
            
            elif algorithm == "Neural Network":
                results = ClassificationModels.run_neural_network(X, y, test_size)
                st.session_state.results['Neural Network'] = results
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                with col2:
                    st.metric("Training Iterations", results['iterations'])
                st.success("Neural Network completed!")

def render_linear_regression(df):
    st.subheader("Select Two Variables")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("X Variable (Independent):", numeric_cols)
    with col2:
        y_col = st.selectbox("Y Variable (Dependent):", [c for c in numeric_cols if c != x_col])
    
    if st.button("Run Linear Regression", width='stretch', type="primary"):
        X = df[[x_col]].values
        y = df[y_col].values
        
        results = RegressionModels.run_linear_regression(X, y)
        st.session_state.results['Linear Regression'] = results
        
        st.success("Linear Regression Complete")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{results['r2']:.4f}")
        with col2:
            st.metric("MSE", f"{results['mse']:.4f}")
        with col3:
            st.metric("Coefficient", f"{results['coefficient']:.4f}")
        
        st.write(f"**Equation:** y = {results['coefficient']:.4f}x + {results['intercept']:.4f}")
        
        fig = Visualizer.plot_linear_regression(
            X, y, 
            X, results['model'].predict(X),
            x_col, y_col
        )
        st.plotly_chart(fig, width='stretch')

def render_multiple_regression(df, target_col):
    st.subheader("Multiple Regression on All Features")
    
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    st.info(f"Using {len(feature_cols)} features to predict {target_col}")
    
    if st.button("Run Multiple Regression", width='stretch', type="primary"):
        results = RegressionModels.run_multiple_regression(X, y)
        st.session_state.results['Multiple Regression'] = results
        
        st.success("Multiple Regression Complete")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{results['r2']:.4f}")
        with col2:
            st.metric("MSE", f"{results['mse']:.4f}")
        
        st.write(f"**Intercept:** {results['intercept']:.4f}")
        
        st.write("**Feature Coefficients:**")
        coef_df = pd.DataFrame({
            'Feature': list(results['coefficients'].keys()),
            'Coefficient': list(results['coefficients'].values())
        }).sort_values('Coefficient', key=abs, ascending=False)
        st.dataframe(coef_df, width='stretch')

def run_all_algorithms(df):
    st.subheader(" Running All Algorithms")
    st.write("This will execute all available algorithms on your dataset")
    
    # Select target variable
    target_col = st.selectbox(" Select Target Variable for Classification/Regression:", df.columns.tolist())
    
    # Test size selection
    test_size = st.slider("Test Set Size (%):", 10, 50, 20, 5) / 100
    
    st.markdown("---")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_algorithms = 6
    current = 0
    
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # 1. KNN
    current += 1
    status_text.text(f"Running KNN... ({current}/{total_algorithms})")
    progress_bar.progress(current / total_algorithms)
    
    with st.spinner("Running KNN..."):
        results = ClassificationModels.run_knn(X, y, test_size)
        st.session_state.results['KNN'] = results
    
    with st.expander("KNN Results", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best K Value", results['best_k'])
        with col2:
            st.metric("Best Accuracy", f"{results['accuracy']:.4f}")
    
    # 2. Naive Bayes
    current += 1
    status_text.text(f"Running Naive Bayes... ({current}/{total_algorithms})")
    progress_bar.progress(current / total_algorithms)
    
    with st.spinner("Running Naive Bayes..."):
        results = ClassificationModels.run_naive_bayes(X, y, test_size)
        st.session_state.results['Naive Bayes'] = results
    
    with st.expander("Naive Bayes Results", expanded=False):
        st.metric("Accuracy", f"{results['accuracy']:.4f}")
    
    # 3. Decision Tree
    current += 1
    status_text.text(f"Running Decision Tree... ({current}/{total_algorithms})")
    progress_bar.progress(current / total_algorithms)
    
    with st.spinner("Running Decision Tree..."):
        results = ClassificationModels.run_decision_tree(X, y, test_size)
        st.session_state.results['Decision Tree'] = results
    
    with st.expander("Decision Tree Results", expanded=False):
        st.metric("Accuracy", f"{results['accuracy']:.4f}")
        fig = Visualizer.plot_feature_importance(
            list(results['feature_importance'].keys()),
            list(results['feature_importance'].values())
        )
        st.plotly_chart(fig, width='stretch')
    
    # 4. Neural Network
    current += 1
    status_text.text(f"Running Neural Network... ({current}/{total_algorithms})")
    progress_bar.progress(current / total_algorithms)
    
    with st.spinner("Running Neural Network..."):
        results = ClassificationModels.run_neural_network(X, y, test_size)
        st.session_state.results['Neural Network'] = results
    
    with st.expander("Neural Network Results", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col2:
            st.metric("Training Iterations", results['iterations'])
    
    # 5. Linear Regression (using first two numeric columns)
    current += 1
    status_text.text(f"Running Linear Regression... ({current}/{total_algorithms})")
    progress_bar.progress(current / total_algorithms)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) >= 2:
        with st.spinner("Running Linear Regression..."):
            X_lr = df[[numeric_cols[1]]].values
            y_lr = df[numeric_cols[2]].values
            results = RegressionModels.run_linear_regression(X_lr, y_lr)
            st.session_state.results['Linear Regression'] = results
        
        with st.expander("Linear Regression Results", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{results['r2']:.4f}")
            with col2:
                st.metric("MSE", f"{results['mse']:.4f}")
            with col3:
                st.metric("Coefficient", f"{results['coefficient']:.4f}")
            
            st.write(f"**Equation:** y = {results['coefficient']:.4f}x + {results['intercept']:.4f}")
            st.write(f"**Variables:** {numeric_cols[0]} vs {numeric_cols[1]}")
    else:
        st.warning("Linear Regression skipped: Need at least 2 numeric columns")
    
    # 6. Multiple Regression
    current += 1
    status_text.text(f"Running Multiple Regression... ({current}/{total_algorithms})")
    progress_bar.progress(current / total_algorithms)
    
    with st.spinner("Running Multiple Regression..."):
        results = RegressionModels.run_multiple_regression(X, y)
        st.session_state.results['Multiple Regression'] = results
    
    with st.expander("Multiple Regression Results", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{results['r2']:.4f}")
        with col2:
            st.metric("MSE", f"{results['mse']:.4f}")
        
        st.write(f"**Intercept:** {results['intercept']:.4f}")
        
        coef_df = pd.DataFrame({
            'Feature': list(results['coefficients'].keys()),
            'Coefficient': list(results['coefficients'].values())
        }).sort_values('Coefficient', key=abs, ascending=False)
        st.dataframe(coef_df, width='stretch')
    
    # Completion
    progress_bar.progress(1.0)
    status_text.text("All algorithms completed!")
    
    st.success(f"Successfully ran {len(st.session_state.results)} algorithms!")
    
    st.info("Click 'Next' to view detailed comparison and results!")