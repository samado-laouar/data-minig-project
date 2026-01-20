import streamlit as st
import pandas as pd
from models.classification import ClassificationModels
from models.regression import RegressionModels
from utils.visualization import Visualizer

def render():
    st.header("🚀 Step 3: Run Machine Learning Algorithms")
    
    if st.session_state.preprocessed_data is None:
        st.error("⚠️ No preprocessed data found. Please complete preprocessing first.")
        return
    
    st.write("Select and run different ML algorithms on your data")
    
    df = st.session_state.preprocessed_data.copy()
    
    # Algorithm selection with cards
    st.subheader("Select Algorithm")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 KNN", use_container_width=True, help="K-Nearest Neighbors Classification"):
            st.session_state.selected_algo = "KNN"
        if st.button("📊 Decision Tree", use_container_width=True, help="C4.5 Decision Tree"):
            st.session_state.selected_algo = "Decision Tree C4.5"
    
    with col2:
        if st.button("📈 Linear Regression", use_container_width=True, help="Simple Linear Regression"):
            st.session_state.selected_algo = "Linear Regression"
        if st.button("🧠 Neural Network", use_container_width=True, help="Multi-layer Perceptron"):
            st.session_state.selected_algo = "Neural Network"
    
    with col3:
        if st.button("🎲 Naive Bayes", use_container_width=True, help="Gaussian Naive Bayes"):
            st.session_state.selected_algo = "Naive Bayes"
        if st.button("📉 Multiple Regression", use_container_width=True, help="Multiple Linear Regression"):
            st.session_state.selected_algo = "Multiple Regression"
    
    if 'selected_algo' not in st.session_state:
        st.info("👆 Click on an algorithm to get started")
        return
    
    algorithm = st.session_state.selected_algo
    st.markdown(f"### Running: **{algorithm}**")
    st.markdown("---")
    
    # Target selection
    target_col = st.selectbox("🎯 Select Target Variable:", df.columns.tolist())
    
    if algorithm in ["KNN", "Naive Bayes", "Decision Tree C4.5", "Neural Network"]:
        render_classification_algorithm(df, target_col, algorithm)
    elif algorithm == "Linear Regression":
        render_linear_regression(df)
    elif algorithm == "Multiple Regression":
        render_multiple_regression(df, target_col)
    
    # Show results summary
    if st.session_state.results:
        st.markdown("---")
        st.subheader("📊 Results So Far")
        cols = st.columns(len(st.session_state.results))
        for idx, (algo_name, results) in enumerate(st.session_state.results.items()):
            with cols[idx]:
                if 'accuracy' in results:
                    st.metric(algo_name, f"{results['accuracy']:.3f}", "Accuracy")
                elif 'r2' in results:
                    st.metric(algo_name, f"{results['r2']:.3f}", "R² Score")
        
        st.info("✨ Click 'Next' to view detailed results!")

def render_classification_algorithm(df, target_col, algorithm):
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_size = st.slider("Test Set Size (%):", 10, 50, 20, 5) / 100
    
    with col2:
        st.metric("Training Size", f"{int((1-test_size)*100)}%")
    
    if st.button(f"▶️ Run {algorithm}", use_container_width=True, type="primary"):
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
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ KNN completed!")
            
            elif algorithm == "Naive Bayes":
                results = ClassificationModels.run_naive_bayes(X, y, test_size)
                st.session_state.results['Naive Bayes'] = results
                
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
                st.success("✅ Naive Bayes completed!")
            
            elif algorithm == "Decision Tree C4.5":
                results = ClassificationModels.run_decision_tree(X, y, test_size)
                st.session_state.results['Decision Tree'] = results
                
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
                
                fig = Visualizer.plot_feature_importance(
                    list(results['feature_importance'].keys()),
                    list(results['feature_importance'].values())
                )
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Decision Tree completed!")
            
            elif algorithm == "Neural Network":
                results = ClassificationModels.run_neural_network(X, y, test_size)
                st.session_state.results['Neural Network'] = results
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                with col2:
                    st.metric("Training Iterations", results['iterations'])
                st.success("✅ Neural Network completed!")

def render_linear_regression(df):
    st.subheader("Select Two Variables")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("📊 X Variable (Independent):", numeric_cols)
    with col2:
        y_col = st.selectbox("📈 Y Variable (Dependent):", [c for c in numeric_cols if c != x_col])
    
    if st.button("▶️ Run Linear Regression", use_container_width=True, type="primary"):
        X = df[[x_col]].values
        y = df[y_col].values
        
        results = RegressionModels.run_linear_regression(X, y)
        st.session_state.results['Linear Regression'] = results
        
        st.success("✅ Linear Regression Complete")
        
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
        st.plotly_chart(fig, use_container_width=True)

def render_multiple_regression(df, target_col):
    st.subheader("Multiple Regression on All Features")
    
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    st.info(f"Using {len(feature_cols)} features to predict {target_col}")
    
    if st.button("▶️ Run Multiple Regression", use_container_width=True, type="primary"):
        results = RegressionModels.run_multiple_regression(X, y)
        st.session_state.results['Multiple Regression'] = results
        
        st.success("✅ Multiple Regression Complete")
        
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
        st.dataframe(coef_df, use_container_width=True)