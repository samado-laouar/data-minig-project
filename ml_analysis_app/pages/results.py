import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.visualization import Visualizer
import numpy as np
def render():
    st.header("Step 4: Results & Analysis")
    
    if not st.session_state.results:
        st.warning("No results yet. Please run some algorithms first.")
        st.info("Go back to the Algorithms step and run at least one algorithm.")
        return
    
    st.success(f"You've run {len(st.session_state.results)} algorithm(s)")
    
    # Summary metrics
    st.subheader("Performance Summary")
    
    cols = st.columns(min(len(st.session_state.results), 4))
    for idx, (algo_name, results) in enumerate(st.session_state.results.items()):
        with cols[idx % 4]:
            if 'accuracy' in results:
                st.metric(
                    algo_name, 
                    f"{results['accuracy']:.4f}",
                    delta="Accuracy",
                    delta_color="off"
                )
            elif 'r2' in results:
                st.metric(
                    algo_name, 
                    f"{results['r2']:.4f}",
                    delta="R² Score",
                    delta_color="off"
                )
    
    st.markdown("---")
    
    # Detailed results for each algorithm
    st.subheader("Detailed Results")
    
    for algo_name, results in st.session_state.results.items():
        with st.expander(f"{algo_name}", expanded=True):
            display_algorithm_results(algo_name, results)
    
    # Comparison chart
    if len(st.session_state.results) > 1:
        st.markdown("---")
        st.subheader("Algorithm Comparison")
        create_comparison_chart()
    
    # Download results
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Download Results Report", width='stretch', type="primary"):
            download_results()
    

def display_algorithm_results(algo_name, results):
    if algo_name == "KNN":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best K Value", results['best_k'])
        with col2:
            st.metric("Best Accuracy", f"{results['accuracy']:.4f}")
        
        st.write("**K Values Tested:**")
        k_df = pd.DataFrame({
            'K': list(results['k_accuracies'].keys()),
            'Accuracy': list(results['k_accuracies'].values())
        })
        st.dataframe(k_df, width='stretch')
    
    elif algo_name in ["Naive Bayes", "Decision Tree", "Neural Network"]:
        st.metric("Accuracy Score", f"{results['accuracy']:.4f}")
        
        # Add Confusion Matrix and ROC for these algorithms too (if available)
        if 'confusion_matrix' in results:
            st.write("**Confusion Matrix:**")
            cm_array = np.array(results['confusion_matrix'])
            fig_cm = Visualizer.plot_confusion_matrix(cm_array, results['classes'])
            st.plotly_chart(fig_cm, width='stretch')
        
        if 'roc_data' in results:
            st.write("**ROC Curve:**")
            fig_roc = Visualizer.plot_roc_curve(results['roc_data'])
            st.plotly_chart(fig_roc, width='stretch')
        
        if 'feature_importance' in results:
            st.write("**Top 5 Important Features:**")
            top_features = sorted(
                results['feature_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            feature_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
            st.dataframe(feature_df, width='stretch')
        if 'iterations' in results:
            st.metric("Training Iterations", results['iterations'])
    
    elif algo_name in ["Linear Regression", "Multiple Regression"]:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{results['r2']:.4f}")
        with col2:
            st.metric("Mean Squared Error", f"{results['mse']:.4f}")
        
        st.metric("Intercept", f"{results['intercept']:.4f}")
        
        if 'coefficient' in results:
            st.metric("Coefficient", f"{results['coefficient']:.4f}")
        
        if 'coefficients' in results:
            st.write("**Feature Coefficients:**")
            coef_df = pd.DataFrame({
                'Feature': list(results['coefficients'].keys()),
                'Coefficient': list(results['coefficients'].values())
            }).sort_values('Coefficient', key=abs, ascending=False)
            st.dataframe(coef_df, width='stretch')

def create_comparison_chart():
    # Create comparison for classification algorithms
    classification_algos = {}
    regression_algos = {}
    
    for algo, results in st.session_state.results.items():
        if 'accuracy' in results:
            classification_algos[algo] = results['accuracy']
        elif 'r2' in results:
            regression_algos[algo] = results['r2']
    
    if classification_algos:
        fig = go.Figure(data=[
            go.Bar(
                x=list(classification_algos.keys()),
                y=list(classification_algos.values()),
                marker_color='#4CAF50',
                text=[f"{v:.4f}" for v in classification_algos.values()],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Classification Algorithms - Accuracy Comparison",
            xaxis_title="Algorithm",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, width='stretch')
    
    if regression_algos:
        fig = go.Figure(data=[
            go.Bar(
                x=list(regression_algos.keys()),
                y=list(regression_algos.values()),
                marker_color='#2196F3',
                text=[f"{v:.4f}" for v in regression_algos.values()],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Regression Algorithms - R² Score Comparison",
            xaxis_title="Algorithm",
            yaxis_title="R² Score"
        )
        st.plotly_chart(fig, width='stretch')

def download_results():
    rows = []
    
    for algo_name, res in st.session_state.results.items():
        row = {'Algorithm': algo_name}
        
        # Add main scalar metrics
        for k, v in res.items():
            if isinstance(v, (int, float, str)):
                row[k] = v
            elif isinstance(v, dict) and k in ['k_accuracies', 'feature_importance', 'coefficients']:
                # Flatten selected important dicts
                for subkey, subval in v.items():
                    row[f"{k}_{subkey}"] = subval
            elif k == 'confusion_matrix' and isinstance(v, list):
                # Simple string representation
                row['confusion_matrix'] = str(np.array(v))
            # You can add more special handling here if needed
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns nicely (optional)
    cols_order = ['Algorithm']
    for col in sorted(df.columns):
        if col != 'Algorithm':
            cols_order.append(col)
    df = df[cols_order]
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="ml_results_summary.csv",
        mime="text/csv"
    )