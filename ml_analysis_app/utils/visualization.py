import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
class Visualizer:
    @staticmethod
    def plot_knn_accuracy(k_accuracies):
        fig = px.line(
            x=list(k_accuracies.keys()), 
            y=list(k_accuracies.values()),
            labels={'x': 'K Value', 'y': 'Accuracy'},
            title='KNN: K Value vs Accuracy',
            markers=True
        )
        fig.update_traces(line_color='#4CAF50', line_width=3)
        return fig
    
    @staticmethod
    def plot_feature_importance(features, importances):
        fig = px.bar(
            x=features, 
            y=importances,
            labels={'x': 'Features', 'y': 'Importance'},
            title='Feature Importance',
            color=importances,
            color_continuous_scale='Greens'
        )
        return fig
    
    @staticmethod
    def plot_linear_regression(X, y, X_pred, y_pred, x_label, y_label):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X.flatten(), 
            y=y, 
            mode='markers', 
            name='Data Points',
            marker=dict(color='#4CAF50', size=8)
        ))
        fig.add_trace(go.Scatter(
            x=X_pred.flatten(), 
            y=y_pred, 
            mode='lines', 
            name='Regression Line',
            line=dict(color='#FF5722', width=3)
        ))
        fig.update_layout(
            title='Linear Regression',
            xaxis_title=x_label,
            yaxis_title=y_label
        )
        return fig

    @staticmethod
    def plot_confusion_matrix(cm, classes):
        # Convert to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                annotations.append({
                    'x': classes[j],
                    'y': classes[i],
                    'text': f'{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)',
                    'showarrow': False,
                    'font': {'size': 12, 'color': 'white' if cm[i][j] > cm.max()/2 else 'black'}
                })
        
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=[str(c) for c in classes],
            y=[str(c) for c in classes],
            colorscale='Greens',
            showscale=True
        )
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            xaxis={'side': 'bottom'},
            height=500
        )
        
        return fig

    @staticmethod
    def plot_roc_curve(roc_data):
        fig = go.Figure()
        
        if roc_data['type'] == 'binary':
            # Binary classification ROC
            fig.add_trace(go.Scatter(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                mode='lines',
                name=f'ROC Curve (AUC = {roc_data["auc"]:.3f})',
                line=dict(color='#4CAF50', width=3)
            ))
        else:
            # Multiclass ROC
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0']
            for idx, (class_label, data) in enumerate(roc_data['classes'].items()):
                fig.add_trace(go.Scatter(
                    x=data['fpr'],
                    y=data['tpr'],
                    mode='lines',
                    name=f'Class {class_label} (AUC = {data["auc"]:.3f})',
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve (Receiver Operating Characteristic)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=500,
            showlegend=True
        )
        
        return fig
