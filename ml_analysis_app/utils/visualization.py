import plotly.express as px
import plotly.graph_objects as go

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