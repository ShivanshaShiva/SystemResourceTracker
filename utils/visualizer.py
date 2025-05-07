import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

class Visualizer:
    """
    Class for generating visualizations for both Sanskrit NLP data
    and data comparison results.
    """
    
    def __init__(self):
        """Initialize the Visualizer class."""
        # Define a color scheme for consistent styling
        self.colors = {
            "primary": "#2D68C4",
            "secondary": "#F2A900",
            "tertiary": "#28A197",
            "quaternary": "#802F2D",
            "background": "#F9F9F9",
            "text": "#333333"
        }
    
    def plot_comparison(self, df, key_col, value1_col, value2_col):
        """
        Plot a comparison between two datasets.
        
        Args:
            df (DataFrame): DataFrame containing the comparison data
            key_col (str): Column containing the keys/labels
            value1_col (str): Column containing first values (e.g., predicted)
            value2_col (str): Column containing second values (e.g., actual)
            
        Returns:
            Figure: Plotly figure object
        """
        # Create a scatter plot
        fig = go.Figure()
        
        # Add the scatter points
        fig.add_trace(go.Scatter(
            x=df[value1_col],
            y=df[value2_col],
            mode='markers',
            marker=dict(
                color=self.colors["primary"],
                size=10,
                opacity=0.7
            ),
            name='Data Points'
        ))
        
        # Add a perfect correlation line
        max_val = max(df[value1_col].max(), df[value2_col].max())
        min_val = min(df[value1_col].min(), df[value2_col].min())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color=self.colors["secondary"], width=2, dash='dash'),
            name='Perfect Correlation'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Comparison between {value1_col} and {value2_col}',
            xaxis_title=value1_col,
            yaxis_title=value2_col,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )
        
        return fig
    
    def plot_distributions(self, df, value1_col, value2_col):
        """
        Plot the distributions of two datasets.
        
        Args:
            df (DataFrame): DataFrame containing the data
            value1_col (str): Column containing first values
            value2_col (str): Column containing second values
            
        Returns:
            Figure: Plotly figure object
        """
        # Create a figure with two subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Add histograms
        fig.add_trace(
            go.Histogram(
                x=df[value1_col],
                name=value1_col,
                marker_color=self.colors["primary"],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=df[value2_col],
                name=value2_col,
                marker_color=self.colors["secondary"],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Distribution Comparison',
            barmode='overlay',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def plot_correlation(self, df, value1_col, value2_col):
        """
        Plot correlation analysis between two datasets.
        
        Args:
            df (DataFrame): DataFrame containing the data
            value1_col (str): Column containing first values
            value2_col (str): Column containing second values
            
        Returns:
            Figure: Plotly figure object
        """
        # Calculate regression line
        x = df[value1_col]
        y = df[value2_col]
        
        # Handle non-numeric data
        if not pd.api.types.is_numeric_dtype(x) or not pd.api.types.is_numeric_dtype(y):
            x = pd.to_numeric(x, errors='coerce')
            y = pd.to_numeric(y, errors='coerce')
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Create the correlation plot
        fig = px.scatter(
            df, x=value1_col, y=value2_col,
            trendline="ols",
            trendline_color_override=self.colors["tertiary"]
        )
        
        # Add correlation coefficient
        correlation = x.corr(y)
        
        # Update layout
        fig.update_layout(
            title=f'Correlation Analysis (r = {correlation:.4f})',
            xaxis_title=value1_col,
            yaxis_title=value2_col,
            template="plotly_white"
        )
        
        return fig
    
    def plot_error_analysis(self, df, key_col, error_col):
        """
        Plot error analysis.
        
        Args:
            df (DataFrame): DataFrame containing the data
            key_col (str): Column containing the keys/labels
            error_col (str): Column containing the error/difference metric
            
        Returns:
            Figure: Plotly figure object
        """
        # Sort by error magnitude for better visualization
        sorted_df = df.sort_values(by=error_col, key=abs)
        
        # Create the error bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sorted_df[key_col],
            y=sorted_df[error_col],
            marker_color=np.where(sorted_df[error_col] >= 0, self.colors["primary"], self.colors["quaternary"]),
            opacity=0.8
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Error Analysis ({error_col})',
            xaxis_title='Data Points',
            yaxis_title=error_col,
            template="plotly_white",
            xaxis=dict(
                tickmode='auto',
                nticks=20,
                tickangle=45
            )
        )
        
        # Add a reference line at y=0
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=0,
            xref="paper",
            line=dict(
                color="black",
                width=1.5,
                dash="dot",
            )
        )
        
        return fig
    
    def plot_training_history(self, history):
        """
        Plot training history for model training.
        
        Args:
            history (dict): Dictionary containing training history
            
        Returns:
            Figure: Plotly figure object
        """
        # Create a figure with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces for loss
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(history['loss']) + 1)),
                y=history['loss'],
                name="Training Loss",
                line=dict(color=self.colors["primary"], width=2)
            ),
            secondary_y=False,
        )
        
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history['val_loss']) + 1)),
                    y=history['val_loss'],
                    name="Validation Loss",
                    line=dict(color=self.colors["quaternary"], width=2, dash='dash')
                ),
                secondary_y=False,
            )
        
        # Add traces for accuracy if available
        if 'accuracy' in history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history['accuracy']) + 1)),
                    y=history['accuracy'],
                    name="Training Accuracy",
                    line=dict(color=self.colors["secondary"], width=2)
                ),
                secondary_y=True,
            )
            
            if 'val_accuracy' in history:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history['val_accuracy']) + 1)),
                        y=history['val_accuracy'],
                        name="Validation Accuracy",
                        line=dict(color=self.colors["tertiary"], width=2, dash='dash')
                    ),
                    secondary_y=True,
                )
        
        # Update layout
        fig.update_layout(
            title='Training History',
            xaxis_title='Epoch',
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Loss", secondary_y=False)
        fig.update_yaxes(title_text="Accuracy", secondary_y=True)
        
        return fig
