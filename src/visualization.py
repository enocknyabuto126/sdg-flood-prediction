"""
Visualization Module for Flash Flood Risk Prediction
Addresses SDG 13 (Climate Action) and SDG 11 (Sustainable Cities and Communities)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FloodRiskVisualizer:
    """
    Comprehensive visualization system for flood risk analysis
    """
    
    def __init__(self):
        self.colors = {
            'low_risk': '#2E8B57',      # Sea Green
            'medium_risk': '#FFA500',   # Orange
            'high_risk': '#DC143C',     # Crimson
            'very_high_risk': '#8B0000' # Dark Red
        }
        
    def plot_data_distribution(self, data, save_path='results/data_distribution.png'):
        """
        Create data distribution plots for the actual dataset
        """
        print("📊 Creating data distribution plots...")
        # Plot distribution of Value by GeoAreaName and TimePeriod
        fig1 = plt.figure()
        data.groupby('GeoAreaName')['Value'].sum().sort_values(ascending=False).plot(kind='bar')
        plt.title('Total People Affected by Disaster by Area')
        plt.ylabel('Total Affected')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig('results/affected_by_area.png')
        plt.show()

        fig2 = plt.figure()
        data.groupby('TimePeriod')['Value'].sum().plot()
        plt.title('Total People Affected by Disaster Over Time')
        plt.ylabel('Total Affected')
        plt.xlabel('Year')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig('results/affected_over_time.png')
        plt.show()
        print("✅ Data distribution plots saved.")
        return fig1, fig2

    def plot_correlation_matrix(self, data, save_path='results/correlation_matrix.png'):
        """
        Create correlation matrix heatmap
        
        Args:
            data: Input dataset
            save_path: Path to save the plot
        """
        print("🔗 Creating correlation matrix...")
        
        # Select numerical columns
        numerical_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix for Flood Risk Prediction', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Correlation matrix saved to {save_path}")
    
    def plot_feature_importance(self, feature_importance_df, save_path='results/feature_importance.png'):
        print("🏆 Creating feature importance plot...")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        top_features = feature_importance_df.head(10)
        plt.barh(top_features['feature'], top_features['importance'], color='lightcoral')
        plt.xlabel('Importance')
        plt.title('Top Features for Disaster Impact Prediction')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        print(f"✅ Feature importance plot saved to {save_path}")
        return fig

    def plot_model_performance(self, performance_df, save_path='results/model_performance.png'):
        print("📈 Creating model performance plot...")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        metrics = ['MAE', 'RMSE', 'R2']
        for metric in metrics:
            plt.bar(performance_df.index, performance_df[metric], label=metric)
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        print(f"✅ Model performance plot saved to {save_path}")
        return fig

    def plot_predicted_vs_actual(self, y_true, y_pred, save_path='results/pred_vs_actual.png'):
        print("🔍 Creating predicted vs actual scatter plot...")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title('Predicted vs Actual Number of People Affected')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            print(f"✅ Predicted vs actual plot saved to {save_path}")
        return fig
    
    def plot_model_performance_comparison(self, performance_df, save_path='results/model_performance.png'):
        """
        Create comprehensive model performance comparison
        
        Args:
            performance_df: DataFrame with model performance metrics
            save_path: Path to save the plot
        """
        print("📈 Creating model performance comparison...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison for Flood Risk Prediction', 
                    fontsize=16, fontweight='bold')
        
        # Metrics to plot
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        titles = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        
        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            bars = ax.bar(performance_df.index, performance_df[metric], 
                         color=color, alpha=0.8)
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Model performance comparison saved to {save_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
        """
        Create confusion matrix visualization
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save the plot
        """
        print("🔍 Creating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix for Flood Risk Prediction', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Risk Level')
        plt.ylabel('Actual Risk Level')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confusion matrix saved to {save_path}")
    
    def create_interactive_risk_map(self, data, predictions, save_path='results/interactive_risk_map.html'):
        """
        Create interactive risk map using Plotly
        
        Args:
            data: Original dataset with coordinates
            predictions: Model predictions
            save_path: Path to save the interactive map
        """
        print("🗺️ Creating interactive risk map...")
        
        # Create risk level mapping
        risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk_colors = {'Low': '#2E8B57', 'Medium': '#FFA500', 'High': '#DC143C'}
        
        # Prepare data for plotting
        plot_data = data.copy()
        plot_data['predicted_risk'] = [risk_mapping.get(pred, 'Unknown') for pred in predictions]
        plot_data['risk_color'] = plot_data['predicted_risk'].map(risk_colors)
        
        # Create interactive scatter plot
        fig = px.scatter_mapbox(
            plot_data,
            lat='latitude',
            lon='longitude',
            color='predicted_risk',
            color_discrete_map=risk_colors,
            hover_data=['town_name', 'elevation', 'rainfall_24h', 'distance_to_river'],
            title='Interactive Flood Risk Map',
            zoom=10
        )
        
        # Update layout
        fig.update_layout(
            mapbox_style="open-street-map",
            title_x=0.5,
            title_font_size=20,
            height=600
        )
        
        # Save interactive plot
        fig.write_html(save_path)
        
        print(f"✅ Interactive risk map saved to {save_path}")
    
    def plot_risk_distribution(self, predictions, save_path='results/risk_distribution.png'):
        """
        Plot distribution of predicted risk levels
        
        Args:
            predictions: Model predictions
            save_path: Path to save the plot
        """
        print("📊 Creating risk distribution plot...")
        
        # Count predictions
        risk_mapping = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        risk_labels = [risk_mapping.get(pred, 'Unknown') for pred in predictions]
        
        # Create count plot
        plt.figure(figsize=(10, 6))
        risk_counts = pd.Series(risk_labels).value_counts()
        
        colors = ['#2E8B57', '#FFA500', '#DC143C']
        bars = plt.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.8)
        
        plt.title('Distribution of Predicted Flood Risk Levels', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Towns')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Risk distribution plot saved to {save_path}")
    
    def create_comprehensive_dashboard(self, data, y_true, y_pred, performance_df, feature_importance_df, save_path='results/comprehensive_dashboard.html'):
        print("📊 Creating comprehensive dashboard...")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Model Performance (MAE, RMSE, R2)',
                'Feature Importance',
                'Predicted vs Actual',
                'Affected by Area (Top 10)'
            )
        )
        # 1. Model Performance
        for metric in ['MAE', 'RMSE', 'R2']:
            fig.add_trace(
                go.Bar(x=performance_df.index, y=performance_df[metric], name=metric),
                row=1, col=1
            )
        # 2. Feature Importance
        top_features = feature_importance_df.head(10)
        fig.add_trace(
            go.Bar(x=top_features['importance'], y=top_features['feature'], orientation='h', name='Importance'),
            row=1, col=2
        )
        # 3. Predicted vs Actual
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', name='Pred vs Actual', marker=dict(color='blue', opacity=0.5)),
            row=2, col=1
        )
        # 4. Affected by Area
        area_agg = data.groupby('GeoAreaName')['Value'].sum().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=area_agg.index, y=area_agg.values, name='Affected by Area'),
            row=2, col=2
        )
        fig.update_layout(title_text="Comprehensive Disaster Impact Analysis Dashboard", title_x=0.5, height=900)
        fig.write_html(save_path)
        print(f"✅ Comprehensive dashboard saved to {save_path}")
    
    def plot_temporal_analysis(self, data, save_path='results/temporal_analysis.png'):
        """
        Create temporal analysis plots
        
        Args:
            data: Dataset with temporal features
            save_path: Path to save the plot
        """
        print("⏰ Creating temporal analysis plots...")
        
        # Create subplots for temporal analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Analysis of Flood Risk Factors', 
                    fontsize=16, fontweight='bold')
        
        # 1. Seasonal patterns
        if 'season' in data.columns:
            seasonal_data = data.groupby('season')['rainfall_24h'].mean()
            axes[0, 0].bar(['Spring', 'Summer', 'Fall', 'Winter'], seasonal_data.values, 
                          color='skyblue', alpha=0.8)
            axes[0, 0].set_title('Average Rainfall by Season')
            axes[0, 0].set_ylabel('Average Rainfall (24h)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Monthly patterns
        if 'month' in data.columns:
            monthly_data = data.groupby('month')['rainfall_24h'].mean()
            axes[0, 1].plot(monthly_data.index, monthly_data.values, 
                           marker='o', linewidth=2, color='orange')
            axes[0, 1].set_title('Average Rainfall by Month')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Average Rainfall (24h)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Day of year patterns
        if 'day_of_year' in data.columns:
            axes[1, 0].scatter(data['day_of_year'], data['rainfall_24h'], 
                              alpha=0.6, color='green')
            axes[1, 0].set_title('Rainfall Patterns Throughout the Year')
            axes[1, 0].set_xlabel('Day of Year')
            axes[1, 0].set_ylabel('Rainfall (24h)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Temperature vs Rainfall
        axes[1, 1].scatter(data['temperature'], data['rainfall_24h'], 
                          alpha=0.6, color='purple')
        axes[1, 1].set_title('Temperature vs Rainfall Relationship')
        axes[1, 1].set_xlabel('Temperature (°C)')
        axes[1, 1].set_ylabel('Rainfall (24h)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Temporal analysis plots saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import FloodDataPreprocessor
    from supervised_models import FloodRiskPredictor
    
    # Load and preprocess data
    preprocessor = FloodDataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(
        'data/sample_data.csv',
        'data/town_characteristics.csv'
    )
    
    # Train models
    predictor = FloodRiskPredictor()
    performance_df = predictor.train_all_models(X_train, y_train, X_test, y_test, feature_names)
    
    # Load original data for visualization
    data = preprocessor.load_data('data/sample_data.csv', 'data/town_characteristics.csv')
    
    # Create visualizations
    visualizer = FloodRiskVisualizer()
    
    # Generate all visualizations
    visualizer.plot_data_distribution(data)
    visualizer.plot_correlation_matrix(data)
    visualizer.plot_feature_importance(predictor.get_feature_importance())
    visualizer.plot_model_performance_comparison(performance_df)
    visualizer.plot_risk_distribution(predictor.predict_risk(X_test)[0])
    visualizer.plot_temporal_analysis(data)
    
    # Create interactive visualizations
    predictions = predictor.predict_risk(X_test)[0]
    visualizer.create_interactive_risk_map(data, predictions)
    visualizer.create_comprehensive_dashboard(data, y_test, predictions, performance_df, 
                                            predictor.get_feature_importance()) 