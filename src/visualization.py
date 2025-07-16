"""
Data Visualization Module

This module provides comprehensive data visualization capabilities for the data preparation pipeline.
It includes functions for exploratory data analysis, data quality visualization, and pipeline results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings

# Configure matplotlib and seaborn
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataVisualizer:
    """
    Comprehensive data visualization class for data preparation pipeline
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the data visualizer
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        
    def plot_missing_values(self, data: pd.DataFrame, title: str = "Missing Values Analysis") -> None:
        """
        Create visualizations for missing values in the dataset
        
        Args:
            data: DataFrame to analyze
            title: Title for the plot
        """
        logger.info("Creating missing values visualization")
        
        # Calculate missing values
        missing_count = data.isnull().sum()
        missing_percentage = (missing_count / len(data)) * 100
        
        # Filter columns with missing values
        missing_data = pd.DataFrame({
            'Column': missing_count.index,
            'Missing_Count': missing_count.values,
            'Missing_Percentage': missing_percentage.values
        })
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if missing_data.empty:
            print("No missing values found in the dataset!")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Bar plot of missing counts
        axes[0, 0].bar(missing_data['Column'], missing_data['Missing_Count'], 
                       color=self.color_palette[0], alpha=0.7)
        axes[0, 0].set_title('Missing Values Count by Column')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Bar plot of missing percentages
        axes[0, 1].bar(missing_data['Column'], missing_data['Missing_Percentage'], 
                       color=self.color_palette[1], alpha=0.7)
        axes[0, 1].set_title('Missing Values Percentage by Column')
        axes[0, 1].set_ylabel('Percentage (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Heatmap of missing values
        sns.heatmap(data.isnull(), cbar=True, cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('Missing Values Heatmap')
        axes[1, 0].set_xlabel('Columns')
        axes[1, 0].set_ylabel('Rows')
        
        # 4. Missing values matrix
        missing_matrix = data.isnull().astype(int)
        axes[1, 1].imshow(missing_matrix.T, cmap='Blues', aspect='auto')
        axes[1, 1].set_title('Missing Values Matrix')
        axes[1, 1].set_xlabel('Rows')
        axes[1, 1].set_ylabel('Columns')
        
        plt.tight_layout()
        plt.show()
        
    def plot_data_distribution(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                              max_columns: int = 6) -> None:
        """
        Create distribution plots for numeric columns
        
        Args:
            data: DataFrame to analyze
            columns: Specific columns to plot (if None, all numeric columns)
            max_columns: Maximum number of columns to plot
        """
        logger.info("Creating data distribution plots")
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        columns = columns[:max_columns]  # Limit number of columns
        
        if not columns:
            print("No numeric columns found for distribution plotting!")
            return
        
        # Calculate subplot dimensions
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            # Create histogram with KDE
            sns.histplot(data[col].dropna(), kde=True, ax=axes[i], 
                        color=self.color_palette[i % len(self.color_palette)])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            
            # Add statistics text
            stats_text = f'Mean: {data[col].mean():.2f}\\nStd: {data[col].std():.2f}\\nMedian: {data[col].median():.2f}'
            axes[i].text(0.7, 0.95, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
    def plot_outliers(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                     max_columns: int = 6) -> None:
        """
        Create box plots to visualize outliers
        
        Args:
            data: DataFrame to analyze
            columns: Specific columns to plot (if None, all numeric columns)
            max_columns: Maximum number of columns to plot
        """
        logger.info("Creating outlier visualization")
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        columns = columns[:max_columns]  # Limit number of columns
        
        if not columns:
            print("No numeric columns found for outlier plotting!")
            return
        
        # Calculate subplot dimensions
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            # Create box plot
            sns.boxplot(y=data[col], ax=axes[i], color=self.color_palette[i % len(self.color_palette)])
            axes[i].set_title(f'Outliers in {col}')
            axes[i].set_ylabel(col)
            
            # Calculate and display outlier statistics
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            outlier_count = len(outliers)
            
            axes[i].text(0.02, 0.98, f'Outliers: {outlier_count}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "Correlation Matrix") -> None:
        """
        Create a correlation matrix heatmap
        
        Args:
            data: DataFrame to analyze
            title: Title for the plot
        """
        logger.info("Creating correlation matrix")
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            print("No numeric columns found for correlation analysis!")
            return
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_categorical_distribution(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                                    max_columns: int = 4) -> None:
        """
        Create distribution plots for categorical columns
        
        Args:
            data: DataFrame to analyze
            columns: Specific columns to plot (if None, all object columns)
            max_columns: Maximum number of columns to plot
        """
        logger.info("Creating categorical distribution plots")
        
        if columns is None:
            columns = data.select_dtypes(include=['object']).columns.tolist()
        
        columns = columns[:max_columns]  # Limit number of columns
        
        if not columns:
            print("No categorical columns found for distribution plotting!")
            return
        
        # Calculate subplot dimensions
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        fig.suptitle('Categorical Distribution Analysis', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            # Get value counts (limit to top 10 categories)
            value_counts = data[col].value_counts().head(10)
            
            # Create bar plot
            axes[i].bar(range(len(value_counts)), value_counts.values, 
                       color=self.color_palette[i % len(self.color_palette)])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            # Add percentage labels on bars
            total = value_counts.sum()
            for j, v in enumerate(value_counts.values):
                axes[i].text(j, v + total * 0.01, f'{v}\\n({v/total*100:.1f}%)', 
                           ha='center', va='bottom', fontsize=9)
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
    def plot_data_quality_summary(self, quality_results: Dict[str, Any]) -> None:
        """
        Create a comprehensive data quality summary visualization
        
        Args:
            quality_results: Results from DataValidator.validate_data_quality()
        """
        logger.info("Creating data quality summary visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Summary', fontsize=16, fontweight='bold')
        
        # 1. Quality Score Gauge
        score = quality_results['quality_score']
        colors = ['red' if score < 60 else 'orange' if score < 80 else 'green']
        
        axes[0, 0].pie([score, 100-score], labels=['Score', ''], colors=colors + ['lightgray'],
                      startangle=90, counterclock=False)
        axes[0, 0].set_title(f'Overall Quality Score: {score:.1f}/100')
        
        # Add score text in center
        circle = plt.Circle((0, 0), 0.7, color='white')
        axes[0, 0].add_patch(circle)
        axes[0, 0].text(0, 0, f'{score:.1f}', ha='center', va='center', fontsize=20, fontweight='bold')
        
        # 2. Missing Values by Column
        missing_data = quality_results['missing_values']['missing_percentage_by_column']
        missing_data = {k: v for k, v in missing_data.items() if v > 0}
        
        if missing_data:
            columns = list(missing_data.keys())[:10]  # Top 10
            percentages = [missing_data[col] for col in columns]
            
            axes[0, 1].barh(columns, percentages, color=self.color_palette[1])
            axes[0, 1].set_title('Missing Values by Column (%)')
            axes[0, 1].set_xlabel('Percentage')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                          transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title('Missing Values by Column (%)')
        
        # 3. Data Quality Issues Count
        issues_count = {
            'Missing Values': quality_results['missing_values']['total_missing'],
            'Duplicates': quality_results['duplicates']['total_duplicates'],
            'Outliers': sum(info['count'] for info in quality_results['outliers']['outlier_info'].values()),
            'Type Issues': len(quality_results['data_types']['potential_issues']),
            'Consistency Issues': len(quality_results['consistency']['consistency_issues'])
        }
        
        issues_count = {k: v for k, v in issues_count.items() if v > 0}
        
        if issues_count:
            axes[1, 0].bar(issues_count.keys(), issues_count.values(), color=self.color_palette[2])
            axes[1, 0].set_title('Data Quality Issues Count')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Quality Issues', ha='center', va='center', 
                          transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Data Quality Issues Count')
        
        # 4. Recommendations
        recommendations = quality_results['recommendations']
        if recommendations:
            axes[1, 1].text(0.05, 0.95, 'Recommendations:', transform=axes[1, 1].transAxes, 
                          fontsize=12, fontweight='bold', verticalalignment='top')
            
            text_content = '\\n'.join([f'• {rec}' for rec in recommendations[:5]])  # Top 5
            axes[1, 1].text(0.05, 0.85, text_content, transform=axes[1, 1].transAxes, 
                          fontsize=10, verticalalignment='top', wrap=True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Recommendations', ha='center', va='center', 
                          transform=axes[1, 1].transAxes, fontsize=14)
        
        axes[1, 1].set_title('Quality Recommendations')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def create_interactive_dashboard(self, data: pd.DataFrame, quality_results: Dict[str, Any]) -> None:
        """
        Create an interactive dashboard using Plotly
        
        Args:
            data: DataFrame to visualize
            quality_results: Results from DataValidator.validate_data_quality()
        """
        logger.info("Creating interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Overview', 'Missing Values', 'Data Quality Score', 'Distribution'),
            specs=[[{"type": "table"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "histogram"}]]
        )
        
        # 1. Data Overview Table
        overview_data = [
            ['Shape', f'{data.shape[0]} rows × {data.shape[1]} columns'],
            ['Memory Usage', f'{data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB'],
            ['Missing Values', f'{data.isnull().sum().sum()}'],
            ['Duplicates', f'{data.duplicated().sum()}'],
            ['Numeric Columns', f'{len(data.select_dtypes(include=[np.number]).columns)}'],
            ['Categorical Columns', f'{len(data.select_dtypes(include=["object"]).columns)}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='paleturquoise'),
                cells=dict(values=list(zip(*overview_data)), fill_color='lavender')
            ),
            row=1, col=1
        )
        
        # 2. Missing Values Bar Chart
        missing_data = quality_results['missing_values']['missing_percentage_by_column']
        missing_data = {k: v for k, v in missing_data.items() if v > 0}
        
        if missing_data:
            fig.add_trace(
                go.Bar(
                    x=list(missing_data.keys()),
                    y=list(missing_data.values()),
                    name='Missing %',
                    marker_color='lightcoral'
                ),
                row=1, col=2
            )
        
        # 3. Quality Score Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_results['quality_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quality Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightcoral"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=1
        )
        
        # 4. Sample Distribution
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sample_col = numeric_cols[0]
            fig.add_trace(
                go.Histogram(
                    x=data[sample_col].dropna(),
                    name=sample_col,
                    marker_color='lightblue'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Data Quality Dashboard",
            showlegend=False,
            height=800
        )
        
        # Show the dashboard
        fig.show()
        
    def save_plots(self, data: pd.DataFrame, quality_results: Dict[str, Any], 
                   output_dir: str = "plots") -> None:
        """
        Save all plots to files
        
        Args:
            data: DataFrame to visualize
            quality_results: Results from DataValidator.validate_data_quality()
            output_dir: Directory to save plots
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving plots to {output_dir}")
        
        # Save missing values plot
        try:
            self.plot_missing_values(data)
            plt.savefig(os.path.join(output_dir, 'missing_values.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not save missing values plot: {e}")
        
        # Save distribution plots
        try:
            self.plot_data_distribution(data)
            plt.savefig(os.path.join(output_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not save distribution plot: {e}")
        
        # Save outliers plot
        try:
            self.plot_outliers(data)
            plt.savefig(os.path.join(output_dir, 'outliers.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not save outliers plot: {e}")
        
        # Save correlation matrix
        try:
            self.plot_correlation_matrix(data)
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not save correlation matrix: {e}")
        
        # Save quality summary
        try:
            self.plot_data_quality_summary(quality_results)
            plt.savefig(os.path.join(output_dir, 'quality_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not save quality summary: {e}")
        
        logger.info("Plots saved successfully")