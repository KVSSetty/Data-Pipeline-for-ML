"""
Data Validation and Quality Checker

This module provides comprehensive data validation and quality assessment functionality
for the data preparation pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation and quality assessment class
    """
    
    def __init__(self):
        self.validation_results = {}
        self.quality_score = 0.0
        self.recommendations = []
        
    def validate_data_structure(self, data: pd.DataFrame, 
                              expected_columns: Optional[List[str]] = None,
                              expected_dtypes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Validate the structure of the dataset
        
        Args:
            data: DataFrame to validate
            expected_columns: List of expected column names
            expected_dtypes: Dictionary of expected data types
            
        Returns:
            Dict containing validation results
        """
        logger.info("Validating data structure")
        
        results = {
            'valid': True,
            'issues': [],
            'summary': {
                'total_columns': len(data.columns),
                'total_rows': len(data),
                'column_names': list(data.columns),
                'data_types': data.dtypes.to_dict()
            }
        }
        
        # Check for empty dataset
        if data.empty:
            results['valid'] = False
            results['issues'].append("Dataset is empty")
            
        # Check expected columns
        if expected_columns:
            missing_columns = set(expected_columns) - set(data.columns)
            extra_columns = set(data.columns) - set(expected_columns)
            
            if missing_columns:
                results['valid'] = False
                results['issues'].append(f"Missing columns: {list(missing_columns)}")
                
            if extra_columns:
                results['issues'].append(f"Unexpected columns: {list(extra_columns)}")
                
        # Check expected data types
        if expected_dtypes:
            for col, expected_dtype in expected_dtypes.items():
                if col in data.columns:
                    actual_dtype = str(data[col].dtype)
                    if expected_dtype not in actual_dtype:
                        results['issues'].append(
                            f"Column '{col}' has type '{actual_dtype}', expected '{expected_dtype}'"
                        )
                        
        # Check for duplicate column names
        if len(data.columns) != len(set(data.columns)):
            results['valid'] = False
            results['issues'].append("Duplicate column names found")
            
        self.validation_results['structure'] = results
        return results
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess the quality of the dataset
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Dict containing quality assessment results
        """
        logger.info("Assessing data quality")
        
        results = {
            'missing_values': self._check_missing_values(data),
            'duplicates': self._check_duplicates(data),
            'outliers': self._check_outliers(data),
            'data_types': self._check_data_types(data),
            'value_ranges': self._check_value_ranges(data),
            'consistency': self._check_consistency(data)
        }
        
        # Calculate overall quality score
        self.quality_score = self._calculate_quality_score(results)
        results['quality_score'] = self.quality_score
        
        # Generate recommendations
        self.recommendations = self._generate_recommendations(results)
        results['recommendations'] = self.recommendations
        
        self.validation_results['quality'] = results
        return results
    
    def _check_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in the dataset"""
        missing_count = data.isnull().sum()
        missing_percentage = (missing_count / len(data)) * 100
        
        return {
            'total_missing': missing_count.sum(),
            'missing_by_column': missing_count.to_dict(),
            'missing_percentage_by_column': missing_percentage.to_dict(),
            'columns_with_missing': missing_count[missing_count > 0].index.tolist(),
            'high_missing_columns': missing_percentage[missing_percentage > 50].index.tolist()
        }
    
    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows in the dataset"""
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(data)) * 100
        
        return {
            'total_duplicates': duplicate_count,
            'duplicate_percentage': duplicate_percentage,
            'has_duplicates': duplicate_count > 0
        }
    
    def _check_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for outliers in numeric columns"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_percentage = (outliers / len(data)) * 100
            
            outlier_info[col] = {
                'count': outliers,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return {
            'outlier_info': outlier_info,
            'columns_with_outliers': [col for col, info in outlier_info.items() if info['count'] > 0]
        }
    
    def _check_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data types and identify potential issues"""
        type_info = {}
        issues = []
        
        for col in data.columns:
            dtype = str(data[col].dtype)
            unique_count = data[col].nunique()
            
            type_info[col] = {
                'dtype': dtype,
                'unique_values': unique_count,
                'sample_values': data[col].dropna().head(5).tolist()
            }
            
            # Check for potential type issues
            if dtype == 'object':
                # Check if numeric values are stored as strings
                non_null_values = data[col].dropna()
                if len(non_null_values) > 0:
                    try:
                        pd.to_numeric(non_null_values.head(100))
                        issues.append(f"Column '{col}' appears to contain numeric values stored as strings")
                    except (ValueError, TypeError):
                        pass
                        
                # Check for date-like strings
                if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    issues.append(f"Column '{col}' might contain date/time values that should be converted")
        
        return {
            'type_info': type_info,
            'potential_issues': issues
        }
    
    def _check_value_ranges(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check value ranges for numeric columns"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        range_info = {}
        
        for col in numeric_columns:
            range_info[col] = {
                'min': data[col].min(),
                'max': data[col].max(),
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'has_negative': (data[col] < 0).any(),
                'has_zero': (data[col] == 0).any()
            }
        
        return range_info
    
    def _check_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data consistency issues"""
        issues = []
        
        # Check for inconsistent string formats
        string_columns = data.select_dtypes(include=['object']).columns
        for col in string_columns:
            unique_values = data[col].dropna().unique()
            if len(unique_values) > 1:
                # Check for case inconsistencies
                lower_values = [str(v).lower() for v in unique_values]
                if len(set(lower_values)) < len(unique_values):
                    issues.append(f"Column '{col}' has case inconsistencies")
                
                # Check for whitespace issues
                stripped_values = [str(v).strip() for v in unique_values]
                if len(set(stripped_values)) < len(unique_values):
                    issues.append(f"Column '{col}' has leading/trailing whitespace issues")
        
        # Check for date consistency
        date_columns = data.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            if data[col].dt.year.min() < 1900 or data[col].dt.year.max() > 2100:
                issues.append(f"Column '{col}' has suspicious date values")
        
        return {
            'consistency_issues': issues,
            'has_consistency_issues': len(issues) > 0
        }
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate an overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct points for missing values
        missing_info = results['missing_values']
        if missing_info['total_missing'] > 0:
            missing_penalty = min(30, (missing_info['total_missing'] / len(missing_info['missing_by_column'])) * 10)
            score -= missing_penalty
        
        # Deduct points for duplicates
        duplicate_info = results['duplicates']
        if duplicate_info['has_duplicates']:
            duplicate_penalty = min(20, duplicate_info['duplicate_percentage'])
            score -= duplicate_penalty
        
        # Deduct points for outliers
        outlier_info = results['outliers']
        if outlier_info['columns_with_outliers']:
            outlier_penalty = min(15, len(outlier_info['columns_with_outliers']) * 3)
            score -= outlier_penalty
        
        # Deduct points for data type issues
        type_info = results['data_types']
        if type_info['potential_issues']:
            type_penalty = min(10, len(type_info['potential_issues']) * 2)
            score -= type_penalty
        
        # Deduct points for consistency issues
        consistency_info = results['consistency']
        if consistency_info['has_consistency_issues']:
            consistency_penalty = min(10, len(consistency_info['consistency_issues']) * 2)
            score -= consistency_penalty
        
        return max(0, score)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Missing values recommendations
        missing_info = results['missing_values']
        if missing_info['high_missing_columns']:
            recommendations.append(
                f"Consider removing columns with >50% missing values: {missing_info['high_missing_columns']}"
            )
        elif missing_info['columns_with_missing']:
            recommendations.append(
                "Handle missing values using appropriate imputation strategies"
            )
        
        # Duplicate recommendations
        if results['duplicates']['has_duplicates']:
            recommendations.append("Remove duplicate rows to improve data quality")
        
        # Outlier recommendations
        outlier_info = results['outliers']
        if outlier_info['columns_with_outliers']:
            recommendations.append(
                f"Investigate outliers in columns: {outlier_info['columns_with_outliers']}"
            )
        
        # Data type recommendations
        type_issues = results['data_types']['potential_issues']
        if type_issues:
            recommendations.extend(type_issues)
        
        # Consistency recommendations
        consistency_issues = results['consistency']['consistency_issues']
        if consistency_issues:
            recommendations.extend(consistency_issues)
        
        return recommendations
    
    def generate_quality_report(self, data: pd.DataFrame, output_format: str = 'dict') -> Any:
        """
        Generate a comprehensive data quality report
        
        Args:
            data: DataFrame to assess
            output_format: Format of the report ('dict', 'json', 'html')
            
        Returns:
            Quality report in the specified format
        """
        logger.info("Generating comprehensive data quality report")
        
        # Run all validations
        structure_results = self.validate_data_structure(data)
        quality_results = self.validate_data_quality(data)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'shape': data.shape,
                'columns': list(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'structure_validation': structure_results,
            'quality_assessment': quality_results,
            'overall_quality_score': self.quality_score,
            'recommendations': self.recommendations
        }
        
        if output_format == 'dict':
            return report
        elif output_format == 'json':
            import json
            return json.dumps(report, indent=2, default=str)
        elif output_format == 'html':
            return self._generate_html_report(report)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate an HTML version of the quality report"""
        html = f"""
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                ul {{ margin: 0; padding-left: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p>Generated: {report['timestamp']}</p>
                <p>Dataset Shape: {report['dataset_info']['shape']}</p>
                <p class="score">Overall Quality Score: 
                    <span class="{'good' if report['overall_quality_score'] >= 80 else 'warning' if report['overall_quality_score'] >= 60 else 'error'}">
                        {report['overall_quality_score']:.1f}/100
                    </span>
                </p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in report['recommendations'])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Missing Values</h2>
                <p>Total Missing: {report['quality_assessment']['missing_values']['total_missing']}</p>
                <p>Columns with Missing Values: {report['quality_assessment']['missing_values']['columns_with_missing']}</p>
            </div>
            
            <div class="section">
                <h2>Duplicates</h2>
                <p>Total Duplicates: {report['quality_assessment']['duplicates']['total_duplicates']}</p>
                <p>Duplicate Percentage: {report['quality_assessment']['duplicates']['duplicate_percentage']:.2f}%</p>
            </div>
            
            <div class="section">
                <h2>Outliers</h2>
                <p>Columns with Outliers: {report['quality_assessment']['outliers']['columns_with_outliers']}</p>
            </div>
        </body>
        </html>
        """
        return html