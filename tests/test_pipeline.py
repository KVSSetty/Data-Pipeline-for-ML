"""
Test suite for the data preparation pipeline
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import DataPreparationPipeline
from validation import DataValidator
from sample_data_generator import SampleDataGenerator


class TestDataPreparationPipeline:
    """Test cases for DataPreparationPipeline class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = DataPreparationPipeline()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 5],  # Duplicate
            'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve'],  # Missing value
            'age': [25, 30, 35, 40, 45, 45],
            'salary': [50000, 60000, 70000, 80000, 90000, 90000],
            'department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'HR'],
            'score': [85.5, 90.2, 78.8, 92.1, 88.7, 88.7]
        })
        
        # Create data with outliers
        self.outlier_data = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'outlier_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]  # 1000 is an outlier
        })
        
    def test_init(self):
        """Test pipeline initialization"""
        assert self.pipeline.data is None
        assert self.pipeline.original_data is None
        assert self.pipeline.preprocessing_steps == []
        assert self.pipeline.data_quality_report == {}
        
    def test_load_data_csv(self):
        """Test loading CSV data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            
            # Test loading
            result = self.pipeline.load_data(f.name)
            
            assert isinstance(result, pd.DataFrame)
            assert result.shape == self.sample_data.shape
            assert list(result.columns) == list(self.sample_data.columns)
            assert self.pipeline.original_data is not None
            assert len(self.pipeline.preprocessing_steps) == 1
            
            # Clean up
            os.unlink(f.name)
            
    def test_load_data_json(self):
        """Test loading JSON data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            self.sample_data.to_json(f.name, orient='records')
            
            # Test loading
            result = self.pipeline.load_data(f.name)
            
            assert isinstance(result, pd.DataFrame)
            assert result.shape == self.sample_data.shape
            
            # Clean up
            os.unlink(f.name)
            
    def test_load_data_file_not_found(self):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            self.pipeline.load_data('non_existent_file.csv')
            
    def test_load_data_unsupported_format(self):
        """Test loading unsupported file format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test data')
            
            with pytest.raises(ValueError):
                self.pipeline.load_data(f.name)
                
            # Clean up
            os.unlink(f.name)
            
    def test_get_data_overview(self):
        """Test data overview generation"""
        self.pipeline.data = self.sample_data
        
        overview = self.pipeline.get_data_overview()
        
        assert 'shape' in overview
        assert 'columns' in overview
        assert 'dtypes' in overview
        assert 'memory_usage' in overview
        assert 'missing_values' in overview
        assert 'duplicate_rows' in overview
        assert 'numeric_columns' in overview
        assert 'categorical_columns' in overview
        
        assert overview['shape'] == self.sample_data.shape
        assert overview['duplicate_rows'] == 1  # We have 1 duplicate
        assert 'name' in overview['missing_values']
        assert overview['missing_values']['name'] == 1
        
    def test_get_data_overview_no_data(self):
        """Test data overview with no data loaded"""
        with pytest.raises(ValueError):
            self.pipeline.get_data_overview()
            
    def test_handle_missing_values_drop(self):
        """Test handling missing values with drop strategy"""
        self.pipeline.data = self.sample_data.copy()
        original_shape = self.pipeline.data.shape
        
        result = self.pipeline.handle_missing_values(strategy='drop')
        
        assert result.shape[0] < original_shape[0]  # Should have fewer rows
        assert result.isnull().sum().sum() == 0  # No missing values
        
    def test_handle_missing_values_fill(self):
        """Test handling missing values with fill strategy"""
        self.pipeline.data = self.sample_data.copy()
        
        result = self.pipeline.handle_missing_values(strategy='fill')
        
        assert result.isnull().sum().sum() == 0  # No missing values
        assert result.shape == self.sample_data.shape  # Same shape
        
    def test_handle_missing_values_invalid_strategy(self):
        """Test handling missing values with invalid strategy"""
        self.pipeline.data = self.sample_data.copy()
        
        with pytest.raises(ValueError):
            self.pipeline.handle_missing_values(strategy='invalid')
            
    def test_handle_missing_values_no_data(self):
        """Test handling missing values with no data"""
        with pytest.raises(ValueError):
            self.pipeline.handle_missing_values()
            
    def test_remove_duplicates(self):
        """Test removing duplicate rows"""
        self.pipeline.data = self.sample_data.copy()
        original_shape = self.pipeline.data.shape
        
        result = self.pipeline.remove_duplicates()
        
        assert result.shape[0] < original_shape[0]  # Should have fewer rows
        assert result.duplicated().sum() == 0  # No duplicates
        
    def test_remove_duplicates_no_data(self):
        """Test removing duplicates with no data"""
        with pytest.raises(ValueError):
            self.pipeline.remove_duplicates()
            
    def test_handle_outliers_iqr(self):
        """Test handling outliers with IQR method"""
        self.pipeline.data = self.outlier_data.copy()
        original_shape = self.pipeline.data.shape
        
        result = self.pipeline.handle_outliers(method='iqr')
        
        assert result.shape[0] < original_shape[0]  # Should have fewer rows
        
    def test_handle_outliers_zscore(self):
        """Test handling outliers with Z-score method"""
        self.pipeline.data = self.outlier_data.copy()
        original_shape = self.pipeline.data.shape
        
        result = self.pipeline.handle_outliers(method='zscore', threshold=2.0)
        
        assert result.shape[0] < original_shape[0]  # Should have fewer rows
        
    def test_handle_outliers_no_data(self):
        """Test handling outliers with no data"""
        with pytest.raises(ValueError):
            self.pipeline.handle_outliers()
            
    def test_encode_categorical_variables_onehot(self):
        """Test one-hot encoding of categorical variables"""
        self.pipeline.data = self.sample_data.copy()
        original_shape = self.pipeline.data.shape
        
        result = self.pipeline.encode_categorical_variables(method='onehot')
        
        assert result.shape[1] > original_shape[1]  # More columns
        assert 'department_HR' in result.columns
        assert 'department_IT' in result.columns
        
    def test_encode_categorical_variables_label(self):
        """Test label encoding of categorical variables"""
        self.pipeline.data = self.sample_data.copy()
        original_shape = self.pipeline.data.shape
        
        result = self.pipeline.encode_categorical_variables(method='label')
        
        assert result.shape == original_shape  # Same shape
        assert result['department'].dtype != 'object'  # Should be numeric
        
    def test_normalize_data_minmax(self):
        """Test min-max normalization"""
        self.pipeline.data = self.sample_data.copy()
        
        result = self.pipeline.normalize_data(method='minmax')
        
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert result[col].min() >= 0
            assert result[col].max() <= 1
            
    def test_normalize_data_standard(self):
        """Test standard normalization"""
        self.pipeline.data = self.sample_data.copy()
        
        result = self.pipeline.normalize_data(method='standard')
        
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(result[col].mean()) < 1e-6  # Mean should be ~0
            # After standardization, values should be within reasonable range
            assert result[col].std() > 0.5  # Standard deviation should be close to 1
            assert result[col].std() < 2.0  # But allowing for small sample size effects
            
    def test_create_features(self):
        """Test feature creation"""
        self.pipeline.data = self.sample_data.copy()
        
        feature_configs = [
            {'name': 'age_squared', 'type': 'polynomial', 'params': {'column': 'age', 'degree': 2}},
            {'name': 'age_salary_interaction', 'type': 'interaction', 'params': {'columns': ['age', 'salary']}}
        ]
        
        result = self.pipeline.create_features(feature_configs)
        
        assert 'age_squared' in result.columns
        assert 'age_salary_interaction' in result.columns
        assert result['age_squared'].equals(result['age'] ** 2)
        assert result['age_salary_interaction'].equals(result['age'] * result['salary'])
        
    def test_split_data(self):
        """Test data splitting"""
        self.pipeline.data = self.sample_data.copy()
        
        X_train, X_test, y_train, y_test = self.pipeline.split_data('score', test_size=0.3)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        total_samples = len(X_train) + len(X_test)
        assert total_samples == len(self.sample_data)
        assert len(X_test) / total_samples == pytest.approx(0.3, rel=0.2)
        
    def test_get_preprocessing_summary(self):
        """Test preprocessing summary"""
        self.pipeline.data = self.sample_data.copy()
        self.pipeline.original_data = self.sample_data.copy()
        self.pipeline.preprocessing_steps = ['step1', 'step2']
        
        summary = self.pipeline.get_preprocessing_summary()
        
        assert 'original_shape' in summary
        assert 'current_shape' in summary
        assert 'preprocessing_steps' in summary
        assert 'data_quality_report' in summary
        
        assert summary['preprocessing_steps'] == ['step1', 'step2']
        
    def test_save_processed_data_csv(self):
        """Test saving processed data to CSV"""
        self.pipeline.data = self.sample_data.copy()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.pipeline.save_processed_data(f.name, format='csv')
            
            # Verify file was created and has correct content
            loaded_data = pd.read_csv(f.name)
            assert loaded_data.shape == self.sample_data.shape
            
            # Clean up
            os.unlink(f.name)
            
    def test_save_processed_data_no_data(self):
        """Test saving data with no data loaded"""
        with pytest.raises(ValueError):
            self.pipeline.save_processed_data('test.csv')
            
    def test_save_processed_data_invalid_format(self):
        """Test saving data with invalid format"""
        self.pipeline.data = self.sample_data.copy()
        
        with pytest.raises(ValueError):
            self.pipeline.save_processed_data('test.txt', format='invalid')


class TestDataValidator:
    """Test cases for DataValidator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],  # Missing value
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['HR', 'IT', 'Finance', 'IT', 'HR']
        })
        
    def test_init(self):
        """Test validator initialization"""
        assert self.validator.validation_results == {}
        assert self.validator.quality_score == 0.0
        assert self.validator.recommendations == []
        
    def test_validate_data_structure(self):
        """Test data structure validation"""
        result = self.validator.validate_data_structure(self.sample_data)
        
        assert 'valid' in result
        assert 'issues' in result
        assert 'summary' in result
        assert result['summary']['total_columns'] == 5
        assert result['summary']['total_rows'] == 5
        
    def test_validate_data_structure_empty_data(self):
        """Test validation with empty data"""
        empty_data = pd.DataFrame()
        result = self.validator.validate_data_structure(empty_data)
        
        assert result['valid'] == False
        assert 'Dataset is empty' in result['issues']
        
    def test_validate_data_structure_expected_columns(self):
        """Test validation with expected columns"""
        expected_columns = ['id', 'name', 'age', 'salary', 'department']
        result = self.validator.validate_data_structure(self.sample_data, expected_columns)
        
        assert result['valid'] == True
        assert len(result['issues']) == 0
        
    def test_validate_data_structure_missing_columns(self):
        """Test validation with missing expected columns"""
        expected_columns = ['id', 'name', 'age', 'salary', 'department', 'bonus']
        result = self.validator.validate_data_structure(self.sample_data, expected_columns)
        
        assert result['valid'] == False
        assert any('Missing columns' in issue for issue in result['issues'])
        
    def test_validate_data_quality(self):
        """Test data quality validation"""
        result = self.validator.validate_data_quality(self.sample_data)
        
        assert 'missing_values' in result
        assert 'duplicates' in result
        assert 'outliers' in result
        assert 'data_types' in result
        assert 'value_ranges' in result
        assert 'consistency' in result
        assert 'quality_score' in result
        assert 'recommendations' in result
        
        assert result['missing_values']['total_missing'] == 1
        assert result['duplicates']['total_duplicates'] == 0
        
    def test_generate_quality_report_dict(self):
        """Test quality report generation in dict format"""
        report = self.validator.generate_quality_report(self.sample_data, output_format='dict')
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'dataset_info' in report
        assert 'structure_validation' in report
        assert 'quality_assessment' in report
        assert 'overall_quality_score' in report
        assert 'recommendations' in report
        
    def test_generate_quality_report_json(self):
        """Test quality report generation in JSON format"""
        report = self.validator.generate_quality_report(self.sample_data, output_format='json')
        
        assert isinstance(report, str)
        import json
        parsed_report = json.loads(report)
        assert 'timestamp' in parsed_report
        
    def test_generate_quality_report_html(self):
        """Test quality report generation in HTML format"""
        report = self.validator.generate_quality_report(self.sample_data, output_format='html')
        
        assert isinstance(report, str)
        assert '<html>' in report
        assert '<title>Data Quality Report</title>' in report
        
    def test_generate_quality_report_invalid_format(self):
        """Test quality report with invalid format"""
        with pytest.raises(ValueError):
            self.validator.generate_quality_report(self.sample_data, output_format='invalid')


class TestSampleDataGenerator:
    """Test cases for SampleDataGenerator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = SampleDataGenerator(seed=42)
        
    def test_init(self):
        """Test generator initialization"""
        assert self.generator.seed == 42
        
    def test_generate_customer_data(self):
        """Test customer data generation"""
        data = self.generator.generate_customer_data(num_records=100)
        
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] >= 100  # Should have at least 100 records (duplicates may be added)
        
        expected_columns = [
            'customer_id', 'first_name', 'last_name', 'email', 'phone',
            'age', 'income', 'registration_date', 'status', 'city', 'state',
            'purchase_amount', 'loyalty_score'
        ]
        
        for col in expected_columns:
            assert col in data.columns
            
        # Check for data quality issues (intentionally added)
        assert data.isnull().sum().sum() > 0  # Should have missing values
        assert data.duplicated().sum() > 0  # Should have duplicates
        
    def test_generate_sales_data(self):
        """Test sales data generation"""
        data = self.generator.generate_sales_data(num_records=100)
        
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] >= 100  # Should have at least 100 records
        
        expected_columns = [
            'transaction_id', 'customer_id', 'product_name', 'category',
            'quantity', 'price', 'total_amount', 'sale_date', 'salesperson',
            'region', 'discount_percent', 'payment_method', 'status'
        ]
        
        for col in expected_columns:
            assert col in data.columns
            
        # Check for data quality issues
        assert data.isnull().sum().sum() > 0  # Should have missing values
        assert data.duplicated().sum() > 0  # Should have duplicates
        
    def test_generate_product_data(self):
        """Test product data generation"""
        data = self.generator.generate_product_data(num_records=50)
        
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] >= 50  # Should have at least 50 records
        
        expected_columns = [
            'product_id', 'product_name', 'category', 'price', 'cost',
            'stock_quantity', 'weight', 'dimensions', 'rating', 'review_count',
            'supplier', 'availability', 'created_date'
        ]
        
        for col in expected_columns:
            assert col in data.columns
            
        # Check for data quality issues
        assert data.isnull().sum().sum() > 0  # Should have missing values
        assert data.duplicated().sum() > 0  # Should have duplicates
        
    def test_save_sample_datasets(self):
        """Test saving sample datasets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = self.generator.save_sample_datasets(output_dir=temp_dir)
            
            assert 'customers' in file_paths
            assert 'sales' in file_paths
            assert 'products' in file_paths
            
            # Check that files were created
            for name, path in file_paths.items():
                assert os.path.exists(path)
                
                # Check that files have content
                data = pd.read_csv(path)
                assert data.shape[0] > 0
                assert data.shape[1] > 0


class TestIntegration:
    """Integration tests for the entire pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = DataPreparationPipeline()
        self.validator = DataValidator()
        self.generator = SampleDataGenerator(seed=42)
        
    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow"""
        # Generate sample data
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = self.generator.save_sample_datasets(output_dir=temp_dir)
            
            # Load data
            data = self.pipeline.load_data(file_paths['customers'])
            assert data is not None
            
            # Validate data
            quality_results = self.validator.validate_data_quality(data)
            assert quality_results['quality_score'] > 0
            
            # Clean data
            self.pipeline.handle_missing_values(strategy='drop')
            self.pipeline.remove_duplicates()
            
            # Get overview
            overview = self.pipeline.get_data_overview()
            # Check that missing values were handled (sum should be 0)
            assert sum(overview['missing_values'].values()) == 0  # Should be cleaned
            assert overview['duplicate_rows'] == 0  # Should be cleaned
            
            # Save processed data
            output_path = os.path.join(temp_dir, 'processed_customers.csv')
            self.pipeline.save_processed_data(output_path)
            assert os.path.exists(output_path)
            
            # Get summary
            summary = self.pipeline.get_preprocessing_summary()
            assert len(summary['preprocessing_steps']) > 0
            
    def test_pipeline_with_different_data_types(self):
        """Test pipeline with different data types"""
        # Test with numeric-heavy data
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = self.generator.save_sample_datasets(output_dir=temp_dir)
            
            # Test sales data (has mixed types)
            sales_data = self.pipeline.load_data(file_paths['sales'])
            
            # Handle outliers in numeric columns
            self.pipeline.handle_outliers(method='iqr', threshold=1.5)
            
            # Normalize numeric data
            self.pipeline.normalize_data(method='standard')
            
            # Encode categorical variables
            self.pipeline.encode_categorical_variables(method='onehot')
            
            # Check that transformations worked
            processed_data = self.pipeline.data
            assert processed_data is not None
            assert processed_data.shape[0] > 0
            
    def test_error_handling(self):
        """Test error handling in pipeline"""
        # Test with no data loaded
        with pytest.raises(ValueError):
            self.pipeline.handle_missing_values()
            
        with pytest.raises(ValueError):
            self.pipeline.remove_duplicates()
            
        with pytest.raises(ValueError):
            self.pipeline.get_data_overview()
            
        # Test with invalid parameters
        self.pipeline.data = pd.DataFrame({'col1': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            self.pipeline.handle_missing_values(strategy='invalid_strategy')
            
        with pytest.raises(ValueError):
            self.pipeline.save_processed_data('test.csv', format='invalid_format')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])