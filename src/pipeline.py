"""
Data Preparation Pipeline

A comprehensive data preparation pipeline for cleaning, transforming, and preparing data
for machine learning and analysis tasks.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparationPipeline:
    """
    A comprehensive data preparation pipeline that handles:
    - Data loading from various sources
    - Data cleaning and validation
    - Data transformation and feature engineering
    - Data quality reporting
    """
    
    def __init__(self):
        self.data = None
        self.original_data = None
        self.preprocessing_steps = []
        self.data_quality_report = {}
        
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various sources (CSV, JSON, Excel, etc.)
        
        Args:
            source: Path to the data file or data source
            **kwargs: Additional parameters for pandas read functions
            
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading data from {source}")
        
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {source}")
            
        # Determine file type and load accordingly
        if file_path.suffix.lower() == '.csv':
            self.data = pd.read_csv(source, **kwargs)
        elif file_path.suffix.lower() == '.json':
            self.data = pd.read_json(source, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            self.data = pd.read_excel(source, **kwargs)
        elif file_path.suffix.lower() == '.parquet':
            self.data = pd.read_parquet(source, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Keep a copy of original data
        self.original_data = self.data.copy()
        
        logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        self.preprocessing_steps.append(f"Loaded data from {source}")
        
        return self.data
    
    def get_data_overview(self) -> Dict[str, Any]:
        """
        Generate a comprehensive overview of the dataset
        
        Returns:
            Dict containing data overview statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        overview = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
            'datetime_columns': list(self.data.select_dtypes(include=['datetime64']).columns)
        }
        
        return overview
    
    def handle_missing_values(self, strategy: str = 'drop', columns: Optional[List[str]] = None, 
                            fill_value: Any = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            strategy: Strategy for handling missing values ('drop', 'fill', 'forward_fill', 'backward_fill')
            columns: Specific columns to apply the strategy to
            fill_value: Value to use for filling (if strategy is 'fill')
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        if columns is None:
            columns = self.data.columns.tolist()
            
        original_shape = self.data.shape
        
        if strategy == 'drop':
            self.data = self.data.dropna(subset=columns)
        elif strategy == 'fill':
            if fill_value is None:
                # Use median for numeric columns, mode for categorical
                for col in columns:
                    if col in self.data.select_dtypes(include=[np.number]).columns:
                        fill_value = self.data[col].median()
                    else:
                        fill_value = self.data[col].mode().iloc[0] if not self.data[col].mode().empty else 'Unknown'
                    self.data[col] = self.data[col].fillna(fill_value)
            else:
                self.data[columns] = self.data[columns].fillna(fill_value)
        elif strategy == 'forward_fill':
            self.data[columns] = self.data[columns].fillna(method='ffill')
        elif strategy == 'backward_fill':
            self.data[columns] = self.data[columns].fillna(method='bfill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        new_shape = self.data.shape
        logger.info(f"Missing values handled. Shape changed from {original_shape} to {new_shape}")
        self.preprocessing_steps.append(f"Handled missing values with strategy: {strategy}")
        
        return self.data
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset
        
        Args:
            subset: Columns to consider for identifying duplicates
            keep: Which duplicate to keep ('first', 'last', False)
            
        Returns:
            pd.DataFrame: Data with duplicates removed
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        logger.info("Removing duplicate rows")
        
        original_shape = self.data.shape
        duplicates_count = self.data.duplicated(subset=subset).sum()
        
        self.data = self.data.drop_duplicates(subset=subset, keep=keep)
        
        new_shape = self.data.shape
        logger.info(f"Removed {duplicates_count} duplicate rows. Shape changed from {original_shape} to {new_shape}")
        self.preprocessing_steps.append(f"Removed {duplicates_count} duplicate rows")
        
        return self.data
    
    def handle_outliers(self, columns: Optional[List[str]] = None, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in numeric columns
        
        Args:
            columns: Columns to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        logger.info(f"Handling outliers using {method} method")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
        original_shape = self.data.shape
        outliers_removed = 0
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outliers_removed += outliers.sum()
                self.data = self.data[~outliers]
                
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = z_scores > threshold
                outliers_removed += outliers.sum()
                self.data = self.data[~outliers]
                
        new_shape = self.data.shape
        logger.info(f"Removed {outliers_removed} outliers. Shape changed from {original_shape} to {new_shape}")
        self.preprocessing_steps.append(f"Removed {outliers_removed} outliers using {method} method")
        
        return self.data
    
    def encode_categorical_variables(self, columns: Optional[List[str]] = None, 
                                   method: str = 'auto', 
                                   ordinal_mappings: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Encode categorical variables with proper handling of ordinal vs nominal variables
        
        Args:
            columns: Columns to encode (if None, auto-detects categorical columns)
            method: Encoding method ('auto', 'onehot', 'ordinal', 'label')
                   - 'auto': Use onehot for nominal, ordinal for ordinal (based on ordinal_mappings)
                   - 'onehot': One-hot encoding for nominal variables
                   - 'ordinal': Ordinal encoding for ordinal variables (requires ordinal_mappings)
                   - 'label': Simple label encoding (not recommended, use ordinal instead)
            ordinal_mappings: Dictionary mapping column names to ordered lists of categories
                            Example: {'size': ['S', 'M', 'L', 'XL', 'XXL'], 
                                     'grade': ['F', 'D', 'C', 'B', 'A']}
            
        Returns:
            pd.DataFrame: Data with encoded categorical variables
            
        Examples:
            # Auto-detect with ordinal mappings
            pipeline.encode_categorical_variables(
                method='auto',
                ordinal_mappings={'size': ['S', 'M', 'L', 'XL', 'XXL']}
            )
            
            # Explicit ordinal encoding
            pipeline.encode_categorical_variables(
                columns=['size'],
                method='ordinal',
                ordinal_mappings={'size': ['S', 'M', 'L', 'XL', 'XXL']}
            )
            
            # One-hot encoding for nominal variables
            pipeline.encode_categorical_variables(
                columns=['color', 'brand'],
                method='onehot'
            )
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        logger.info(f"Encoding categorical variables using {method} method")
        
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns.tolist()
            
        if ordinal_mappings is None:
            ordinal_mappings = {}
            
        original_shape = self.data.shape
        
        if method == 'auto':
            # Split columns into ordinal and nominal
            ordinal_cols = [col for col in columns if col in ordinal_mappings]
            nominal_cols = [col for col in columns if col not in ordinal_mappings]
            
            # Apply ordinal encoding to ordinal columns
            if ordinal_cols:
                self._encode_ordinal_variables(ordinal_cols, ordinal_mappings)
                logger.info(f"Applied ordinal encoding to: {ordinal_cols}")
            
            # Apply one-hot encoding to nominal columns
            if nominal_cols:
                self.data = pd.get_dummies(self.data, columns=nominal_cols, prefix=nominal_cols)
                logger.info(f"Applied one-hot encoding to: {nominal_cols}")
                
        elif method == 'onehot':
            # One-hot encoding for nominal variables
            self.data = pd.get_dummies(self.data, columns=columns, prefix=columns)
            
        elif method == 'ordinal':
            # Ordinal encoding for ordinal variables
            if not ordinal_mappings:
                raise ValueError("ordinal_mappings is required for ordinal encoding")
            
            missing_mappings = [col for col in columns if col not in ordinal_mappings]
            if missing_mappings:
                raise ValueError(f"Missing ordinal mappings for columns: {missing_mappings}")
            
            self._encode_ordinal_variables(columns, ordinal_mappings)
            
        elif method == 'label':
            # Simple label encoding (not recommended)
            logger.warning("Label encoding doesn't preserve ordinal relationships. Consider using 'ordinal' method instead.")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in columns:
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                
        else:
            raise ValueError(f"Unknown encoding method: {method}")
                
        new_shape = self.data.shape
        logger.info(f"Encoded categorical variables. Shape changed from {original_shape} to {new_shape}")
        self.preprocessing_steps.append(f"Encoded categorical variables using {method} method")
        
        return self.data
    
    def _encode_ordinal_variables(self, columns: List[str], ordinal_mappings: Dict[str, List[str]]) -> None:
        """
        Helper method to encode ordinal variables with proper ordering
        
        Args:
            columns: List of column names to encode
            ordinal_mappings: Dictionary mapping column names to ordered category lists
        """
        from sklearn.preprocessing import OrdinalEncoder
        
        for col in columns:
            if col not in ordinal_mappings:
                raise ValueError(f"Missing ordinal mapping for column: {col}")
                
            # Get the ordered categories for this column
            categories = ordinal_mappings[col]
            
            # Create ordinal encoder with specified categories
            ordinal_encoder = OrdinalEncoder(categories=[categories], handle_unknown='error')
            
            # Check if all values in the column are in the mapping
            unique_values = set(self.data[col].dropna().astype(str).unique())
            mapping_values = set(categories)
            
            if not unique_values.issubset(mapping_values):
                unknown_values = unique_values - mapping_values
                raise ValueError(f"Unknown values in column '{col}': {unknown_values}. "
                               f"Expected values: {categories}")
            
            # Apply ordinal encoding
            self.data[col] = ordinal_encoder.fit_transform(self.data[[col]].astype(str))
    
    def standardize_data(self, columns: Optional[List[str]] = None, 
                        method: str = 'minmax') -> pd.DataFrame:
        """
        Standardize numeric columns using various scaling methods
        
        Args:
            columns: Columns to standardize
            method: Standardization method ('minmax', 'standard', 'robust')
            
        Returns:
            pd.DataFrame: Data with standardized columns
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        logger.info(f"Standardizing data using {method} method")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            self.data[columns] = scaler.fit_transform(self.data[columns])
        elif method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.data[columns] = scaler.fit_transform(self.data[columns])
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            self.data[columns] = scaler.fit_transform(self.data[columns])
            
        logger.info(f"Standardized {len(columns)} columns using {method} method")
        self.preprocessing_steps.append(f"Standardized data using {method} method")
        
        return self.data
    
    def create_features(self, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create new features based on existing columns
        
        Args:
            feature_configs: List of feature configurations
            Each config should have: {'name': str, 'type': str, 'params': dict}
            
        Returns:
            pd.DataFrame: Data with new features
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        logger.info("Creating new features")
        
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config['type']
            params = config.get('params', {})
            
            if feature_type == 'polynomial':
                column = params['column']
                degree = params.get('degree', 2)
                self.data[feature_name] = self.data[column] ** degree
                
            elif feature_type == 'interaction':
                col1, col2 = params['columns']
                self.data[feature_name] = self.data[col1] * self.data[col2]
                
            elif feature_type == 'binning':
                column = params['column']
                bins = params['bins']
                labels = params.get('labels', None)
                self.data[feature_name] = pd.cut(self.data[column], bins=bins, labels=labels)
                
            elif feature_type == 'aggregate':
                columns = params['columns']
                operation = params['operation']
                if operation == 'sum':
                    self.data[feature_name] = self.data[columns].sum(axis=1)
                elif operation == 'mean':
                    self.data[feature_name] = self.data[columns].mean(axis=1)
                elif operation == 'max':
                    self.data[feature_name] = self.data[columns].max(axis=1)
                elif operation == 'min':
                    self.data[feature_name] = self.data[columns].min(axis=1)
                    
        logger.info(f"Created {len(feature_configs)} new features")
        self.preprocessing_steps.append(f"Created {len(feature_configs)} new features")
        
        return self.data
    
    def split_data(self, target_column: str, test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        logger.info(f"Splitting data into train/test sets with test_size={test_size}")
        
        from sklearn.model_selection import train_test_split
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        self.preprocessing_steps.append(f"Split data into train/test sets")
        
        return X_train, X_test, y_train, y_test
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all preprocessing steps performed
        
        Returns:
            Dict containing preprocessing summary
        """
        return {
            'original_shape': self.original_data.shape if self.original_data is not None else None,
            'current_shape': self.data.shape if self.data is not None else None,
            'preprocessing_steps': self.preprocessing_steps,
            'data_quality_report': self.data_quality_report
        }
    
    def save_processed_data(self, filepath: str, format: str = 'csv') -> None:
        """
        Save the processed data to a file
        
        Args:
            filepath: Path to save the data
            format: File format ('csv', 'json', 'excel', 'parquet')
        """
        if self.data is None:
            raise ValueError("No data to save. Please load and process data first.")
            
        logger.info(f"Saving processed data to {filepath}")
        
        if format == 'csv':
            self.data.to_csv(filepath, index=False)
        elif format == 'json':
            self.data.to_json(filepath, orient='records', indent=2)
        elif format == 'excel':
            self.data.to_excel(filepath, index=False)
        elif format == 'parquet':
            self.data.to_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Data saved successfully to {filepath}")
        self.preprocessing_steps.append(f"Saved processed data to {filepath}")