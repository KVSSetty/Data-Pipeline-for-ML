"""
Example Usage of the Simple Data Preparation Pipeline

This script shows simple, practical examples of how to use the pipeline for common data preparation tasks.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from pipeline import DataPreparationPipeline
from validation import DataValidator
from visualization import DataVisualizer


def basic_data_cleaning_example():
    """
    Basic example: Load data, clean it, and save results
    """
    print("=== BASIC DATA CLEANING EXAMPLE ===")
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline()
    
    # Load data
    data = pipeline.load_data("data/customers.csv")
    print(f"Original data shape: {data.shape}")
    
    # Clean data
    pipeline.handle_missing_values(strategy='drop')
    pipeline.remove_duplicates()
    print(f"After cleaning shape: {pipeline.data.shape}")
    
    # Save cleaned data
    pipeline.save_processed_data("data/cleaned_customers.csv")
    print("Cleaned data saved to data/cleaned_customers.csv")
    
    # Get processing summary
    summary = pipeline.get_preprocessing_summary()
    print(f"Processing steps: {len(summary['preprocessing_steps'])}")
    print()


def data_validation_example():
    """
    Example: Validate data quality and generate report
    """
    print("=== DATA VALIDATION EXAMPLE ===")
    
    # Initialize validator
    validator = DataValidator()
    
    # Load data for validation
    pipeline = DataPreparationPipeline()
    data = pipeline.load_data("data/customers.csv")
    
    # Validate data quality
    quality_results = validator.validate_data_quality(data)
    print(f"Data quality score: {quality_results['quality_score']:.1f}/100")
    
    # Show recommendations
    print("Top 3 recommendations:")
    for i, rec in enumerate(quality_results['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    # Generate and save quality report
    report = validator.generate_quality_report(data, output_format='json')
    with open("data/example_quality_report.json", "w") as f:
        f.write(report)
    print("Quality report saved to data/example_quality_report.json")
    print()


def feature_engineering_example():
    """
    Example: Create new features from existing data
    """
    print("=== FEATURE ENGINEERING EXAMPLE ===")
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline()
    
    # Load and prepare data
    data = pipeline.load_data("data/customers.csv")
    pipeline.handle_missing_values(strategy='fill')
    
    # Create new features
    feature_configs = [
        # Square the age column
        {'name': 'age_squared', 'type': 'polynomial', 'params': {'column': 'age', 'degree': 2}},
        
        # Create interaction between age and income
        {'name': 'age_income_interaction', 'type': 'interaction', 'params': {'columns': ['age', 'income']}},
        
        # Create income categories
        {'name': 'income_category', 'type': 'binning', 'params': {
            'column': 'income', 
            'bins': [0, 50000, 100000, 200000], 
            'labels': ['Low', 'Medium', 'High']
        }},
        
        # Aggregate features
        {'name': 'age_income_sum', 'type': 'aggregate', 'params': {
            'columns': ['age', 'income'], 
            'operation': 'sum'
        }}
    ]
    
    pipeline.create_features(feature_configs)
    print(f"Original columns: {data.shape[1]}")
    print(f"After feature engineering: {pipeline.data.shape[1]}")
    
    # Show new feature names
    new_features = [config['name'] for config in feature_configs]
    print(f"New features created: {new_features}")
    print()


def categorical_encoding_example():
    """
    Example: Demonstrate proper ordinal vs nominal encoding
    """
    print("=== CATEGORICAL ENCODING EXAMPLE ===")
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline()
    
    # Load and prepare data
    data = pipeline.load_data("data/customers.csv")
    pipeline.handle_missing_values(strategy='fill')
    
    print(f"Original data shape: {data.shape}")
    print(f"Categorical columns: {data.select_dtypes(include=['object']).columns.tolist()}")
    
    # Example 1: Auto-detect with ordinal mappings
    print("\n1. Auto-detect method with ordinal mappings:")
    temp_pipeline = DataPreparationPipeline()
    temp_pipeline.data = data.copy()
    
    # Define ordinal mappings for columns that have natural order
    ordinal_mappings = {
        # Example: if we had education levels
        # 'education': ['Elementary', 'High School', 'Bachelor', 'Master', 'PhD'],
        # Example: if we had size categories
        # 'size': ['S', 'M', 'L', 'XL', 'XXL']
    }
    
    temp_pipeline.encode_categorical_variables(
        method='auto',
        ordinal_mappings=ordinal_mappings
    )
    print(f"  Result shape: {temp_pipeline.data.shape}")
    
    # Example 2: Explicit one-hot encoding for nominal variables
    print("\n2. One-hot encoding for nominal variables:")
    temp_pipeline = DataPreparationPipeline()
    temp_pipeline.data = data.copy()
    
    # Get categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    temp_pipeline.encode_categorical_variables(
        columns=cat_cols,
        method='onehot'
    )
    print(f"  Result shape: {temp_pipeline.data.shape}")
    
    # Example 3: Demonstrate ordinal encoding with custom order
    print("\n3. Ordinal encoding example (simulated data):")
    # Create sample data with ordinal categories
    import pandas as pd
    sample_data = pd.DataFrame({
        'size': ['S', 'M', 'L', 'XL', 'M', 'L', 'S', 'XXL'],
        'grade': ['A', 'B', 'C', 'A', 'B', 'A', 'C', 'B'],
        'satisfaction': ['Low', 'Medium', 'High', 'Medium', 'High', 'Low', 'Medium', 'High']
    })
    
    temp_pipeline = DataPreparationPipeline()
    temp_pipeline.data = sample_data.copy()
    
    # Define proper ordinal mappings
    ordinal_mappings = {
        'size': ['S', 'M', 'L', 'XL', 'XXL'],  # Small to Extra Extra Large
        'grade': ['F', 'D', 'C', 'B', 'A'],    # Failing to Excellent
        'satisfaction': ['Low', 'Medium', 'High']  # Low to High satisfaction
    }
    
    temp_pipeline.encode_categorical_variables(
        method='ordinal',
        ordinal_mappings=ordinal_mappings
    )
    
    print(f"  Original: {sample_data.shape}")
    print(f"  After ordinal encoding: {temp_pipeline.data.shape}")
    print(f"  Sample encoded values:")
    print(f"    Size 'S'→{temp_pipeline.data['size'].iloc[0]}, 'M'→{temp_pipeline.data['size'].iloc[1]}, 'L'→{temp_pipeline.data['size'].iloc[2]}")
    print(f"    Grade 'A'→{temp_pipeline.data['grade'].iloc[0]}, 'B'→{temp_pipeline.data['grade'].iloc[1]}, 'C'→{temp_pipeline.data['grade'].iloc[2]}")
    
    print()


def standardization_example():
    """
    Example: Standardize data using different methods
    """
    print("=== DATA STANDARDIZATION EXAMPLE ===")
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline()
    
    # Load and prepare data
    data = pipeline.load_data("data/customers.csv")
    pipeline.handle_missing_values(strategy='fill')
    
    # Show original data statistics
    numeric_cols = data.select_dtypes(include=['number']).columns
    print(f"Numeric columns: {list(numeric_cols)}")
    
    # Try different standardization methods
    methods = ['minmax', 'standard', 'robust']
    
    for method in methods:
        # Create a copy for each method
        temp_pipeline = DataPreparationPipeline()
        temp_pipeline.data = data.copy()
        
        # Apply standardization
        temp_pipeline.standardize_data(method=method)
        
        # Show results for first numeric column
        col = numeric_cols[0]
        standardized_col = temp_pipeline.data[col]
        print(f"Method '{method}' for {col}: min={standardized_col.min():.3f}, max={standardized_col.max():.3f}, mean={standardized_col.mean():.3f}")
    
    print()


def complete_pipeline_example():
    """
    Example: Complete pipeline workflow
    """
    print("=== COMPLETE PIPELINE EXAMPLE ===")
    
    # Initialize components
    pipeline = DataPreparationPipeline()
    validator = DataValidator()
    
    # Step 1: Load data
    data = pipeline.load_data("data/customers.csv")
    print(f"1. Data loaded: {data.shape}")
    
    # Step 2: Validate data quality
    quality_results = validator.validate_data_quality(data)
    print(f"2. Data quality score: {quality_results['quality_score']:.1f}/100")
    
    # Step 3: Clean data
    pipeline.handle_missing_values(strategy='drop')
    pipeline.remove_duplicates()
    print(f"3. After cleaning: {pipeline.data.shape}")
    
    # Step 4: Handle outliers
    pipeline.handle_outliers(method='iqr', threshold=2.0)
    print(f"4. After outlier removal: {pipeline.data.shape}")
    
    # Step 5: Encode categorical variables
    pipeline.encode_categorical_variables(method='onehot')
    print(f"5. After encoding: {pipeline.data.shape}")
    
    # Step 6: Standardize data
    pipeline.standardize_data(method='standard')
    print(f"6. Data standardized")
    
    # Step 7: Create features
    feature_configs = [
        {'name': 'age_income_ratio', 'type': 'interaction', 'params': {'columns': ['age', 'income']}}
    ]
    pipeline.create_features(feature_configs)
    print(f"7. After feature creation: {pipeline.data.shape}")
    
    # Step 8: Save processed data
    pipeline.save_processed_data("data/complete_pipeline_output.csv")
    print("8. Processed data saved")
    
    # Step 9: Generate summary
    summary = pipeline.get_preprocessing_summary()
    print(f"9. Processing completed with {len(summary['preprocessing_steps'])} steps")
    print()


def main():
    """
    Run all examples
    """
    print("Simple Data Preparation Pipeline - Example Usage")
    print("=" * 60)
    print()
    
    # Run all examples
    basic_data_cleaning_example()
    data_validation_example()
    feature_engineering_example()
    categorical_encoding_example()
    standardization_example()
    complete_pipeline_example()
    
    print("All examples completed successfully!")
    print("Check the data/ directory for output files.")


if __name__ == "__main__":
    main()