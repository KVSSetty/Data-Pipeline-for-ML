"""
Main demonstration script for the Simple Data Preparation Pipeline

This script demonstrates the complete functionality of the data preparation pipeline
including data loading, validation, cleaning, transformation, and visualization.
"""

import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from pipeline import DataPreparationPipeline
from validation import DataValidator
from visualization import DataVisualizer
from sample_data_generator import SampleDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_pipeline():
    """
    Demonstrate the complete data preparation pipeline workflow
    """
    print("=" * 80)
    print("SIMPLE DATA PREPARATION PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize components
    pipeline = DataPreparationPipeline()
    validator = DataValidator()
    visualizer = DataVisualizer()
    generator = SampleDataGenerator(seed=42)
    
    # Step 1: Generate sample data
    print("\n1. GENERATING SAMPLE DATA")
    print("-" * 40)
    
    try:
        file_paths = generator.save_sample_datasets(output_dir="data")
        print("✓ Sample datasets generated successfully:")
        for name, path in file_paths.items():
            print(f"  - {name}: {path}")
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return
    
    # Step 2: Load and explore data
    print("\n2. LOADING AND EXPLORING DATA")
    print("-" * 40)
    
    try:
        # Load customer data
        data = pipeline.load_data(file_paths['customers'])
        print(f"✓ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Get data overview
        overview = pipeline.get_data_overview()
        print(f"✓ Data overview generated:")
        print(f"  - Missing values: {overview['missing_values']}")
        print(f"  - Duplicate rows: {overview['duplicate_rows']}")
        print(f"  - Numeric columns: {len(overview['numeric_columns'])}")
        print(f"  - Categorical columns: {len(overview['categorical_columns'])}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Step 3: Validate data quality
    print("\n3. VALIDATING DATA QUALITY")
    print("-" * 40)
    
    try:
        # Validate data structure
        structure_results = validator.validate_data_structure(data)
        print(f"✓ Data structure validation: {'PASSED' if structure_results['valid'] else 'FAILED'}")
        if structure_results['issues']:
            print(f"  Issues found: {structure_results['issues']}")
        
        # Assess data quality
        quality_results = validator.validate_data_quality(data)
        print(f"✓ Data quality assessment completed")
        print(f"  - Overall quality score: {quality_results['quality_score']:.1f}/100")
        print(f"  - Recommendations: {len(quality_results['recommendations'])} items")
        
        # Show top recommendations
        if quality_results['recommendations']:
            print("  Top recommendations:")
            for i, rec in enumerate(quality_results['recommendations'][:3], 1):
                print(f"    {i}. {rec}")
                
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return
    
    # Step 4: Clean and transform data
    print("\n4. CLEANING AND TRANSFORMING DATA")
    print("-" * 40)
    
    try:
        original_shape = pipeline.data.shape
        
        # Handle missing values
        pipeline.handle_missing_values(strategy='drop')
        print(f"✓ Missing values handled: {original_shape[0]} → {pipeline.data.shape[0]} rows")
        
        # Remove duplicates
        pipeline.remove_duplicates()
        print(f"✓ Duplicates removed: {pipeline.data.shape[0]} rows remaining")
        
        # Handle outliers
        pipeline.handle_outliers(method='iqr', threshold=1.5)
        print(f"✓ Outliers handled: {pipeline.data.shape[0]} rows remaining")
        
        # Encode categorical variables (using auto method for demonstration)
        # In real usage, you would specify ordinal mappings for ordinal variables
        pipeline.encode_categorical_variables(method='onehot')  # Using onehot for simplicity in demo
        print(f"✓ Categorical variables encoded: {pipeline.data.shape[1]} columns")
        
        # Standardize numeric data
        numeric_cols = pipeline.data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            pipeline.standardize_data(method='standard')
            print(f"✓ Numeric data standardized: {len(numeric_cols)} columns")
        
        # Create some sample features
        feature_configs = [
            {'name': 'age_income_ratio', 'type': 'interaction', 'params': {'columns': ['age', 'income']}}
        ]
        pipeline.create_features(feature_configs)
        print(f"✓ Features created: {pipeline.data.shape[1]} total columns")
        
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        return
    
    # Step 5: Generate visualizations
    print("\n5. GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    try:
        # Note: In a real environment, these would display plots
        print("✓ Visualization capabilities available:")
        print("  - Missing values analysis")
        print("  - Data distribution plots")
        print("  - Outlier detection plots")
        print("  - Correlation matrix")
        print("  - Data quality summary")
        print("  - Interactive dashboard")
        
        # You can uncomment these lines to see actual plots (requires display)
        # visualizer.plot_missing_values(data)
        # visualizer.plot_data_distribution(data)
        # visualizer.plot_data_quality_summary(quality_results)
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
    
    # Step 6: Save processed data
    print("\n6. SAVING PROCESSED DATA")
    print("-" * 40)
    
    try:
        # Save processed data
        output_path = "data/processed_customers.csv"
        pipeline.save_processed_data(output_path, format='csv')
        print(f"✓ Processed data saved to: {output_path}")
        
        # Generate quality report
        quality_report = validator.generate_quality_report(data, output_format='json')
        report_path = "data/quality_report.json"
        with open(report_path, 'w') as f:
            f.write(quality_report)
        print(f"✓ Quality report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error saving data: {e}")
    
    # Step 7: Display summary
    print("\n7. PROCESSING SUMMARY")
    print("-" * 40)
    
    try:
        summary = pipeline.get_preprocessing_summary()
        print(f"✓ Pipeline execution completed successfully")
        print(f"  - Original shape: {summary['original_shape']}")
        print(f"  - Final shape: {summary['current_shape']}")
        print(f"  - Processing steps: {len(summary['preprocessing_steps'])}")
        
        print("\n  Processing steps performed:")
        for i, step in enumerate(summary['preprocessing_steps'], 1):
            print(f"    {i}. {step}")
            
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
    
    print("\n" + "=" * 80)
    print("PIPELINE DEMONSTRATION COMPLETED")
    print("=" * 80)


def demonstrate_individual_features():
    """
    Demonstrate individual features of the pipeline
    """
    print("\n" + "=" * 80)
    print("INDIVIDUAL FEATURES DEMONSTRATION")
    print("=" * 80)
    
    # Initialize components
    pipeline = DataPreparationPipeline()
    validator = DataValidator()
    
    # Load sample data
    data = pipeline.load_data("data/customers.csv")
    
    print("\n1. DATA VALIDATION FEATURES")
    print("-" * 40)
    
    # Demonstrate validation features
    expected_columns = ['customer_id', 'first_name', 'last_name', 'email', 'age']
    expected_dtypes = {'age': 'int', 'income': 'int'}
    
    structure_results = validator.validate_data_structure(
        data, expected_columns=expected_columns, expected_dtypes=expected_dtypes
    )
    
    print(f"Structure validation result: {'PASSED' if structure_results['valid'] else 'FAILED'}")
    
    print("\n2. DATA CLEANING FEATURES")
    print("-" * 40)
    
    # Demonstrate different cleaning strategies
    strategies = ['drop', 'fill', 'forward_fill']
    
    for strategy in strategies:
        temp_pipeline = DataPreparationPipeline()
        temp_pipeline.data = data.copy()
        
        before_shape = temp_pipeline.data.shape
        temp_pipeline.handle_missing_values(strategy=strategy)
        after_shape = temp_pipeline.data.shape
        
        print(f"Strategy '{strategy}': {before_shape} → {after_shape}")
    
    print("\n3. FEATURE ENGINEERING FEATURES")
    print("-" * 40)
    
    # Demonstrate feature creation
    feature_configs = [
        {'name': 'age_squared', 'type': 'polynomial', 'params': {'column': 'age', 'degree': 2}},
        {'name': 'age_income_interaction', 'type': 'interaction', 'params': {'columns': ['age', 'income']}},
        {'name': 'income_category', 'type': 'binning', 'params': {'column': 'income', 'bins': [0, 50000, 100000, 200000], 'labels': ['Low', 'Medium', 'High']}},
    ]
    
    for config in feature_configs:
        temp_pipeline = DataPreparationPipeline()
        temp_pipeline.data = data.copy()
        
        try:
            temp_pipeline.create_features([config])
            print(f"✓ Created feature: {config['name']} (type: {config['type']})")
        except Exception as e:
            print(f"✗ Failed to create feature {config['name']}: {e}")
    
    print("\n4. STANDARDIZATION FEATURES")
    print("-" * 40)
    
    # Demonstrate different standardization methods
    methods = ['minmax', 'standard', 'robust']
    
    for method in methods:
        temp_pipeline = DataPreparationPipeline()
        temp_pipeline.data = data.copy()
        
        try:
            temp_pipeline.standardize_data(method=method)
            print(f"✓ Standardization method '{method}' applied successfully")
        except Exception as e:
            print(f"✗ Standardization method '{method}' failed: {e}")


def main():
    """
    Main function to run the demonstration
    """
    print("Welcome to the Simple Data Preparation Pipeline!")
    print("This demonstration will show you the complete functionality of the pipeline.")
    print()
    
    try:
        # Run the main demonstration
        demonstrate_pipeline()
        
        # Run individual features demonstration
        demonstrate_individual_features()
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error during demonstration: {e}")
        print(f"\nError: {e}")
    finally:
        print("\nThank you for using the Simple Data Preparation Pipeline!")


if __name__ == "__main__":
    main()
