# Data Pipeline for ML

A comprehensive, easy-to-use data preparation pipeline for cleaning, transforming, and preparing data for machine learning and analysis tasks. This pipeline provides a complete set of tools for data validation, quality assessment, cleaning, feature engineering, and visualization.

> **Note**: This project was developed using Claude Code CLI tool as an AI assistant to demonstrate modern data preprocessing techniques and best practices.

## Features

### Core Pipeline Components

- **Data Loading**: Support for CSV, JSON, Excel, and Parquet files
- **Data Cleaning**: Handle missing values, duplicates, and outliers
- **Data Transformation**: Normalization, encoding, and feature engineering
- **Data Validation**: Comprehensive quality assessment and validation
- **Data Visualization**: Interactive plots and quality dashboards
- **Extensible Design**: Easy to extend with custom transformations

### Data Quality Assessment

- **Missing Value Analysis**: Detect and handle missing data
- **Duplicate Detection**: Identify and remove duplicate records
- **Outlier Detection**: Statistical outlier identification using IQR and Z-score
- **Data Type Validation**: Ensure data types are appropriate
- **Consistency Checks**: Validate data consistency and format
- **Quality Scoring**: Overall data quality score with recommendations

### Data Cleaning and Transformation

- **Missing Value Handling**: Drop, fill, forward fill, backward fill strategies
- **Duplicate Removal**: Remove duplicate rows with various criteria
- **Outlier Treatment**: IQR and Z-score based outlier removal
- **Categorical Encoding**: One-hot and ordinal encoding with proper handling
- **Standardization**: Min-Max, Standard, and Robust scaling
- **Feature Engineering**: Polynomial, interaction, binning, and aggregation features

### Visualization and Reporting

- **Missing Value Plots**: Heatmaps and bar charts for missing data
- **Distribution Analysis**: Histograms and box plots
- **Correlation Analysis**: Correlation matrices and heatmaps
- **Quality Dashboards**: Interactive Plotly dashboards
- **Comprehensive Reports**: JSON and HTML quality reports

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/Data-Pipeline-for-ML.git
cd Data-Pipeline-for-ML

# Install dependencies using uv (recommended)
uv sync

# OR install using pip
pip install -r requirements.txt

# Run the demonstration
python src/main.py  # or: uv run python src/main.py
```

## Quick Start

### Basic Usage

```python
from src.pipeline import DataPreparationPipeline
from src.validation import DataValidator

# Initialize pipeline
pipeline = DataPreparationPipeline()

# Load data
data = pipeline.load_data("your_data.csv")

# Clean data
pipeline.handle_missing_values(strategy='drop')
pipeline.remove_duplicates()
pipeline.handle_outliers(method='iqr')

# Transform data
pipeline.encode_categorical_variables(method='onehot')
pipeline.standardize_data(method='standard')

# Save processed data
pipeline.save_processed_data("processed_data.csv")
```

### Data Validation

```python
from src.validation import DataValidator

# Initialize validator
validator = DataValidator()

# Validate data quality
quality_results = validator.validate_data_quality(data)
print(f"Quality Score: {quality_results['quality_score']}/100")

# Generate comprehensive report
report = validator.generate_quality_report(data, output_format='html')
with open('quality_report.html', 'w') as f:
    f.write(report)
```

### Enhanced Categorical Encoding

```python
# Auto-detect ordinal vs nominal variables
pipeline.encode_categorical_variables(
    method='auto',
    ordinal_mappings={
        'size': ['S', 'M', 'L', 'XL', 'XXL'],
        'grade': ['F', 'D', 'C', 'B', 'A'],
        'satisfaction': ['Low', 'Medium', 'High']
    }
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
```

### Feature Engineering

```python
# Create new features
feature_configs = [
    {'name': 'age_squared', 'type': 'polynomial', 'params': {'column': 'age', 'degree': 2}},
    {'name': 'age_income_interaction', 'type': 'interaction', 'params': {'columns': ['age', 'income']}},
    {'name': 'income_category', 'type': 'binning', 'params': {'column': 'income', 'bins': [0, 50000, 100000, 200000]}}
]

pipeline.create_features(feature_configs)
```

### Visualization

```python
from src.visualization import DataVisualizer

# Initialize visualizer
visualizer = DataVisualizer()

# Create visualizations
visualizer.plot_missing_values(data)
visualizer.plot_data_distribution(data)
visualizer.plot_correlation_matrix(data)
visualizer.plot_data_quality_summary(quality_results)

# Create interactive dashboard
visualizer.create_interactive_dashboard(data, quality_results)
```

## Examples

### Running the Demonstrations

```bash
# Run the main demonstration
python src/main.py

# Run example usage scripts
python src/example_usage.py

# Generate sample data
python src/sample_data_generator.py
```

### Complete Pipeline Example

```python
from src.pipeline import DataPreparationPipeline
from src.validation import DataValidator
from src.visualization import DataVisualizer

# Initialize components
pipeline = DataPreparationPipeline()
validator = DataValidator()
visualizer = DataVisualizer()

# Load and validate data
data = pipeline.load_data("data/customers.csv")
quality_results = validator.validate_data_quality(data)

# Clean and transform data
pipeline.handle_missing_values(strategy='drop')
pipeline.remove_duplicates()
pipeline.handle_outliers(method='iqr')
pipeline.encode_categorical_variables(method='onehot')
pipeline.standardize_data(method='standard')

# Create features
feature_configs = [
    {'name': 'age_income_ratio', 'type': 'interaction', 'params': {'columns': ['age', 'income']}}
]
pipeline.create_features(feature_configs)

# Visualize results
visualizer.plot_data_quality_summary(quality_results)

# Save results
pipeline.save_processed_data("processed_data.csv")

# Generate summary
summary = pipeline.get_preprocessing_summary()
print(f"Processing completed: {len(summary['preprocessing_steps'])} steps")
```

## API Reference

### DataPreparationPipeline

The main pipeline class for data preparation tasks.

#### Methods

- `load_data(source, **kwargs)`: Load data from various file formats
- `get_data_overview()`: Get comprehensive data overview
- `handle_missing_values(strategy, columns, fill_value)`: Handle missing values
- `remove_duplicates(subset, keep)`: Remove duplicate rows
- `handle_outliers(columns, method, threshold)`: Handle outliers
- `encode_categorical_variables(columns, method, ordinal_mappings)`: Encode categorical variables
- `standardize_data(columns, method)`: Standardize numeric data
- `create_features(feature_configs)`: Create new features
- `split_data(target_column, test_size, random_state)`: Split data for ML
- `save_processed_data(filepath, format)`: Save processed data

### DataValidator

Comprehensive data validation and quality assessment.

#### Methods

- `validate_data_structure(data, expected_columns, expected_dtypes)`: Validate data structure
- `validate_data_quality(data)`: Assess data quality
- `generate_quality_report(data, output_format)`: Generate quality reports

### DataVisualizer

Create visualizations for data analysis.

#### Methods

- `plot_missing_values(data)`: Visualize missing values
- `plot_data_distribution(data)`: Plot data distributions
- `plot_outliers(data)`: Visualize outliers
- `plot_correlation_matrix(data)`: Create correlation heatmap
- `plot_data_quality_summary(quality_results)`: Quality summary plots
- `create_interactive_dashboard(data, quality_results)`: Interactive dashboard

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
Data-Pipeline-for-ML/
├── src/
│   ├── pipeline.py              # Main pipeline class
│   ├── validation.py            # Data validation and quality assessment
│   ├── visualization.py         # Data visualization tools
│   ├── sample_data_generator.py # Generate sample datasets
│   ├── main.py                  # Main demonstration script
│   └── example_usage.py         # Usage examples
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         # Comprehensive test suite
├── data/                        # Sample and processed data
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── CLAUDE.md                    # Project documentation
```

## Sample Data

The pipeline comes with a sample data generator that creates realistic datasets with various data quality issues:

- **Customer Data**: 1000+ records with missing values, duplicates, and outliers
- **Sales Data**: 2000+ transaction records with data quality issues
- **Product Data**: 500+ product records with inconsistent formats

Generate sample data:

```bash
python src/sample_data_generator.py
```

## Key Features

### Enhanced Categorical Encoding

This pipeline provides advanced categorical variable handling:

- **Automatic Detection**: Distinguishes between ordinal and nominal variables
- **Ordinal Encoding**: Preserves natural order (e.g., S < M < L < XL < XXL)
- **Custom Mappings**: User-defined ordinal relationships
- **Validation**: Ensures all data values are covered in mappings

### Proper Standardization

- **Terminology**: Uses "standardization" instead of "normalization" (aligns with sklearn)
- **Multiple Methods**: MinMax, Standard, and Robust scaling
- **Column-wise**: Operates on features (columns), not samples

### Modern ML Practices

- **Scikit-learn Integration**: Uses proper sklearn transformers
- **Interactive Visualizations**: Plotly for modern, interactive charts
- **Comprehensive Testing**: 44 test cases covering all functionality
- **Documentation**: Extensive examples and API reference

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using pandas, scikit-learn, matplotlib, seaborn, and plotly
- Inspired by best practices in data science and machine learning
- Developed with Claude Code CLI tool as AI assistant

## Support

For questions, issues, or contributions, please:
1. Check the documentation and examples
2. Review the test suite for usage patterns
3. Open an issue on GitHub
4. Contact the development team

## Changelog

### v0.1.0 (Current)
- Initial release with core pipeline functionality
- Enhanced categorical encoding with ordinal/nominal distinction
- Proper standardization terminology and methods
- Comprehensive data validation and quality assessment
- Multiple data cleaning and transformation methods
- Interactive Plotly visualization and reporting capabilities
- Complete test suite with 44 test cases
- Sample data generation for testing
- Comprehensive documentation and examples