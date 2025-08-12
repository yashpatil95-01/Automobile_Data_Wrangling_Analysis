# Automobile Data Wrangling and Analysis

A comprehensive data analysis project focusing on automobile dataset from the UCI Machine Learning Repository. This project demonstrates advanced data wrangling techniques, exploratory data analysis, and statistical insights.

## Project Structure

```
Automobile_Data_Wrangling_Analysis/
│
├── automobile_data_analysis.py      # Main analysis script
├── automobile_analysis.ipynb        # Interactive Jupyter notebook
├── results/                         # Generated outputs
│   ├── automobile_data_cleaned.csv  # Cleaned dataset
│   ├── numeric_summary.csv          # Statistical summaries
│   ├── categorical_summary.csv      # Categorical analysis
│   ├── dataset_insights.csv         # Key metrics
│   ├── basic_analysis.png           # Basic visualizations
│   ├── correlation_heatmap.png      # Correlation matrix
│   ├── price_relationships.png      # Price analysis
│   ├── advanced_analysis.png        # Advanced insights
│   └── missing_data_analysis.png    # Missing data patterns
│
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Dataset Information

**Source**: UCI Machine Learning Repository - Automobile Dataset
**URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

**Features**: 26 attributes including:
- **Categorical**: make, fuel-type, aspiration, body-style, drive-wheels, etc.
- **Numerical**: price, horsepower, engine-size, city-mpg, highway-mpg, etc.
- **Target Variable**: price (primary focus for analysis)

## Key Features

### Data Wrangling
- **Missing Data Handling**: Comprehensive analysis and treatment of missing values
- **Data Type Conversion**: Proper conversion of numerical and categorical variables
- **Feature Engineering**: Creation of new variables (price categories, engine size categories)
- **Unit Conversion**: MPG to L/100km for international standards

### Exploratory Data Analysis
- **Descriptive Statistics**: Comprehensive summary of all variables
- **Distribution Analysis**: Understanding data patterns and outliers
- **Categorical Analysis**: Frequency analysis of categorical variables
- **Correlation Analysis**: Relationship analysis between numerical features

### Advanced Visualizations
- **Price Distribution Analysis**: Histograms, box plots, and statistical summaries
- **Correlation Heatmaps**: Visual representation of feature relationships
- **Scatter Plot Analysis**: Price relationships with key features
- **Categorical Insights**: Make analysis, fuel type comparisons
- **Missing Data Visualization**: Patterns and impact assessment

### Statistical Analysis
- **Correlation Coefficients**: Quantified relationships with price
- **Summary Statistics**: Mean, median, standard deviation for all numerical features
- **Category Analysis**: Value counts and distributions for categorical variables

## Key Insights Generated

1. **Price Predictors**: Identification of strongest correlations with automobile price
2. **Market Analysis**: Most common makes, body styles, and fuel types
3. **Performance Metrics**: Average horsepower, engine size, and fuel efficiency
4. **Missing Data Impact**: Analysis of data quality and completeness
5. **Feature Relationships**: Understanding how different attributes interact

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation
```bash
# Clone or download the project
cd Automobile_Data_Wrangling_Analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Complete Analysis
```bash
python automobile_data_analysis.py
```

### Interactive Analysis
```bash
jupyter notebook automobile_analysis.ipynb
```

## Output Files

All analysis results are saved in the `results/` directory:

### Data Files
- **automobile_data_cleaned.csv**: Fully processed and cleaned dataset
- **numeric_summary.csv**: Statistical summaries of numerical features
- **categorical_summary.csv**: Analysis of categorical variables
- **dataset_insights.csv**: Key metrics and insights

### Visualizations
- **basic_analysis.png**: Price distribution, make analysis, body style breakdown
- **correlation_heatmap.png**: Correlation matrix of numerical features
- **price_relationships.png**: Price vs key features (engine size, horsepower, weight)
- **advanced_analysis.png**: Fuel efficiency, aspiration types, cylinder analysis
- **missing_data_analysis.png**: Missing data patterns and percentages

## Technical Highlights

### Data Quality Assessment
- Systematic handling of missing values represented as '?'
- Data type validation and conversion
- Outlier detection and analysis

### Feature Engineering
- Price categorization (Budget, Mid-range, Luxury)
- Engine size classification (Small, Medium, Large, Very Large)
- Fuel efficiency standardization (MPG to L/100km)

### Statistical Rigor
- Correlation analysis with significance testing
- Comprehensive descriptive statistics
- Missing data impact assessment

## Business Applications

This analysis provides valuable insights for:
- **Automotive Industry**: Understanding price drivers and market segments
- **Insurance Companies**: Risk assessment based on vehicle characteristics
- **Consumers**: Informed decision-making for vehicle purchases
- **Researchers**: Baseline analysis for automotive studies

## Future Enhancements

Potential extensions to this analysis:
- **Predictive Modeling**: Machine learning models for price prediction
- **Time Series Analysis**: Historical price trends (with temporal data)
- **Geographic Analysis**: Regional market differences
- **Advanced Statistics**: Regression analysis and hypothesis testing

## Author

**Yash Patil** - Manufacturing Engineer transitioning to Data Analytics

## License

This project is open source and available under the MIT License.

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Python data science community for excellent libraries
- Automotive industry domain knowledge contributors