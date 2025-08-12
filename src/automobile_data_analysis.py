#!/usr/bin/env python3
"""
Automobile Data Wrangling and Analysis

This script performs comprehensive analysis of automobile data including:
- Data loading and cleaning
- Exploratory data analysis
- Statistical analysis and correlations
- Advanced visualizations
- Feature engineering
- Missing data analysis

Author: Yash Patil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_automobile_data():
    """Load automobile dataset from UCI repository."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    
    # Column names based on UCI dataset documentation
    column_names = [
        "symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
        "num-of-doors", "body-style", "drive-wheels", "engine-location",
        "wheel-base", "length", "width", "height", "curb-weight", 
        "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
        "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", 
        "city-mpg", "highway-mpg", "price"
    ]
    
    df = pd.read_csv(url, header=None, names=column_names)
    return df

def clean_data(df):
    """Comprehensive data cleaning and preprocessing."""
    print("="*60)
    print("DATA CLEANING AND PREPROCESSING")
    print("="*60)
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Replace '?' with NaN
    df_clean = df_clean.replace('?', np.nan)
    
    # Convert numeric columns
    numeric_columns = [
        'symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 
        'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
        'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 
        'highway-mpg', 'price'
    ]
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Convert fuel efficiency from MPG to L/100km for international standard
    df_clean['city-L/100km'] = 235 / df_clean['city-mpg']
    df_clean['highway-L/100km'] = 235 / df_clean['highway-mpg']
    
    # Create price categories
    df_clean['price_category'] = pd.cut(df_clean['price'], 
                                       bins=[0, 10000, 20000, 50000], 
                                       labels=['Budget', 'Mid-range', 'Luxury'])
    
    # Create engine size categories
    df_clean['engine_size_category'] = pd.cut(df_clean['engine-size'], 
                                             bins=[0, 100, 150, 200, 500], 
                                             labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Missing values per column:")
    print(df_clean.isnull().sum().sort_values(ascending=False))
    
    return df_clean

def exploratory_data_analysis(df):
    """Perform comprehensive exploratory data analysis."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print("\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Data types:\n{df.dtypes.value_counts()}")
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns: {len(categorical_cols)}")
    
    for col in categorical_cols[:5]:  # Show first 5 categorical columns
        print(f"\n{col} - Unique values: {df[col].nunique()}")
        print(df[col].value_counts().head())
    
    # Numerical analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumerical columns: {len(numerical_cols)}")
    print("\nNumerical summary:")
    print(df[numerical_cols].describe())
    
    return categorical_cols, numerical_cols

def correlation_analysis(df, numerical_cols):
    """Perform correlation analysis with price."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS WITH PRICE")
    print("="*60)
    
    # Filter numerical columns that have price data
    price_data = df.dropna(subset=['price'])
    correlations = []
    
    for col in numerical_cols:
        if col != 'price' and col in price_data.columns:
            # Calculate correlation only for non-null pairs
            valid_data = price_data[[col, 'price']].dropna()
            if len(valid_data) > 10:  # Minimum data points for meaningful correlation
                corr_coef = valid_data[col].corr(valid_data['price'])
                correlations.append({
                    'Feature': col,
                    'Correlation': corr_coef,
                    'Abs_Correlation': abs(corr_coef),
                    'Data_Points': len(valid_data)
                })
    
    # Sort by absolute correlation
    corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
    
    print("Top correlations with price:")
    print(corr_df.head(10).to_string(index=False))
    
    return corr_df

def create_visualizations(df, viz_dir):
    """Create comprehensive visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Price distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    df['price'].dropna().hist(bins=30, edgecolor='black', alpha=0.7)
    plt.title('Price Distribution')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    df.boxplot(column='price', ax=plt.gca())
    plt.title('Price Box Plot')
    plt.ylabel('Price ($)')
    
    plt.subplot(2, 2, 3)
    make_counts = df['make'].value_counts().head(10)
    make_counts.plot(kind='bar')
    plt.title('Top 10 Car Makes')
    plt.xlabel('Make')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    df['body-style'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Body Style Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'basic_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(14, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Price vs key features
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Price vs Engine Size
    axes[0, 0].scatter(df['engine-size'], df['price'], alpha=0.6)
    axes[0, 0].set_xlabel('Engine Size')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('Price vs Engine Size')
    
    # Price vs Horsepower
    axes[0, 1].scatter(df['horsepower'], df['price'], alpha=0.6, color='orange')
    axes[0, 1].set_xlabel('Horsepower')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Price vs Horsepower')
    
    # Price vs Curb Weight
    axes[1, 0].scatter(df['curb-weight'], df['price'], alpha=0.6, color='green')
    axes[1, 0].set_xlabel('Curb Weight')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Price vs Curb Weight')
    
    # Price by Drive Wheels
    df.boxplot(column='price', by='drive-wheels', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Drive Wheels')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].set_title('Price by Drive Wheels')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'price_relationships.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Advanced analysis
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 3, 1)
    df.groupby('make')['price'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title('Average Price by Make (Top 10)')
    plt.xlabel('Make')
    plt.ylabel('Average Price ($)')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 2)
    fuel_price = df.groupby('fuel-type')['price'].mean()
    fuel_price.plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Average Price by Fuel Type')
    plt.xlabel('Fuel Type')
    plt.ylabel('Average Price ($)')
    
    plt.subplot(2, 3, 3)
    df['city-L/100km'].dropna().hist(bins=20, alpha=0.7, color='purple')
    plt.title('City Fuel Consumption Distribution')
    plt.xlabel('City L/100km')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 4)
    aspiration_counts = df['aspiration'].value_counts()
    plt.pie(aspiration_counts.values, labels=aspiration_counts.index, autopct='%1.1f%%')
    plt.title('Aspiration Type Distribution')
    
    plt.subplot(2, 3, 5)
    df.plot.scatter(x='city-L/100km', y='highway-L/100km', alpha=0.6, ax=plt.gca())
    plt.title('City vs Highway Fuel Consumption')
    plt.xlabel('City L/100km')
    plt.ylabel('Highway L/100km')
    
    plt.subplot(2, 3, 6)
    df['num-of-cylinders'].value_counts().sort_index().plot(kind='bar', color='gold')
    plt.title('Number of Cylinders Distribution')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'advanced_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def missing_data_analysis(df, viz_dir):
    """Analyze missing data patterns."""
    print("\n" + "="*60)
    print("MISSING DATA ANALYSIS")
    print("="*60)
    
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percentage': missing_percent
    })
    
    # Only show columns with missing data
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    print("Missing data summary:")
    print(missing_df)
    
    # Visualize missing data
    if len(missing_df) > 0:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        missing_df['Missing_Count'].plot(kind='bar')
        plt.title('Missing Data Count by Column')
        plt.xlabel('Columns')
        plt.ylabel('Missing Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        missing_df['Missing_Percentage'].plot(kind='bar', color='orange')
        plt.title('Missing Data Percentage by Column')
        plt.xlabel('Columns')
        plt.ylabel('Missing Percentage (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'missing_data_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

def generate_insights(df, corr_df):
    """Generate business insights from the analysis."""
    print("\n" + "="*60)
    print("KEY BUSINESS INSIGHTS")
    print("="*60)
    
    insights = []
    
    # Price insights
    avg_price = df['price'].mean()
    median_price = df['price'].median()
    insights.append(f"Average car price: ${avg_price:,.2f}")
    insights.append(f"Median car price: ${median_price:,.2f}")
    
    # Top correlations with price
    if len(corr_df) > 0:
        top_corr = corr_df.iloc[0]
        insights.append(f"Strongest price predictor: {top_corr['Feature']} (correlation: {top_corr['Correlation']:.3f})")
    
    # Make analysis
    most_common_make = df['make'].mode()[0]
    make_count = df['make'].value_counts().iloc[0]
    insights.append(f"Most common make: {most_common_make} ({make_count} cars)")
    
    # Fuel efficiency
    avg_city_mpg = df['city-mpg'].mean()
    if not np.isnan(avg_city_mpg):
        insights.append(f"Average city fuel efficiency: {avg_city_mpg:.1f} MPG")
    
    # Engine analysis
    avg_horsepower = df['horsepower'].mean()
    if not np.isnan(avg_horsepower):
        insights.append(f"Average horsepower: {avg_horsepower:.0f} HP")
    
    print("\nKey Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    return insights

def save_processed_data(df, data_dir, reports_dir):
    """Save processed datasets."""
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)
    
    # Save full cleaned dataset
    df.to_csv(os.path.join(data_dir, 'automobile_data_cleaned.csv'), index=False)
    
    # Save summary statistics
    numeric_summary = df.describe()
    numeric_summary.to_csv(os.path.join(reports_dir, 'numeric_summary.csv'))
    
    # Save categorical summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    cat_summary = {}
    for col in categorical_cols:
        cat_summary[col] = df[col].value_counts().to_dict()
    
    cat_summary_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cat_summary.items()]))
    cat_summary_df.to_csv(os.path.join(reports_dir, 'categorical_summary.csv'))
    
    # Save key insights
    insights_df = pd.DataFrame({
        'Metric': ['Total_Records', 'Total_Features', 'Missing_Data_Columns', 'Price_Range_Min', 'Price_Range_Max'],
        'Value': [len(df), len(df.columns), df.isnull().any().sum(), df['price'].min(), df['price'].max()]
    })
    insights_df.to_csv(os.path.join(reports_dir, 'dataset_insights.csv'), index=False)
    
    print(f"Data saved to organized directories:")
    print(f"Processed Data: {data_dir}/")
    print(f"Reports: {reports_dir}/")
    print(f"Files created:")
    print("- automobile_data_cleaned.csv (processed_data/)")
    print("- numeric_summary.csv (reports/)") 
    print("- categorical_summary.csv (reports/)")
    print("- dataset_insights.csv (reports/)")

def main():
    """Main analysis pipeline."""
    print("AUTOMOBILE DATA WRANGLING AND ANALYSIS")
    print("="*60)
    
    # Create organized results directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    viz_dir = os.path.join(results_dir, "visualizations")
    data_dir = os.path.join(results_dir, "processed_data")
    reports_dir = os.path.join(results_dir, "reports")
    
    for directory in [results_dir, viz_dir, data_dir, reports_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load data
    print("Loading automobile dataset...")
    df = load_automobile_data()
    
    # Clean data
    df_clean = clean_data(df)
    
    # Exploratory data analysis
    categorical_cols, numerical_cols = exploratory_data_analysis(df_clean)
    
    # Correlation analysis
    corr_df = correlation_analysis(df_clean, numerical_cols)
    
    # Missing data analysis
    missing_data_analysis(df_clean, viz_dir)
    
    # Create visualizations
    create_visualizations(df_clean, viz_dir)
    
    # Generate insights
    insights = generate_insights(df_clean, corr_df)
    
    # Save processed data
    save_processed_data(df_clean, data_dir, reports_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Check the 'results/' directory for all outputs:")
    print("- results/visualizations/ - Charts and plots")
    print("- results/processed_data/ - Clean datasets") 
    print("- results/reports/ - Analysis summaries")

if __name__ == "__main__":
    main()