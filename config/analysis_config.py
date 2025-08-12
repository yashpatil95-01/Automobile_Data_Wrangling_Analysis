"""
Configuration file for Automobile Data Analysis
"""

# Data source
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

# Column names for the dataset
COLUMN_NAMES = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", 
    "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", 
    "city-mpg", "highway-mpg", "price"
]

# Numeric columns for conversion
NUMERIC_COLUMNS = [
    'symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 
    'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
    'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 
    'highway-mpg', 'price'
]

# Key features for analysis
KEY_FEATURES = ['price', 'horsepower', 'engine-size', 'city-mpg', 'highway-mpg']

# Price categories
PRICE_BINS = [0, 10000, 20000, 50000]
PRICE_LABELS = ['Budget', 'Mid-range', 'Luxury']

# Engine size categories  
ENGINE_BINS = [0, 100, 150, 200, 500]
ENGINE_LABELS = ['Small', 'Medium', 'Large', 'Very Large']

# Visualization settings
FIGURE_SIZE = (15, 10)
DPI = 300
PLOT_STYLE = 'seaborn-v0_8'