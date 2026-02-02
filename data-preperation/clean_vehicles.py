# Craigslist used cars data cleaning script
# https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

# This script cleans the raw Craiglist vehicles dataset and prepares it for use across all four ML frameworks.

# Input: data/raw/vehicles.csv (downloaded from kaggle)
# Output: data/processed/vehicles_clean.csv

import pandas as pd # library for data manipulation using Dataframes
import numpy as np  # library for numerical operations on arrays and matrices
import os           # built-in python module for interacting with operating systems (file paths)

# Setting random seed for reproducability - using 113 as defined in the project rules
RANDOM_SEED = 113

# Define file paths using os.path.join() for cross-platform compatibility
# os.path.join() automatically uses the correct slash for the OS
RAW_DATA_PATH = os.path.join("data", "raw", "vehicles.csv")     # The path we expect the raw data to be at

PROCESSED_DATA_PATH = os.path.join("data", "processed", "vehicles_clean.csv")  #where we will save our cleaned dataset

# Step 1: loading the raw data
print("Loading raw data...") 
print("Current directory", os.getcwd())
print()
df = pd.read_csv(RAW_DATA_PATH) # reads CSV file and convers it into a pandas dataframe

# Returns a tuple: (number_of_rows, number_of_columns)
# this tell us how big our dataset is before cleaning
print(f"Raw dataset shape: {df.shape}")

# convers the column names from a pandas index to a list
# shows us all available columns in the dataset
print(f"Columns: {df.columns.tolist()}")

# Step 2: selecting relevant columns

# We won't need all 25 columns from the raw data
# Selecting only the columns useful for predicting car prices removes unwanted variables
# This reduces memory usage and simplifies our analysis

# When running the code from step 1, we recieved:
# Raw dataset shape: (426880, 26)
# Columns: ['id', 'url', 'region', 'region_url', 'price', 'year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel',
#  'odometer', 'title_status', 'transmission', 'VIN', 'drive', 'size', 'type', 'paint_color', 'image_url', 'description',
#  'county', 'state', 'lat', 'long', 'posting_date']

# List of columns names we want to keep for our model:
COLUMNS_TO_KEEP = [
    'price',            # Target variable - what we want to predict
    'year',             # Numeric - model year of the vehicle
    'manufacturer',     # Categorical - brand name (ford, toyota, ect.) 
    'model',            # Categorical - specific model name
    'condition',        # Categorical - excellent, good, fair, ect.
    'cylinders',        # Categorical - engine cylinders (4, 6, 8)
    'fuel',             # Categorical - gas, diesal, hybrid, electric
    'odometer',         # Numeric - miles driven  
    'title_status',     # Categorical - clean, salvage, rebuilt
    'transmission',     # Categorical - automatic, manual, other
    'drive',            # Categorical - fwd, rwd, 4wd
    'type',             # Categorical - Sedan, SUV, truck, coupe
    'state'             # Categorical - US state where listed
]

# Using double brackers to select multiple columns from the dataframe
# .copy() creates an independent copy so changes don't affect original df
df = df[COLUMNS_TO_KEEP].copy()

# verify our column selection worked
print(f"\nAfter column selection: {df.shape}")      
# After column selection: (426880, 13)

