# Craigslist used cars data cleaning script

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