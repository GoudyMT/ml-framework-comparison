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

# Step 3: Initial data exploration

# Before cleaning, we want to understand what we're working with

# df.info prints:
# - Column names
# - Non-null counts (help identify missing data)
# - Data types (int, float, object/string)
print("\n--- Data Types amd Non-Null Counts ---")
print(df.info())

"""
RangeIndex: 426880 entries, 0 to 426879
Data columns (total 13 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----
 0   price         426880 non-null  int64
 1   year          425675 non-null  float64
 2   manufacturer  409234 non-null  str
 3   model         421603 non-null  str
 4   condition     252776 non-null  str
 5   cylinders     249202 non-null  str
 6   fuel          423867 non-null  str
 7   odometer      422480 non-null  float64
 8   title_status  418638 non-null  str
 9   transmission  424324 non-null  str
 10  drive         296313 non-null  str
 11  type          334022 non-null  str
 12  state         426880 non-null  str
"""

# df.isnull().sum() counts missing values in each column
# .isnull() creates a True/False mask where True = missing
# .sum() adds up the Trues 
print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())

"""
price                0
year              1205
manufacturer     17646
model             5277
condition       174104
cylinders       177678
fuel              3013
odometer          4400
title_status      8242
transmission      2556
drive           130567
type             92858
state                0
"""

# df.describe() shows statistics for numeric columns
# This helps identify outliers (price of 1$ or $1,000,000 which skew the data)
print("\n--- Numeric Column Statistics ---")
print(df.describe())

"""
              price           year      odometer
count  4.268800e+05  425675.000000  4.224800e+05
mean   7.519903e+04    2011.235191  9.804333e+04
std    1.218228e+07       9.452120  2.138815e+05
min    0.000000e+00    1900.000000  0.000000e+00
25%    5.900000e+03    2008.000000  3.770400e+04
50%    1.395000e+04    2013.000000  8.554800e+04
75%    2.648575e+04    2017.000000  1.335425e+05
max    3.736929e+09    2022.000000  1.000000e+07
"""

# Step 4: Filter invalid and outlier values

# The raw data contains many invalid entries that would hurt our models
# Remove rows with unrealistic values before training

# Store the starting row count so we can report how many rows we removed
rows_before = len(df) # 426880

# Price Filtering

# We'll keep prices between $500 and $100,000
# $500 minimum removes junk listings
# $100,000 maximum removes exotic cards and data errors
MIN_PRICE = 500
MAX_PRICE = 100000

# Filter rows where price is within our acceptable range
df = df[(df['price'] >= MIN_PRICE) & (df['price'] <= MAX_PRICE)]

# Report how many rows remain after price filtering
print(f"\nAfter price filter (${MIN_PRICE}-{MAX_PRICE}){len(df)} rows") # 384131 rows

# Year Filtering
# Cars before 1990 are classics/antiques - different pricing dynamics
# We want to focus on "normal" used car market
MIN_YEAR = 1990 
MAX_YEAR = 2026

# Filter to keep only years within our range
df = df[(df['year'] >= MIN_YEAR) & (df['year'] <= MAX_YEAR)]

print(f"\nAfter year filter ({MIN_YEAR}-{MAX_YEAR}):{len(df)} rows") # 371172 rows

# Odometer Filtering
# - Min: 0 Miles (Could be new, but often means missing data)
# - Max: 10 million miles (Not possible)
# Average car is driven 12,000 miles/year
MIN_ODOMETER = 100
MAX_ODOMETER = 500000

# Filter odometer to realstic range
df = df[(df['odometer'] >= MIN_ODOMETER) & (df['odometer'] <= MAX_ODOMETER)]

print(f"\nAfter odometer filter ({MIN_ODOMETER}-{MAX_ODOMETER}): {len(df)} rows") # 364546 rows

# Summary of filtering
rows_after = len(df)
rows_removed = rows_before - rows_after

# Calculate percentage of data retained
percent_retained = (rows_after / rows_before) * 100

print(f"\n--- Filtering Summary ---")
print(f"Rows before filtering:  {rows_before}")             # 426880
print(f"Rows after filtering:   {rows_after}")              # 364546
print(f"Rows removed:           {rows_removed}")            # 62334
print(f"Data retained:          {percent_retained:.1f}%")   # 85.4%

# Step 5: Handle missing values

# My strategy:
# - Required Columns (price, year, odometer, manufacturer): drop rows if missing
# - Optional categorical columns: Fill missing with "unknown"

REQUIRED_COLUMNS = ['price', 'year', 'odometer', 'manufacturer']

# Store count before dropping so we can report the difference
rows_before_drop = len(df)

# dropna() removes rows with missing values
# subset= specifies which columns to check for missing values
df = df.dropna(subset=REQUIRED_COLUMNS)

# Report how many rows were dropped due to missing required values
rows_after_drop = len(df)
print(f"\nDropped {rows_before_drop - rows_after_drop:,} rows with missing required values") # 11,367
print(f"Rows remaining: {rows_after_drop}") # 353179

# List of categorical columns where missing values become "unknown"
CATEGORICAL_COLUMNS = [
    'model',            
    'condition',        
    'cylinders',        
    'fuel',
    'title_status',     
    'transmission',     
    'drive',            
    'type',             
    'state'
]

# Loop through each categorical column and fill missing values
for col in CATEGORICAL_COLUMNS:
    # Count missing values in this column before filling
    missing_count = df[col].isnull().sum()

    # fillna() replaces NaN/null values with specified value
    df[col] = df[col].fillna('unknown')

    # Only print if there were actually missing values to fill
    if missing_count > 0:
        print(f"\nFilled {missing_count:,} missing values in '{col}' with 'unknown'")

# Verify no missing values remain
total_missing = df.isnull().sum().sum() # Sum of all missing across all columns
print("\n --- Missing Value Check ---")
print(f"Total missing values remaining: {total_missing}") # Total missing values remaining: 0