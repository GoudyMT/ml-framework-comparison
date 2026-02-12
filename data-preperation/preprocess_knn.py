"""
Preprocessing script for KNN - Covertype Dataset.

Downloads the Forest Cover Type dataset from sklearn, applies stratified
train/test split, scales features, and saves to data/processed/knn/.

Dataset: 581,012 samples, 54 features, 7 forest cover types.
Source: UCI ML Repository via sklearn.datasets.fetch_covtype

Run once before training any KNN models.
"""

import numpy as np
import json
import os
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Project-wide random seed for reproducibility
RANDOM_SEED = 113