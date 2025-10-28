"""
STEP 02: Data Preprocessing and Cleaning
Cleans raw data, handles outliers, standardizes formats
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from loguru import logger

# INSERT CODE HERE
# Extract and implement Phase 2 logic from run_full_pipeline.py
#
# Expected implementation:
# 1. Load configuration
# 2. Initialize DataCleaner
# 3. For each ceramic system:
#    - Load raw data from data/raw/materials_project/{system}_raw.csv
#    - Clean dataframe (remove duplicates, handle missing values, detect outliers)
#    - Save cleaned data to data/processed/{system}/{system}_clean.csv
