"""
STEP 01: Data Collection from Materials Project and Other Sources
Collects experimental and DFT data for all ceramic systems
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger

# INSERT CODE HERE
# Extract and implement Phase 1 logic from run_full_pipeline.py
#
# Expected implementation:
# 1. Load configuration from config/config.yaml
# 2. Load API keys from config/api_keys.yaml
# 3. Initialize MaterialsProjectCollector
# 4. Collect data for each ceramic system (SiC, Al2O3, B4C, WC, TiC)
# 5. Optionally collect from AFLOW, JARVIS, NIST
# 6. Save raw data to data/raw/{source}/{system}_raw.csv
