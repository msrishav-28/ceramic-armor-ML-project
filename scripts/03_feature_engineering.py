"""
STEP 03: Feature Engineering
Calculates derived properties and phase stability
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from loguru import logger

from src.feature_engineering.derived_properties import DerivedPropertiesCalculator
from src.feature_engineering.phase_stability import PhaseStabilityClassifier

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Initialize calculators
    derived_calc = DerivedPropertiesCalculator()
    phase_classifier = PhaseStabilityClassifier(api_key=None)  # Set API key if available
    
    for system in config['ceramic_systems']['primary']:
        logger.info(f"\nProcessing features for {system}...")
        
        # Load processed data
        processed_path = Path(config['paths']['processed']) / system.lower() / f"{system.lower()}_processed.csv"
        
        if not processed_path.exists():
            logger.warning(f"Processed data not found for {system}, skipping...")
            continue
        
        df = pd.read_csv(processed_path)
        
        # Calculate derived properties
        df = derived_calc.add_all_derived_properties(df)
        
        # Add phase stability (if Materials Project API available)
        try:
            df = phase_classifier.analyze_dataframe(df)
        except Exception as e:
            logger.warning(f"Phase stability analysis failed: {e}")
        
        # Save features
        features_dir = Path(config['paths']['features']) / system.lower()
        features_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = features_dir / f"{system.lower()}_features.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Features saved: {output_path}")
        logger.info(f"  Total features: {len(df.columns)}")
        logger.info(f"  Total samples: {len(df)}")

if __name__ == "__main__":
    main()
