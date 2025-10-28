"""
STEP 04: Model Training
Trains XGBoost, CatBoost, RF, GB, and Ensemble models for all systems
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger

from src.utils.intel_optimizer import apply_optimizations
from src.training.trainer import CeramicPropertyTrainer

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    # Apply Intel optimizations
    logger.info("Applying Intel MKL optimizations...")
    apply_optimizations(n_jobs=20)
    
    # Load config
    config = load_config()
    
    # Initialize trainer
    trainer = CeramicPropertyTrainer(config)
    
    # Train all systems
    trainer.train_all_systems()
    
    logger.info("\n✓ Model training complete!")

if __name__ == "__main__":
    main()
