"""
ONE-COMMAND EXECUTION SCRIPT
Runs complete pipeline from data collection to final report
"""

import sys
sys.path.append('.')

import yaml
from pathlib import Path
from loguru import logger

# Import pipeline components
from src.utils.intel_optimizer import intel_opt
from src.data_collection.materials_project_collector import MaterialsProjectCollector
from src.preprocessing.data_cleaner import DataCleaner
from src.feature_engineering.derived_properties import DerivedPropertiesCalculator
from src.feature_engineering.phase_stability import PhaseStabilityAnalyzer
from src.training.trainer import CeramicPropertyTrainer
from src.interpretation.shap_analyzer import SHAPAnalyzer
from src.evaluation.metrics import PerformanceChecker

def load_config(config_path='config/config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Execute complete ML pipeline"""
    
    # ASCII Banner
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  CERAMIC ARMOR ML PIPELINE - COMPLETE EXECUTION              ║
    ║  Tree-based Models for Ballistic Property Prediction         ║
    ║  Intel i7-12700K Optimized | 20-Week Research Program        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    
    # Apply Intel optimizations
    logger.info("Applying Intel CPU optimizations...")
    intel_opt.apply_optimizations()
    
    # ========================================================================
    # PHASE 1: DATA COLLECTION (Weeks 1-4)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("="*80)
    
    api_keys_path = 'config/api_keys.yaml'
    if Path(api_keys_path).exists():
        with open(api_keys_path, 'r') as f:
            api_keys = yaml.safe_load(f)
        
        collector = MaterialsProjectCollector(api_keys['materials_project'])
        
        for system in config['ceramic_systems']['primary']:
            logger.info(f"\nCollecting data for {system}...")
            collector.collect_ceramic_data(system)
    else:
        logger.warning("API keys not found - skipping data collection")
    
    # ========================================================================
    # PHASE 2: PREPROCESSING (Weeks 5-6)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: DATA PREPROCESSING")
    logger.info("="*80)
    
    cleaner = DataCleaner()
    
    for system in config['ceramic_systems']['primary']:
        logger.info(f"\nPreprocessing {system} data...")
        
        raw_file = Path(config['paths']['data']['raw']) / 'materials_project' / f"{system.lower()}_raw.csv"
        if raw_file.exists():
            import pandas as pd
            df = pd.read_csv(raw_file)
            df_clean = cleaner.clean_dataframe(df)
            
            output_dir = Path(config['paths']['data']['processed']) / system.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(output_dir / f"{system.lower()}_clean.csv", index=False)
    
    # ========================================================================
    # PHASE 3: FEATURE ENGINEERING (Weeks 7-8)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("="*80)
    
    derived_calc = DerivedPropertiesCalculator()
    stability_analyzer = PhaseStabilityAnalyzer()
    
    for system in config['ceramic_systems']['primary']:
        logger.info(f"\nEngineering features for {system}...")
        
        clean_file = Path(config['paths']['data']['processed']) / system.lower() / f"{system.lower()}_clean.csv"
        if clean_file.exists():
            import pandas as pd
            df = pd.read_csv(clean_file)
            
            # Calculate derived properties
            df_derived = derived_calc.calculate_all_derived_properties(df)
            
            # Analyze phase stability
            df_final = stability_analyzer.analyze_dataframe(df_derived)
            
            # Save features
            output_dir = Path(config['paths']['data']['features']) / system.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            df_final.to_csv(output_dir / f"{system.lower()}_features.csv", index=False)
    
    # ========================================================================
    # PHASE 4: MODEL TRAINING (Weeks 11-14)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: MODEL TRAINING")
    logger.info("="*80)
    
    trainer = CeramicPropertyTrainer(config)
    trainer.train_all_systems()
    
    # ========================================================================
    # PHASE 5: EVALUATION (Weeks 15-17)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: MODEL EVALUATION")
    logger.info("="*80)
    
    checker = PerformanceChecker(config)
    results = checker.check_all_targets()
    
    # ========================================================================
    # PHASE 6: INTERPRETATION (Week 16)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 6: SHAP INTERPRETATION")
    logger.info("="*80)
    
    # Generate SHAP analysis for key properties
    # (Implementation depends on trained models - see scripts/06_interpret_results.py)
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("✓ PIPELINE EXECUTION COMPLETE")
    logger.info("="*80)
    
    logger.info("\nResults saved to:")
    logger.info(f"  Models: {config['paths']['models']}")
    logger.info(f"  Predictions: {config['paths']['predictions']}")
    logger.info(f"  Figures: {config['paths']['figures']}")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review evaluation metrics in results/metrics/")
    logger.info("  2. Examine SHAP plots in results/figures/shap/")
    logger.info("  3. Run scripts/07_generate_report.py for publication figures")

if __name__ == "__main__":
    main()
