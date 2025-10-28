"""
SHAP (SHapley Additive exPlanations) Analysis
Provides interpretable explanations for model predictions
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

class SHAPAnalyzer:
    """
    SHAP-based model interpretation for ceramic property prediction
    
    Generates:
    1. Feature importance (global)
    2. Dependence plots (feature interactions)
    3. Force plots (individual predictions)
    4. Waterfall plots (prediction breakdown)
    """
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained model
            model_type: 'tree' for tree-based models, 'linear' for linear models
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
        logger.info(f"SHAP Analyzer initialized for {model_type} model")
    
    def create_explainer(self, X_background: np.ndarray = None,
                        feature_names: list = None):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background dataset for TreeExplainer (optional)
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        
        if self.model_type == 'tree':
            # TreeExplainer for XGBoost, RF, CatBoost, GB
            if X_background is not None:
                # Use subset of data for faster computation
                background = shap.sample(X_background, min(100, len(X_background)))
                self.explainer = shap.TreeExplainer(self.model, background)
            else:
                self.explainer = shap.TreeExplainer(self.model)
            
            logger.info("✓ TreeExplainer created")
            
        elif self.model_type == 'linear':
            # LinearExplainer for linear models
            self.explainer = shap.LinearExplainer(self.model, X_background)
            logger.info("✓ LinearExplainer created")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def calculate_shap_values(self, X: np.ndarray, 
                            n_samples: Optional[int] = None) -> np.ndarray:
        """
        Calculate SHAP values for dataset
        
        Args:
            X: Feature matrix
            n_samples: Number of samples to calculate (None = all)
        
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Subsample if requested
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        logger.info(f"Calculating SHAP values for {len(X_sample)} samples...")
        
        self.shap_values = self.explainer.shap_values(X_sample)
        
        logger.info("✓ SHAP values calculated")
        
        return self.shap_values
    
    def plot_summary(self, save_path: str = None, 
                    plot_type: str = 'dot', max_display: int = 20):
        """
        Create SHAP summary plot (feature importance)
        
        Args:
            save_path: Path to save figure
            plot_type: 'dot', 'bar', or 'violin'
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        plt.figure(figsize=(10, 8))
        
        shap.summary_plot(
            self.shap_values,
            features=X_sample if hasattr(self, 'X_sample') else None,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Summary plot saved: {save_path}")
        
        plt.close()
    
    def plot_dependence(self, feature_name: str, 
                       interaction_feature: str = None,
                       save_path: str = None):
        """
        Create SHAP dependence plot for specific feature
        
        Args:
            feature_name: Feature to plot
            interaction_feature: Feature to color by (auto-detected if None)
            save_path: Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        plt.figure(figsize=(10, 6))
        
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            features=X_sample if hasattr(self, 'X_sample') else None,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Dependence plot saved: {save_path}")
        
        plt.close()
    
    def plot_waterfall(self, sample_idx: int, save_path: str = None):
        """
        Create waterfall plot for individual prediction
        
        Args:
            sample_idx: Index of sample to explain
            save_path: Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        plt.figure(figsize=(10, 6))
        
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=X_sample[sample_idx] if hasattr(self, 'X_sample') else None,
                feature_names=self.feature_names
            ),
            show=False
        )
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Waterfall plot saved: {save_path}")
        
        plt.close()
    
    def get_feature_importance(self, importance_type: str = 'mean_abs') -> pd.DataFrame:
        """
        Calculate feature importance from SHAP values
        
        Args:
            importance_type: 'mean_abs' or 'mean'
        
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        if importance_type == 'mean_abs':
            importance = np.abs(self.shap_values).mean(axis=0)
        elif importance_type == 'mean':
            importance = self.shap_values.mean(axis=0)
        else:
            raise ValueError(f"Unknown importance type: {importance_type}")
        
        feature_names = self.feature_names or [f"Feature {i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def generate_all_plots(self, X: np.ndarray, output_dir: str,
                          top_features: int = 10):
        """
        Generate complete SHAP analysis with all plot types
        
        Args:
            X: Feature matrix
            output_dir: Directory to save plots
            top_features: Number of top features for dependence plots
        """
        logger.info("Generating complete SHAP analysis...")
        
        # Calculate SHAP values
        self.calculate_shap_values(X, n_samples=min(1000, len(X)))
        self.X_sample = X[:len(self.shap_values)]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Summary plot (dot)
        self.plot_summary(
            save_path=str(output_path / 'shap_summary_dot.png'),
            plot_type='dot'
        )
        
        # 2. Summary plot (bar)
        self.plot_summary(
            save_path=str(output_path / 'shap_summary_bar.png'),
            plot_type='bar'
        )
        
        # 3. Dependence plots for top features
        importance_df = self.get_feature_importance()
        top_feature_names = importance_df.head(top_features)['feature'].tolist()
        
        for feature in top_feature_names:
            safe_name = feature.replace('/', '_').replace(' ', '_')
            self.plot_dependence(
                feature,
                save_path=str(output_path / f'shap_dependence_{safe_name}.png')
            )
        
        # 4. Waterfall plots for representative samples
        sample_indices = [0, len(self.shap_values)//2, len(self.shap_values)-1]
        for idx in sample_indices:
            self.plot_waterfall(
                idx,
                save_path=str(output_path / f'shap_waterfall_sample_{idx}.png')
            )
        
        logger.info(f"✓ Complete SHAP analysis saved to {output_dir}")
