"""
Trader Clustering Model - Prediction Script

This script loads the trained trader clustering model and provides functions
to predict clusters for new trader data.

Usage:
    from trader_cluster_predictor import predict_trader_cluster, load_model
    
    # Predict cluster for a trader
    trader_data = {
        'total_trades_count': 1500,
        'total_sol_spent': 25000,
        'unique_coins_traded': 45,
        'win_rate': 0.65,
        'avg_pnl_per_position': 15.5,
        # ... include all required features
    }
    
    cluster_id = predict_trader_cluster(trader_data)
    print(f"Trader belongs to cluster: {cluster_id}")
"""

import pickle
import joblib
import json
import numpy as np
import pandas as pd
import os
from typing import Dict, Any, Tuple

# Default model directory
DEFAULT_MODEL_DIR = '/Users/noel/projects/trading_eda/solana/models'

class TraderClusterPredictor:
    """
    Trader clustering model predictor class
    """
    
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        """
        Initialize the predictor with model files
        
        Args:
            model_dir: Directory containing the saved model files
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.preprocessing = None
        self.metadata = None
        self._load_model()
    
    def _load_model(self):
        """Load all model components"""
        try:
            # Load K-means model
            with open(f"{self.model_dir}/trader_clustering_kmeans.pkl", 'rb') as f:
                self.model = pickle.load(f)
            
            # Load preprocessing components
            with open(f"{self.model_dir}/preprocessing_pipeline.pkl", 'rb') as f:
                self.preprocessing = pickle.load(f)
            
            # Load scaler
            self.scaler = joblib.load(f"{self.model_dir}/robust_scaler.pkl")
            
            # Load metadata
            with open(f"{self.model_dir}/model_metadata.json", 'r') as f:
                self.metadata = json.load(f)
            
            print(f"✓ Loaded trader clustering model")
            print(f"  - Trained on {self.metadata['n_training_samples']:,} traders")
            print(f"  - {self.metadata['n_clusters']} clusters with {self.metadata['n_features']} features")
            print(f"  - Training date: {self.metadata['training_date']}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found in {self.model_dir}. "
                                  f"Please ensure the model has been trained and saved. Error: {e}")
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature transformations used during training
        
        Args:
            df: DataFrame with trader features in original scale
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        def signed_log_transform(x):
            """Handle negative values with signed log transform"""
            return np.sign(x) * np.log1p(np.abs(x))
        
        # 1. Volume features - log1p transform
        for feature in self.preprocessing['transformation_params']['volume_log_features']:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature])
        
        # 2. Performance features with signed log transform
        for feature in self.preprocessing['transformation_params']['performance_signed_log_features']:
            if feature in df.columns:
                df[feature] = signed_log_transform(df[feature])
        
        # 3. Diversification features
        for feature in self.preprocessing['transformation_params']['sqrt_features']:
            if feature in df.columns:
                df[feature] = np.sqrt(df[feature])
        
        for feature in self.preprocessing['transformation_params']['log_per_coin_features']:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature])
        
        # 4. Timing features
        for feature in self.preprocessing['transformation_params']['timing_log_features']:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature])
        
        for feature in self.preprocessing['transformation_params']['timing_sqrt_features']:
            if feature in df.columns:
                df[feature] = np.sqrt(df[feature])
        
        # 5. Trade features
        for feature in self.preprocessing['transformation_params']['trade_log_features']:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature])
        
        for feature in self.preprocessing['transformation_params']['trade_sqrt_features']:
            if feature in df.columns:
                df[feature] = np.sqrt(df[feature])
        
        # 6. Performance log features
        for feature in self.preprocessing['transformation_params']['performance_log_features']:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature])
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using the same strategy as training
        
        Args:
            df: DataFrame with transformed features
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Handle avg_roi missing values (insider traders)
        if 'avg_roi' in df.columns:
            missing_roi_mask = pd.isna(df['avg_roi'])
            df.loc[missing_roi_mask, 'avg_roi'] = self.preprocessing['missing_value_fixes']['avg_roi_insider_marker']
            df['has_buy_history'] = (~missing_roi_mask).astype(int)
        else:
            df['has_buy_history'] = 1  # Default to having buy history
        
        # Handle avg_hours_between_trades missing values (single-trade accounts)
        if 'avg_hours_between_trades' in df.columns:
            missing_hours_mask = pd.isna(df['avg_hours_between_trades'])
            df.loc[missing_hours_mask, 'avg_hours_between_trades'] = self.preprocessing['missing_value_fixes']['avg_hours_max_plus_one']
            df['is_multi_trader'] = (~missing_hours_mask).astype(int)
        else:
            df['is_multi_trader'] = 1  # Default to multi-trader
        
        return df
    
    def predict(self, trader_features: Dict[str, Any]) -> int:
        """
        Predict cluster for a single trader
        
        Args:
            trader_features: Dictionary with trader features in original scale
            
        Returns:
            Predicted cluster ID (0 to n_clusters-1)
        """
        # Convert to DataFrame
        df = pd.DataFrame([trader_features])
        
        # Apply transformations
        df_transformed = self._apply_transformations(df)
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_transformed)
        
        # Ensure all required features are present
        feature_order = self.preprocessing['feature_names']
        for feature in feature_order:
            if feature not in df_processed.columns:
                df_processed[feature] = 0  # Default value for missing features
        
        # Select and order features correctly
        df_features = df_processed[feature_order]
        
        # Scale features
        X_scaled = self.scaler.transform(df_features)
        
        # Predict cluster
        cluster_pred = self.model.predict(X_scaled)[0]
        
        return int(cluster_pred)
    
    def predict_with_confidence(self, trader_features: Dict[str, Any]) -> Tuple[int, np.ndarray]:
        """
        Predict cluster with confidence scores
        
        Args:
            trader_features: Dictionary with trader features in original scale
            
        Returns:
            Tuple of (predicted_cluster_id, confidence_scores)
            confidence_scores: Array of scores for each cluster (higher = more confident)
        """
        # Convert to DataFrame and process
        df = pd.DataFrame([trader_features])
        df_transformed = self._apply_transformations(df)
        df_processed = self._handle_missing_values(df_transformed)
        
        # Prepare features
        feature_order = self.preprocessing['feature_names']
        for feature in feature_order:
            if feature not in df_processed.columns:
                df_processed[feature] = 0
        
        df_features = df_processed[feature_order]
        X_scaled = self.scaler.transform(df_features)
        
        # Predict cluster
        cluster_pred = self.model.predict(X_scaled)[0]
        
        # Calculate distances to each cluster center
        distances = self.model.transform(X_scaled)[0]
        
        # Convert distances to confidence scores (inverse of distance)
        confidence_scores = 1 / (1 + distances)
        confidence_scores = confidence_scores / confidence_scores.sum()  # Normalize
        
        return int(cluster_pred), confidence_scores
    
    def get_cluster_interpretation(self, cluster_id: int) -> str:
        """
        Get human-readable interpretation of cluster
        
        Args:
            cluster_id: Cluster ID (0 to n_clusters-1)
            
        Returns:
            String description of cluster characteristics
        """
        cluster_interpretations = {
            0: "Specialized/Niche Traders - Focused on specific strategies or tokens",
            1: "Active Retail Traders - Regular trading activity with moderate volumes", 
            2: "Casual/Small Traders - Occasional trading with small volumes",
            3: "Serious/Semi-Pro Traders - High activity with significant volumes",
            4: "Elite/Whale Traders - Very high volume professional trading"
        }
        
        return cluster_interpretations.get(cluster_id, f"Cluster {cluster_id}")
    
    def get_required_features(self) -> list:
        """
        Get list of required features for prediction
        
        Returns:
            List of feature names required for prediction
        """
        return self.preprocessing['feature_names']


# Convenience functions for direct usage
_predictor = None

def load_model(model_dir: str = DEFAULT_MODEL_DIR) -> TraderClusterPredictor:
    """
    Load the trader clustering model
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        TraderClusterPredictor instance
    """
    global _predictor
    _predictor = TraderClusterPredictor(model_dir)
    return _predictor

def predict_trader_cluster(trader_features: Dict[str, Any], model_dir: str = DEFAULT_MODEL_DIR) -> int:
    """
    Predict trader cluster (convenience function)
    
    Args:
        trader_features: Dictionary with trader features
        model_dir: Directory containing model files
        
    Returns:
        Predicted cluster ID
    """
    global _predictor
    if _predictor is None:
        _predictor = TraderClusterPredictor(model_dir)
    
    return _predictor.predict(trader_features)

def get_cluster_info(cluster_id: int) -> str:
    """
    Get cluster interpretation (convenience function)
    
    Args:
        cluster_id: Cluster ID
        
    Returns:
        Cluster description
    """
    global _predictor
    if _predictor is None:
        raise ValueError("Model not loaded. Call load_model() first.")
    
    return _predictor.get_cluster_interpretation(cluster_id)


if __name__ == "__main__":
    # Example usage
    print("=== Trader Cluster Predictor Example ===")
    
    # Load model
    predictor = load_model()
    
    # Example trader data
    example_trader = {
        'total_trades_count': 1500,
        'total_sol_spent': 25000,
        'total_sol_received': 26500,
        'unique_coins_traded': 45,
        'trading_span_days': 90,
        'win_rate': 0.65,
        'avg_pnl_per_position': 15.5,
        'avg_trades_per_coin': 33.3,
        'avg_sol_trade_size': 16.7,
        'median_sol_trade_size': 12.0,
        'max_single_sol_trade': 500.0,
        'min_sol_trade_size': 0.1,
        'sol_trade_size_std_dev': 25.8,
        'trade_size_coefficient_variation': 1.5,
        'net_sol_pnl': 1500.0,
        'trade_concentration_ratio': 0.3,
        'trades_per_day': 16.7,
        'avg_hours_between_trades': 1.5,
        'active_hours': 12,
        'active_days': 85,
        'trades_per_active_hour': 1.4,
        'round_number_preference': 0.1,
        'sol_to_token_trades': 750,
        'token_to_sol_trades': 750,
        'token_to_token_trades': 0,
        'unique_from_tokens_non_sol': 0,
        'unique_to_tokens_non_sol': 0,
        'sol_to_token_percentage': 0.5,
        'token_to_sol_percentage': 0.5,
        'token_to_token_percentage': 0.0,
        'buy_sell_ratio': 1.0,
        'total_positions': 45,
        'avg_roi': 0.06
    }
    
    # Predict cluster
    cluster = predictor.predict(example_trader)
    interpretation = predictor.get_cluster_interpretation(cluster)
    
    print(f"\nPredicted cluster: {cluster}")
    print(f"Interpretation: {interpretation}")
    
    # Predict with confidence
    cluster_conf, confidence_scores = predictor.predict_with_confidence(example_trader)
    print(f"\nConfidence scores for each cluster:")
    for i, score in enumerate(confidence_scores):
        print(f"  Cluster {i}: {score:.3f}")
    
    print(f"\nRequired features ({len(predictor.get_required_features())}):")
    for feature in predictor.get_required_features():
        print(f"  - {feature}")