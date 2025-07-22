#!/usr/bin/env python3
"""
Regime Classification Inference
Provides regime labels and numerical values (distances, probabilities) from the clustering model
"""

import pandas as pd
import numpy as np
import joblib
import json
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class RegimeClassifier:
    """
    Enhanced regime classifier that provides both categorical labels and numerical values
    """
    
    def __init__(self, model_dir: str = "solana/models/regime_clustering"):
        """
        Initialize the regime classifier
        
        Args:
            model_dir: Directory containing the clustering model files
        """
        self.model_dir = model_dir
        self.regime_classifier = None
        self.feature_scaler = None
        self.feature_names = None
        self.metadata = None
        self.n_regimes = None
        self._load_models()
    
    def _load_models(self):
        """Load all clustering model components"""
        print(f"Loading regime classification models from: {self.model_dir}")
        
        try:
            # Load clustering model
            self.regime_classifier = joblib.load(f"{self.model_dir}/regime_classifier.pkl")
            
            # Load feature scaler
            self.feature_scaler = joblib.load(f"{self.model_dir}/feature_scaler.pkl")
            
            # Load feature names
            self.feature_names = joblib.load(f"{self.model_dir}/feature_names.pkl")
            
            # Load metadata
            with open(f"{self.model_dir}/clustering_metadata.json", 'r') as f:
                self.metadata = json.load(f)
            
            self.n_regimes = self.metadata.get('n_regimes', 3)
            
            print(f"✅ Models loaded successfully")
            print(f"   Method: {self.metadata.get('method', 'unknown')}")
            print(f"   Number of regimes: {self.n_regimes}")
            print(f"   Number of features: {len(self.feature_names)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load clustering models: {e}")
    
    def predict_regime_labels(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regime labels only
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of regime labels (0, 1, 2, ...)
        """
        X_processed = self._preprocess_features(X)
        return self.regime_classifier.predict(X_processed)
    
    def predict_regime_distances(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get distances to each regime center
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of shape (n_samples, n_regimes) with distances to each center
        """
        X_processed = self._preprocess_features(X)
        
        if hasattr(self.regime_classifier, 'transform'):
            # For K-means, transform gives distances to centers
            distances = self.regime_classifier.transform(X_processed)
        else:
            raise NotImplementedError("Distance computation not available for this clustering method")
        
        return distances
    
    def predict_regime_probabilities(self, X: pd.DataFrame, temperature: float = 1.0) -> np.ndarray:
        """
        Convert distances to soft probability assignments
        
        Args:
            X: Feature dataframe
            temperature: Temperature parameter for softmax (lower = more confident)
            
        Returns:
            Array of shape (n_samples, n_regimes) with probabilities for each regime
        """
        distances = self.predict_regime_distances(X)
        
        # Convert distances to probabilities using negative exponential
        # Smaller distance = higher probability
        neg_distances = -distances / temperature
        
        # Apply softmax to get probabilities
        exp_distances = np.exp(neg_distances)
        probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
        
        return probabilities
    
    def predict_full_output(self, X: pd.DataFrame, temperature: float = 1.0) -> Dict:
        """
        Get comprehensive regime classification output
        
        Args:
            X: Feature dataframe
            temperature: Temperature for probability calculation
            
        Returns:
            Dictionary containing:
            - labels: Hard regime assignments
            - distances: Distances to each regime center
            - probabilities: Soft probability assignments
            - confidence: Confidence scores (max probability - second max)
            - closest_regime: Most likely regime for each sample
        """
        X_processed = self._preprocess_features(X)
        
        # Get labels
        labels = self.regime_classifier.predict(X_processed)
        
        # Get distances
        distances = self.predict_regime_distances(X)
        
        # Get probabilities
        probabilities = self.predict_regime_probabilities(X, temperature)
        
        # Calculate confidence (difference between top 2 probabilities)
        sorted_probs = np.sort(probabilities, axis=1)
        confidence = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # Get closest regime (should match labels, but computed from probabilities)
        closest_regime = np.argmax(probabilities, axis=1)
        
        return {
            'labels': labels,
            'distances': distances,
            'probabilities': probabilities,
            'confidence': confidence,
            'closest_regime': closest_regime,
            'n_samples': len(X),
            'n_regimes': self.n_regimes
        }
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features for clustering
        
        Args:
            X: Raw feature dataframe
            
        Returns:
            Preprocessed feature array
        """
        # Ensure we have the right features
        if not all(col in X.columns for col in self.feature_names):
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            raise ValueError(f"Missing required features: {missing_cols}")
        
        # Select and order features
        X_features = X[self.feature_names].copy()
        
        # Clean data (same as training)
        X_features = X_features.fillna(0)
        X_features = X_features.replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X_features)
        
        return X_scaled
    
    def analyze_regime_characteristics(self, X: pd.DataFrame, 
                                     regime_output: Optional[Dict] = None) -> pd.DataFrame:
        """
        Analyze characteristics of each regime
        
        Args:
            X: Feature dataframe
            regime_output: Pre-computed regime output (optional)
            
        Returns:
            DataFrame with regime characteristics
        """
        if regime_output is None:
            regime_output = self.predict_full_output(X)
        
        labels = regime_output['labels']
        probabilities = regime_output['probabilities']
        
        characteristics = []
        
        for regime in range(self.n_regimes):
            regime_mask = labels == regime
            regime_samples = regime_mask.sum()
            
            if regime_samples == 0:
                continue
            
            # Basic statistics
            regime_data = X[regime_mask]
            avg_confidence = probabilities[regime_mask, regime].mean()
            
            # Feature means for this regime
            feature_means = regime_data[self.feature_names].mean()
            
            characteristics.append({
                'regime': regime,
                'sample_count': regime_samples,
                'percentage': regime_samples / len(X) * 100,
                'avg_confidence': avg_confidence,
                'top_features': feature_means.nlargest(5).to_dict(),
                'bottom_features': feature_means.nsmallest(5).to_dict()
            })
        
        return pd.DataFrame(characteristics)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'method': self.metadata.get('method', 'unknown'),
            'n_regimes': self.n_regimes,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names.tolist(),
            'silhouette_score': self.metadata.get('silhouette_score'),
            'target_separation': self.metadata.get('target_separation'),
            'training_date': self.metadata.get('training_date')
        }


def demo_usage():
    """Demonstrate usage of the regime classifier"""
    print("=== REGIME CLASSIFIER INFERENCE DEMO ===")
    
    # Initialize classifier
    classifier = RegimeClassifier()
    
    # Load some sample data (you can replace this with your own data)
    try:
        # Try to load sample data
        data_file = "/Volumes/Extreme SSD/trading_data/solana/classification_features/combined_features_sql.csv"
        print(f"Loading sample data from: {data_file}")
        df = pd.read_csv(data_file)
        
        # Take a small sample for demo
        sample_size = 1000
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"Using {len(df_sample)} samples for demo")
        
        # Get comprehensive output
        print("\n1. Getting comprehensive regime classification...")
        regime_output = classifier.predict_full_output(df_sample, temperature=0.5)
        
        print(f"   Processed {regime_output['n_samples']} samples")
        print(f"   Found {regime_output['n_regimes']} regimes")
        
        # Show regime distribution
        unique_regimes, counts = np.unique(regime_output['labels'], return_counts=True)
        print("\n2. Regime distribution:")
        for regime, count in zip(unique_regimes, counts):
            pct = count / len(df_sample) * 100
            print(f"   Regime {regime}: {count} samples ({pct:.1f}%)")
        
        # Show confidence statistics
        print(f"\n3. Confidence statistics:")
        print(f"   Mean confidence: {regime_output['confidence'].mean():.3f}")
        print(f"   Min confidence: {regime_output['confidence'].min():.3f}")
        print(f"   Max confidence: {regime_output['confidence'].max():.3f}")
        
        # Show sample outputs
        print(f"\n4. Sample outputs (first 5 rows):")
        sample_df = pd.DataFrame({
            'regime_label': regime_output['labels'][:5],
            'confidence': regime_output['confidence'][:5],
            'prob_regime_0': regime_output['probabilities'][:5, 0],
            'prob_regime_1': regime_output['probabilities'][:5, 1],
            'prob_regime_2': regime_output['probabilities'][:5, 2]
        })
        print(sample_df.round(3))
        
        # Analyze regime characteristics
        print(f"\n5. Regime characteristics analysis:")
        char_df = classifier.analyze_regime_characteristics(df_sample, regime_output)
        for _, row in char_df.iterrows():
            print(f"   Regime {row['regime']}: {row['sample_count']} samples ({row['percentage']:.1f}%), "
                  f"avg confidence: {row['avg_confidence']:.3f}")
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please ensure the data file exists and contains the required features")


if __name__ == "__main__":
    demo_usage()