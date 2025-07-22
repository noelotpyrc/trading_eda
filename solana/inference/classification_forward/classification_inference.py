#!/usr/bin/env python3
"""
Classification Forward Inference
Load classification models and score inference datasets for trading predictions
"""

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, List, Optional, Union
import os
import warnings
warnings.filterwarnings('ignore')

class ClassificationInference:
    """
    Classification inference engine for trading signal prediction
    """
    
    def __init__(self, model_dir: str = "solana/models/classification_forward"):
        """
        Initialize the classification inference engine
        
        Args:
            model_dir: Directory containing the classification model files
        """
        self.model_dir = model_dir
        self.model = None
        self.feature_names = None
        self.metadata = None
        self.feature_importance = None
        self._load_models()
    
    def _load_models(self):
        """Load all classification model components"""
        print(f"Loading classification models from: {self.model_dir}")
        
        try:
            # Load the trained model
            model_path = f"{self.model_dir}/best_model.pkl"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"✅ Model loaded: {type(self.model).__name__}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load feature names
            feature_names_path = f"{self.model_dir}/feature_names.pkl"
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
                print(f"✅ Feature names loaded: {len(self.feature_names)} features")
            else:
                print("⚠️  Feature names file not found, will infer from data")
            
            # Load metadata
            metadata_path = f"{self.model_dir}/model_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Metadata loaded")
            else:
                print("⚠️  Metadata file not found")
            
            # Load feature importance
            importance_path = f"{self.model_dir}/feature_importance.csv"
            if os.path.exists(importance_path):
                self.feature_importance = pd.read_csv(importance_path, index_col=0)
                print(f"✅ Feature importance loaded")
            else:
                print("⚠️  Feature importance file not found")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load classification models: {e}")
    
    def predict_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for trading signals
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [unprofitable, profitable]
        """
        X_processed = self._preprocess_features(X)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_processed)
        else:
            raise NotImplementedError("Model does not support probability prediction")
        
        return probabilities
    
    def predict_labels(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary trading signals (0=unprofitable, 1=profitable)
        
        Args:
            X: Feature dataframe
            threshold: Probability threshold for positive prediction
            
        Returns:
            Array of binary predictions
        """
        probabilities = self.predict_probabilities(X)
        return (probabilities[:, 1] >= threshold).astype(int)
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get continuous trading scores (probability of profitable trade)
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of scores between 0 and 1
        """
        probabilities = self.predict_probabilities(X)
        return probabilities[:, 1]  # Probability of profitable class
    
    def predict_full_output(self, X: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """
        Get comprehensive prediction output
        
        Args:
            X: Feature dataframe
            threshold: Classification threshold
            
        Returns:
            Dictionary containing:
            - scores: Probability of profitable trade
            - labels: Binary predictions
            - probabilities: Full probability matrix
            - confidence: Prediction confidence scores
            - high_confidence_mask: Boolean mask for high-confidence predictions
        """
        probabilities = self.predict_probabilities(X)
        scores = probabilities[:, 1]
        labels = (scores >= threshold).astype(int)
        
        # Calculate confidence as max probability
        confidence = np.max(probabilities, axis=1)
        
        # High confidence mask (e.g., confidence > 0.7)
        high_confidence_threshold = 0.7
        high_confidence_mask = confidence > high_confidence_threshold
        
        return {
            'scores': scores,
            'labels': labels,
            'probabilities': probabilities,
            'confidence': confidence,
            'high_confidence_mask': high_confidence_mask,
            'n_samples': len(X),
            'n_high_confidence': high_confidence_mask.sum(),
            'threshold_used': threshold
        }
    
    def batch_inference(self, data_files: List[str], 
                       output_dir: Optional[str] = None,
                       threshold: float = 0.5,
                       save_results: bool = True) -> Dict:
        """
        Run batch inference on multiple data files
        
        Args:
            data_files: List of CSV file paths to process
            output_dir: Directory to save results (optional)
            threshold: Classification threshold
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with results for each file
        """
        print(f"Running batch inference on {len(data_files)} files...")
        
        all_results = {}
        
        for i, file_path in enumerate(data_files, 1):
            print(f"\nProcessing file {i}/{len(data_files)}: {file_path}")
            
            try:
                # Load data
                df = pd.read_csv(file_path)
                print(f"  Loaded {len(df):,} samples")
                
                # Run inference
                results = self.predict_full_output(df, threshold=threshold)
                
                # Add file info
                results['file_path'] = file_path
                results['file_name'] = os.path.basename(file_path)
                
                # Calculate summary statistics
                profitable_pct = (results['labels'] == 1).mean() * 100
                high_conf_pct = results['high_confidence_mask'].mean() * 100
                avg_score = results['scores'].mean()
                
                results['summary'] = {
                    'profitable_predictions_pct': profitable_pct,
                    'high_confidence_pct': high_conf_pct,
                    'average_score': avg_score,
                    'min_score': results['scores'].min(),
                    'max_score': results['scores'].max()
                }
                
                print(f"  Profitable predictions: {profitable_pct:.1f}%")
                print(f"  High confidence predictions: {high_conf_pct:.1f}%")
                print(f"  Average score: {avg_score:.3f}")
                
                all_results[file_path] = results
                
                # Save individual results if requested
                if save_results and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'trading_score': results['scores'],
                        'predicted_label': results['labels'],
                        'prob_unprofitable': results['probabilities'][:, 0],
                        'prob_profitable': results['probabilities'][:, 1],
                        'confidence': results['confidence'],
                        'high_confidence': results['high_confidence_mask']
                    })
                    
                    # Save results
                    output_file = os.path.join(output_dir, f"inference_{os.path.basename(file_path)}")
                    results_df.to_csv(output_file, index=False)
                    print(f"  Results saved to: {output_file}")
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path}: {e}")
                all_results[file_path] = {'error': str(e)}
        
        print(f"\n✅ Batch inference completed!")
        return all_results
    
    def evaluate_predictions(self, X: pd.DataFrame, y_true: np.ndarray, 
                           threshold: float = 0.5) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Args:
            X: Feature dataframe
            y_true: True labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        results = self.predict_full_output(X, threshold=threshold)
        y_pred = results['labels']
        y_scores = results['scores']
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_scores),
            'threshold_used': threshold,
            'n_samples': len(y_true),
            'n_positive': (y_true == 1).sum(),
            'n_predicted_positive': (y_pred == 1).sum()
        }
        
        return metrics
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for model inference
        
        Args:
            X: Raw feature dataframe
            
        Returns:
            Preprocessed feature dataframe
        """
        # If we have feature names, ensure we use the right columns
        if self.feature_names is not None:
            if not all(col in X.columns for col in self.feature_names):
                missing_cols = [col for col in self.feature_names if col not in X.columns]
                available_cols = [col for col in self.feature_names if col in X.columns]
                print(f"⚠️  Missing {len(missing_cols)} features: {missing_cols[:5]}...")
                print(f"   Using {len(available_cols)} available features")
                
                # Use only available features
                X_features = X[available_cols].copy()
            else:
                # Use all required features in correct order
                X_features = X[self.feature_names].copy()
        else:
            # Use all numeric columns if feature names not available
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_features = X[numeric_cols].copy()
        
        # Clean data (handle missing values and infinities)
        X_features = X_features.fillna(0)
        X_features = X_features.replace([np.inf, -np.inf], 0)
        
        return X_features
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        info = {
            'model_type': type(self.model).__name__ if self.model else 'Not loaded',
            'n_features': len(self.feature_names) if self.feature_names else 'Unknown',
            'has_feature_importance': self.feature_importance is not None,
            'has_metadata': self.metadata is not None
        }
        
        if self.metadata:
            info.update(self.metadata)
        
        return info
    
    def get_top_features(self, n_features: int = 10) -> pd.DataFrame:
        """
        Get top important features
        
        Args:
            n_features: Number of top features to return
            
        Returns:
            DataFrame with top features and their importance scores
        """
        if self.feature_importance is None:
            return pd.DataFrame({'message': ['Feature importance not available']})
        
        # Assume the importance dataframe has an 'Average' column or similar
        importance_cols = [col for col in self.feature_importance.columns if 'average' in col.lower()]
        if not importance_cols:
            # Try to find any numeric column
            importance_cols = self.feature_importance.select_dtypes(include=[np.number]).columns.tolist()
        
        if importance_cols:
            importance_col = importance_cols[0]
            top_features = self.feature_importance.nlargest(n_features, importance_col)
            return top_features[[importance_col]].reset_index()
        else:
            return pd.DataFrame({'message': ['No suitable importance column found']})


def demo_usage():
    """Demonstrate usage of the classification inference engine"""
    print("=== CLASSIFICATION INFERENCE DEMO ===")
    
    # Initialize inference engine
    try:
        inference = ClassificationInference()
        
        # Show model info
        print("\n1. Model Information:")
        info = inference.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Show top features
        print("\n2. Top 10 Most Important Features:")
        top_features = inference.get_top_features(10)
        if 'message' not in top_features.columns:
            for _, row in top_features.iterrows():
                print(f"   {row.iloc[0]}: {row.iloc[1]:.4f}")
        else:
            print(f"   {top_features.iloc[0]['message']}")
        
        # Try to load sample data for inference
        try:
            data_file = "/Volumes/Extreme SSD/trading_data/solana/classification_features/combined_features_sql.csv"
            print(f"\n3. Loading sample data: {data_file}")
            df = pd.read_csv(data_file)
            
            # Take a small sample
            sample_size = 1000
            df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
            print(f"   Using {len(df_sample)} samples for demo")
            
            # Run inference
            print("\n4. Running inference...")
            results = inference.predict_full_output(df_sample, threshold=0.6)
            
            print(f"   Processed {results['n_samples']} samples")
            print(f"   Profitable predictions: {(results['labels'] == 1).mean()*100:.1f}%")
            print(f"   High confidence predictions: {results['n_high_confidence']} ({results['n_high_confidence']/results['n_samples']*100:.1f}%)")
            print(f"   Average trading score: {results['scores'].mean():.3f}")
            print(f"   Score range: [{results['scores'].min():.3f}, {results['scores'].max():.3f}]")
            
            # Show sample predictions
            print(f"\n5. Sample predictions (first 10 rows):")
            sample_df = pd.DataFrame({
                'trading_score': results['scores'][:10],
                'predicted_label': results['labels'][:10],
                'confidence': results['confidence'][:10],
                'high_confidence': results['high_confidence_mask'][:10]
            })
            print(sample_df.round(3))
            
            # If we have ground truth, evaluate
            if 'is_profitable_300s' in df_sample.columns:
                print(f"\n6. Model Evaluation:")
                y_true = df_sample['is_profitable_300s'].values
                metrics = inference.evaluate_predictions(df_sample, y_true, threshold=0.6)
                
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"   {metric}: {value:.3f}")
                    else:
                        print(f"   {metric}: {value}")
        
        except Exception as e:
            print(f"❌ Data loading/inference failed: {e}")
        
        print(f"\n✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")


if __name__ == "__main__":
    demo_usage()