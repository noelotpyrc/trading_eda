#!/usr/bin/env python3
"""
Part 3: Train Regime-Specific Models
Train all four models (RF, XGBoost, LightGBM, Logistic) for each regime subset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = "/Volumes/Extreme SSD/trading_data/solana/classification_features/combined_features_sql.csv"
CLUSTERING_DIR = "solana/models/regime_clustering"
# CORRELATION_DIR = "solana/analysis/regime_correlations"  # No longer needed
OUTPUT_DIR = "/Volumes/Extreme SSD/trading_data/solana/models/regime_specific"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SAMPLE_SIZE = None  # Set to integer (e.g., 500000) to sample data for testing, None for full dataset

print("=== PART 3: REGIME-SPECIFIC MODEL TRAINING ===")
print("Objective: Train specialized models for each market regime")
print(f"Data source: {FEATURES_FILE}")
print(f"Clustering model: {CLUSTERING_DIR}")
if SAMPLE_SIZE is not None:
    print(f"Sample size: {SAMPLE_SIZE:,} records (testing mode)")
else:
    print("Sample size: Full dataset (production mode)")
print()

def load_data_and_setup():
    """Load data, regime labels, and high-signal regime info"""
    print("Loading data and regime configuration...")
    
    # Load clustering components
    regime_classifier = joblib.load(f"{CLUSTERING_DIR}/regime_classifier.pkl")
    scaler = joblib.load(f"{CLUSTERING_DIR}/feature_scaler.pkl")
    feature_names = joblib.load(f"{CLUSTERING_DIR}/feature_names.pkl")
    
    # Train models for all regimes (no high-signal filtering)
    high_signal_regimes = []  # Train on all regimes
    print("Training models for all regimes")
    
    # Load and prepare data
    df = pd.read_csv(FEATURES_FILE)
    print(f"Original dataset: {len(df):,} records")
    
    # Optional sampling for testing
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE:,} records for testing...")
        sample_indices = np.random.choice(len(df), SAMPLE_SIZE, replace=False)
        df = df.iloc[sample_indices].reset_index(drop=True)
        print(f"Using sampled dataset: {len(df):,} records")
    else:
        print(f"Using full dataset: {len(df):,} records")
    
    # Prepare features
    X = df[feature_names].copy()
    y = df['is_profitable_300s'].copy()
    
    # Clean data
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Apply regime clustering to dataset
    print("Applying regime clustering to dataset...")
    X_scaled = scaler.transform(X)
    regime_labels = regime_classifier.predict(X_scaled)
    
    # Show regime distribution
    unique_regimes, counts = np.unique(regime_labels, return_counts=True)
    print(f"Regime distribution in dataset:")
    for regime, count in zip(unique_regimes, counts):
        pct = count / len(regime_labels) * 100
        signal_type = "HIGH SIGNAL" if regime in high_signal_regimes else "Normal"
        print(f"  Regime {regime}: {count:,} samples ({pct:.1f}%) - {signal_type}")
    
    return X, y, regime_labels, high_signal_regimes, unique_regimes

def create_model_configs():
    """Create model configurations for training"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            verbosity=0
        ),
        
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            verbose=-1
        ),
        
        'Logistic Regression': LogisticRegression(
            C=0.1,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
    return models

def train_regime_models(X, y, regime_labels, regimes_to_train):
    """Train models for each specified regime"""
    print(f"\n=== TRAINING MODELS FOR EACH REGIME ===")
    
    regime_results = {}
    
    for regime in regimes_to_train:
        print(f"\n--- TRAINING MODELS FOR REGIME {regime} ---")
        
        # Get regime data
        regime_mask = regime_labels == regime
        regime_samples = regime_mask.sum()
        
        if regime_samples < 1000:
            print(f"Skipping Regime {regime}: insufficient samples ({regime_samples})")
            continue
        
        X_regime = X[regime_mask]
        y_regime = y[regime_mask]
        
        print(f"Regime {regime} samples: {regime_samples:,}")
        print(f"Target distribution: {y_regime.value_counts().to_dict()}")
        
        # Check if we have both classes
        if y_regime.nunique() < 2:
            print(f"Skipping Regime {regime}: only one class present")
            continue
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_regime, y_regime, test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, stratify=y_regime
        )
        
        regime_model_results = {}
        
        # Create fresh model instances for this regime to avoid cross-contamination
        models = create_model_configs()
        
        # Train each model type
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            test_auc = roc_auc_score(y_test, y_prob)
            test_accuracy = (y_pred == y_test).mean()
            
            # Cross-validation
            if len(X_train) > 5000:  # Only CV on larger datasets
                cv_scores = cross_val_score(model, X_train, y_train, 
                                          cv=CV_FOLDS, scoring='roc_auc')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                cv_mean, cv_std = test_auc, 0.0
            
            training_time = time.time() - start_time
            
            regime_model_results[model_name] = {
                'model': model,
                'test_auc': test_auc,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                # Store the exact train-test splits used for this model
                'X_train': X_train.copy(),
                'X_test': X_test.copy(), 
                'y_train': y_train.copy(),
                'y_test': y_test.copy()
            }
            
            print(f"    AUC: {test_auc:.4f}, CV: {cv_mean:.4f}±{cv_std:.4f}, Time: {training_time:.1f}s")
        
        regime_results[regime] = regime_model_results
    
    return regime_results

def load_existing_baseline_results():
    """Load existing baseline model results instead of retraining"""
    print(f"\n=== LOADING EXISTING BASELINE MODEL RESULTS ===")
    
    # Use existing model performance from classification_forward
    baseline_results = {
        'XGBoost': {
            'test_auc': 0.6683,  # From existing model metadata
            'test_accuracy': 0.6113,
            'cv_mean': 0.6665,
            'cv_std': 0.0021,
            'training_time': 0,  # Already trained
            'train_samples': 2487093,  # 80% of 3.1M
            'test_samples': 621773     # 20% of 3.1M  
        }
    }
    
    print("✅ Using existing XGBoost baseline (AUC: 0.668)")
    print("Note: Skipping RF, LightGBM, Logistic baseline retraining")
    
    return baseline_results

def train_baseline_models_original(X, y):
    """Original baseline training function (unused)"""
    print(f"\n=== TRAINING BASELINE MODELS (FULL DATASET) ===")
    
    models = create_model_configs()
    baseline_results = {}
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Baseline training samples: {len(X_train):,}")
    print(f"Baseline test samples: {len(X_test):,}")
    
    for model_name, model in models.items():
        print(f"Training baseline {model_name}...")
        
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_prob)
        test_accuracy = (y_pred == y_test).mean()
        
        # CV on sample (full CV would be too slow)
        if len(X_train) > 50000:
            # Sample for CV
            sample_size = 50000
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_cv_sample = X_train.iloc[indices]
            y_cv_sample = y_train.iloc[indices]
        else:
            X_cv_sample = X_train
            y_cv_sample = y_train
        
        cv_scores = cross_val_score(model, X_cv_sample, y_cv_sample, 
                                  cv=CV_FOLDS, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        training_time = time.time() - start_time
        
        baseline_results[model_name] = {
            'model': model,
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"  AUC: {test_auc:.4f}, CV: {cv_mean:.4f}±{cv_std:.4f}, Time: {training_time:.1f}s")
    
    return baseline_results

def save_datasets_and_splits(X, y, regime_labels, regime_results, regimes_trained):
    """Save datasets and train-test splits for each regime"""
    print(f"\n=== SAVING DATASETS AND SPLITS ===")
    
    import os
    datasets_dir = f"{OUTPUT_DIR}/datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Save full dataset and regime labels
    print("Saving regime labels...")
    np.save(f"{datasets_dir}/regime_labels.npy", regime_labels)
    
    # Save dataset metadata
    dataset_metadata = {
        'total_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': X.columns.tolist(),
        'target_distribution': y.value_counts().to_dict(),
        'regime_distribution': {int(regime): int(count) for regime, count in zip(*np.unique(regime_labels, return_counts=True))},
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
        'save_date': pd.Timestamp.now().isoformat()
    }
    
    with open(f"{datasets_dir}/dataset_metadata.json", 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    
    # Save train-test splits for each regime using the exact splits from model training
    print("Saving train-test splits for each regime...")
    for regime in regimes_trained:
        if regime not in regime_results:
            continue
            
        print(f"  Saving splits for Regime {regime}...")
        regime_dir = f"{datasets_dir}/regime_{regime}"
        os.makedirs(regime_dir, exist_ok=True)
        
        # Get the exact train-test splits that were used during training
        # Use the splits from the first model (all models use the same splits)
        first_model_name = list(regime_results[regime].keys())[0]
        model_result = regime_results[regime][first_model_name]
        
        X_train = model_result['X_train']
        X_test = model_result['X_test']
        y_train = model_result['y_train']
        y_test = model_result['y_test']
        
        # Save the exact splits that were used during training
        X_train.to_csv(f"{regime_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{regime_dir}/X_test.csv", index=False) 
        y_train.to_csv(f"{regime_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{regime_dir}/y_test.csv", index=False)
        
        # Create and save regime subset (combine train+test back)
        X_regime = pd.concat([X_train, X_test], ignore_index=True)
        y_regime = pd.concat([y_train, y_test], ignore_index=True)
        
        X_regime.to_csv(f"{regime_dir}/X_regime_full.csv", index=False)
        y_regime.to_csv(f"{regime_dir}/y_regime_full.csv", index=False)
        
        # Save split metadata
        split_metadata = {
            'regime': int(regime),
            'total_regime_samples': len(X_regime),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_target_distribution': y_train.value_counts().to_dict(),
            'test_target_distribution': y_test.value_counts().to_dict(),
            'regime_percentage': len(X_regime) / len(X) * 100,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'note': 'These are the exact train-test splits used during model training'
        }
        
        with open(f"{regime_dir}/split_metadata.json", 'w') as f:
            json.dump(split_metadata, f, indent=2)
    
    print(f"✅ Datasets and splits saved to: {datasets_dir}")
    return datasets_dir

def save_regime_models(regime_results, baseline_results, regimes_trained):
    """Save all trained models and results"""
    print(f"\n=== SAVING REGIME-SPECIFIC MODELS ===")
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save regime-specific models
    for regime, model_results in regime_results.items():
        regime_dir = f"{OUTPUT_DIR}/regime_{regime}"
        os.makedirs(regime_dir, exist_ok=True)
        
        for model_name, result in model_results.items():
            model_filename = f"{regime_dir}/{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(result['model'], model_filename)
        
        print(f"✅ Saved models for Regime {regime}")
    
    # Skip saving baseline models (using existing ones)
    baseline_dir = f"{OUTPUT_DIR}/baseline"
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Check if baseline results have model objects to save
    baseline_models_saved = 0
    for model_name, result in baseline_results.items():
        if 'model' in result:
            model_filename = f"{baseline_dir}/{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(result['model'], model_filename)
            baseline_models_saved += 1
    
    if baseline_models_saved > 0:
        print(f"✅ Saved {baseline_models_saved} baseline models")
    else:
        print(f"✅ Using existing baseline models (no new models to save)")
    
    # Prepare results for JSON (remove model objects and dataset splits)
    def prepare_results_for_json(results):
        json_results = {}
        for key, model_results in results.items():
            json_results[key] = {}
            for model_name, result in model_results.items():
                json_results[key][model_name] = {
                    k: v for k, v in result.items() 
                    if k not in ['model', 'X_train', 'X_test', 'y_train', 'y_test']
                }
        return json_results
    
    # Save results metadata (convert numpy types for JSON)
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(key): convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    all_results = {
        'regime_results': convert_numpy_types(prepare_results_for_json(regime_results)),
        'baseline_results': convert_numpy_types({k: {kk: vv for kk, vv in v.items() if kk != 'model'} 
                           for k, v in baseline_results.items()}),
        'regimes_trained': [int(r) for r in regimes_trained],  # Convert to standard int
        'training_date': pd.Timestamp.now().isoformat(),
        'model_types': list(create_model_configs().keys()),
        'config': {
            'test_size': TEST_SIZE,
            'cv_folds': CV_FOLDS,
            'random_state': RANDOM_STATE
        }
    }
    
    with open(f"{OUTPUT_DIR}/training_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Saved training results metadata")
    
    return all_results

def create_training_summary(regime_results, baseline_results, regimes_trained):
    """Create a summary of training results"""
    print(f"\n=== TRAINING SUMMARY ===")
    
    print(f"Models trained: {list(create_model_configs().keys())}")
    print(f"Regimes trained: {regimes_trained}")
    
    print(f"\n{'Model':<20} {'Baseline AUC':<12} {'Best Regime AUC':<15} {'Improvement':<12}")
    print("-" * 65)
    
    model_names = list(create_model_configs().keys())
    
    for model_name in model_names:
        # Check if baseline exists for this model
        if model_name in baseline_results:
            baseline_auc = baseline_results[model_name]['test_auc']
        else:
            baseline_auc = 0.0  # No baseline available
        
        # Find best regime performance for this model
        best_regime_auc = 0
        for regime in regimes_trained:
            if regime in regime_results and model_name in regime_results[regime]:
                regime_auc = regime_results[regime][model_name]['test_auc']
                best_regime_auc = max(best_regime_auc, regime_auc)
        
        improvement = best_regime_auc - baseline_auc
        
        if baseline_auc > 0:
            print(f"{model_name:<20} {baseline_auc:<12.4f} {best_regime_auc:<15.4f} {improvement:+<12.4f}")
        else:
            print(f"{model_name:<20} {'N/A':<12} {best_regime_auc:<15.4f} {'N/A':<12}")
    
    # Overall best performance
    all_regime_aucs = []
    all_baseline_aucs = []
    
    for model_name in model_names:
        if model_name in baseline_results:
            all_baseline_aucs.append(baseline_results[model_name]['test_auc'])
        
        for regime in regimes_trained:
            if regime in regime_results and model_name in regime_results[regime]:
                all_regime_aucs.append(regime_results[regime][model_name]['test_auc'])
    
    if all_regime_aucs:
        print(f"\nOverall Performance:")
        if all_baseline_aucs:
            print(f"  Best baseline AUC: {max(all_baseline_aucs):.4f}")
            print(f"  Best regime-specific AUC: {max(all_regime_aucs):.4f}")
            print(f"  Maximum improvement: {max(all_regime_aucs) - max(all_baseline_aucs):+.4f}")
        else:
            print(f"  Best regime-specific AUC: {max(all_regime_aucs):.4f}")
            print(f"  No baseline comparison available")

def main():
    """Main regime-specific model training pipeline"""
    print("Starting regime-specific model training...\n")
    
    # Load data and setup
    X, y, regime_labels, high_signal_regimes, all_regimes = load_data_and_setup()
    
    # Determine which regimes to train (prioritize high-signal regimes)
    regimes_to_train = high_signal_regimes if high_signal_regimes else all_regimes
    print(f"Training models for regimes: {regimes_to_train}")
    
    # Train regime-specific models
    regime_results = train_regime_models(X, y, regime_labels, regimes_to_train)
    
    # Load existing baseline results (skip retraining)
    baseline_results = load_existing_baseline_results()
    
    # Save datasets and train-test splits
    datasets_dir = save_datasets_and_splits(X, y, regime_labels, regime_results, regimes_to_train)
    
    # Save models and results
    all_results = save_regime_models(regime_results, baseline_results, regimes_to_train)
    
    # Create summary
    create_training_summary(regime_results, baseline_results, regimes_to_train)
    
    print(f"\n" + "="*60)
    print("✅ REGIME-SPECIFIC MODEL TRAINING COMPLETED!")
    print("="*60)
    print(f"Regimes trained: {len(regime_results)}")
    print(f"Models per regime: {len(create_model_configs())}")
    print(f"Total models trained: {len(regime_results) * len(create_model_configs()) + len(create_model_configs())}")
    print(f"Models saved to: {OUTPUT_DIR}")
    print(f"Datasets saved to: {datasets_dir}")
    print("="*60)
    print("\nNext step:")
    print("1. Run 04_regime_model_comparison.py to compare performance")

if __name__ == "__main__":
    main()