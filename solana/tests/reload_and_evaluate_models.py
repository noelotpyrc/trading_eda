#!/usr/bin/env python3
"""
Reload and Evaluate Saved Models
Load all saved datasets, models, and regenerate evaluation results
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for consistency
np.random.seed(42)

# Configuration
SAVED_MODELS_DIR = "/Volumes/Extreme SSD/trading_data/solana/models/regime_specific"
RANDOM_STATE = 42
CV_FOLDS = 5

print("=== RELOAD AND EVALUATE SAVED MODELS ===")
print("Objective: Load saved models and datasets, regenerate evaluation results")
print(f"Saved models directory: {SAVED_MODELS_DIR}")
print()

def load_saved_datasets():
    """Load all saved datasets and metadata"""
    print("Loading saved datasets...")
    
    datasets_dir = f"{SAVED_MODELS_DIR}/datasets"
    
    # Load dataset metadata
    with open(f"{datasets_dir}/dataset_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Dataset metadata loaded:")
    print(f"  Total samples: {metadata['total_samples']:,}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Regimes: {list(metadata['regime_distribution'].keys())}")
    
    # Load regime labels
    regime_labels = np.load(f"{datasets_dir}/regime_labels.npy")
    
    return metadata, regime_labels, datasets_dir

def load_regime_datasets(datasets_dir, regimes):
    """Load train-test splits for each regime"""
    print("\nLoading regime-specific datasets...")
    
    regime_datasets = {}
    
    for regime in regimes:
        regime_dir = f"{datasets_dir}/regime_{regime}"
        
        if not os.path.exists(regime_dir):
            print(f"  Regime {regime}: Directory not found, skipping")
            continue
        
        print(f"  Loading Regime {regime} datasets...")
        
        # Load split metadata
        with open(f"{regime_dir}/split_metadata.json", 'r') as f:
            split_metadata = json.load(f)
        
        # Load train-test splits
        X_train = pd.read_csv(f"{regime_dir}/X_train.csv")
        X_test = pd.read_csv(f"{regime_dir}/X_test.csv")
        y_train = pd.read_csv(f"{regime_dir}/y_train.csv").iloc[:, 0]  # First column
        y_test = pd.read_csv(f"{regime_dir}/y_test.csv").iloc[:, 0]    # First column
        
        # Apply same data cleaning as in training script
        X_train = X_train.fillna(0)
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        
        # Load full regime data
        X_regime_full = pd.read_csv(f"{regime_dir}/X_regime_full.csv")
        y_regime_full = pd.read_csv(f"{regime_dir}/y_regime_full.csv").iloc[:, 0]
        
        # Apply same data cleaning to full regime data
        X_regime_full = X_regime_full.fillna(0)
        X_regime_full = X_regime_full.replace([np.inf, -np.inf], 0)
        
        regime_datasets[regime] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_regime_full': X_regime_full,
            'y_regime_full': y_regime_full,
            'metadata': split_metadata
        }
        
        print(f"    Train: {len(X_train):,}, Test: {len(X_test):,}, Total: {len(X_regime_full):,}")
    
    return regime_datasets

def load_saved_models(regimes):
    """Load all saved models for each regime"""
    print("\nLoading saved models...")
    
    model_types = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
    model_name_map = {
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM', 
        'logistic_regression': 'Logistic Regression'
    }
    
    loaded_models = {}
    
    for regime in regimes:
        regime_models_dir = f"{SAVED_MODELS_DIR}/regime_{regime}"
        
        if not os.path.exists(regime_models_dir):
            print(f"  Regime {regime}: Models directory not found, skipping")
            continue
        
        print(f"  Loading models for Regime {regime}...")
        regime_models = {}
        
        for model_file in model_types:
            model_path = f"{regime_models_dir}/{model_file}.pkl"
            
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    model_name = model_name_map[model_file]
                    regime_models[model_name] = model
                    print(f"    ✅ {model_name}")
                except Exception as e:
                    print(f"    ❌ {model_file}: {e}")
            else:
                print(f"    ❌ {model_file}: File not found")
        
        if regime_models:
            loaded_models[regime] = regime_models
    
    return loaded_models

def evaluate_loaded_models(regime_datasets, loaded_models):
    """Evaluate all loaded models on their respective test sets"""
    print("\n=== EVALUATING LOADED MODELS ===")
    
    evaluation_results = {}
    
    for regime, models in loaded_models.items():
        print(f"\n--- EVALUATING REGIME {regime} MODELS ---")
        
        if regime not in regime_datasets:
            print(f"No dataset found for Regime {regime}, skipping")
            continue
        
        datasets = regime_datasets[regime]
        X_train = datasets['X_train']
        X_test = datasets['X_test']
        y_train = datasets['y_train']
        y_test = datasets['y_test']
        
        print(f"Test set: {len(X_test):,} samples")
        print(f"Target distribution: {y_test.value_counts().to_dict()}")
        
        regime_results = {}
        
        for model_name, model in models.items():
            print(f"  Evaluating {model_name}...")
            
            try:
                # Predictions on test set
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                test_auc = roc_auc_score(y_test, y_prob)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation on training set (use same logic as training script)
                if len(X_train) > 5000:  # Only CV on larger datasets (same threshold as training)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='roc_auc')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean, cv_std = test_auc, 0.0
                
                regime_results[model_name] = {
                    'test_auc': float(test_auc),
                    'test_accuracy': float(test_accuracy),
                    'cv_mean': float(cv_mean),
                    'cv_std': float(cv_std),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                
                print(f"    AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}, CV: {cv_mean:.4f}±{cv_std:.4f}")
                
            except Exception as e:
                print(f"    Error evaluating {model_name}: {e}")
        
        if regime_results:
            evaluation_results[regime] = regime_results
    
    return evaluation_results

def create_baseline_results():
    """Create baseline results (using known XGBoost performance)"""
    return {
        'XGBoost': {
            'test_auc': 0.6683,
            'test_accuracy': 0.6113,
            'cv_mean': 0.6665,
            'cv_std': 0.0021,
            'train_samples': 2487093,
            'test_samples': 621773
        }
    }

def analyze_regime_performance(evaluation_results):
    """Analyze and display regime performance insights"""
    print(f"\n=== REGIME PERFORMANCE ANALYSIS ===")
    
    baseline_auc = 0.6683  # Known XGBoost baseline
    
    # Find best models per regime
    print(f"\nBest model per regime:")
    for regime, models in evaluation_results.items():
        best_model = max(models.keys(), key=lambda x: models[x]['test_auc'])
        best_auc = models[best_model]['test_auc']
        improvement = best_auc - baseline_auc
        
        print(f"  Regime {regime}: {best_model} (AUC: {best_auc:.4f}, {improvement:+.4f} vs baseline)")
    
    # Overall best performance
    all_results = []
    for regime, models in evaluation_results.items():
        for model_name, results in models.items():
            all_results.append({
                'regime': regime,
                'model': model_name,
                'auc': results['test_auc'],
                'accuracy': results['test_accuracy'],
                'improvement': results['test_auc'] - baseline_auc
            })
    
    # Sort by AUC
    all_results.sort(key=lambda x: x['auc'], reverse=True)
    
    print(f"\nTop 5 regime-model combinations:")
    print(f"{'Rank':<5} {'Regime':<8} {'Model':<18} {'AUC':<8} {'Improvement':<12}")
    print("-" * 55)
    
    for i, result in enumerate(all_results[:5], 1):
        print(f"{i:<5} {result['regime']:<8} {result['model']:<18} {result['auc']:<8.4f} {result['improvement']:+<12.4f}")
    
    # Regime comparison
    print(f"\nRegime performance summary:")
    for regime in sorted(evaluation_results.keys()):
        models = evaluation_results[regime]
        regime_aucs = [m['test_auc'] for m in models.values()]
        avg_auc = np.mean(regime_aucs)
        best_auc = max(regime_aucs)
        
        print(f"  Regime {regime}: Avg AUC = {avg_auc:.4f}, Best AUC = {best_auc:.4f}")
    
    return all_results

def create_evaluation_summary(evaluation_results, baseline_results):
    """Create summary of regenerated evaluation results"""
    print(f"\n=== REGENERATED EVALUATION SUMMARY ===")
    
    print(f"Regimes evaluated: {list(evaluation_results.keys())}")
    if evaluation_results:
        print(f"Models per regime: {list(list(evaluation_results.values())[0].keys())}")
    
    print(f"\n{'Regime':<8} {'Model':<18} {'AUC':<8} {'Accuracy':<10} {'CV AUC':<10}")
    print("-" * 60)
    
    all_aucs = []
    for regime, models in evaluation_results.items():
        for model_name, results in models.items():
            auc = results['test_auc']
            accuracy = results['test_accuracy']
            cv_auc = results['cv_mean']
            all_aucs.append(auc)
            
            print(f"{regime:<8} {model_name:<18} {auc:<8.4f} {accuracy:<10.4f} {cv_auc:<10.4f}")
    
    if all_aucs:
        baseline_auc = baseline_results['XGBoost']['test_auc']
        best_regime_auc = max(all_aucs)
        improvement = best_regime_auc - baseline_auc
        
        print(f"\nPerformance Summary:")
        print(f"  Baseline XGBoost AUC: {baseline_auc:.4f}")
        print(f"  Best regime-specific AUC: {best_regime_auc:.4f}")
        print(f"  Maximum improvement: {improvement:+.4f} ({improvement/baseline_auc*100:+.1f}%)")

def main():
    """Main evaluation pipeline"""
    print("Starting model reload and evaluation...\n")
    
    # Check if saved models directory exists
    if not os.path.exists(SAVED_MODELS_DIR):
        print(f"❌ Saved models directory not found: {SAVED_MODELS_DIR}")
        print("Please run the training script first to generate saved models.")
        return
    
    # Load saved datasets
    dataset_metadata, regime_labels, datasets_dir = load_saved_datasets()
    
    # Get available regimes
    regimes = list(dataset_metadata['regime_distribution'].keys())
    regimes = [int(r) for r in regimes]  # Convert to int
    
    # Load regime datasets
    regime_datasets = load_regime_datasets(datasets_dir, regimes)
    
    # Load saved models
    loaded_models = load_saved_models(regimes)
    
    if not loaded_models:
        print("❌ No models found to evaluate")
        return
    
    # Evaluate models
    evaluation_results = evaluate_loaded_models(regime_datasets, loaded_models)
    
    if not evaluation_results:
        print("❌ No evaluation results generated")
        return
    
    # Load original training results for comparison
    original_results_path = f"{SAVED_MODELS_DIR}/training_results.json"
    if os.path.exists(original_results_path):
        with open(original_results_path, 'r') as f:
            original_results = json.load(f).get("regime_results", {})
        
        print("\n=== COMPARING RELOADED VS ORIGINAL AUC SCORES ===")
        
        for regime in sorted(evaluation_results.keys()):
            orig_regime = str(regime)
            if orig_regime not in original_results:
                print(f"Regime {regime} not in original results")
                continue
            
            print(f"\nRegime {regime}:")
            
            for model_name in sorted(evaluation_results[regime].keys()):
                if model_name not in original_results[orig_regime]:
                    print(f"  {model_name}: Not in original")
                    continue
                
                orig_auc = original_results[orig_regime][model_name].get("test_auc", None)
                reload_auc = evaluation_results[regime][model_name]["test_auc"]
                
                if orig_auc is not None:
                    diff = reload_auc - orig_auc
                    print(f"  {model_name}: Original {orig_auc:.4f} vs Reloaded {reload_auc:.4f} (diff {diff:+.4f})")
                else:
                    print(f"  {model_name}: No test_auc in original")
    else:
        print(f"❌ Original training_results.json not found at {original_results_path}")
    
    # Create baseline results
    baseline_results = create_baseline_results()
    
    # Analyze performance
    top_results = analyze_regime_performance(evaluation_results)
    
    # Create summary
    create_evaluation_summary(evaluation_results, baseline_results)
    
    print(f"\n" + "="*70)
    print("✅ MODEL RELOAD AND EVALUATION COMPLETED!")
    print("="*70)
    print(f"📊 {len(evaluation_results)} regimes evaluated")
    print(f"🏆 Best overall: Regime {top_results[0]['regime']} - {top_results[0]['model']} (AUC: {top_results[0]['auc']:.4f})")
    print(f"📈 Max improvement: {max(r['improvement'] for r in top_results):+.4f} vs baseline")
    print("="*70)

if __name__ == "__main__":
    main()