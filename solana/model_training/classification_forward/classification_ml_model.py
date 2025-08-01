#!/usr/bin/env python3
"""
Classification Forward ML Model
Production script for training ensemble models on Solana classification features dataset
Based on solana/analysis/batch578_sample_analysis/04-3_ml_ensemble_signals.ipynb methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import os
import pickle
import json
import time

# ML imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = "/Volumes/Extreme SSD/trading_data/solana/classification_features/combined_features_sql.csv"
MODEL_OUTPUT_DIR = "/Volumes/Extreme SSD/trading_data/solana/models/classification_forward"
RANDOM_STATE = 42

# Model parameters
TEST_SIZE = 0.2
VAL_SIZE = 0.15  # 15% of coins for validation (never used in training)
CV_FOLDS = 5
MAX_FEATURES_SELECTION = 80
SAMPLE_SIZE = 10000000  # For faster training, adjust as needed

print("=== SOLANA CLASSIFICATION FORWARD ML MODEL ===")
print("Based on ensemble methodology from Phase 4 analysis")
print(f"Data source: {FEATURES_FILE}")
print()

def load_and_prepare_data(sample_size=SAMPLE_SIZE):
    """
    Load and prepare the classification features dataset
    """
    print("Loading classification features dataset...")
    
    # Load data
    df = pd.read_csv(FEATURES_FILE)
    print(f"Original dataset: {len(df):,} records, {df.shape[1]} features")
    
    # Add first sample timestamp for each coin_id without changing data types or order
    print("Recording first sample timestamp for each coin...")
    df['first_sample_timestamp'] = df.groupby('coin_id')['sample_timestamp'].transform('first')
    
    # Sample for manageable training (if dataset is large)
    if len(df) > sample_size:
        # Take every nth record to maintain temporal distribution
        step = len(df) // sample_size
        df = df.iloc[::step].reset_index(drop=True)
        print(f"Sampled dataset: {len(df):,} records (every {step}th record)")
    
    # Prepare features and target
    metadata_cols = ['coin_id', 'sample_timestamp', 'first_sample_timestamp', 'total_transactions']
    target_col = 'is_profitable_300s'
    forward_cols = ['forward_buy_volume_300s', 'forward_sell_volume_300s']
    
    # Separate features, target, metadata, and forward volume features
    feature_cols = [col for col in df.columns 
                   if col not in metadata_cols + [target_col] + forward_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    metadata = df[metadata_cols + [target_col]].copy()
    forward_volumes = df[forward_cols].copy()
    additional_features = df[['coin_id', 'sample_timestamp', 'first_sample_timestamp', 'total_transactions']].copy()
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(X):,}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Forward volume features: {forward_volumes.shape[1]}")
    print(f"Additional metadata features: {additional_features.shape[1]}")
    print(f"Unique coins: {additional_features['coin_id'].nunique():,}")
    print(f"Total transactions range: {additional_features['total_transactions'].min():.0f} - {additional_features['total_transactions'].max():.0f}")
    
    # Data quality checks
    print(f"\nData Quality:")
    print(f"Missing values: {X.isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(X.values).sum()}")
    
    # Handle data quality issues
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Handle forward volume data quality
    forward_volumes = forward_volumes.fillna(0)
    forward_volumes = forward_volumes.replace([np.inf, -np.inf], 0)
    
    # Remove constant features
    constant_features = X.columns[X.std() == 0]
    if len(constant_features) > 0:
        print(f"Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)
    
    print(f"Final features: {X.shape[1]}")
    return X, y, metadata, forward_volumes, additional_features

def analyze_baseline_performance(X, y):
    """
    Analyze individual feature performance as baseline
    """
    print("\n=== BASELINE: INDIVIDUAL FEATURE ANALYSIS ===")
    
    # Calculate correlations
    feature_correlations = []
    for col in X.columns:
        corr = np.corrcoef(X[col], y)[0, 1]
        if not np.isnan(corr):
            feature_correlations.append((col, abs(corr)))
    
    # Sort by correlation strength
    feature_correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top 15 individual features by correlation:")
    for i, (feature, corr) in enumerate(feature_correlations[:15]):
        print(f"{i+1:2d}. {feature:<35} | r = {corr:.4f}")
    
    # Statistical significance test
    significant_features = []
    for feature, corr in feature_correlations:
        if corr > 0.01:  # Only test features with some correlation
            try:
                _, p_value = stats.pearsonr(X[feature], y)
                if p_value < 0.05:
                    significant_features.append((feature, corr, p_value))
            except:
                continue
    
    print(f"\nStatistically significant features (p<0.05): {len(significant_features)}")
    if significant_features:
        print("Top 10 significant features:")
        for i, (feature, corr, p_val) in enumerate(significant_features[:10]):
            print(f"{i+1:2d}. {feature:<35} | r = {corr:.4f}, p = {p_val:.4f}")
    
    return feature_correlations, significant_features



def create_ensemble_models():
    """
    Create ensemble models based on notebook best practices
    """
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

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate all models with comprehensive metrics
    """
    print("\n=== MODEL EVALUATION ===")
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        model_start = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        train_accuracy = (y_pred_train == y_train).mean()
        test_accuracy = (y_pred_test == y_test).mean()
        auc_score = roc_auc_score(y_test, y_prob_test)
        
        # Cross-validation with proper time series handling
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc')
        
        results[name] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test,
            'probabilities': y_prob_test
        }
        
        model_time = time.time() - model_start
        print(f"  Time for {name}: {model_time:.2f} seconds")
        
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy:  {test_accuracy:.4f}")
        print(f"  AUC Score:      {auc_score:.4f}")
        print(f"  CV AUC:         {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Summary table
    print(f"\n=== MODEL COMPARISON SUMMARY ===")
    print(f"{'Model':<20} {'Train Acc':<10} {'Test Acc':<10} {'AUC':<8} {'CV AUC':<8} {'CV Std':<8}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<20} {result['train_accuracy']:<10.4f} {result['test_accuracy']:<10.4f} "
              f"{result['auc_score']:<8.4f} {result['cv_mean']:<8.4f} {result['cv_std']:<8.4f}")
    
    return results

def analyze_feature_importance(results, feature_names):
    """
    Analyze feature importance across models
    """
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    importance_data = {}
    
    for name, result in results.items():
        model = result['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_data[name] = model.feature_importances_
    
    if not importance_data:
        print("No tree-based models with feature importance found")
        return None
    
    # Create importance DataFrame
    importance_df = pd.DataFrame(importance_data, index=feature_names)
    importance_df['Average'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values('Average', ascending=False)
    
    print("Top 20 most important features:")
    print(f"{'Feature':<40} {'Average':<10}")
    print("-" * 50)
    
    for i, (feature, row) in enumerate(importance_df.head(20).iterrows()):
        print(f"{feature:<40} {row['Average']:<10.4f}")
    
    return importance_df

def create_voting_ensemble(results, X_train, y_train):
    """
    Create voting ensemble from best models
    """
    print("\n=== CREATING VOTING ENSEMBLE ===")
    
    # Select top 3 models by AUC
    sorted_models = sorted(results.items(), 
                          key=lambda x: x[1]['auc_score'], 
                          reverse=True)
    
    top_models = sorted_models[:4]
    print("Selected models for voting ensemble:")
    for name, result in top_models:
        print(f"  {name}: AUC = {result['auc_score']:.4f}")
    
    # Create voting classifier
    estimators = [(name, result['model']) for name, result in top_models]
    voting_ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    # Train voting ensemble
    voting_ensemble.fit(X_train, y_train)
    
    return voting_ensemble, top_models

def save_production_model(model, feature_names, importance_df, results, metadata):
    """
    Save the best model and metadata for production use
    """
    print("\n=== SAVING PRODUCTION MODEL ===")
    
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Select best model by AUC
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
    best_result = results[best_model_name]
    
    print(f"Selected model: {best_model_name}")
    print(f"AUC Score: {best_result['auc_score']:.4f}")
    
    # Save model
    model_file = f"{MODEL_OUTPUT_DIR}/best_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(best_result['model'], f)
    
    # Save feature names
    features_file = f"{MODEL_OUTPUT_DIR}/feature_names.pkl"
    with open(features_file, 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save feature importance
    if importance_df is not None:
        importance_file = f"{MODEL_OUTPUT_DIR}/feature_importance.csv"
        importance_df.to_csv(importance_file)
    
    # Save metadata
    metadata_dict = {
        'model_type': best_model_name,
        'auc_score': best_result['auc_score'],
        'test_accuracy': best_result['test_accuracy'],
        'cv_mean': best_result['cv_mean'],
        'cv_std': best_result['cv_std'],
        'n_features': len(feature_names),
        'training_date': datetime.now().isoformat(),
        'data_source': FEATURES_FILE,
        'sample_size': len(metadata),
        'target_distribution': metadata['is_profitable_300s'].value_counts().to_dict()
    }
    
    metadata_file = f"{MODEL_OUTPUT_DIR}/model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f"✅ Model saved to: {MODEL_OUTPUT_DIR}/")
    print(f"Files created:")
    print(f"  - best_model.pkl (trained model)")
    print(f"  - feature_names.pkl (feature list)")
    print(f"  - feature_importance.csv (feature rankings)")
    print(f"  - model_metadata.json (model info)")
    
    return model_file, metadata_dict

def main():
    """
    Main training pipeline
    """
    print("Starting ML model training pipeline...\n")
    
    overall_start = time.time()
    
    # Load and prepare data
    section_start = time.time()
    X, y, metadata, forward_volumes, additional_features = load_and_prepare_data()
    print(f"Data loading and preparation time: {time.time() - section_start:.2f} seconds")
    
    # Baseline analysis
    section_start = time.time()
    feature_correlations, significant_features = analyze_baseline_performance(X, y)
    print(f"Baseline analysis time: {time.time() - section_start:.2f} seconds")
    
    # Use all features without advanced feature engineering
    print(f"\n=== USING FULL FEATURE SET ===")
    print(f"Training with all {X.shape[1]} original features")
    X_final = X
    
    # Create train/validation/test split by sampling coin_ids
    print(f"\n=== CREATING TRAIN/VALIDATION/TEST SPLITS BY COIN_ID ===")
    
    # Get unique coin_ids
    unique_coins = additional_features['coin_id'].unique()
    np.random.seed(RANDOM_STATE)
    
    # First split: separate validation coins
    val_coins = np.random.choice(unique_coins, size=int(len(unique_coins) * VAL_SIZE), replace=False)
    remaining_coins = np.setdiff1d(unique_coins, val_coins)
    
    # Second split: separate test coins from remaining
    test_coins = np.random.choice(remaining_coins, size=int(len(remaining_coins) * TEST_SIZE), replace=False)
    train_coins = np.setdiff1d(remaining_coins, test_coins)
    
    print(f"Total unique coins: {len(unique_coins):,}")
    print(f"Validation coins: {len(val_coins):,} ({len(val_coins)/len(unique_coins)*100:.1f}%)")
    print(f"Test coins: {len(test_coins):,} ({len(test_coins)/len(unique_coins)*100:.1f}%)")
    print(f"Train coins: {len(train_coins):,} ({len(train_coins)/len(unique_coins)*100:.1f}%)")
    
    # Create masks for each split
    val_mask = additional_features['coin_id'].isin(val_coins)
    test_mask = additional_features['coin_id'].isin(test_coins)
    train_mask = additional_features['coin_id'].isin(train_coins)
    
    # Split the data
    X_train = X_final[train_mask].copy()
    X_val = X_final[val_mask].copy()
    X_test = X_final[test_mask].copy()
    
    y_train = y[train_mask].copy()
    y_val = y[val_mask].copy()
    y_test = y[test_mask].copy()
    
    forward_train = forward_volumes[train_mask].copy()
    forward_val = forward_volumes[val_mask].copy()
    forward_test = forward_volumes[test_mask].copy()
    
    additional_train = additional_features[train_mask].copy()
    additional_val = additional_features[val_mask].copy()
    additional_test = additional_features[test_mask].copy()
    
    print(f"\nTraining set: {len(X_train):,} samples ({y_train.mean():.3f} positive rate)")
    print(f"Validation set: {len(X_val):,} samples ({y_val.mean():.3f} positive rate)")
    print(f"Test set: {len(X_test):,} samples ({y_test.mean():.3f} positive rate)")
    
    # Persist train/validation/test sets
    print(f"\n=== PERSISTING TRAIN/VALIDATION/TEST SETS ===")
    datasets_dir = f"{MODEL_OUTPUT_DIR}/datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Save as pickle files
    train_data = {
        'X_train': X_train, 
        'y_train': y_train,
        'forward_buy_volume_300s_train': forward_train['forward_buy_volume_300s'],
        'forward_sell_volume_300s_train': forward_train['forward_sell_volume_300s'],
        'metadata_train': additional_train  # Save as DataFrame
    }
    
    val_data = {
        'X_val': X_val, 
        'y_val': y_val,
        'forward_buy_volume_300s_val': forward_val['forward_buy_volume_300s'],
        'forward_sell_volume_300s_val': forward_val['forward_sell_volume_300s'],
        'metadata_val': additional_val  # Save as DataFrame
    }
    
    test_data = {
        'X_test': X_test, 
        'y_test': y_test,
        'forward_buy_volume_300s_test': forward_test['forward_buy_volume_300s'],
        'forward_sell_volume_300s_test': forward_test['forward_sell_volume_300s'],
        'metadata_test': additional_test  # Save as DataFrame
    }
    
    train_file = f"{datasets_dir}/train_set.pkl"
    val_file = f"{datasets_dir}/val_set.pkl"
    test_file = f"{datasets_dir}/test_set.pkl"
    
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)
    with open(val_file, 'wb') as f:
        pickle.dump(val_data, f)
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"✅ Train set saved: {train_file} ({len(X_train):,} samples)")
    print(f"✅ Validation set saved: {val_file} ({len(X_val):,} samples)")
    print(f"✅ Test set saved: {test_file} ({len(X_test):,} samples)")
    print(f"✅ Coins are completely separated between splits")
    print(f"   Train coins: {len(train_coins):,}")
    print(f"   Validation coins: {len(val_coins):,}")
    print(f"   Test coins: {len(test_coins):,}")
    
    # Create and evaluate only XGBoost model
    section_start = time.time()
    models = create_ensemble_models()
    xgboost_model = {'XGBoost': models['XGBoost']}  # Only use XGBoost
    results = evaluate_models(xgboost_model, X_train, X_test, y_train, y_test)
    print(f"XGBoost model creation and evaluation time: {time.time() - section_start:.2f} seconds")
    
    # Feature importance analysis
    section_start = time.time()
    importance_df = analyze_feature_importance(results, X_final.columns)
    print(f"Feature importance analysis time: {time.time() - section_start:.2f} seconds")
    
    # Save XGBoost model for production
    section_start = time.time()
    model_file, model_metadata = save_production_model(
        None, X_final.columns, importance_df, results, metadata
    )
    print(f"Model saving time: {time.time() - section_start:.2f} seconds")
    
    overall_time = time.time() - overall_start
    print(f"\nTotal execution time: {overall_time:.2f} seconds")
    print(f"Processed {len(X):,} samples")
    print(f"Time per sample: {overall_time / len(X):.6f} seconds")
    
    print("\n" + "="*80)
    print("✅ ML MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Best Model: {model_metadata['model_type']}")
    print(f"AUC Score: {model_metadata['auc_score']:.4f}")
    print(f"Features: {model_metadata['n_features']}")
    print(f"Model saved: {model_file}")
    print("="*80)

if __name__ == "__main__":
    main()