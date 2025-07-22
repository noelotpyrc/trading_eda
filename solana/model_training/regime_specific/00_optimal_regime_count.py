#!/usr/bin/env python3
"""
Part 0: Optimal Regime Count Detection
Systematically test different numbers of regimes to find the optimal clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = "/Volumes/Extreme SSD/trading_data/solana/classification_features/combined_features_sql.csv"
OUTPUT_DIR = "solana/analysis/optimal_regimes"
RANDOM_STATE = 42
SAMPLE_SIZE = 200000  # Manageable for testing multiple regime counts
MIN_REGIMES = 2
MAX_REGIMES = 10

print("=== PART 0: OPTIMAL REGIME COUNT DETECTION ===")
print(f"Objective: Find optimal number of regimes for market segmentation")
print(f"Testing regime counts: {MIN_REGIMES} to {MAX_REGIMES}")
print(f"Sample size: {SAMPLE_SIZE:,} records")
print()

def load_and_prepare_data(sample_size=SAMPLE_SIZE):
    """Load and prepare data for regime count testing"""
    print("Loading data for regime count optimization...")
    
    df = pd.read_csv(FEATURES_FILE)
    print(f"Original dataset: {len(df):,} records, {df.shape[1]} features")
    
    # Time-stratified sampling
    if len(df) > sample_size:
        df['sample_timestamp'] = pd.to_datetime(df['sample_timestamp'])
        df = df.sort_values('sample_timestamp')
        step = len(df) // sample_size
        df = df.iloc[::step].reset_index(drop=True)
        print(f"Sampled dataset: {len(df):,} records")
    
    # Prepare features
    metadata_cols = ['coin_id', 'sample_timestamp', 'total_transactions']
    target_col = 'is_profitable_300s'
    forward_cols = ['forward_buy_volume_300s', 'forward_sell_volume_300s']
    
    feature_cols = [col for col in df.columns 
                   if col not in metadata_cols + [target_col] + forward_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Clean data
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Remove constant features
    constant_features = X.columns[X.std() == 0]
    if len(constant_features) > 0:
        print(f"Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)
    
    print(f"Features for analysis: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def test_clustering_metrics(X_scaled, y, regime_range):
    """Test clustering quality metrics for different numbers of regimes"""
    print(f"\n=== TESTING CLUSTERING QUALITY METRICS ===")
    
    results = {}
    
    print(f"{'N_Regimes':<10} {'Method':<8} {'Silhouette':<12} {'Calinski':<12} {'Davies':<12} {'AIC':<12} {'BIC':<12}")
    print("-" * 80)
    
    for n_regimes in regime_range:
        results[n_regimes] = {}
        
        # Test GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=RANDOM_STATE)
        gmm_labels = gmm.fit_predict(X_scaled)
        
        # Calculate clustering metrics
        silhouette_gmm = silhouette_score(X_scaled, gmm_labels)
        calinski_gmm = calinski_harabasz_score(X_scaled, gmm_labels)
        davies_gmm = davies_bouldin_score(X_scaled, gmm_labels)
        aic_gmm = gmm.aic(X_scaled)
        bic_gmm = gmm.bic(X_scaled)
        
        results[n_regimes]['gmm'] = {
            'model': gmm,
            'labels': gmm_labels,
            'silhouette': silhouette_gmm,
            'calinski': calinski_gmm,
            'davies': davies_gmm,
            'aic': aic_gmm,
            'bic': bic_gmm
        }
        
        print(f"{n_regimes:<10} {'GMM':<8} {silhouette_gmm:<12.4f} {calinski_gmm:<12.2f} "
              f"{davies_gmm:<12.4f} {aic_gmm:<12.0f} {bic_gmm:<12.0f}")
        
        # Test K-Means
        kmeans = KMeans(n_clusters=n_regimes, random_state=RANDOM_STATE, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
        calinski_kmeans = calinski_harabasz_score(X_scaled, kmeans_labels)
        davies_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)
        
        results[n_regimes]['kmeans'] = {
            'model': kmeans,
            'labels': kmeans_labels,
            'silhouette': silhouette_kmeans,
            'calinski': calinski_kmeans,
            'davies': davies_kmeans,
            'inertia': kmeans.inertia_
        }
        
        print(f"{n_regimes:<10} {'K-Means':<8} {silhouette_kmeans:<12.4f} {calinski_kmeans:<12.2f} "
              f"{davies_kmeans:<12.4f} {'N/A':<12} {'N/A':<12}")
    
    return results

def test_predictive_power(X, y, X_scaled, results, regime_range):
    """Test how well each regime count separates the target variable"""
    print(f"\n=== TESTING PREDICTIVE POWER BY REGIME COUNT ===")
    
    predictive_results = {}
    
    print(f"{'N_Regimes':<10} {'Method':<8} {'Target_Sep':<12} {'Best_AUC':<12} {'Avg_AUC':<12} {'CV_Score':<12}")
    print("-" * 75)
    
    for n_regimes in regime_range:
        predictive_results[n_regimes] = {}
        
        for method in ['gmm', 'kmeans']:
            labels = results[n_regimes][method]['labels']
            
            # Target separation (standard deviation of regime target means)
            regime_target_means = []
            regime_aucs = []
            
            for regime in np.unique(labels):
                regime_mask = labels == regime
                if regime_mask.sum() >= 50:  # Minimum samples for meaningful AUC
                    regime_target_mean = y[regime_mask].mean()
                    regime_target_means.append(regime_target_mean)
                    
                    # Quick AUC estimate using simple model
                    X_regime = X[regime_mask]
                    y_regime = y[regime_mask]
                    
                    if len(X_regime) >= 100 and y_regime.nunique() > 1:
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_regime, y_regime, test_size=0.3, random_state=RANDOM_STATE, stratify=y_regime
                            )
                            
                            # Use LightGBM for regime AUC testing
                            model = lgb.LGBMClassifier(
                                n_estimators=100,
                                max_depth=6,
                                learning_rate=0.1,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=RANDOM_STATE,
                                verbose=-1
                            )
                            model.fit(X_train, y_train)
                            y_prob = model.predict_proba(X_test)[:, 1]
                            regime_auc = roc_auc_score(y_test, y_prob)
                            regime_aucs.append(regime_auc)
                        except:
                            regime_aucs.append(0.5)
            
            target_separation = np.std(regime_target_means) if len(regime_target_means) > 1 else 0
            avg_auc = np.mean(regime_aucs) if regime_aucs else 0.5
            best_auc = max(regime_aucs) if regime_aucs else 0.5
            
            # Cross-validation score using regime labels as features
            try:
                # Create regime feature matrix
                regime_features = np.eye(n_regimes)[labels]
                X_with_regimes = np.column_stack([X.values, regime_features])
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_with_regimes, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
                )
                
                # Use LightGBM for cross-validation
                model_cv = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    verbose=-1
                )
                model_cv.fit(X_train, y_train)
                y_prob_cv = model_cv.predict_proba(X_test)[:, 1]
                cv_auc = roc_auc_score(y_test, y_prob_cv)
            except:
                cv_auc = 0.5
            
            predictive_results[n_regimes][method] = {
                'target_separation': target_separation,
                'avg_auc': avg_auc,
                'best_auc': best_auc,
                'cv_auc': cv_auc,
                'n_valid_regimes': len(regime_aucs)
            }
            
            print(f"{n_regimes:<10} {method.upper():<8} {target_separation:<12.4f} "
                  f"{best_auc:<12.4f} {avg_auc:<12.4f} {cv_auc:<12.4f}")
    
    return predictive_results

def calculate_regime_stability(X_scaled, results, regime_range):
    """Test regime stability using bootstrap sampling"""
    print(f"\n=== TESTING REGIME STABILITY ===")
    
    stability_results = {}
    n_bootstrap = 5  # Limited for speed
    
    print(f"{'N_Regimes':<10} {'Method':<8} {'Stability':<12} {'Consistency':<12}")
    print("-" * 50)
    
    for n_regimes in regime_range:
        stability_results[n_regimes] = {}
        
        for method in ['gmm', 'kmeans']:
            original_labels = results[n_regimes][method]['labels']
            similarities = []
            
            # Bootstrap stability test
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
                X_bootstrap = X_scaled[indices]
                
                # Fit model on bootstrap sample
                if method == 'gmm':
                    model = GaussianMixture(n_components=n_regimes, random_state=RANDOM_STATE)
                else:
                    model = KMeans(n_clusters=n_regimes, random_state=RANDOM_STATE, n_init=10)
                
                try:
                    bootstrap_labels = model.fit_predict(X_bootstrap)
                    
                    # Measure similarity (adjusted rand index approximation)
                    # Simple consistency measure: same regime assignment percentage
                    original_bootstrap = original_labels[indices]
                    
                    # Find best label mapping
                    best_consistency = 0
                    for perm in range(min(10, n_regimes**2)):  # Limited permutations
                        # Simple mapping attempt
                        mapped_labels = bootstrap_labels.copy()
                        consistency = (mapped_labels == original_bootstrap).mean()
                        best_consistency = max(best_consistency, consistency)
                    
                    similarities.append(best_consistency)
                except:
                    similarities.append(0)
            
            avg_stability = np.mean(similarities)
            consistency = np.std(similarities)  # Lower std = more consistent
            
            stability_results[n_regimes][method] = {
                'stability': avg_stability,
                'consistency': 1 - consistency  # Convert to positive metric
            }
            
            print(f"{n_regimes:<10} {method.upper():<8} {avg_stability:<12.4f} {1-consistency:<12.4f}")
    
    return stability_results

def find_optimal_regime_count(results, predictive_results, stability_results, regime_range):
    """Find optimal regime count using composite scoring"""
    print(f"\n=== FINDING OPTIMAL REGIME COUNT ===")
    
    composite_scores = {}
    
    print(f"{'N_Regimes':<10} {'Method':<8} {'Silhouette':<12} {'Predictive':<12} {'Stability':<12} {'Composite':<12}")
    print("-" * 80)
    
    for n_regimes in regime_range:
        composite_scores[n_regimes] = {}
        
        for method in ['gmm', 'kmeans']:
            # Normalize metrics to 0-1 scale
            
            # Silhouette score (already 0-1, higher better)
            silhouette_norm = results[n_regimes][method]['silhouette']
            
            # Target separation (normalize by max, higher better)
            target_sep = predictive_results[n_regimes][method]['target_separation']
            max_target_sep = max([predictive_results[r][method]['target_separation'] 
                                for r in regime_range])
            target_sep_norm = target_sep / (max_target_sep + 1e-10)
            
            # Best AUC (already 0-1, higher better, subtract 0.5 to center around 0)
            best_auc = predictive_results[n_regimes][method]['best_auc']
            auc_norm = (best_auc - 0.5) * 2  # Scale to 0-1
            
            # Stability (already 0-1, higher better)
            stability = stability_results[n_regimes][method]['stability']
            
            # Composite score (weighted average)
            weights = {
                'silhouette': 0.25,
                'target_separation': 0.35,  # Most important for trading
                'auc': 0.25,
                'stability': 0.15
            }
            
            composite = (weights['silhouette'] * silhouette_norm +
                        weights['target_separation'] * target_sep_norm +
                        weights['auc'] * auc_norm +
                        weights['stability'] * stability)
            
            composite_scores[n_regimes][method] = {
                'silhouette_norm': silhouette_norm,
                'target_sep_norm': target_sep_norm,
                'auc_norm': auc_norm,
                'stability': stability,
                'composite': composite
            }
            
            print(f"{n_regimes:<10} {method.upper():<8} {silhouette_norm:<12.4f} "
                  f"{target_sep_norm:<12.4f} {stability:<12.4f} {composite:<12.4f}")
    
    # Find best combination
    best_score = -1
    best_n_regimes = None
    best_method = None
    
    for n_regimes in regime_range:
        for method in ['gmm', 'kmeans']:
            score = composite_scores[n_regimes][method]['composite']
            if score > best_score:
                best_score = score
                best_n_regimes = n_regimes
                best_method = method
    
    print(f"\n🏆 OPTIMAL CONFIGURATION:")
    print(f"  Best regime count: {best_n_regimes}")
    print(f"  Best method: {best_method.upper()}")
    print(f"  Composite score: {best_score:.4f}")
    
    return best_n_regimes, best_method, composite_scores

def create_optimization_visualizations(results, predictive_results, stability_results, composite_scores, regime_range):
    """Create comprehensive visualizations of regime count optimization"""
    print(f"\n=== CREATING OPTIMIZATION VISUALIZATIONS ===")
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Optimal Regime Count Analysis', fontsize=16, fontweight='bold')
    
    # 1. Silhouette scores
    ax1 = axes[0, 0]
    gmm_silhouette = [results[r]['gmm']['silhouette'] for r in regime_range]
    kmeans_silhouette = [results[r]['kmeans']['silhouette'] for r in regime_range]
    
    ax1.plot(regime_range, gmm_silhouette, 'o-', label='GMM', linewidth=2)
    ax1.plot(regime_range, kmeans_silhouette, 's-', label='K-Means', linewidth=2)
    ax1.set_xlabel('Number of Regimes')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Clustering Quality (Silhouette)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. AIC/BIC for GMM
    ax2 = axes[0, 1]
    gmm_aic = [results[r]['gmm']['aic'] for r in regime_range]
    gmm_bic = [results[r]['gmm']['bic'] for r in regime_range]
    
    ax2.plot(regime_range, gmm_aic, 'o-', label='AIC', linewidth=2)
    ax2.plot(regime_range, gmm_bic, 's-', label='BIC', linewidth=2)
    ax2.set_xlabel('Number of Regimes')
    ax2.set_ylabel('Information Criterion')
    ax2.set_title('GMM Model Selection (Lower=Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Target separation
    ax3 = axes[0, 2]
    gmm_target_sep = [predictive_results[r]['gmm']['target_separation'] for r in regime_range]
    kmeans_target_sep = [predictive_results[r]['kmeans']['target_separation'] for r in regime_range]
    
    ax3.plot(regime_range, gmm_target_sep, 'o-', label='GMM', linewidth=2)
    ax3.plot(regime_range, kmeans_target_sep, 's-', label='K-Means', linewidth=2)
    ax3.set_xlabel('Number of Regimes')
    ax3.set_ylabel('Target Separation')
    ax3.set_title('Predictive Power (Target Separation)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Best AUC by regime count
    ax4 = axes[1, 0]
    gmm_best_auc = [predictive_results[r]['gmm']['best_auc'] for r in regime_range]
    kmeans_best_auc = [predictive_results[r]['kmeans']['best_auc'] for r in regime_range]
    
    ax4.plot(regime_range, gmm_best_auc, 'o-', label='GMM', linewidth=2)
    ax4.plot(regime_range, kmeans_best_auc, 's-', label='K-Means', linewidth=2)
    ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Target (0.70)')
    ax4.set_xlabel('Number of Regimes')
    ax4.set_ylabel('Best Regime AUC')
    ax4.set_title('Best Individual Regime Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Stability scores
    ax5 = axes[1, 1]
    gmm_stability = [stability_results[r]['gmm']['stability'] for r in regime_range]
    kmeans_stability = [stability_results[r]['kmeans']['stability'] for r in regime_range]
    
    ax5.plot(regime_range, gmm_stability, 'o-', label='GMM', linewidth=2)
    ax5.plot(regime_range, kmeans_stability, 's-', label='K-Means', linewidth=2)
    ax5.set_xlabel('Number of Regimes')
    ax5.set_ylabel('Stability Score')
    ax5.set_title('Regime Stability')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Composite scores
    ax6 = axes[1, 2]
    gmm_composite = [composite_scores[r]['gmm']['composite'] for r in regime_range]
    kmeans_composite = [composite_scores[r]['kmeans']['composite'] for r in regime_range]
    
    ax6.plot(regime_range, gmm_composite, 'o-', label='GMM', linewidth=2, markersize=8)
    ax6.plot(regime_range, kmeans_composite, 's-', label='K-Means', linewidth=2, markersize=8)
    
    # Highlight best combination
    best_regime = max(regime_range, key=lambda r: max(composite_scores[r]['gmm']['composite'], 
                                                     composite_scores[r]['kmeans']['composite']))
    best_method = 'gmm' if composite_scores[best_regime]['gmm']['composite'] > composite_scores[best_regime]['kmeans']['composite'] else 'kmeans'
    best_score = composite_scores[best_regime][best_method]['composite']
    
    ax6.scatter([best_regime], [best_score], color='red', s=200, marker='*', zorder=5, label='Optimal')
    ax6.set_xlabel('Number of Regimes')
    ax6.set_ylabel('Composite Score')
    ax6.set_title('Overall Performance (Composite Score)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/regime_count_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Optimization visualizations saved to: {OUTPUT_DIR}/")

def save_optimization_results(results, predictive_results, stability_results, composite_scores, 
                            best_n_regimes, best_method, regime_range):
    """Save optimization results"""
    print(f"\n=== SAVING OPTIMIZATION RESULTS ===")
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Prepare results for JSON
    optimization_results = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'regime_range_tested': list(regime_range),
        'optimal_configuration': {
            'n_regimes': int(best_n_regimes),
            'method': best_method,
            'composite_score': float(composite_scores[best_n_regimes][best_method]['composite'])
        },
        'detailed_scores': {
            str(n_regimes): {
                method: {
                    'clustering_metrics': {
                        'silhouette': float(results[n_regimes][method]['silhouette']),
                        'calinski': float(results[n_regimes][method]['calinski']),
                        'davies': float(results[n_regimes][method]['davies'])
                    },
                    'predictive_metrics': {
                        'target_separation': float(predictive_results[n_regimes][method]['target_separation']),
                        'best_auc': float(predictive_results[n_regimes][method]['best_auc']),
                        'avg_auc': float(predictive_results[n_regimes][method]['avg_auc']),
                        'cv_auc': float(predictive_results[n_regimes][method]['cv_auc'])
                    },
                    'stability_metrics': {
                        'stability': float(stability_results[n_regimes][method]['stability']),
                        'consistency': float(stability_results[n_regimes][method]['consistency'])
                    },
                    'composite_score': float(composite_scores[n_regimes][method]['composite'])
                }
                for method in ['gmm', 'kmeans']
            }
            for n_regimes in regime_range
        },
        'recommendations': [
            f"Use {best_n_regimes} regimes with {best_method.upper()} clustering",
            f"Expected target separation: {predictive_results[best_n_regimes][best_method]['target_separation']:.4f}",
            f"Best regime AUC potential: {predictive_results[best_n_regimes][best_method]['best_auc']:.4f}",
            f"Stability score: {stability_results[best_n_regimes][best_method]['stability']:.4f}"
        ]
    }
    
    import json
    with open(f"{OUTPUT_DIR}/regime_optimization_results.json", 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    # Create summary CSV
    summary_data = []
    for n_regimes in regime_range:
        for method in ['gmm', 'kmeans']:
            summary_data.append({
                'n_regimes': n_regimes,
                'method': method,
                'silhouette': results[n_regimes][method]['silhouette'],
                'target_separation': predictive_results[n_regimes][method]['target_separation'],
                'best_auc': predictive_results[n_regimes][method]['best_auc'],
                'stability': stability_results[n_regimes][method]['stability'],
                'composite_score': composite_scores[n_regimes][method]['composite'],
                'is_optimal': (n_regimes == best_n_regimes and method == best_method)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{OUTPUT_DIR}/regime_optimization_summary.csv", index=False)
    
    print(f"✅ Optimization results saved:")
    print(f"  Detailed results: regime_optimization_results.json")
    print(f"  Summary table: regime_optimization_summary.csv")
    print(f"  Directory: {OUTPUT_DIR}")
    
    return optimization_results

def main():
    """Main regime count optimization pipeline"""
    print("Starting optimal regime count detection...\n")
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    regime_range = range(MIN_REGIMES, MAX_REGIMES + 1)
    
    # Test clustering metrics
    results = test_clustering_metrics(X_scaled, y, regime_range)
    
    # Test predictive power
    predictive_results = test_predictive_power(X, y, X_scaled, results, regime_range)
    
    # Test stability
    stability_results = calculate_regime_stability(X_scaled, results, regime_range)
    
    # Find optimal regime count
    best_n_regimes, best_method, composite_scores = find_optimal_regime_count(
        results, predictive_results, stability_results, regime_range
    )
    
    # Create visualizations
    create_optimization_visualizations(results, predictive_results, stability_results, 
                                     composite_scores, regime_range)
    
    # Save results
    optimization_results = save_optimization_results(results, predictive_results, stability_results, 
                                                   composite_scores, best_n_regimes, best_method, regime_range)
    
    print(f"\n" + "="*80)
    print("✅ OPTIMAL REGIME COUNT DETECTION COMPLETED!")
    print("="*80)
    print(f"🎯 RECOMMENDATION: Use {best_n_regimes} regimes with {best_method.upper()}")
    print(f"📊 Composite Score: {composite_scores[best_n_regimes][best_method]['composite']:.4f}")
    print(f"🎪 Target Separation: {predictive_results[best_n_regimes][best_method]['target_separation']:.4f}")
    print(f"🚀 Best Regime AUC: {predictive_results[best_n_regimes][best_method]['best_auc']:.4f}")
    print(f"📈 Results saved to: {OUTPUT_DIR}")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Update N_REGIMES = {best_n_regimes} in 01_regime_clustering.py")
    print(f"2. Run the complete 4-part pipeline with optimal settings")

if __name__ == "__main__":
    main()