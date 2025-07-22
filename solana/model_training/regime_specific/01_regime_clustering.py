#!/usr/bin/env python3
"""
Part 1: Regime Clustering Model
Train clustering model to identify market regimes in Solana trading data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = "/Volumes/Extreme SSD/trading_data/solana/classification_features/combined_features_sql.csv"
OUTPUT_DIR = "solana/models/regime_clustering"
RANDOM_STATE = 42
SAMPLE_SIZE = 3000000  # Manageable size for clustering
N_REGIMES = 3

print("=== PART 1: REGIME CLUSTERING MODEL ===")
print(f"Objective: Identify {N_REGIMES} distinct market regimes (optimized count)")
print(f"Method: Best clustering method determined by data evaluation")
print(f"Data source: {FEATURES_FILE}")
print(f"Sample size: {SAMPLE_SIZE:,} records")
print()

def load_and_sample_data(sample_size=SAMPLE_SIZE):
    """Load and prepare data for regime clustering"""
    print("Loading data for regime clustering...")
    
    df = pd.read_csv(FEATURES_FILE)
    print(f"Original dataset: {len(df):,} records, {df.shape[1]} features")
    
    # Time-stratified sampling to preserve temporal patterns
    if len(df) > sample_size:
        df['sample_timestamp'] = pd.to_datetime(df['sample_timestamp'])
        df = df.sort_values('sample_timestamp')
        step = len(df) // sample_size
        df = df.iloc[::step].reset_index(drop=True)
        print(f"Sampled dataset: {len(df):,} records (every {step}th record)")
    
    # Prepare features (exclude metadata and target)
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
    
    print(f"Features for clustering: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, df[metadata_cols + [target_col]]

def train_clustering_models(X, n_regimes=N_REGIMES):
    """Train multiple clustering models and select the best one"""
    print(f"\n=== TRAINING CLUSTERING MODELS ===")
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train different clustering models
    models = {}
    
    # 1. Gaussian Mixture Model
    print(f"Training Gaussian Mixture Model (n_components={n_regimes})...")
    gmm = GaussianMixture(n_components=n_regimes, random_state=RANDOM_STATE)
    gmm_labels = gmm.fit_predict(X_scaled)
    models['gmm'] = {
        'model': gmm,
        'labels': gmm_labels,
        'aic': gmm.aic(X_scaled),
        'bic': gmm.bic(X_scaled)
    }
    
    # 2. K-Means
    print(f"Training K-Means (n_clusters={n_regimes})...")
    kmeans = KMeans(n_clusters=n_regimes, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    models['kmeans'] = {
        'model': kmeans,
        'labels': kmeans_labels,
        'inertia': kmeans.inertia_
    }
    
    # Analyze clustering results
    print(f"\n--- Clustering Results ---")
    for method, info in models.items():
        labels = info['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print(f"\n{method.upper()}:")
        if method == 'gmm':
            print(f"  AIC: {info['aic']:.2f}, BIC: {info['bic']:.2f}")
        else:
            print(f"  Inertia: {info['inertia']:.2f}")
        
        print(f"  Cluster distribution:")
        for label, count in zip(unique_labels, counts):
            pct = count / len(labels) * 100
            print(f"    Cluster {label}: {count:,} samples ({pct:.1f}%)")
    
    return models, scaler

def evaluate_cluster_quality(models, X, y, sample_size=100000):
    """Evaluate clustering quality using fast metrics and target separation"""
    print(f"\n=== CLUSTER QUALITY EVALUATION ===")
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    results = {}
    
    # Sample data for silhouette score calculation (too slow on full dataset)
    if len(X) > sample_size:
        print(f"Using sample of {sample_size:,} points for silhouette score calculation")
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_indices]
    else:
        X_sample = X
        sample_indices = np.arange(len(X))
    
    for method, info in models.items():
        labels = info['labels']
        labels_sample = labels[sample_indices] if len(X) > sample_size else labels
        
        # Fast clustering metrics
        sil_score = silhouette_score(X_sample, labels_sample)  # On sample only
        calinski_score = calinski_harabasz_score(X, labels)    # Fast on full data
        davies_score = davies_bouldin_score(X, labels)         # Fast on full data
        
        # Target separation (how well clusters separate the target) - on full data
        target_separation = 0
        unique_labels = np.unique(labels)
        
        if len(unique_labels) > 1:
            cluster_target_means = []
            for label in unique_labels:
                cluster_mask = labels == label
                if cluster_mask.sum() > 0:
                    cluster_target_mean = y[cluster_mask].mean()
                    cluster_target_means.append(cluster_target_mean)
            
            if len(cluster_target_means) > 1:
                target_separation = np.std(cluster_target_means)
        
        # Composite quality score (emphasize target separation for trading)
        composite_score = (0.4 * target_separation * 10 +  # Scale up target separation
                          0.3 * (sil_score + 1) / 2 +       # Normalize silhouette to 0-1
                          0.2 * min(calinski_score / 10000, 1) + # Normalize Calinski
                          0.1 * (2 - davies_score) / 2)     # Invert Davies-Bouldin (lower is better)
        
        results[method] = {
            'silhouette_score': sil_score,
            'calinski_harabasz': calinski_score,
            'davies_bouldin': davies_score,
            'target_separation': target_separation,
            'composite_score': composite_score,
            'n_clusters': len(unique_labels)
        }
        
        print(f"{method.upper()}:")
        print(f"  Silhouette Score: {sil_score:.4f} (sampled)")
        print(f"  Calinski-Harabasz: {calinski_score:.2f}")
        print(f"  Davies-Bouldin: {davies_score:.4f}")
        print(f"  Target Separation: {target_separation:.4f}")
        print(f"  Composite Score: {composite_score:.4f}")
        print(f"  Clusters Found: {len(unique_labels)}")
    
    # Select best model based on composite score
    best_method = max(results.keys(), key=lambda x: results[x]['composite_score'])
    print(f"\nBest clustering method: {best_method.upper()} (composite score: {results[best_method]['composite_score']:.4f})")
    
    return results, best_method

def visualize_clusters(models, X, best_method):
    """Create visualizations of the clustering results"""
    print(f"\n=== CREATING CLUSTER VISUALIZATIONS ===")
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Sample data for visualization (PCA on full 3M points is slow)
    viz_sample_size = 50000
    if len(X) > viz_sample_size:
        print(f"Using sample of {viz_sample_size:,} points for visualization")
        viz_indices = np.random.choice(len(X), viz_sample_size, replace=False)
        X_viz = X.iloc[viz_indices]
    else:
        X_viz = X
        viz_indices = np.arange(len(X))
    
    # Use PCA for 2D visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X_viz))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot all clustering methods
    methods = ['gmm', 'kmeans']
    colors = ['viridis', 'Set1']
    
    for i, method in enumerate(methods):
        if method in models:
            labels = models[method]['labels']
            labels_viz = labels[viz_indices] if len(X) > viz_sample_size else labels
            
            scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_viz, 
                                    cmap=colors[i], alpha=0.6, s=1)
            axes[i].set_title(f'{method.upper()} Clustering')
            axes[i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {OUTPUT_DIR}/cluster_visualization.png")

def save_clustering_model(models, scaler, best_method, X, results):
    """Save the best clustering model and results"""
    print(f"\n=== SAVING CLUSTERING MODEL ===")
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    best_model_info = models[best_method]
    
    # Save model and scaler
    joblib.dump(best_model_info['model'], f"{OUTPUT_DIR}/regime_classifier.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/feature_scaler.pkl")
    
    # Save feature names
    joblib.dump(X.columns.tolist(), f"{OUTPUT_DIR}/feature_names.pkl")
    
    # Save regime labels
    np.save(f"{OUTPUT_DIR}/regime_labels.npy", best_model_info['labels'])
    
    # Save metadata
    metadata = {
        'method': best_method,
        'n_regimes': N_REGIMES,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'feature_names': X.columns.tolist(),
        'silhouette_score': results[best_method]['silhouette_score'],
        'target_separation': results[best_method]['target_separation'],
        'training_date': datetime.now().isoformat(),
        'data_source': FEATURES_FILE
    }
    
    with open(f"{OUTPUT_DIR}/clustering_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Clustering model saved:")
    print(f"  Method: {best_method.upper()}")
    print(f"  Model: regime_classifier.pkl")
    print(f"  Scaler: feature_scaler.pkl")
    print(f"  Labels: regime_labels.npy")
    print(f"  Metadata: clustering_metadata.json")
    print(f"  Directory: {OUTPUT_DIR}")
    
    return metadata

def main():
    """Main clustering pipeline"""
    print("Starting regime clustering pipeline...\n")
    
    # Load data
    X, y, metadata = load_and_sample_data()
    
    # Train clustering models
    models, scaler = train_clustering_models(X)
    
    # Evaluate cluster quality
    results, best_method = evaluate_cluster_quality(models, X, y)
    
    # Create visualizations
    visualize_clusters(models, X, best_method)
    
    # Save best model
    saved_metadata = save_clustering_model(models, scaler, best_method, X, results)
    
    print(f"\n" + "="*60)
    print("✅ REGIME CLUSTERING COMPLETED!")
    print("="*60)
    print(f"Best Method: {best_method.upper()}")
    print(f"Silhouette Score: {results[best_method]['silhouette_score']:.4f}")
    print(f"Target Separation: {results[best_method]['target_separation']:.4f}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 02_regime_correlation_analysis.py")
    print("2. Run 03_regime_specific_models.py") 
    print("3. Run 04_regime_model_comparison.py")

if __name__ == "__main__":
    main()