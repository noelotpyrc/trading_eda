#!/usr/bin/env python3
"""
Regime Characteristics Analysis
Analyze what each market regime represents in intuitive terms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = "/Volumes/Extreme SSD/trading_data/solana/classification_features/combined_features_sql.csv"
CLUSTERING_DIR = "solana/models/regime_clustering"
OUTPUT_DIR = "solana/analysis/regime_characteristics"
SAMPLE_SIZE = 500000  # Sample for analysis

print("=== REGIME CHARACTERISTICS ANALYSIS ===")
print("Objective: Understand what each market regime represents")
print()

def load_regime_data():
    """Load clustering results and sample data for analysis"""
    print("Loading regime clustering results...")
    
    # Load clustering components
    regime_classifier = joblib.load(f"{CLUSTERING_DIR}/regime_classifier.pkl")
    scaler = joblib.load(f"{CLUSTERING_DIR}/feature_scaler.pkl") 
    feature_names = joblib.load(f"{CLUSTERING_DIR}/feature_names.pkl")
    regime_labels = np.load(f"{CLUSTERING_DIR}/regime_labels.npy")
    
    print(f"Loaded clustering with {len(np.unique(regime_labels))} regimes")
    
    # Load and sample original data
    df = pd.read_csv(FEATURES_FILE)
    print(f"Original dataset: {len(df):,} records")
    
    # Time-stratified sampling 
    if len(df) > SAMPLE_SIZE:
        try:
            df['sample_timestamp'] = pd.to_datetime(df['sample_timestamp'], utc=True)
        except:
            # If timestamp conversion fails, use simple random sampling
            print("Timestamp conversion failed, using random sampling instead")
            sample_indices = np.random.choice(len(df), SAMPLE_SIZE, replace=False)
            df = df.iloc[sample_indices].reset_index(drop=True)
            regime_labels = regime_labels[sample_indices]
        else:
            df = df.sort_values('sample_timestamp')
            step = len(df) // SAMPLE_SIZE
            df = df.iloc[::step].reset_index(drop=True)
            regime_labels = regime_labels[::step]
        print(f"Sampled dataset: {len(df):,} records")
    
    # Prepare features and target
    X = df[feature_names].copy()
    y = df['is_profitable_300s'].copy()
    
    # Handle timestamps safely
    try:
        timestamps = pd.to_datetime(df['sample_timestamp'], utc=True)
    except:
        timestamps = None  # Skip timestamp analysis if conversion fails
    
    # Clean data
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, y, regime_labels, timestamps, feature_names

def analyze_regime_characteristics(X, y, regime_labels, feature_names):
    """Analyze what makes each regime unique"""
    print("\n=== REGIME CHARACTERISTICS ANALYSIS ===")
    
    unique_regimes = np.unique(regime_labels)
    regime_analysis = {}
    
    # Overall statistics
    overall_stats = {
        'profitable_pct': y.mean() * 100,
        'feature_means': X.mean(),
        'feature_stds': X.std()
    }
    
    print(f"Overall profitability: {overall_stats['profitable_pct']:.1f}%")
    print()
    
    for regime in unique_regimes:
        print(f"--- REGIME {regime} ANALYSIS ---")
        
        regime_mask = regime_labels == regime
        regime_X = X[regime_mask]
        regime_y = y[regime_mask]
        
        n_samples = regime_mask.sum()
        pct_of_total = n_samples / len(regime_labels) * 100
        profitable_pct = regime_y.mean() * 100
        
        print(f"Sample size: {n_samples:,} ({pct_of_total:.1f}% of total)")
        print(f"Profitability: {profitable_pct:.1f}% (vs {overall_stats['profitable_pct']:.1f}% overall)")
        
        # Calculate feature deviations from overall mean
        regime_means = regime_X.mean()
        feature_ratios = regime_means / (overall_stats['feature_means'] + 1e-10)
        
        # Find most distinctive features (furthest from 1.0 ratio)
        feature_deviations = np.abs(np.log(feature_ratios + 1e-10))
        top_features_idx = np.argsort(feature_deviations)[-10:][::-1]  # Top 10 most distinctive
        
        print(f"\nTop distinctive characteristics:")
        regime_characteristics = []
        
        for i, feat_idx in enumerate(top_features_idx):
            feature = feature_names[feat_idx]
            ratio = feature_ratios.iloc[feat_idx]
            regime_val = regime_means.iloc[feat_idx]
            overall_val = overall_stats['feature_means'].iloc[feat_idx]
            
            if ratio > 1.5:
                description = "HIGH"
            elif ratio < 0.67:
                description = "LOW" 
            else:
                continue  # Skip features that aren't very different
                
            regime_characteristics.append({
                'feature': feature,
                'regime_value': regime_val,
                'overall_value': overall_val,
                'ratio': ratio,
                'description': description
            })
            
            print(f"  {i+1:2d}. {feature:<35} {description:<4} ({ratio:.2f}x)")
        
        # Market activity level classification
        volume_features = [f for f in feature_names if 'volume' in f and '30s' in f]
        txn_features = [f for f in feature_names if 'txns' in f and '30s' in f]
        whale_features = [f for f in feature_names if 'whale' in f]
        
        avg_volume_ratio = regime_X[volume_features].mean().mean() / X[volume_features].mean().mean()
        avg_txn_ratio = regime_X[txn_features].mean().mean() / X[txn_features].mean().mean()
        avg_whale_ratio = regime_X[whale_features].mean().mean() / X[whale_features].mean().mean()
        
        # Regime interpretation
        if avg_volume_ratio > 1.5 and avg_whale_ratio > 1.5:
            regime_type = "🐋 HIGH WHALE ACTIVITY"
        elif avg_volume_ratio > 1.3:
            regime_type = "📈 HIGH VOLUME TRADING"
        elif avg_volume_ratio < 0.7:
            regime_type = "😴 LOW ACTIVITY / QUIET"
        elif profitable_pct > overall_stats['profitable_pct'] + 5:
            regime_type = "💰 HIGH PROFIT POTENTIAL"
        elif profitable_pct < overall_stats['profitable_pct'] - 5:
            regime_type = "⚠️ LOW PROFIT / RISKY"
        else:
            regime_type = "📊 MODERATE ACTIVITY"
        
        print(f"\nRegime interpretation: {regime_type}")
        print(f"Volume activity: {avg_volume_ratio:.2f}x normal")
        print(f"Transaction activity: {avg_txn_ratio:.2f}x normal")
        print(f"Whale activity: {avg_whale_ratio:.2f}x normal")
        
        regime_analysis[regime] = {
            'n_samples': n_samples,
            'pct_of_total': pct_of_total,
            'profitable_pct': profitable_pct,
            'characteristics': regime_characteristics,
            'activity_ratios': {
                'volume': avg_volume_ratio,
                'transactions': avg_txn_ratio,
                'whale': avg_whale_ratio
            },
            'interpretation': regime_type
        }
        
        print()
    
    return regime_analysis, overall_stats

def create_regime_comparison_plots(X, y, regime_labels, regime_analysis):
    """Create comprehensive visualization comparing regimes"""
    print("=== CREATING REGIME COMPARISON VISUALIZATIONS ===")
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    unique_regimes = np.unique(regime_labels)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Profitability by regime
    ax1 = plt.subplot(3, 4, 1)
    profitable_pcts = [regime_analysis[r]['profitable_pct'] for r in unique_regimes]
    regime_names = [f"R{r}\\n{regime_analysis[r]['interpretation'].split()[1:]}" for r in unique_regimes]
    
    bars = ax1.bar(range(len(unique_regimes)), profitable_pcts, 
                   color=['red', 'orange', 'green'][:len(unique_regimes)], alpha=0.7)
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('Profitability %')
    ax1.set_title('Profitability by Regime')
    ax1.set_xticks(range(len(unique_regimes)))
    ax1.set_xticklabels([f'R{r}' for r in unique_regimes])
    
    # Add value labels on bars
    for bar, pct in zip(bars, profitable_pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Sample distribution
    ax2 = plt.subplot(3, 4, 2)
    sample_pcts = [regime_analysis[r]['pct_of_total'] for r in unique_regimes]
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_regimes)))
    
    wedges, texts, autotexts = ax2.pie(sample_pcts, labels=[f'R{r}' for r in unique_regimes],
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Sample Distribution by Regime')
    
    # 3-5. Activity level comparisons
    activity_types = ['volume', 'transactions', 'whale']
    for i, activity in enumerate(activity_types):
        ax = plt.subplot(3, 4, 3 + i)
        ratios = [regime_analysis[r]['activity_ratios'][activity] for r in unique_regimes]
        
        bars = ax.bar(range(len(unique_regimes)), ratios,
                      color=colors, alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Normal')
        ax.set_xlabel('Regime')
        ax.set_ylabel('Activity Ratio')
        ax.set_title(f'{activity.title()} Activity Level')
        ax.set_xticks(range(len(unique_regimes)))
        ax.set_xticklabels([f'R{r}' for r in unique_regimes])
        ax.legend()
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # 6-8. Feature heatmaps for each regime
    key_features = ['total_volume_30s', 'whale_txn_ratio_30s', 'buy_ratio_30s', 
                   'volume_imbalance_30s', 'unique_traders_30s', 'avg_txn_size_30s']
    
    for i, regime in enumerate(unique_regimes):
        ax = plt.subplot(3, 4, 6 + i)
        regime_mask = regime_labels == regime
        regime_X = X[regime_mask]
        
        # Normalize features to show relative to overall mean
        feature_data = []
        feature_labels = []
        
        for feature in key_features:
            if feature in X.columns:
                regime_mean = regime_X[feature].mean()
                overall_mean = X[feature].mean()
                ratio = regime_mean / (overall_mean + 1e-10)
                feature_data.append(ratio)
                feature_labels.append(feature.replace('_30s', '').replace('_', ' ').title())
        
        # Create horizontal bar chart
        y_pos = np.arange(len(feature_labels))
        bars = ax.barh(y_pos, feature_data, color=colors[i], alpha=0.7)
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=8)
        ax.set_xlabel('Ratio vs Overall')
        ax.set_title(f'Regime {regime} Key Features')
        
        # Add value labels
        for bar, ratio in zip(bars, feature_data):
            width = bar.get_width()
            ax.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                   f'{ratio:.2f}x', ha='left', va='center', fontweight='bold', fontsize=8)
    
    # 9. Time series of regime occurrence (if we have timestamps)
    ax9 = plt.subplot(3, 4, 9)
    
    # Create regime occurrence over time (simplified)
    regime_counts = []
    regime_colors_list = []
    for regime in unique_regimes:
        count = (regime_labels == regime).sum()
        regime_counts.append(count)
        regime_colors_list.append(colors[regime])
    
    bars = ax9.bar(range(len(unique_regimes)), regime_counts, color=regime_colors_list, alpha=0.7)
    ax9.set_xlabel('Regime')
    ax9.set_ylabel('Total Occurrences')
    ax9.set_title('Regime Frequency')
    ax9.set_xticks(range(len(unique_regimes)))
    ax9.set_xticklabels([f'R{r}' for r in unique_regimes])
    
    # 10-12. Regime interpretations as text
    for i, regime in enumerate(unique_regimes):
        ax = plt.subplot(3, 4, 10 + i)
        ax.axis('off')
        
        interpretation = regime_analysis[regime]['interpretation']
        characteristics = regime_analysis[regime]['characteristics'][:5]  # Top 5
        
        text_content = f"REGIME {regime}\n{interpretation}\n\n"
        text_content += f"Sample: {regime_analysis[regime]['n_samples']:,} ({regime_analysis[regime]['pct_of_total']:.1f}%)\n"
        text_content += f"Profitability: {regime_analysis[regime]['profitable_pct']:.1f}%\n\n"
        text_content += "Key Characteristics:\n"
        
        for char in characteristics[:3]:  # Top 3 characteristics
            text_content += f"• {char['feature'].replace('_', ' ').title()}: {char['description']}\n"
        
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor=colors[i], alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/regime_characteristics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comprehensive regime analysis saved to: {OUTPUT_DIR}/regime_characteristics_analysis.png")

def save_regime_analysis(regime_analysis, overall_stats):
    """Save regime analysis results"""
    print("\n=== SAVING REGIME ANALYSIS ===")
    
    import os
    import json
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    analysis_results = {
        'analysis_date': datetime.now().isoformat(),
        'regime_analysis': {str(k): convert_numpy_types(v) for k, v in regime_analysis.items()},
        'overall_stats': convert_numpy_types(overall_stats),
        'summary': {
            'total_regimes': len(regime_analysis),
            'most_profitable_regime': int(max(regime_analysis.keys(), 
                                        key=lambda x: regime_analysis[x]['profitable_pct'])),
            'largest_regime': int(max(regime_analysis.keys(),
                                key=lambda x: regime_analysis[x]['n_samples'])),
            'regime_interpretations': {
                str(regime): analysis['interpretation'] 
                for regime, analysis in regime_analysis.items()
            }
        }
    }
    
    with open(f"{OUTPUT_DIR}/regime_characteristics_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Create simple summary
    print("✅ REGIME SUMMARY:")
    for regime, analysis in regime_analysis.items():
        print(f"  Regime {regime}: {analysis['interpretation']}")
        print(f"    └─ {analysis['n_samples']:,} samples ({analysis['pct_of_total']:.1f}%)")
        print(f"    └─ {analysis['profitable_pct']:.1f}% profitable")
    
    print(f"\n📁 Analysis saved to: {OUTPUT_DIR}/regime_characteristics_analysis.json")
    
    return analysis_results

def main():
    """Main regime characteristics analysis pipeline"""
    print("Starting regime characteristics analysis...\n")
    
    # Load data
    X, y, regime_labels, timestamps, feature_names = load_regime_data()
    
    # Analyze regime characteristics  
    regime_analysis, overall_stats = analyze_regime_characteristics(X, y, regime_labels, feature_names)
    
    # Create visualizations
    create_regime_comparison_plots(X, y, regime_labels, regime_analysis)
    
    # Save results
    results = save_regime_analysis(regime_analysis, overall_stats)
    
    print(f"\n" + "="*80)
    print("✅ REGIME CHARACTERISTICS ANALYSIS COMPLETED!")
    print("="*80)
    print("KEY INSIGHTS:")
    for regime, analysis in regime_analysis.items():
        print(f"  🎯 Regime {regime}: {analysis['interpretation']}")
    print(f"\n📊 Results saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()