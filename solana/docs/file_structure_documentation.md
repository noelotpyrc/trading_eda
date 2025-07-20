# Solana Trading Data Analysis - File Structure Documentation

## Overview
This document provides a comprehensive overview of the Solana trading analysis project structure, including all files, their purposes, dependencies, and usage patterns.

**Last Updated**: July 20, 2025  
**Total Files Documented**: 40+ files across 8 directories  
**Data Coverage**: 325M+ trades, 10M+ traders, 5.9K tokens

---

## 📁 Project Structure

```
solana/
├── analysis/                    # Jupyter notebooks for data analysis
│   ├── batch578_sample_analysis/   # Analysis on sample batch 578 data
│   └── full_data/                  # Analysis on complete dataset
├── docs/                        # Documentation
├── feature_engineering/         # Data processing and feature extraction
│   ├── classification_forward/     # Features for ML classification
│   └── to_ohlc/                   # OHLC data conversion tools
├── inference/                   # Model inference scripts
│   ├── classification_forward/     # Classification model inference
│   └── trader_clustering/         # Trader clustering inference
├── models/                      # Trained ML models and metadata
│   ├── classification_forward/     # Classification models (empty)
│   └── trader_clustering/         # Trader clustering models
├── tests/                       # Test scripts and validation
│   └── output/                    # Test output files
└── utils/                       # Utility functions and helpers
```

---

## 📊 Analysis Directory

### `/analysis/batch578_sample_analysis/` - Sample Data Analysis (10 notebooks)

**Purpose**: Comprehensive analysis of batch 578 sample data to develop and validate analytical approaches before applying to full dataset.

| File | Purpose | Key Features |
|------|---------|--------------|
| `01_data_loading_structure.ipynb` | Data exploration and structure analysis | Schema validation, data quality checks |
| `02_trader_analysis.ipynb` | Trader behavior profiling | Volume patterns, trading frequency |
| `03-1_temporal_patterns_analysis.ipynb` | Time-based trading pattern discovery | Hourly/daily patterns, seasonality |
| `03-2_temporal_window_discovery.ipynb` | Optimal time window identification | Feature window optimization |
| `04-1_feature_engineering_signals.ipynb` | ML feature creation and validation | 30s/60s/120s aggregated features |
| `04-2_statistical_validation.ipynb` | Statistical testing of features | Significance tests, correlation analysis |
| `04-3_ml_ensemble_signals.ipynb` | ML model ensemble development | Multi-model predictions |
| `05_regime_analysis.ipynb` | Market regime identification | Volatility regimes, state detection |
| `06_order_flow_analysis.ipynb` | Order flow and microstructure analysis | Buy/sell pressure, flow imbalances |
| `07_microstructure_mean_reversion.ipynb` | Mean reversion strategy development | Short-term price prediction |

### `/analysis/full_data/` - Complete Dataset Analysis (5 notebooks)

**Purpose**: Scale validated approaches to the complete 325M+ trade dataset for production-ready insights.

| File | Purpose | Dataset Size | Output |
|------|---------|--------------|--------|
| `01_coin_selection_post2024.ipynb` | Token filtering for post-2024 analysis | 5.9K tokens → filtered subset | Coin selection criteria |
| `02_trader_profiling_features.ipynb` | Large-scale trader feature engineering | 10M+ traders | 33 behavioral features |
| `03_session_based_features.ipynb` | First 2hrs trading session analysis | Session-level aggregation | Session features |
| `04-00_trader_clustering_model_prep.ipynb` | Model preparation and data preprocessing | 10M+ traders | Preprocessed features |
| `04-01_trader_clustering_model_run.ipynb` | K-means clustering model training | 10M+ traders → 5 clusters | Trained clustering model |

---

## 🔧 Feature Engineering Directory

### `/feature_engineering/classification_forward/` - ML Feature Extraction

**Purpose**: Extract time-series features from raw trading data for classification models.

| File | Type | Purpose | Performance |
|------|------|---------|-------------|
| `batch_feature_extraction.sql` | SQL | DuckDB-based feature extraction query | Core extraction logic |
| `csv_feature_extractor.py` | Python | CSV-based feature processor (memory-safe) | 351s for batch 578 |
| `sql_csv_processor.py` | Python | SQL-based batch processor (high-performance) | 48s for batch 578 (7.3x faster) |

**Features Extracted**:
- 30s/60s/120s price and volume aggregations
- Volume-weighted metrics
- Transaction count statistics

### `/feature_engineering/to_ohlc/` - OHLC Data Conversion

**Purpose**: Convert raw trade-by-trade data into standard OHLC (Open, High, Low, Close) format for financial analysis.

| File | Type | Purpose | Dataset Scope |
|------|------|---------|---------------|
| `aggregate_to_ohlc.sql` | SQL | DuckDB query for OHLC aggregation | Single-table processing |
| `aggregate_to_ohlc.py` | Python | Python implementation of OHLC logic | Batch processing |
| `convert_to_ohlc.py` | Python | Full dataset OHLC converter | Complete dataset (67M+ records) |
| `csv_ohlc_processor.py` | Python | CSV-based OHLC processor | Batch CSV processing (589 files) |

**Output Format**: 
- 67.3M OHLC records
- 5,822 unique tokens
- 721M SOL total volume
- Time range: 2021-2025

---

## 🤖 Models Directory

### `/models/trader_clustering/` - Trader Behavioral Clustering

**Purpose**: K-means clustering model to segment traders based on behavioral patterns.

| File | Type | Purpose |
|------|------|---------|
| `model_metadata.json` | Metadata | Model configuration and performance metrics |
| `trader_clustering_kmeans.pkl` | Model | Trained K-means model (5 clusters) |
| `preprocessing_pipeline.pkl` | Pipeline | Feature preprocessing pipeline |
| `robust_scaler.pkl` | Scaler | Feature scaling model |

**Model Specifications**:
- **Algorithm**: K-means clustering
- **Clusters**: 5 trader segments
- **Features**: 33 behavioral features
- **Training Data**: 10.06M traders
- **Silhouette Score**: 0.31 (moderate cluster quality)

**Cluster Distribution**:
- Cluster 0: 745K traders (7.4%)
- Cluster 1: 1.59M traders (15.8%)
- Cluster 2: 5.96M traders (59.3%) - Largest segment
- Cluster 3: 1.14M traders (11.4%)
- Cluster 4: 619K traders (6.2%)

---

## 🔍 Inference Directory

### `/inference/trader_clustering/` - Model Deployment

| File | Purpose |
|------|---------|
| `trader_cluster_predictor.py` | Production inference script for trader classification |

---

## 🧪 Tests Directory

**Purpose**: Validation and quality assurance for data processing pipelines.

| File | Purpose | Validation |
|------|---------|------------|
| `simple_ohlc_test.py` | OHLC SQL vs Python comparison | 3,879/3,879 exact matches |
| `test_csv_vs_sql.py` | Feature extraction method validation | Performance and accuracy testing |
| `output/combined_features.csv` | Test output data | Sample feature extraction results |

**Test Results**:
- ✅ OHLC implementations produce identical results
- ✅ SQL approach 7.3x faster than CSV approach
- ✅ Feature extraction accuracy validated

---

## 🛠️ Utils Directory

**Purpose**: Shared utilities and helper functions across the project.

| File | Purpose | Usage |
|------|---------|-------|
| `solana_eda_utils.py` | Core EDA utilities and data processing functions | Analysis notebooks |
| `dexpaprika_enrichment.py` | External data enrichment from DexScreener API | Token metadata enhancement |

---

## 📋 Documentation Directory

### `/docs/`
| File | Purpose |
|------|---------|
| `trader_feature_engineering.md` | Detailed feature engineering documentation |
| `file_structure_documentation.md` | This comprehensive file structure guide |

---

## 💡 Usage Recommendations

### For New Analysis:
1. Start with `/analysis/batch578_sample_analysis/` notebooks for methodology
2. Scale to full dataset using `/analysis/full_data/` notebooks
3. Use `/feature_engineering/` tools for new feature creation

### For Production:
1. Use `sql_csv_processor.py` for high-performance feature extraction
2. Use `csv_ohlc_processor.py` for OHLC data generation
3. Use `/inference/` scripts for model deployment

### For Testing:
1. Use `/tests/` scripts to validate new implementations
2. Compare SQL vs Python approaches for accuracy
3. Benchmark performance improvements

---

## 🔗 Dependencies

**Core Technologies**:
- **DuckDB**: High-performance SQL processing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models
- **Jupyter**: Interactive analysis notebooks

**External Data Sources**:
- Solana blockchain trading data
- Dexparika API (via dexpaprika_enrichment.py)

---

*This documentation covers the complete Solana trading analysis project structure as of July 20, 2025. For specific implementation details, refer to individual file docstrings and notebook markdown cells.*