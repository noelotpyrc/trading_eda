import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SolanaDataAnalyzer:
    """
    Utility class for analyzing the large Solana swap dataset efficiently
    """
    
    def __init__(self, db_path='/Volumes/Extreme SSD/DuckDB/solana.duckdb', table_name='first_day_trades'):
        """Initialize connection to DuckDB"""
        self.db_path = db_path
        self.table_name = table_name
        self.con = None
        self.connect()
        
    def connect(self):
        """Connect to DuckDB database"""
        try:
            self.con = duckdb.connect(database=self.db_path, read_only=True)
            print(f"Connected to database: {self.db_path}")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            
    def close(self):
        """Close database connection"""
        if self.con:
            self.con.close()
            
    def execute_query(self, query, fetch_df=True):
        """Execute SQL query and return results"""
        try:
            if fetch_df:
                return self.con.execute(query).fetchdf()
            else:
                return self.con.execute(query).fetchall()
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
            
    def get_sample(self, sample_size=10000, random_seed=42):
        """Get random sample of data"""
        query = f"""
        SELECT * FROM {self.table_name} 
        USING SAMPLE {sample_size} ROWS (bernoulli, {random_seed})
        """
        return self.execute_query(query)
        
    def get_time_range_data(self, start_date, end_date, columns='*'):
        """Get data for specific time range"""
        query = f"""
        SELECT {columns} FROM {self.table_name}
        WHERE block_timestamp >= '{start_date}' 
        AND block_timestamp <= '{end_date}'
        """
        return self.execute_query(query)
        
    def get_daily_stats(self):
        """Get daily trading statistics"""
        query = f"""
        SELECT 
            DATE(block_timestamp) as trade_date,
            COUNT(*) as trade_count,
            COUNT(DISTINCT swapper) as unique_traders,
            COUNT(DISTINCT mint) as unique_tokens,
            SUM(CASE WHEN succeeded THEN 1 ELSE 0 END) as successful_trades,
            AVG(swap_from_amount) as avg_from_amount,
            AVG(swap_to_amount) as avg_to_amount,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY swap_from_amount) as median_from_amount
        FROM {self.table_name}
        GROUP BY DATE(block_timestamp)
        ORDER BY trade_date
        """
        return self.execute_query(query)
        
    def get_hourly_patterns(self):
        """Get hourly trading patterns"""
        query = f"""
        SELECT 
            EXTRACT(hour FROM block_timestamp) as hour,
            COUNT(*) as trade_count,
            AVG(swap_from_amount) as avg_trade_size
        FROM {self.table_name}
        GROUP BY EXTRACT(hour FROM block_timestamp)
        ORDER BY hour
        """
        return self.execute_query(query)
        
    def get_token_stats(self, limit=50):
        """Get token trading statistics"""
        query = f"""
        SELECT 
            mint,
            COUNT(*) as trade_count,
            COUNT(DISTINCT swapper) as unique_traders,
            SUM(swap_to_amount) as total_volume_received,
            AVG(swap_to_amount) as avg_amount_received,
            MIN(block_timestamp) as first_trade,
            MAX(block_timestamp) as last_trade
        FROM {self.table_name}
        GROUP BY mint
        ORDER BY trade_count DESC
        LIMIT {limit}
        """
        return self.execute_query(query)
        
    def get_trader_stats(self, limit=100):
        """Get trader statistics"""
        query = f"""
        SELECT 
            swapper,
            COUNT(*) as trade_count,
            COUNT(DISTINCT mint) as tokens_traded,
            SUM(swap_from_amount) as total_volume_sent,
            AVG(swap_from_amount) as avg_trade_size,
            MIN(block_timestamp) as first_trade,
            MAX(block_timestamp) as last_trade
        FROM {self.table_name}
        GROUP BY swapper
        ORDER BY trade_count DESC
        LIMIT {limit}
        """
        return self.execute_query(query)
        
    def get_failed_trades_analysis(self):
        """Analyze failed trades"""
        query = f"""
        SELECT 
            DATE(block_timestamp) as date,
            COUNT(*) as failed_count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as pct_of_total_failures
        FROM {self.table_name}
        WHERE succeeded = false
        GROUP BY DATE(block_timestamp)
        ORDER BY failed_count DESC
        """
        return self.execute_query(query)
        
    def get_token_pair_analysis(self, limit=50):
        """Analyze most common token pairs"""
        query = f"""
        SELECT 
            swap_from_mint,
            swap_to_mint,
            COUNT(*) as pair_count,
            AVG(swap_from_amount) as avg_from_amount,
            AVG(swap_to_amount) as avg_to_amount
        FROM {self.table_name}
        WHERE succeeded = true
        GROUP BY swap_from_mint, swap_to_mint
        ORDER BY pair_count DESC
        LIMIT {limit}
        """
        return self.execute_query(query)
        
    def plot_daily_volume(self, data=None):
        """Plot daily trading volume"""
        if data is None:
            data = self.get_daily_stats()
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['trade_date'],
            y=data['trade_count'],
            mode='lines',
            name='Daily Trades',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title='Daily Trading Volume Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Trades',
            template='plotly_white'
        )
        return fig
        
    def plot_hourly_patterns(self):
        """Plot hourly trading patterns"""
        data = self.get_hourly_patterns()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data['hour'],
            y=data['trade_count'],
            name='Hourly Trade Count'
        ))
        
        fig.update_layout(
            title='Trading Activity by Hour of Day',
            xaxis_title='Hour',
            yaxis_title='Number of Trades',
            template='plotly_white'
        )
        return fig
        
    def plot_token_distribution(self, limit=20):
        """Plot top token distribution"""
        data = self.get_token_stats(limit)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data['trade_count'][:limit],
            y=data['mint'][:limit],
            orientation='h',
            name='Trade Count'
        ))
        
        fig.update_layout(
            title=f'Top {limit} Tokens by Trade Count',
            xaxis_title='Number of Trades',
            yaxis_title='Token',
            template='plotly_white',
            height=600
        )
        return fig
        
    def get_summary_stats(self):
        """Get comprehensive summary statistics"""
        query = f"""
        SELECT 
            COUNT(*) as total_trades,
            COUNT(DISTINCT swapper) as unique_traders,
            COUNT(DISTINCT mint) as unique_tokens,
            COUNT(DISTINCT swap_from_mint) as unique_from_tokens,
            COUNT(DISTINCT swap_to_mint) as unique_to_tokens,
            SUM(CASE WHEN succeeded THEN 1 ELSE 0 END) as successful_trades,
            MIN(block_timestamp) as earliest_trade,
            MAX(block_timestamp) as latest_trade,
            AVG(swap_from_amount) as avg_from_amount,
            AVG(swap_to_amount) as avg_to_amount,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY swap_from_amount) as median_from_amount,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY swap_from_amount) as p95_from_amount
        FROM {self.table_name}
        """
        return self.execute_query(query)
        
    def create_analysis_cache(self, cache_name, query):
        """Create cached table for repeated analysis"""
        create_query = f"CREATE OR REPLACE TABLE {cache_name} AS {query}"
        self.con.execute(create_query)
        print(f"Created cache table: {cache_name}")

# Utility functions
def format_large_number(num):
    """Format large numbers for display"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(int(num))

def truncate_address(address, start=8, end=4):
    """Truncate blockchain addresses for display"""
    if len(address) <= start + end:
        return address
    return f"{address[:start]}...{address[-end:]}"

def safe_divide(a, b):
    """Safe division that handles division by zero"""
    return a / b if b != 0 else 0

# Data validation functions
def validate_timestamp_range(df, timestamp_col='block_timestamp'):
    """Validate timestamp data"""
    if timestamp_col not in df.columns:
        return False, "Timestamp column not found"
    
    try:
        min_time = df[timestamp_col].min()
        max_time = df[timestamp_col].max()
        print(f"Timestamp range: {min_time} to {max_time}")
        return True, f"Valid range: {min_time} to {max_time}"
    except Exception as e:
        return False, f"Error validating timestamps: {e}"

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """Detect outliers in numerical columns"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > threshold]
        return outliers
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

# Constants
SOL_TOKEN = 'So11111111111111111111111111111111111111112'
USDC_TOKEN = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'

# Common token mappings (can be expanded)
TOKEN_SYMBOLS = {
    SOL_TOKEN: 'SOL',
    USDC_TOKEN: 'USDC'
} 