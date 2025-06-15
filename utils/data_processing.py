import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import streamlit as st

class DataProcessor:
    """Utility class for data processing and validation."""
    
    @staticmethod
    def validate_ohlc_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate OHLC data format and quality."""
        required_columns = ['Open', 'High', 'Low', 'Close']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for non-numeric data
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"Column {col} must be numeric"
        
        # Check for negative prices
        for col in required_columns:
            if (df[col] <= 0).any():
                return False, f"Column {col} contains non-positive values"
        
        # Check OHLC logic
        invalid_ohlc = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        )
        
        if invalid_ohlc.any():
            return False, "Invalid OHLC data: High < Low or prices outside High-Low range"
        
        # Check for sufficient data
        if len(df) < 100:
            return False, f"Insufficient data: {len(df)} rows. Need at least 100 rows for meaningful analysis."
        
        return True, "Data validation passed"
    
    @staticmethod
    def load_and_process_data(uploaded_file) -> Tuple[pd.DataFrame, str]:
        """Load and process uploaded OHLC data."""
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Try to identify date column
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_columns:
                date_col = date_columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            else:
                # Try to parse first column as date
                first_col = df.columns[0]
                try:
                    df[first_col] = pd.to_datetime(df[first_col])
                    df.set_index(first_col, inplace=True)
                except:
                    # Create a simple date index
                    df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Validate data
            is_valid, message = DataProcessor.validate_ohlc_data(df)
            
            if not is_valid:
                return None, f"Data validation failed: {message}"
            
            return df, "Data loaded and processed successfully"
            
        except Exception as e:
            return None, f"Error loading data: {str(e)}"
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        summary = {
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'days': (df.index.max() - df.index.min()).days
            },
            'price_summary': {
                'min_close': df['Close'].min(),
                'max_close': df['Close'].max(),
                'mean_close': df['Close'].mean(),
                'std_close': df['Close'].std()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'columns': list(df.columns)
        }
        
        # Calculate returns
        returns = df['Close'].pct_change().dropna()
        summary['returns'] = {
            'mean_daily_return': returns.mean(),
            'volatility': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_daily_return': returns.max(),
            'min_daily_return': returns.min()
        }
        
        return summary
    
    @staticmethod
    def detect_data_frequency(df: pd.DataFrame) -> str:
        """Detect the frequency of the data (daily, hourly, etc.)."""
        if len(df) < 2:
            return "Unknown"
        
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff <= pd.Timedelta(minutes=5):
            return "5-minute or less"
        elif median_diff <= pd.Timedelta(minutes=15):
            return "15-minute"
        elif median_diff <= pd.Timedelta(hours=1):
            return "Hourly"
        elif median_diff <= pd.Timedelta(hours=4):
            return "4-hour"
        elif median_diff <= pd.Timedelta(days=1):
            return "Daily"
        elif median_diff <= pd.Timedelta(days=7):
            return "Weekly"
        else:
            return "Monthly or longer"
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers."""
        df_clean = df.copy()
        
        # Forward fill missing values
        df_clean = df_clean.fillna(method='ffill')
        
        # Remove any remaining NaN values
        df_clean = df_clean.dropna()
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_clean.columns:
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                outlier_mask = np.abs(df_clean[col] - mean_val) > 3 * std_val
                
                if outlier_mask.sum() > 0:
                    st.warning(f"Removed {outlier_mask.sum()} outliers from {col}")
                    # Replace outliers with median
                    df_clean.loc[outlier_mask, col] = df_clean[col].median()
        
        return df_clean
