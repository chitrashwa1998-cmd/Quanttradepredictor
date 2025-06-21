#!/usr/bin/env python3
"""
Load sample OHLC data into the database for dashboard testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append('.')

from utils.database_adapter import get_trading_database

def create_sample_nifty_data(days=30):
    """Create realistic sample NIFTY 50 data for the last N days"""
    
    # Generate 5-minute intervals for market hours (9:15 AM to 3:30 PM IST)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create date range for trading days only (Monday to Friday)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_dates = [d for d in all_dates if d.weekday() < 5]  # Monday=0, Friday=4
    
    # Generate 5-minute intervals for each trading day (9:15 AM to 3:30 PM)
    timestamps = []
    for date in trading_dates:
        # Market hours: 9:15 AM to 3:30 PM (375 minutes = 75 intervals of 5 minutes)
        start_time = date.replace(hour=9, minute=15, second=0, microsecond=0)
        end_time = date.replace(hour=15, minute=30, second=0, microsecond=0)
        
        day_intervals = pd.date_range(start=start_time, end=end_time, freq='5T')
        timestamps.extend(day_intervals)
    
    # Convert to DataFrame index
    df_index = pd.DatetimeIndex(timestamps)
    
    # Generate realistic NIFTY 50 data
    np.random.seed(42)  # For reproducible data
    
    # Starting price around current NIFTY levels
    base_price = 22500.0
    num_points = len(df_index)
    
    # Generate price movements with realistic volatility
    returns = np.random.normal(0, 0.0015, num_points)  # ~0.15% volatility per 5-min interval
    
    # Add some trend and mean reversion
    trend = np.linspace(-0.02, 0.02, num_points)  # Slight overall trend
    mean_reversion = -0.1 * np.cumsum(returns)  # Mean reversion factor
    
    combined_returns = returns + trend/num_points + mean_reversion/num_points
    
    # Generate prices using cumulative returns
    prices = base_price * np.exp(np.cumsum(combined_returns))
    
    # Generate OHLC data
    ohlc_data = []
    
    for i, price in enumerate(prices):
        # Add some intraday volatility
        volatility = abs(np.random.normal(0, 0.003))  # ~0.3% intraday volatility
        
        high = price * (1 + volatility * np.random.uniform(0.3, 1.0))
        low = price * (1 - volatility * np.random.uniform(0.3, 1.0))
        
        # Ensure logical OHLC relationships
        if i == 0:
            open_price = base_price
        else:
            open_price = ohlc_data[i-1]['Close']  # Open = previous close
        
        close_price = price
        
        # Adjust high/low to make sense
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Generate volume (higher during market open/close)
        hour = df_index[i].hour
        if hour in [9, 10, 14, 15]:  # Higher volume at open/close
            base_volume = np.random.uniform(800000, 1500000)
        else:
            base_volume = np.random.uniform(300000, 800000)
        
        volume = int(base_volume * (1 + np.random.normal(0, 0.3)))
        volume = max(volume, 100000)  # Minimum volume
        
        ohlc_data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(ohlc_data, index=df_index)
    
    return df

def load_sample_data():
    """Load sample data into the database"""
    try:
        print("Creating sample NIFTY 50 data...")
        
        # Generate 30 days of 5-minute data
        sample_data = create_sample_nifty_data(days=30)
        print(f"Generated {len(sample_data)} data points")
        
        # Get database connection
        db = get_trading_database()
        
        # Save the data as main_dataset
        success = db.save_ohlc_data(sample_data, "main_dataset", preserve_full_data=True)
        
        if success:
            print("✅ Sample data loaded successfully into database")
            print(f"Data range: {sample_data.index.min()} to {sample_data.index.max()}")
            print(f"Latest price: ₹{sample_data['Close'].iloc[-1]:,.2f}")
            return True
        else:
            print("❌ Failed to save sample data")
            return False
            
    except Exception as e:
        print(f"Error loading sample data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    load_sample_data()