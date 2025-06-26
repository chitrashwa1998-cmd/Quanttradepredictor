
import pandas as pd
import numpy as np

def create_volatility_features(df):
    """Create volatility-specific custom features"""
    
    # Calculate rolling volatility
    returns = df['Close'].pct_change()
    df['volatility_10'] = returns.rolling(10).std()
    df['volatility_20'] = returns.rolling(20).std()
    
    # Volatility ratios
    df['volatility_ratio'] = df['volatility_10'] / df['volatility_20'].replace(0, np.nan)
    
    # Volatility regime detection
    df['volatility_regime'] = np.where(df['volatility_10'] > df['volatility_20'].rolling(50).quantile(0.7), 'high', 'low')
    df['volatility_regime'] = np.where(df['volatility_10'] < df['volatility_20'].rolling(50).quantile(0.3), 'low', df['volatility_regime'])
    df['volatility_regime'] = np.where((df['volatility_10'] >= df['volatility_20'].rolling(50).quantile(0.3)) & 
                                     (df['volatility_10'] <= df['volatility_20'].rolling(50).quantile(0.7)), 'medium', df['volatility_regime'])
    
    # Convert to numeric for model training
    volatility_regime_map = {'low': 0, 'medium': 1, 'high': 2}
    df['volatility_regime'] = df['volatility_regime'].map(volatility_regime_map)
    
    return df

def create_candle_behavior_features(df):
    """Create basic candle behavior features for volatility context"""
    
    # Basic candle metrics
    df['body_size'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['total_range'] = df['High'] - df['Low']
    
    # Ratios for volatility analysis
    df['body_ratio'] = df['body_size'] / df['total_range'].replace(0, np.nan)
    df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['total_range'].replace(0, np.nan)
    
    return df

def create_all_custom_features(df):
    """Create all volatility-focused custom features"""
    
    df = create_volatility_features(df)
    df = create_candle_behavior_features(df)
    
    return df
