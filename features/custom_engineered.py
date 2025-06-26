
import pandas as pd
import numpy as np

def create_custom_volatility_features(df):
    """Create only volatility-related custom features"""
    
    # Volatility_10 calculation (rolling standard deviation of log returns)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_10'] = df['log_return'].rolling(window=10).std()
    
    # Volatility regime classification based on percentiles
    df['volatility_regime'] = pd.cut(
        df['volatility_10'].fillna(0), 
        bins=3, 
        labels=['Low', 'Medium', 'High']
    ).astype(str)
    
    # High-Low ratio (volatility measure)
    df['high_low_ratio'] = (df['high'] / df['low']) - 1
    
    # EMA 5 for volatility model
    df['ema_5'] = df['close'].ewm(span=5).mean()
    
    # Bollinger Band position
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # VWAP proxy using average price
    df['vwap_proxy'] = (df['high'] + df['low'] + df['close']) / 3
    df['price_vs_vwap'] = df['close'] - df['vwap_proxy']
    
    # Momentum acceleration for volatility
    df['price_momentum_1'] = df['close'].pct_change(1)
    df['price_momentum_3'] = df['close'].pct_change(3)
    df['momentum_acceleration'] = df['price_momentum_1'] - df['price_momentum_3']
    
    return df
