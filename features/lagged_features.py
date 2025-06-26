
import pandas as pd
import numpy as np

def create_lagged_volatility_features(df):
    """Create lagged features for volatility prediction"""
    
    # Log returns (if not already present)
    if 'log_return' not in df.columns:
        if 'Close' in df.columns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        elif 'close' in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Rolling volatility
    if 'log_return' in df.columns:
        df['vol_rolling_10'] = df['log_return'].rolling(window=10).std()
        
        # Lagged volatility values
        df['vol_lag_1'] = df['vol_rolling_10'].shift(1)
        df['vol_lag_3'] = df['vol_rolling_10'].shift(3)
        df['vol_lag_5'] = df['vol_rolling_10'].shift(5)

    # Lagged ATR and BB width (if they exist)
    if 'atr' in df.columns:
        df['atr_lag_1'] = df['atr'].shift(1)
        df['atr_lag_3'] = df['atr'].shift(3)
    
    if 'bb_width' in df.columns:
        df['bb_width_lag_1'] = df['bb_width'].shift(1)
        df['bb_width_lag_5'] = df['bb_width'].shift(5)

    # Lagged log returns (1 to 5 bars) for volatility context
    if 'log_return' in df.columns:
        for i in range(1, 6):
            df[f'log_return_lag_{i}'] = df['log_return'].shift(i)

    # Volatility delta (change from previous bar)
    if 'vol_rolling_10' in df.columns and 'vol_lag_1' in df.columns:
        df['vol_delta_1'] = df['vol_rolling_10'] - df['vol_lag_1']

    # Volatility regime lag (if defined elsewhere)
    if 'volatility_regime' in df.columns:
        df['volatility_regime_lag'] = df['volatility_regime'].shift(1)

    return df

def create_all_lagged_features(df):
    """Create all volatility-focused lagged features"""
    
    df = create_lagged_volatility_features(df)
    
    return df
