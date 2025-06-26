import pandas as pd
import numpy as np

def create_lagged_volatility_features(df):
    """Create only volatility-related lagged features"""

    # Ensure volatility_10 exists
    if 'volatility_10' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_10'] = df['log_return'].rolling(window=10).std()

    # Lagged volatility values
    df['vol_lag_1'] = df['volatility_10'].shift(1)
    df['vol_lag_3'] = df['volatility_10'].shift(3)

    # Lagged ATR (if available)
    if 'atr' in df.columns:
        df['atr_lag_1'] = df['atr'].shift(1)

    # Lagged BB width (if available)
    if 'bb_width' in df.columns:
        df['bb_width_lag_1'] = df['bb_width'].shift(1)

    # Volatility regime lag (if available)
    if 'volatility_regime' in df.columns:
        df['volatility_regime_lag'] = df['volatility_regime'].shift(1)

    return df