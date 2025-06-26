import pandas as pd
import numpy as np

def create_lagged_volatility_features(df):
    """Create comprehensive lagged volatility features for volatility prediction model."""
    df = df.copy()

    # Ensure column names are lowercase for consistency
    column_mapping = {}
    for col in df.columns:
        if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
            column_mapping[col] = col.lower()

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Use Close if close doesn't exist
    if 'close' not in df.columns and 'Close' in df.columns:
        df['close'] = df['Close']

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_rolling_10'] = df['log_return'].rolling(window=10).std()

    # Lagged volatility
    df['vol_lag_1'] = df['vol_rolling_10'].shift(1)
    df['vol_lag_3'] = df['vol_rolling_10'].shift(3)
    df['vol_lag_5'] = df['vol_rolling_10'].shift(5)

    # Lagged ATR and BB width (assume already calculated elsewhere)
    if 'atr' in df.columns:
        df['atr_lag_1'] = df['atr'].shift(1)
        df['atr_lag_3'] = df['atr'].shift(3)
    if 'bb_width' in df.columns:
        df['bb_width_lag_1'] = df['bb_width'].shift(1)
        df['bb_width_lag_5'] = df['bb_width'].shift(5)

    # Lagged log returns
    for i in range(1, 6):
        df[f'log_return_lag_{i}'] = df['log_return'].shift(i)

    # Volatility delta and slope
    df['vol_delta_1'] = df['vol_rolling_10'] - df['vol_lag_1']
    df['vol_rolling_10_slope'] = (
        df['vol_rolling_10'].rolling(window=5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True
        )
    )

    # Lagged regime (if created elsewhere)
    if 'volatility_regime' in df.columns:
        df['volatility_regime_lag'] = df['volatility_regime'].shift(1)

    return df