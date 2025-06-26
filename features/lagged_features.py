import pandas as pd
import numpy as np

def create_lagged_volatility_features(df):
    # Log returns (if not already present)
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Rolling volatility
    df['vol_rolling_10'] = df['log_return'].rolling(window=10).std()

    # Lagged volatility values
    df['vol_lag_1'] = df['vol_rolling_10'].shift(1)
    df['vol_lag_3'] = df['vol_rolling_10'].shift(3)
    df['vol_lag_5'] = df['vol_rolling_10'].shift(5)

    # Lagged ATR and BB width (assume they're calculated earlier in pipeline)
    if 'atr' in df.columns:
        df['atr_lag_1'] = df['atr'].shift(1)
        df['atr_lag_3'] = df['atr'].shift(3)
    if 'bb_width' in df.columns:
        df['bb_width_lag_1'] = df['bb_width'].shift(1)
        df['bb_width_lag_5'] = df['bb_width'].shift(5)

    # Lagged log returns (1 to 5 bars)
    for i in range(1, 6):
        df[f'log_return_lag_{i}'] = df['log_return'].shift(i)

    # Volatility delta (change from previous bar)
    df['vol_delta_1'] = df['vol_rolling_10'] - df['vol_lag_1']

    # Rolling slope of volatility (linear trend)
    df['vol_rolling_10_slope'] = df['vol_rolling_10'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if x.notna().all() else np.nan, raw=False
    )

    # Volatility regime lag (if defined elsewhere)
    if 'volatility_regime' in df.columns:
        df['volatility_regime_lag'] = df['volatility_regime'].shift(1)

    return df
