import pandas as pd
import numpy as np

def add_lagged_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log return if not present
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Lagged prices and returns
    df['lag_close_1'] = df['close'].shift(1)
    df['lag_close_3'] = df['close'].shift(3)
    df['lag_close_5'] = df['close'].shift(5)
    df['lag_return_1'] = df['log_return'].shift(1)
    df['lag_return_3'] = df['log_return'].shift(3)

    # Candle structure dependencies
    df['body_size'] = abs(df['close'] - df['open'])
    df['candle_strength'] = df['body_size'] / (df['high'] - df['low']).replace(0, np.nan)
    df['momentum_shift'] = df['close'].diff(1) - df['close'].diff(2)
    df['bar_position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['wick_upper'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['wick_lower'] = df[['close', 'open']].min(axis=1) - df['low']
    df['wick_ratio'] = (df['wick_upper'] + df['wick_lower']) / df['body_size'].replace(0, np.nan)

    # Lagged structure features
    df['lag_body_size_1'] = df['body_size'].shift(1)
    df['lag_body_size_3'] = df['body_size'].shift(3)
    df['lag_candle_strength_1'] = df['candle_strength'].shift(1)
    df['lag_momentum_shift'] = df['momentum_shift'].shift(1)
    df['lag_bar_position_in_range'] = df['bar_position_in_range'].shift(1)

    # EMA comparison lags if available
    for period in [5, 20]:
        if f'ema_{period}' in df.columns:
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
            df[f'lag_price_vs_ema_{period}'] = df[f'price_vs_ema_{period}'].shift(1)
        else:
            df[f'lag_price_vs_ema_{period}'] = np.nan

    # Rolling window stats
    df['rolling_high_10'] = df['high'].rolling(10).max()
    df['rolling_low_10'] = df['low'].rolling(10).min()
    df['rolling_std_body_5'] = df['body_size'].rolling(5).std()
    df['rolling_std_return_5'] = df['log_return'].rolling(5).std()
    df['rolling_max_wick_ratio_3'] = df['wick_ratio'].rolling(3).max()

    return df
