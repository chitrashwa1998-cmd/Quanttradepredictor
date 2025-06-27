import pandas as pd
import numpy as np

def add_lagged_direction_features(df: pd.DataFrame, ema=None) -> pd.DataFrame:
    df = df.copy()
    
    # Determine column names (handle both uppercase and lowercase)
    close_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'

    # Lagged close prices
    df['lag_close_1'] = df[close_col].shift(1)
    df['lag_close_3'] = df[close_col].shift(3)
    df['lag_close_5'] = df[close_col].shift(5)

    # Lagged returns
    df['return_1'] = df[close_col].pct_change(1)
    df['log_return_1'] = np.log(df[close_col] / df[close_col].shift(1))

    # Lagged body size
    df['body_size'] = abs(df[close_col] - df[open_col])
    df['lag_body_size_1'] = df['body_size'].shift(1)

    # Candle direction
    df['candle_direction'] = (df[close_col] > df[open_col]).astype(int)
    df['lag_candle_direction_1'] = df['candle_direction'].shift(1)

    # Rolling return stats
    df['rolling_return_mean_5'] = df['log_return_1'].rolling(5).mean()
    df['rolling_return_std_5'] = df['log_return_1'].rolling(5).std()

    # Rolling high/low
    df['rolling_high_10'] = df[high_col].rolling(10).max()
    df['rolling_low_10'] = df[low_col].rolling(10).min()

    # EMA diff lagged - use EMA_20 if available, otherwise create simple moving average
    if 'ema_20' in df.columns:
        df['ema_diff_lagged_1'] = (df[close_col] - df['ema_20']).shift(1)
    elif ema is not None:
        df['ema_diff_lagged_1'] = (df[close_col] - ema).shift(1)
    else:
        # Create a simple 20-period moving average as fallback
        simple_ma_20 = df[close_col].rolling(20).mean()
        df['ema_diff_lagged_1'] = (df[close_col] - simple_ma_20).shift(1)

    # Candle pattern shifted — placeholder (you can plug in real pattern detector)
    df['candle_pattern_shifted'] = 0  # set actual logic later if needed

    return df
