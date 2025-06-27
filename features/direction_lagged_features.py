import pandas as pd
import numpy as np

def add_lagged_direction_features(df: pd.DataFrame, ema: pd.Series = None) -> pd.DataFrame:
    df = df.copy()

    # Lagged close prices
    df['lag_close_1'] = df['close'].shift(1)
    df['lag_close_3'] = df['close'].shift(3)
    df['lag_close_5'] = df['close'].shift(5)

    # Lagged returns
    df['return_1'] = df['close'].pct_change(1)
    df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))

    # Lagged body size
    df['body_size'] = abs(df['close'] - df['open'])
    df['lag_body_size_1'] = df['body_size'].shift(1)

    # Candle direction
    df['candle_direction'] = (df['close'] > df['open']).astype(int)
    df['lag_candle_direction_1'] = df['candle_direction'].shift(1)

    # Rolling return stats
    df['rolling_return_mean_5'] = df['log_return_1'].rolling(5).mean()
    df['rolling_return_std_5'] = df['log_return_1'].rolling(5).std()

    # Rolling high/low
    df['rolling_high_10'] = df['high'].rolling(10).max()
    df['rolling_low_10'] = df['low'].rolling(10).min()

    # EMA diff lagged (optional: pass in EMA externally)
    if ema is not None:
        df['ema_diff_lagged_1'] = (df['close'] - ema).shift(1)
    else:
        df['ema_diff_lagged_1'] = np.nan  # placeholder if no EMA provided

    # Candle pattern shifted — placeholder (you can plug in real pattern detector)
    df['candle_pattern_shifted'] = 0  # set actual logic later if needed

    return df
