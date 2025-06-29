import pandas as pd
import numpy as np

def add_custom_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Determine column names (handle both uppercase and lowercase)
    close_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'

    # Candle structure
    df['body_size'] = abs(df[close_col] - df[open_col])
    df['wick_upper'] = df[high_col] - df[[close_col, open_col]].max(axis=1)
    df['wick_lower'] = df[[close_col, open_col]].min(axis=1) - df[low_col]
    df['wick_ratio'] = (df['wick_upper'] + df['wick_lower']) / df['body_size'].replace(0, np.nan)

    # Candle direction
    df['candle_direction'] = (df[close_col] > df[open_col]).astype(int)
    df['previous_direction'] = df['candle_direction'].shift(1)

    # Price momentum
    df['momentum_1'] = df[close_col] - df[close_col].shift(1)
    df['momentum_3'] = df[close_col] - df[close_col].shift(3)
    df['momentum_5'] = df[close_col] - df[close_col].shift(5)
    df['momentum_acceleration'] = df['momentum_3'] - df['momentum_5']

    # High-low ratio (raw volatility)
    df['high_low_ratio'] = (df[high_col] / df[low_col]) - 1

    # Price acceleration (second derivative)
    df['price_acceleration'] = df[close_col] - 2 * df[close_col].shift(1) + df[close_col].shift(2)

    # Approx VWAP (typical price)
    df['vwap_approx'] = (df[high_col] + df[low_col] + df[close_col]) / 3
    df['price_vs_vwap'] = (df[close_col] - df['vwap_approx']) / df['vwap_approx']

    # OBV approximation (direction-based pressure)
    df['obv_proxy'] = ((df[close_col] > df[close_col].shift(1)).astype(int)
                       - (df[close_col] < df[close_col].shift(1)).astype(int)).cumsum()

    # Volume spike approximation (range spike)
    df['range'] = df[high_col] - df[low_col]
    df['vol_spike_proxy'] = df['range'] > df['range'].rolling(10).mean() + 2 * df['range'].rolling(10).std()

    # MFI proxy
    df['typical_price'] = df['vwap_approx']
    df['tp_change'] = df['typical_price'] - df['typical_price'].shift(1)
    df['mfi_direction'] = ((df['tp_change'] > 0).astype(int)
                           - (df['tp_change'] < 0).astype(int)).rolling(10).sum()

    return df
