import pandas as pd
import numpy as np

def add_custom_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Auto-detect column names
    close_col = None
    open_col = None
    high_col = None
    low_col = None
    
    for col in df.columns:
        if col.lower() == 'close':
            close_col = col
        elif col.lower() == 'open':
            open_col = col
        elif col.lower() == 'high':
            high_col = col
        elif col.lower() == 'low':
            low_col = col
    
    if not all([close_col, open_col, high_col, low_col]):
        missing = [name for name, col in [('Close', close_col), ('Open', open_col), ('High', high_col), ('Low', low_col)] if col is None]
        raise ValueError(f"Missing required OHLC columns: {missing}. Available columns: {list(df.columns)}")

    # Body and wick features
    df['body_size'] = abs(df[close_col] - df[open_col])
    df['wick_upper'] = df[high_col] - df[[close_col, open_col]].max(axis=1)
    df['wick_lower'] = df[[close_col, open_col]].min(axis=1) - df[low_col]
    df['wick_ratio'] = (df['wick_upper'] + df['wick_lower']) / df['body_size'].replace(0, np.nan)
    df['candle_strength'] = df['body_size'] / (df[high_col] - df[low_col]).replace(0, np.nan)

    # Candle pattern flags
    df['engulfing_pattern'] = ((df[close_col] > df[open_col].shift(1)) & (df[open_col] < df[close_col].shift(1))).astype(int)
    df['doji_candle'] = (df['body_size'] / (df[high_col] - df[low_col]).replace(0, np.nan) < 0.1).astype(int)
    df['hammer_pattern'] = ((df['wick_lower'] > 2 * df['body_size']) & (df['wick_upper'] < df['body_size'])).astype(int)
    df['shooting_star_pattern'] = ((df['wick_upper'] > 2 * df['body_size']) & (df['wick_lower'] < df['body_size'])).astype(int)

    # Price vs EMA features
    for period in [5, 20]:
        if f'ema_{period}' in df.columns:
            df[f'price_vs_ema_{period}'] = (df[close_col] - df[f'ema_{period}']) / df[f'ema_{period}']
        else:
            df[f'price_vs_ema_{period}'] = 0.0  # Use neutral value instead of NaN

    # Momentum and trend shift features
    df['momentum_shift'] = df[close_col].diff(1) - df[close_col].diff(2)
    df['trend_strength'] = df['body_size'].rolling(3).mean()
    df['trend_strength_drop'] = df['trend_strength'] - df['trend_strength'].shift(1)

    # Bar position in range
    df['bar_position_in_range'] = (df[close_col] - df[low_col]) / (df[high_col] - df[low_col]).replace(0, np.nan)

    # Range contraction
    df['range_contraction_3'] = (df[high_col] - df[low_col]).rolling(3).std()

    return df
