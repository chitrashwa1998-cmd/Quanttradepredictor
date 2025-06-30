import pandas as pd
import numpy as np

def add_custom_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Body and wick features
    df['body_size'] = abs(df['close'] - df['open'])
    df['wick_upper'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['wick_lower'] = df[['close', 'open']].min(axis=1) - df['low']
    df['wick_ratio'] = (df['wick_upper'] + df['wick_lower']) / df['body_size'].replace(0, np.nan)
    df['candle_strength'] = df['body_size'] / (df['high'] - df['low']).replace(0, np.nan)

    # Candle pattern flags
    df['engulfing_pattern'] = ((df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))).astype(int)
    df['doji_candle'] = (df['body_size'] / (df['high'] - df['low']).replace(0, np.nan) < 0.1).astype(int)
    df['hammer_pattern'] = ((df['wick_lower'] > 2 * df['body_size']) & (df['wick_upper'] < df['body_size'])).astype(int)
    df['shooting_star_pattern'] = ((df['wick_upper'] > 2 * df['body_size']) & (df['wick_lower'] < df['body_size'])).astype(int)

    # Price vs EMA features
    for period in [5, 20]:
        if f'ema_{period}' in df.columns:
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        else:
            df[f'price_vs_ema_{period}'] = np.nan

    # Momentum and trend shift features
    df['momentum_shift'] = df['close'].diff(1) - df['close'].diff(2)
    df['trend_strength'] = df['body_size'].rolling(3).mean()
    df['trend_strength_drop'] = df['trend_strength'] - df['trend_strength'].shift(1)

    # Bar position in range
    df['bar_position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)

    # Range contraction
    df['range_contraction_3'] = (df['high'] - df['low']).rolling(3).std()

    return df
