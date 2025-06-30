import pandas as pd
import numpy as np

def add_custom_profit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Body and wick calculations
    df['body_size'] = abs(df['close'] - df['open'])
    df['wick_upper'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['wick_lower'] = df[['close', 'open']].min(axis=1) - df['low']
    df['wick_ratio'] = (df['wick_upper'] + df['wick_lower']) / df['body_size'].replace(0, np.nan)
    df['candle_strength'] = df['body_size'] / (df['high'] - df['low']).replace(0, np.nan)

    # Price vs EMA (assumes EMAs already computed)
    for period in [5, 10, 20]:
        if f'ema_{period}' in df.columns:
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        else:
            df[f'price_vs_ema_{period}'] = np.nan

    # Momentum acceleration
    df['momentum_3'] = df['close'] - df['close'].shift(3)
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_acceleration'] = df['momentum_3'] - df['momentum_5']

    # High-low ratio and distance from open
    df['high_low_ratio'] = (df['high'] / df['low']) - 1
    df['close_vs_open_distance'] = (df['close'] - df['open']) / df['open']

    # Trend consistency: % of last 5 candles that closed green
    df['green_candle'] = (df['close'] > df['open']).astype(int)
    df['trend_consistency_score'] = df['green_candle'].rolling(5).sum() / 5

    # Position in day's range
    df['bar_position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)

    # Rolling return stats
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['rolling_return_mean_5'] = df['log_return'].rolling(5).mean()
    df['rolling_return_std_5'] = df['log_return'].rolling(5).std()

    return df
