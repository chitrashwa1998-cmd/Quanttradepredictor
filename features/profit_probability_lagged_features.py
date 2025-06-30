import pandas as pd
import numpy as np

def add_lagged_features_profit_prob(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log return (if not present)
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Lagged price and return features
    df['lag_close_1'] = df['close'].shift(1)
    df['lag_close_3'] = df['close'].shift(3)
    df['lag_close_5'] = df['close'].shift(5)

    df['lag_return_1'] = df['log_return'].shift(1)
    df['lag_return_3'] = df['log_return'].shift(3)

    # Lagged body and momentum features
    df['body_size'] = abs(df['close'] - df['open'])
    df['candle_strength'] = df['body_size'] / (df['high'] - df['low']).replace(0, np.nan)
    df['momentum_3'] = df['close'] - df['close'].shift(3)
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_acceleration'] = df['momentum_3'] - df['momentum_5']

    df['lag_body_size_1'] = df['body_size'].shift(1)
    df['lag_candle_strength_1'] = df['candle_strength'].shift(1)
    df['lag_momentum_acceleration'] = df['momentum_acceleration'].shift(1)

    # Trend consistency
    df['green_candle'] = (df['close'] > df['open']).astype(int)
    df['trend_consistency_score'] = df['green_candle'].rolling(5).sum() / 5
    df['lag_trend_consistency_score'] = df['trend_consistency_score'].shift(1)

    # Price vs EMA 5 (if present)
    if 'ema_5' in df.columns:
        df['price_vs_ema_5'] = (df['close'] - df['ema_5']) / df['ema_5']
        df['lag_price_vs_ema_5'] = df['price_vs_ema_5'].shift(1)

    # Bar position in range (if high != low)
    df['bar_position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['lag_bb_position'] = df['bar_position_in_range'].shift(1)

    # Rolling high/low and return stats
    df['rolling_max_high_10'] = df['high'].rolling(10).max()
    df['rolling_min_low_10'] = df['low'].rolling(10).min()
    df['rolling_body_mean_3'] = df['body_size'].rolling(3).mean()
    df['rolling_return_skew_5'] = df['log_return'].rolling(5).skew()

    return df
