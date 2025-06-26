import pandas as pd
import numpy as np

def create_custom_volatility_features(df):
    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Realized volatility (rolling standard deviation of returns)
    df['realized_volatility'] = df['log_return'].rolling(window=10).std()

    # Parkinson volatility estimate
    df['parkinson_volatility'] = 0.5 * (np.log(df['high'] / df['low']) ** 2)

    # High-Low ratio
    df['high_low_ratio'] = (df['high'] / df['low']) - 1

    # Gap percentage
    df['prev_close'] = df['close'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']

    # Candle body size
    df['body_size'] = abs(df['close'] - df['open'])

    # Upper and lower wick percentage
    df['wick_upper_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-6)
    df['wick_lower_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-6)

    # Body-to-range ratio
    df['body_to_range_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)

    # Candle asymmetry score (difference between upper and lower wick pct)
    df['candle_asymmetry_score'] = abs(df['wick_upper_pct'] - df['wick_lower_pct'])

    # VWAP proxy using average price (since no volume data)
    df['vwap_proxy'] = (df['high'] + df['low'] + df['close']) / 3
    df['price_vs_vwap'] = df['close'] - df['vwap_proxy']

    # Volatility spike flag (high log return or high HL ratio)
    df['volatility_spike_flag'] = ((df['log_return'].abs() > df['log_return'].rolling(10).mean()) |
                                    (df['high_low_ratio'] > df['high_low_ratio'].rolling(10).mean())).astype(int)

    return df
