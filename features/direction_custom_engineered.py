
import pandas as pd
import numpy as np

def compute_direction_custom_features(df):
    """Compute custom engineered features specifically for direction prediction."""
    df = df.copy()

    # Ensure we have the right column names
    close_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'

    # Price momentum features
    df['price_momentum_1'] = df[close_col] / df[close_col].shift(1) - 1
    df['price_momentum_3'] = df[close_col] / df[close_col].shift(3) - 1
    df['price_momentum_5'] = df[close_col] / df[close_col].shift(5) - 1

    # Moving average crossovers
    sma_10 = df[close_col].rolling(10).mean()
    sma_20 = df[close_col].rolling(20).mean()
    df['sma_crossover'] = (sma_10 > sma_20).astype(int)
    df['price_above_sma20'] = (df[close_col] > sma_20).astype(int)

    # EMA crossovers (using pre-calculated EMAs if available)
    if 'ema_fast' in df.columns and 'ema_slow' in df.columns:
        df['ema_crossover'] = (df['ema_fast'] > df['ema_slow']).astype(int)
        df['ema_divergence'] = df['ema_fast'] - df['ema_slow']
    else:
        ema_12 = df[close_col].ewm(span=12).mean()
        ema_26 = df[close_col].ewm(span=26).mean()
        df['ema_crossover'] = (ema_12 > ema_26).astype(int)
        df['ema_divergence'] = ema_12 - ema_26

    # Candlestick patterns
    body_size = abs(df[close_col] - df[open_col])
    candle_range = df[high_col] - df[low_col]
    
    # Avoid division by zero
    candle_range = candle_range.replace(0, np.nan)
    df['candle_body_pct'] = body_size / candle_range
    
    # Bullish/Bearish candle
    df['bullish_candle'] = (df[close_col] > df[open_col]).astype(int)
    
    # Doji pattern (small body relative to range)
    df['doji_pattern'] = (df['candle_body_pct'] < 0.1).astype(int)

    # Gap analysis
    df['gap_up'] = (df[open_col] > df[close_col].shift(1)).astype(int)
    df['gap_down'] = (df[open_col] < df[close_col].shift(1)).astype(int)

    # Support and resistance levels
    df['near_high_20'] = (df[close_col] >= df[high_col].rolling(20).max() * 0.98).astype(int)
    df['near_low_20'] = (df[close_col] <= df[low_col].rolling(20).min() * 1.02).astype(int)

    # Volume-price analysis (if volume available)
    volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
    if volume_col in df.columns:
        avg_volume = df[volume_col].rolling(20).mean()
        df['high_volume'] = (df[volume_col] > avg_volume * 1.5).astype(int)
        df['volume_price_trend'] = df['price_momentum_1'] * df['high_volume']
    else:
        df['high_volume'] = 0
        df['volume_price_trend'] = 0

    # Price volatility regime
    returns = df[close_col].pct_change()
    rolling_vol = returns.rolling(10).std()
    vol_median = rolling_vol.rolling(50).median()
    df['high_volatility_regime'] = (rolling_vol > vol_median * 1.5).astype(int)

    # Trend strength
    df['uptrend_strength'] = df['price_momentum_5'] * df['ema_crossover']
    df['downtrend_strength'] = -df['price_momentum_5'] * (1 - df['ema_crossover'])

    # Replace infinities with NaN
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
