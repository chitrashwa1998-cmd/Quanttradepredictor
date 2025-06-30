
import pandas as pd
import numpy as np

def add_custom_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Handle column name compatibility
    close_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'

    # Candle pattern features for reversal detection
    df['body_size'] = abs(df[close_col] - df[open_col])
    df['upper_wick'] = df[high_col] - df[[close_col, open_col]].max(axis=1)
    df['lower_wick'] = df[[close_col, open_col]].min(axis=1) - df[low_col]
    df['total_wick'] = df['upper_wick'] + df['lower_wick']
    
    # Wick to body ratios (key reversal patterns)
    df['upper_wick_ratio'] = df['upper_wick'] / df['body_size'].replace(0, np.nan)
    df['lower_wick_ratio'] = df['lower_wick'] / df['body_size'].replace(0, np.nan)
    df['wick_balance'] = (df['upper_wick'] - df['lower_wick']) / (df['upper_wick'] + df['lower_wick']).replace(0, np.nan)

    # Doji and reversal candle patterns
    candle_range = df[high_col] - df[low_col]
    df['doji_score'] = 1 - (df['body_size'] / candle_range.replace(0, np.nan))
    df['hammer_score'] = (df['lower_wick'] > df['body_size'] * 2).astype(int) * (df['upper_wick'] < df['body_size'] * 0.5).astype(int)
    df['shooting_star_score'] = (df['upper_wick'] > df['body_size'] * 2).astype(int) * (df['lower_wick'] < df['body_size'] * 0.5).astype(int)

    # Price position in candle range
    df['close_position_in_range'] = (df[close_col] - df[low_col]) / candle_range.replace(0, np.nan)
    df['open_position_in_range'] = (df[open_col] - df[low_col]) / candle_range.replace(0, np.nan)

    # Momentum divergence features
    price_change_3 = df[close_col].pct_change(3)
    price_change_5 = df[close_col].pct_change(5)
    price_change_10 = df[close_col].pct_change(10)
    
    df['momentum_3'] = price_change_3
    df['momentum_5'] = price_change_5
    df['momentum_10'] = price_change_10
    df['momentum_acceleration'] = price_change_3 - price_change_5
    df['momentum_deceleration'] = price_change_5 - price_change_10

    # Price extremes detection
    df['high_5'] = df[high_col].rolling(5).max()
    df['low_5'] = df[low_col].rolling(5).min()
    df['high_10'] = df[high_col].rolling(10).max()
    df['low_10'] = df[low_col].rolling(10).min()
    df['high_20'] = df[high_col].rolling(20).max()
    df['low_20'] = df[low_col].rolling(20).min()

    # Position relative to recent extremes
    df['position_in_5_range'] = (df[close_col] - df['low_5']) / (df['high_5'] - df['low_5']).replace(0, np.nan)
    df['position_in_10_range'] = (df[close_col] - df['low_10']) / (df['high_10'] - df['low_10']).replace(0, np.nan)
    df['position_in_20_range'] = (df[close_col] - df['low_20']) / (df['high_20'] - df['low_20']).replace(0, np.nan)

    # Near extremes flags (potential reversal zones)
    df['near_5_high'] = (df[close_col] >= df['high_5'] * 0.95).astype(int)
    df['near_5_low'] = (df[close_col] <= df['low_5'] * 1.05).astype(int)
    df['near_10_high'] = (df[close_col] >= df['high_10'] * 0.98).astype(int)
    df['near_10_low'] = (df[close_col] <= df['low_10'] * 1.02).astype(int)

    # Volatility expansion (often precedes reversals)
    returns = df[close_col].pct_change()
    df['volatility_5'] = returns.rolling(5).std()
    df['volatility_10'] = returns.rolling(10).std()
    df['volatility_20'] = returns.rolling(20).std()
    
    df['volatility_expansion_5'] = (df['volatility_5'] / df['volatility_20'].replace(0, np.nan)) > 1.5
    df['volatility_expansion_10'] = (df['volatility_10'] / df['volatility_20'].replace(0, np.nan)) > 1.2

    # Gap analysis (overnight reversals)
    df['overnight_gap'] = (df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1)
    df['gap_size'] = abs(df['overnight_gap'])
    df['gap_up'] = (df['overnight_gap'] > 0.002).astype(int)  # 0.2% gap up
    df['gap_down'] = (df['overnight_gap'] < -0.002).astype(int)  # 0.2% gap down

    # Price vs moving averages convergence/divergence
    if 'sma_5' in df.columns and 'sma_20' in df.columns:
        df['sma_convergence'] = abs(df['sma_5'] - df['sma_20']) / df['sma_20']
        df['price_ma_divergence'] = abs(df[close_col] - df['sma_5']) / df['sma_5']
    else:
        # Calculate simple moving averages if not present
        df['sma_5'] = df[close_col].rolling(5).mean()
        df['sma_20'] = df[close_col].rolling(20).mean()
        df['sma_convergence'] = abs(df['sma_5'] - df['sma_20']) / df['sma_20']
        df['price_ma_divergence'] = abs(df[close_col] - df['sma_5']) / df['sma_5']

    # Volume-price relationship (if volume available)
    if 'Volume' in df.columns or 'volume' in df.columns:
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
        df['volume_ma_10'] = df[volume_col].rolling(10).mean()
        df['volume_spike'] = (df[volume_col] > df['volume_ma_10'] * 1.5).astype(int)
        df['price_volume_divergence'] = (
            (price_change_3 > 0) & (df[volume_col] < df['volume_ma_10'])
        ).astype(int) + (
            (price_change_3 < 0) & (df[volume_col] < df['volume_ma_10'])
        ).astype(int)
    else:
        df['volume_spike'] = 0
        df['price_volume_divergence'] = 0

    # Trend exhaustion signals
    df['consecutive_green'] = (df[close_col] > df[open_col]).astype(int)
    df['consecutive_red'] = (df[close_col] < df[open_col]).astype(int)
    
    # Count consecutive candles
    df['green_streak'] = df['consecutive_green'].groupby((df['consecutive_green'] != df['consecutive_green'].shift()).cumsum()).cumsum()
    df['red_streak'] = df['consecutive_red'].groupby((df['consecutive_red'] != df['consecutive_red'].shift()).cumsum()).cumsum()
    
    # Trend exhaustion flags
    df['green_exhaustion'] = (df['green_streak'] >= 5).astype(int)
    df['red_exhaustion'] = (df['red_streak'] >= 5).astype(int)

    # Support/Resistance levels
    df['pivot_high'] = (
        (df[high_col] > df[high_col].shift(1)) & 
        (df[high_col] > df[high_col].shift(-1))
    ).astype(int)
    df['pivot_low'] = (
        (df[low_col] < df[low_col].shift(1)) & 
        (df[low_col] < df[low_col].shift(-1))
    ).astype(int)

    # Clean infinite and NaN values
    for col in df.columns:
        if col not in [close_col, open_col, high_col, low_col]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
