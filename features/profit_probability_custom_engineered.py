
import pandas as pd
import numpy as np

def add_custom_profit_probability_features(df):
    """Add custom engineered features specifically for profit probability prediction."""
    df = df.copy()

    # Ensure we have the right column names
    close_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    volume_col = 'Volume' if 'Volume' in df.columns else 'volume'

    # MACD momentum features (if MACD exists)
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_momentum'] = df['macd'].diff()
        df['macd_signal_momentum'] = df['macd_signal'].diff()
        df['macd_crossover_bullish'] = ((df['macd'] > df['macd_signal']) & 
                                       (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_crossover_bearish'] = ((df['macd'] < df['macd_signal']) & 
                                       (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

    # RSI divergence and momentum
    if 'rsi' in df.columns:
        df['rsi_momentum'] = df['rsi'].diff()
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)

    # Bollinger Band squeeze and expansion
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8).astype(int)
        df['bb_expansion'] = (df['bb_width'] > df['bb_width'].rolling(20).mean() * 1.2).astype(int)
        
        # Bollinger Band breakout signals
        df['bb_upper_breakout'] = (df[close_col] > df['bb_upper']).astype(int)
        df['bb_lower_breakout'] = (df[close_col] < df['bb_lower']).astype(int)

    # Stochastic momentum and signals
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        df['stoch_momentum'] = df['stoch_k'].diff()
        df['stoch_crossover_bullish'] = ((df['stoch_k'] > df['stoch_d']) & 
                                        (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
        df['stoch_crossover_bearish'] = ((df['stoch_k'] < df['stoch_d']) & 
                                        (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))).astype(int)

    # Price action patterns
    df['doji'] = (abs(df[close_col] - df[open_col]) / (df[high_col] - df[low_col]) < 0.1).astype(int)
    df['hammer'] = ((df[low_col] < df[open_col]) & (df[low_col] < df[close_col]) & 
                   ((df[high_col] - np.maximum(df[open_col], df[close_col])) < 
                    (np.maximum(df[open_col], df[close_col]) - df[low_col]) * 0.3)).astype(int)

    # Volume-price relationship (if volume available)
    if volume_col in df.columns:
        df['volume_price_trend'] = (df[close_col].diff() * df[volume_col]).rolling(5).mean()
        df['volume_momentum'] = df[volume_col].pct_change(5)
    else:
        # Proxy volume indicators using price/range
        df['volume_price_trend'] = df[close_col].diff() * (df[high_col] - df[low_col])
        df['volume_momentum'] = (df[high_col] - df[low_col]).pct_change(5)

    # Multi-timeframe momentum
    df['momentum_3'] = df[close_col].pct_change(3)
    df['momentum_8'] = df[close_col].pct_change(8)
    df['momentum_13'] = df[close_col].pct_change(13)

    # Volatility-adjusted momentum
    if 'atr' in df.columns:
        df['volatility_adjusted_momentum'] = df[close_col].diff() / df['atr']
    else:
        typical_range = (df[high_col] - df[low_col]).rolling(14).mean()
        df['volatility_adjusted_momentum'] = df[close_col].diff() / typical_range

    # Support/Resistance levels
    df['resistance_5'] = df[high_col].rolling(5).max()
    df['support_5'] = df[low_col].rolling(5).min()
    df['near_resistance'] = (df[close_col] >= df['resistance_5'] * 0.99).astype(int)
    df['near_support'] = (df[close_col] <= df['support_5'] * 1.01).astype(int)

    # Trend strength
    if 'ema_fast' in df.columns and 'ema_slow' in df.columns:
        df['trend_strength'] = abs(df['ema_fast'] - df['ema_slow']) / df[close_col]
        df['trend_acceleration'] = (df['ema_fast'] - df['ema_slow']).diff()

    # Market regime indicators
    returns_5 = df[close_col].pct_change(5)
    df['return_volatility_5'] = returns_5.rolling(10).std()
    df['return_skew_5'] = returns_5.rolling(10).skew()

    # Price gap analysis
    df['gap_up'] = ((df[open_col] > df[close_col].shift(1)) & 
                   ((df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1) > 0.001)).astype(int)
    df['gap_down'] = ((df[open_col] < df[close_col].shift(1)) & 
                     ((df[close_col].shift(1) - df[open_col]) / df[close_col].shift(1) > 0.001)).astype(int)

    # Replace inf and nan values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
