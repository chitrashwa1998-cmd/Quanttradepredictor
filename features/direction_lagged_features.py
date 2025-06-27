
import pandas as pd
import numpy as np

def add_direction_lagged_features(df):
    """Add lagged features specifically for direction prediction."""
    df = df.copy()

    # Ensure we have the right column names
    close_col = 'Close' if 'Close' in df.columns else 'close'

    # Lagged RSI features
    if 'rsi' in df.columns:
        df['lag_rsi_1'] = df['rsi'].shift(1)
        df['lag_rsi_3'] = df['rsi'].shift(3)
        df['lag_rsi_5'] = df['rsi'].shift(5)
        
        # RSI momentum
        df['rsi_momentum'] = df['rsi'] - df['rsi'].shift(3)

    # Lagged MACD features
    if 'macd' in df.columns:
        df['lag_macd_1'] = df['macd'].shift(1)
        df['lag_macd_3'] = df['macd'].shift(3)
        
        # MACD momentum
        df['macd_momentum'] = df['macd'] - df['macd'].shift(2)

    # Lagged price momentum
    if 'price_momentum_1' in df.columns:
        df['lag_momentum_1'] = df['price_momentum_1'].shift(1)
        df['lag_momentum_3'] = df['price_momentum_1'].shift(3)

    # Lagged moving average signals
    if 'ema_crossover' in df.columns:
        df['lag_ema_crossover_1'] = df['ema_crossover'].shift(1)
        df['lag_ema_crossover_3'] = df['ema_crossover'].shift(3)

    # Lagged Stochastic
    if 'stoch_k' in df.columns:
        df['lag_stoch_k_1'] = df['stoch_k'].shift(1)
        df['lag_stoch_d_1'] = df['stoch_d'].shift(1) if 'stoch_d' in df.columns else 0

    # Direction persistence features
    returns = df[close_col].pct_change()
    direction = (returns > 0).astype(int)
    
    # Count consecutive up/down days
    df['consecutive_up'] = direction.groupby((direction != direction.shift()).cumsum()).cumsum()
    df['consecutive_down'] = (1 - direction).groupby(((1 - direction) != (1 - direction).shift()).cumsum()).cumsum()
    
    # Direction change frequency
    direction_changes = (direction != direction.shift()).astype(int)
    df['direction_changes_5d'] = direction_changes.rolling(5).sum()
    df['direction_changes_10d'] = direction_changes.rolling(10).sum()

    # Trend consistency
    if 'ema_crossover' in df.columns:
        df['trend_consistency_5d'] = df['ema_crossover'].rolling(5).mean()
        df['trend_consistency_10d'] = df['ema_crossover'].rolling(10).mean()

    # Momentum regime classification
    if 'price_momentum_5' in df.columns:
        momentum_20 = df['price_momentum_5'].rolling(20).mean()
        momentum_std = df['price_momentum_5'].rolling(20).std()

        conditions = [
            df['price_momentum_5'] < (momentum_20 - 0.5 * momentum_std),
            df['price_momentum_5'] > (momentum_20 + 0.5 * momentum_std)
        ]
        choices = [0, 2]  # 0=bearish, 1=neutral, 2=bullish
        df['momentum_regime'] = np.select(conditions, choices, default=1)

    return df
