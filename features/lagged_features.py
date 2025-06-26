import pandas as pd
import numpy as np

# Assumes df has 'close', 'high', 'low', and datetime index

def add_volatility_lagged_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()

    # Log return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Realized volatility (rolling std of log returns)
    df['realized_volatility'] = df['log_return'].rolling(window).std()

    # Lagged realized volatility
    df['lag_volatility_1'] = df['realized_volatility'].shift(1)
    df['lag_volatility_3'] = df['realized_volatility'].shift(3)
    df['lag_volatility_5'] = df['realized_volatility'].shift(5)

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window).mean()

    # Lagged ATR
    df['lag_atr_1'] = df['atr'].shift(1)
    df['lag_atr_3'] = df['atr'].shift(3)

    # Bollinger Band Width
    rolling_mean = df['close'].rolling(window).mean()
    rolling_std = df['close'].rolling(window).std()
    df['bb_upper'] = rolling_mean + 2 * rolling_std
    df['bb_lower'] = rolling_mean - 2 * rolling_std
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['lag_bb_width'] = df['bb_width'].shift(1)

    # Volatility regime (simple threshold-based)
    df['volatility_regime'] = pd.qcut(df['realized_volatility'], q=3, labels=["low", "medium", "high"])

    return df
