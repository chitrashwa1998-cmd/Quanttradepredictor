import pandas as pd
import numpy as np

# Assumes df has 'close', 'high', 'low', and datetime index

def add_volatility_lagged_features(df):
    """Add lagged features specifically for volatility prediction with data preservation."""
    df = df.copy()

    # Ensure we have the right column names
    close_col = 'Close' if 'Close' in df.columns else 'close'

    # Ensure we have realized volatility with smaller window for less data loss
    if 'realized_volatility' not in df.columns:
        returns = df[close_col].pct_change()
        df['realized_volatility'] = returns.rolling(5).std()  # Reduced from 10 to 5

    # Lagged volatility features with forward fill to preserve data
    df['lag_volatility_1'] = df['realized_volatility'].shift(1).ffill()
    df['lag_volatility_3'] = df['realized_volatility'].shift(3).ffill()
    df['lag_volatility_5'] = df['realized_volatility'].shift(5).ffill()

    # Lagged ATR if available with forward fill
    if 'atr' in df.columns:
        df['lag_atr_1'] = df['atr'].shift(1).ffill()
        df['lag_atr_3'] = df['atr'].shift(3).ffill()

    # Lagged Bollinger Band width if available with forward fill
    if 'bb_width' in df.columns:
        df['lag_bb_width'] = df['bb_width'].shift(1).ffill()

    # Volatility regime classification with smaller window
    if 'realized_volatility' in df.columns:
        vol_10 = df['realized_volatility'].rolling(10).mean()  # Reduced from 20 to 10
        vol_std = df['realized_volatility'].rolling(10).std()

        conditions = [
            df['realized_volatility'] < (vol_10 - 0.5 * vol_std),
            df['realized_volatility'] > (vol_10 + 0.5 * vol_std)
        ]
        choices = [0, 2]  # 0=low, 1=medium, 2=high
        df['volatility_regime'] = np.select(conditions, choices, default=1)

    return df