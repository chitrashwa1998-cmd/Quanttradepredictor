import pandas as pd
import numpy as np

# Assumes df has 'close', 'high', 'low', and datetime index

def add_volatility_lagged_features(df):
    """Add lagged features specifically for volatility prediction."""
    df = df.copy()

    # Ensure we have the right column names
    close_col = 'Close' if 'Close' in df.columns else 'close'

    # Ensure we have realized volatility
    if 'realized_volatility' not in df.columns:
        returns = df[close_col].pct_change()
        df['realized_volatility'] = returns.rolling(10).std()

    # Lagged volatility features
    df['lag_volatility_1'] = df['realized_volatility'].shift(1)
    df['lag_volatility_3'] = df['realized_volatility'].shift(3)
    df['lag_volatility_5'] = df['realized_volatility'].shift(5)

    # Lagged ATR if available
    if 'atr' in df.columns:
        df['lag_atr_1'] = df['atr'].shift(1)
        df['lag_atr_3'] = df['atr'].shift(3)

    # Lagged Bollinger Band width if available
    if 'bb_width' in df.columns:
        df['lag_bb_width'] = df['bb_width'].shift(1)

    # Volatility regime classification (5 regimes)
    if 'realized_volatility' in df.columns:
        vol_20 = df['realized_volatility'].rolling(20).mean()
        vol_std = df['realized_volatility'].rolling(20).std()

        conditions = [
            df['realized_volatility'] < (vol_20 - 1.5 * vol_std),  # Very Low
            df['realized_volatility'] < (vol_20 - 0.5 * vol_std),  # Low
            df['realized_volatility'] > (vol_20 + 1.5 * vol_std),  # Very High
            df['realized_volatility'] > (vol_20 + 0.5 * vol_std)   # High
        ]
        choices = [0, 1, 4, 3]  # 0=very low, 1=low, 2=medium, 3=high, 4=very high
        df['volatility_regime'] = np.select(conditions, choices, default=2)

    return df