import pandas as pd
import numpy as np

def compute_custom_volatility_features(df):
    """Compute custom engineered features for volatility prediction."""
    df = df.copy()

    # Ensure we have the right column names
    close_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'

    # Log returns
    df['log_return'] = np.log(df[close_col] / df[close_col].shift(1))

    # Realized volatility (rolling standard deviation of log returns)
    df['realized_volatility'] = df['log_return'].rolling(window=10).std()

    # Parkinson volatility estimator
    df['parkinson_volatility'] = np.sqrt((1/(4*np.log(2))) * np.log(df[high_col]/df[low_col])**2)

    # High-Low ratio
    df['high_low_ratio'] = df[high_col] / df[low_col]

    # Gap percentage (current open vs previous close)
    df['gap_pct'] = (df[open_col] / df[close_col].shift(1) - 1) * 100

    # Price vs VWAP approximation
    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
    df['price_vs_vwap'] = df[close_col] / typical_price.rolling(20).mean() - 1

    # Volatility spike flag
    rolling_vol = df['realized_volatility'].rolling(20).mean()
    df['volatility_spike_flag'] = (df['realized_volatility'] > rolling_vol * 1.5).astype(int)

    # Candle body to range ratio
    body_size = abs(df[close_col] - df[open_col])
    candle_range = df[high_col] - df[low_col]

    # Avoid division by zero
    candle_range = candle_range.replace(0, np.nan)
    df['candle_body_ratio'] = body_size / candle_range

    # Replace infinities with NaN
    df['candle_body_ratio'] = df['candle_body_ratio'].replace([np.inf, -np.inf], np.nan)

    # Keep candle_body_ratio as part of the 27 features
    # Remove any legacy candle_asymmetry_ratio if it exists
    if 'candle_asymmetry_ratio' in df.columns:
        df = df.drop(columns=['candle_asymmetry_ratio'])

    return df