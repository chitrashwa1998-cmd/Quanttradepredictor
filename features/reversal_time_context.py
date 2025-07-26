import pandas as pd
import numpy as np

def add_time_context_features_reversal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Auto-detect column names
    close_col = None
    open_col = None
    high_col = None
    low_col = None
    
    for col in df.columns:
        if col.lower() == 'close':
            close_col = col
        elif col.lower() == 'open':
            open_col = col
        elif col.lower() == 'high':
            high_col = col
        elif col.lower() == 'low':
            low_col = col
    
    if not all([close_col, open_col, high_col, low_col]):
        missing = [name for name, col in [('Close', close_col), ('Open', open_col), ('High', high_col), ('Low', low_col)] if col is None]
        raise ValueError(f"Missing required OHLC columns: {missing}. Available columns: {list(df.columns)}")
    
    # Handle timestamp - check if it's in columns or use index
    if 'timestamp' in df.columns:
        timestamp_col = pd.to_datetime(df['timestamp'])
    else:
        # Use index as timestamp
        timestamp_col = pd.to_datetime(df.index)

    # Basic time breakdown
    df['hour_of_day'] = timestamp_col.hour
    df['minute_of_hour'] = timestamp_col.minute
    df['day_of_week'] = timestamp_col.dayofweek

    # Opening and closing ranges (simplified using hour)
    hour = timestamp_col.hour
    minute = timestamp_col.minute
    df['is_opening_range'] = ((hour == 9) & (minute >= 15) | (hour == 10) & (minute == 0)).astype(int)
    df['is_closing_range'] = (hour >= 15).astype(int)

    # Session phase (simplified numeric encoding)
    hour = timestamp_col.hour
    df['session_phase_numeric'] = np.where(hour < 10, 0,  # open
                                  np.where(hour < 14.5, 1,  # mid
                                          2))  # close

    # Previous day context
    df['date'] = timestamp_col.date
    df['prev_close'] = df[close_col].shift(1)
    df['overnight_gap'] = df[open_col] - df['prev_close']

    # Simplified time features to avoid groupby complexity
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_hour'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_hour'] / 60)

    # Consecutive green/red candles
    df['green_candle'] = (df[close_col] > df[open_col]).astype(int)
    df['red_candle'] = (df[close_col] < df[open_col]).astype(int)
    
    # Simple momentum features
    df['price_momentum_3'] = df[close_col].diff(3)
    df['price_momentum_5'] = df[close_col].diff(5)

    # Market volatility & heat context
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df[close_col] / df[close_col].shift(1))
    df['recent_volatility'] = df['log_return'].rolling(5).std()
    df['market_heat'] = df['log_return'].rolling(5).mean()

    return df
