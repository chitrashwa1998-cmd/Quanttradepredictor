import pandas as pd
import numpy as np

def add_time_context_features_profit_prob(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Handle column name compatibility
    close_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    
    # Handle timestamp column
    timestamp_col = 'timestamp'
    if 'timestamp' not in df.columns and df.index.name == 'timestamp':
        df = df.reset_index()
    elif 'timestamp' not in df.columns:
        # If no timestamp column, create one from index
        df['timestamp'] = df.index
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Store timestamp for processing but will remove it later
    temp_timestamp = df['timestamp'].copy()

    # Basic time breakdown
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['minute_of_hour'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday = 0

    # Opening and closing range flags
    df['is_opening_range'] = df['timestamp'].dt.time.between(pd.to_datetime('09:15').time(), pd.to_datetime('10:00').time()).astype(int)
    df['is_closing_range'] = df['timestamp'].dt.time >= pd.to_datetime('15:00').time()
    df['is_closing_range'] = df['is_closing_range'].astype(int)

    # Session phase
    def session_phase(t):
        if t < pd.to_datetime('10:00').time():
            return 'open'
        elif t < pd.to_datetime('14:30').time():
            return 'mid'
        else:
            return 'close'

    df['session_phase'] = df['timestamp'].dt.time.apply(session_phase)

    # Overnight gap (requires previous day close)
    df['prev_close'] = df[close_col].shift(1)
    df['overnight_gap'] = df[open_col] - df['prev_close']

    # Simplified time features to avoid complex groupby operations
    try:
        # Basic price momentum features (much faster)
        df['price_change_1h'] = df[close_col] - df[close_col].shift(12)  # 1 hour ago (assuming 5min data)
        df['price_change_4h'] = df[close_col] - df[close_col].shift(48)  # 4 hours ago
        
        # Simple gap detection
        df['gap_from_prev'] = df[open_col] - df[close_col].shift(1)
        df['gap_pct'] = df['gap_from_prev'] / df[close_col].shift(1).replace(0, np.nan)
        df['is_large_gap'] = (abs(df['gap_pct']) > 0.005).astype(int)  # 0.5% gap threshold
        
    except Exception as e:
        print(f"Simplified time features error: {e}")
        # Fallback to basic features
        df['price_change_1h'] = 0.0
        df['price_change_4h'] = 0.0
        df['gap_from_prev'] = 0.0
        df['gap_pct'] = 0.0
        df['is_large_gap'] = 0

    # Simple candle direction features (much faster)
    df['green_candle'] = (df[close_col] > df[open_col]).astype(int)
    df['red_candle'] = (df[close_col] < df[open_col]).astype(int)
    
    # Simple rolling sum instead of complex groupby
    df['green_candles_last_5'] = df['green_candle'].rolling(5).sum()
    df['red_candles_last_5'] = df['red_candle'].rolling(5).sum()

    # Remove timestamp column to prevent it from being used as a feature
    if 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)

    return df
