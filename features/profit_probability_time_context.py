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

    # Previous day range and return
    df['date'] = df['timestamp'].dt.date
    prev_day_high = df.groupby('date')[df.columns[df.columns.str.contains('igh', case=False)][0]].transform('max').shift(1) if any(df.columns.str.contains('igh', case=False)) else df.groupby('date')['High'].transform('max').shift(1)
    prev_day_low = df.groupby('date')[df.columns[df.columns.str.contains('ow', case=False)][0]].transform('min').shift(1) if any(df.columns.str.contains('ow', case=False)) else df.groupby('date')['Low'].transform('min').shift(1)
    prev_day_close = df.groupby('date')[close_col].transform('last').shift(1)
    prev_prev_day_close = prev_day_close.shift(1)

    df['prev_day_range'] = prev_day_high - prev_day_low
    df['prev_day_return'] = (prev_day_close - prev_prev_day_close) / prev_prev_day_close

    # Weekend gap detection (if current date - previous date > 1 day)
    df['prev_date'] = df['date'].shift(1)
    df['is_weekend_gap'] = (df['date'] - df['prev_date']).dt.days > 1
    df['is_weekend_gap'] = df['is_weekend_gap'].astype(int)

    # Number of consecutive green or red candles
    df['green_candle'] = (df['close'] > df['open']).astype(int)
    df['red_candle'] = (df['close'] < df['open']).astype(int)
    df['num_consecutive_green_red'] = df['green_candle'].groupby((df['green_candle'] != df['green_candle'].shift()).cumsum()).cumcount() + 1

    return df
