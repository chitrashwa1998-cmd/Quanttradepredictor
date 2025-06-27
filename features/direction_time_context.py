import pandas as pd

def add_time_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Try to find timestamp column
    timestamp_col = None
    for col in ['timestamp', 'Timestamp', 'datetime', 'Datetime', 'Date']:
        if col in df.columns:
            timestamp_col = col
            break
    
    # If no timestamp column found, use index if it's datetime
    if timestamp_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index
            timestamp_col = 'timestamp'
        else:
            # Create basic time features with default values
            df['hour'] = 10  # Default to mid-session
            df['minute'] = 30
            df['day_of_week'] = 2  # Default to Wednesday
            df['is_opening_range'] = 0
            df['is_mid_session'] = 1
            df['is_closing_phase'] = 0
            df['bar_number_in_day'] = range(1, len(df) + 1)
            df['is_friday'] = 0
            return df

    # Ensure timestamp is in datetime format
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Extract time-based features
    df['hour'] = df[timestamp_col].dt.hour
    df['minute'] = df[timestamp_col].dt.minute
    df['day_of_week'] = df[timestamp_col].dt.dayofweek  # Monday = 0, Friday = 4

    # Session flags (assumes Indian market: 9:15 to 15:30)
    df['is_opening_range'] = ((df['hour'] == 9) & (df['minute'] >= 15)) | ((df['hour'] == 10) & (df['minute'] < 0))
    df['is_mid_session'] = ((df['hour'] >= 11) & (df['hour'] < 13))
    df['is_closing_phase'] = ((df['hour'] >= 14) & (df['minute'] >= 45)) | (df['hour'] == 15)

    # Bar number in day (assumes 5-min bars)
    df['bar_number_in_day'] = df.groupby(df[timestamp_col].dt.date).cumcount() + 1

    # Is Friday
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    return df
