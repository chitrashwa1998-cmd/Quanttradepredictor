import pandas as pd

def add_time_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Create timestamp from index if it doesn't exist
    if 'timestamp' not in df.columns:
        df['timestamp'] = df.index

    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday = 0, Friday = 4

    # Session flags (assumes Indian market: 9:15 to 15:30)
    df['is_opening_range'] = (((df['hour'] == 9) & (df['minute'] >= 15)) | ((df['hour'] == 10) & (df['minute'] < 0))).astype(int)
    df['is_mid_session'] = ((df['hour'] >= 11) & (df['hour'] < 13)).astype(int)
    df['is_closing_phase'] = (((df['hour'] >= 14) & (df['minute'] >= 45)) | (df['hour'] == 15)).astype(int)

    # Bar number in day (assumes 5-min bars)
    df['bar_number_in_day'] = df.groupby(df['timestamp'].dt.date).cumcount() + 1

    # Is Friday
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    return df