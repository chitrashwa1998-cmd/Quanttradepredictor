import pandas as pd

def add_time_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Handle timestamp column if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert existing index to datetime
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime. Index type: {type(df.index)}. Sample values: {df.index[:3].tolist()}. Error: {str(e)}")

    # Now we have a guaranteed DatetimeIndex, extract time features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6

    # Market session flags (for Indian market: 9:15 AM to 3:30 PM)
    df['is_post_10am'] = df.index.time >= pd.to_datetime("10:00").time()
    df['is_opening_range'] = df.index.time.between(pd.to_datetime("09:15").time(), pd.to_datetime("10:00").time())
    df['is_closing_phase'] = df.index.time >= pd.to_datetime("14:30").time()

    # Weekend flag
    df['is_weekend'] = df['day_of_week'] >= 5

    return df