import pandas as pd

# Assumes df has a datetime index or a column named 'timestamp'
def add_time_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Time components
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6

    # Market session flags (for Indian market: 9:15 AM to 3:30 PM)
    df['is_post_10am'] = df.index.time >= pd.to_datetime("10:00").time()
    df['is_opening_range'] = df.index.time.between(pd.to_datetime("09:15").time(), pd.to_datetime("10:00").time())
    df['is_closing_phase'] = df.index.time >= pd.to_datetime("14:30").time()

    # Session phase bucket
    def get_session_phase(t):
        if t < pd.to_datetime("10:00").time():
            return 'morning'
        elif t < pd.to_datetime("14:30").time():
            return 'mid'
        else:
            return 'close'
    df['session_phase'] = df.index.time.map(get_session_phase)

    # Optional: weekend flag (in case using calendar dates)
    df['is_weekend'] = df['day_of_week'] >= 5

    return df
