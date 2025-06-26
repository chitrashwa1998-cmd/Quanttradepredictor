import pandas as pd
import numpy as np

def create_time_context_features(df):
    """Create simplified time context features focused on volatility"""

    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except:
                print("Warning: Could not convert index to datetime")

    # Basic time features for volatility analysis
    df['hour_of_day'] = df.index.hour
    df['minute_of_hour'] = df.index.minute

    # Market phase (affects volatility patterns)
    def get_market_phase(hour, minute):
        time = hour + minute / 60
        if 9.25 <= time < 10.0:
            return 'open'
        elif 10.0 <= time < 14.5:
            return 'midday'
        else:
            return 'close'

    df['market_phase'] = [get_market_phase(h, m) for h, m in zip(df['hour_of_day'], df['minute_of_hour'])]

    # Time since market open (volatility often higher at open)
    df['time_since_open'] = (df.index - df.index.normalize() - pd.Timedelta(minutes=555)).total_seconds() / 60
    df['time_since_open'] = df['time_since_open'].clip(lower=0)

    return df

def filter_intraday_window(df):
    """Filter data to intraday trading window"""
    df['datetime'] = pd.to_datetime(df['datetime'])  # if not already datetime
    df['time'] = df['datetime'].dt.time

    market_start = pd.to_datetime("10:00").time()
    market_end = pd.to_datetime("15:15").time()

    # Keep rows between 10:00 and 15:15 only
    df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
    return df