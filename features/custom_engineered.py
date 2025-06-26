import pandas as pd
import numpy as np

def filter_intraday_window(df):
    df['datetime'] = pd.to_datetime(df['datetime'])  # if not already datetime
    df['time'] = df['datetime'].dt.time

    market_start = pd.to_datetime("10:00").time()
    market_end = pd.to_datetime("15:15").time()

    # Keep rows between 10:00 and 15:15 only
    df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
    return df