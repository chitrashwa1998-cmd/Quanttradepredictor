import pandas as pd
import numpy as np

def add_time_context_features_reversal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Basic time breakdown
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['minute_of_hour'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Opening and closing ranges
    df['is_opening_range'] = df['timestamp'].dt.time.between(pd.to_datetime('09:15').time(), pd.to_datetime('10:00').time()).astype(int)
    df['is_closing_range'] = (df['timestamp'].dt.time >= pd.to_datetime('15:00').time()).astype(int)

    # Session phase
    def session_phase(t):
        if t < pd.to_datetime('10:00').time():
            return 'open'
        elif t < pd.to_datetime('14:30').time():
            return 'mid'
        else:
            return 'close'
    df['session_phase'] = df['timestamp'].dt.time.apply(session_phase)

    # Previous day context
    df['date'] = df['timestamp'].dt.date
    df['prev_close'] = df['close'].shift(1)
    df['overnight_gap'] = df['open'] - df['prev_close']

    prev_day_high = df.groupby('date')['high'].transform('max').shift(1)
    prev_day_low = df.groupby('date')['low'].transform('min').shift(1)
    prev_day_close = df.groupby('date')['close'].transform('last').shift(1)
    prev_prev_day_close = prev_day_close.shift(1)

    df['prev_day_range'] = prev_day_high - prev_day_low
    df['prev_day_return'] = (prev_day_close - prev_prev_day_close) / prev_prev_day_close

    # Weekend gap detection
    df['prev_date'] = df['date'].shift(1)
    df['is_weekend_gap'] = (df['date'] - df['prev_date']).dt.days > 1
    df['is_weekend_gap'] = df['is_weekend_gap'].astype(int)

    # Consecutive green/red candles
    df['green_candle'] = (df['close'] > df['open']).astype(int)
    df['red_candle'] = (df['close'] < df['open']).astype(int)
    df['num_consecutive_green_red'] = df['green_candle'].groupby((df['green_candle'] != df['green_candle'].shift()).cumsum()).cumcount() + 1

    # Market volatility & heat context
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['recent_volatility'] = df['log_return'].rolling(5).std()
    df['market_heat'] = df['log_return'].rolling(5).mean()

    return df
