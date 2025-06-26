import pandas as pd
import numpy as np

def create_time_context_features(df):
    # Ensure datetime index or column exists
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        else:
            # Try to convert index to datetime if possible
            try:
                df.index = pd.to_datetime(df.index)
            except:
                print("Warning: Could not convert index to datetime")

    df['hour_of_day'] = df.index.hour
    df['minute_of_hour'] = df.index.minute

    df['is_post_10am'] = (df['hour_of_day'] > 10) | ((df['hour_of_day'] == 10) & (df['minute_of_hour'] >= 0))

    df['day_of_week'] = df.index.dayofweek  # Monday = 0, Sunday = 6

    df['is_opening_range'] = ((df['hour_of_day'] == 9) & (df['minute_of_hour'] >= 15)) | \
                              ((df['hour_of_day'] == 10) & (df['minute_of_hour'] < 0))

    df['is_closing_phase'] = (df['hour_of_day'] == 15) & (df['minute_of_hour'] >= 0)

    # Categorize market phases
    def get_market_phase(hour, minute):
        time = hour + minute / 60
        if 9.25 <= time < 10.0:
            return 'open'
        elif 10.0 <= time < 14.5:
            return 'midday'
        else:
            return 'close'

    df['market_phase'] = [get_market_phase(h, m) for h, m in zip(df['hour_of_day'], df['minute_of_hour'])]

    df['time_since_open'] = (df.index - df.index.normalize() - pd.Timedelta(minutes=555)).total_seconds() / 60
    df['time_since_open'] = df['time_since_open'].clip(lower=0)

    # Calculate log returns if not present
    if 'log_return' not in df.columns:
        if 'Close' in df.columns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        elif 'close' in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Placeholder: use past session volatility if available
    if 'log_return' in df.columns:
        df['prev_session_volatility'] = df['log_return'].rolling('1D').std().shift(1)

        # Intraday percentile volatility rank
        df['rolling_vol_today'] = df.groupby(df.index.date)['log_return'].transform(lambda x: x.rolling(10).std())
        df['intraday_volatility_rank'] = df.groupby(df.index.date)['rolling_vol_today'].transform(
            lambda x: x.rank(pct=True)
        )

    return df