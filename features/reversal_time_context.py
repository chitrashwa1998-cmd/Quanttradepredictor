
import pandas as pd
import numpy as np

def add_time_context_features_reversal(df: pd.DataFrame) -> pd.DataFrame:
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

    # Market session phases (critical for reversal timing)
    df['is_pre_market'] = df['timestamp'].dt.time < pd.to_datetime('09:15').time()
    df['is_opening_range'] = df['timestamp'].dt.time.between(
        pd.to_datetime('09:15').time(), 
        pd.to_datetime('10:00').time()
    ).astype(int)
    df['is_morning_session'] = df['timestamp'].dt.time.between(
        pd.to_datetime('10:00').time(), 
        pd.to_datetime('12:00').time()
    ).astype(int)
    df['is_midday_session'] = df['timestamp'].dt.time.between(
        pd.to_datetime('12:00').time(), 
        pd.to_datetime('14:00').time()
    ).astype(int)
    df['is_closing_range'] = df['timestamp'].dt.time >= pd.to_datetime('14:30').time()
    df['is_closing_range'] = df['is_closing_range'].astype(int)
    df['is_last_hour'] = df['timestamp'].dt.time >= pd.to_datetime('14:30').time()
    df['is_last_hour'] = df['is_last_hour'].astype(int)

    # Convert boolean to int for pre_market
    df['is_pre_market'] = df['is_pre_market'].astype(int)

    # Session phase (numeric encoding for reversal timing)
    def session_phase_reversal(t):
        if t < pd.to_datetime('09:15').time():
            return 0  # pre-market
        elif t < pd.to_datetime('10:00').time():
            return 1  # opening
        elif t < pd.to_datetime('12:00').time():
            return 2  # morning
        elif t < pd.to_datetime('14:00').time():
            return 3  # midday
        elif t < pd.to_datetime('15:00').time():
            return 4  # afternoon
        else:
            return 5  # closing

    df['session_phase'] = df['timestamp'].dt.time.apply(session_phase_reversal)

    # Reversal-prone time windows
    df['reversal_prone_open'] = df['timestamp'].dt.time.between(
        pd.to_datetime('09:15').time(), 
        pd.to_datetime('09:45').time()
    ).astype(int)
    
    df['reversal_prone_close'] = df['timestamp'].dt.time.between(
        pd.to_datetime('15:00').time(), 
        pd.to_datetime('15:30').time()
    ).astype(int)
    
    df['reversal_prone_lunch'] = df['timestamp'].dt.time.between(
        pd.to_datetime('12:30').time(), 
        pd.to_datetime('13:30').time()
    ).astype(int)

    # Day-of-week reversal patterns
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)  # Monday reversals
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)  # Friday reversals
    df['is_midweek'] = df['day_of_week'].isin([1, 2, 3]).astype(int)

    # Overnight gap analysis (previous day close vs current open)
    df['prev_close'] = df[close_col].shift(1)
    df['overnight_gap'] = df[open_col] - df['prev_close']
    df['overnight_gap_pct'] = df['overnight_gap'] / df['prev_close'].replace(0, np.nan)
    
    # Gap size categories for reversal probability
    df['small_gap'] = (abs(df['overnight_gap_pct']) < 0.005).astype(int)  # <0.5%
    df['medium_gap'] = (abs(df['overnight_gap_pct']).between(0.005, 0.02)).astype(int)  # 0.5-2%
    df['large_gap'] = (abs(df['overnight_gap_pct']) > 0.02).astype(int)  # >2%

    # Time-based momentum features
    try:
        # Intraday high/low progression
        df['time_to_high'] = df.groupby(df['timestamp'].dt.date)['close'].transform(
            lambda x: (x == x.max()).astype(int)
        )
        df['time_to_low'] = df.groupby(df['timestamp'].dt.date)['close'].transform(
            lambda x: (x == x.min()).astype(int)
        )
        
        # Morning vs afternoon performance
        morning_mask = df['timestamp'].dt.time < pd.to_datetime('12:00').time()
        df['morning_return'] = np.where(
            morning_mask,
            df[close_col] / df[open_col] - 1,
            np.nan
        )
        df['afternoon_return'] = np.where(
            ~morning_mask,
            df[close_col] / df[open_col] - 1,
            np.nan
        )

    except Exception as e:
        print(f"Time-based momentum error: {e}")
        df['time_to_high'] = 0
        df['time_to_low'] = 0
        df['morning_return'] = 0.0
        df['afternoon_return'] = 0.0

    # Volume patterns by time (if volume available)
    if 'Volume' in df.columns or 'volume' in df.columns:
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
        
        # Average volume by hour
        try:
            hourly_avg_volume = df.groupby(df['hour_of_day'])[volume_col].transform('mean')
            df['volume_vs_hourly_avg'] = df[volume_col] / hourly_avg_volume.replace(0, np.nan)
            
            # High volume reversal signals
            df['high_volume_reversal'] = (df['volume_vs_hourly_avg'] > 1.5).astype(int)
            
        except Exception as e:
            print(f"Volume time analysis error: {e}")
            df['volume_vs_hourly_avg'] = 1.0
            df['high_volume_reversal'] = 0
    else:
        df['volume_vs_hourly_avg'] = 1.0
        df['high_volume_reversal'] = 0

    # First/last hour volatility
    try:
        returns = df[close_col].pct_change()
        
        first_hour_vol = returns[df['is_opening_range'] == 1].rolling(12).std()  # 5min data
        last_hour_vol = returns[df['is_last_hour'] == 1].rolling(12).std()
        
        df['first_hour_volatility'] = np.where(df['is_opening_range'] == 1, first_hour_vol, np.nan)
        df['last_hour_volatility'] = np.where(df['is_last_hour'] == 1, last_hour_vol, np.nan)
        
        # Fill NaN with forward fill
        df['first_hour_volatility'] = df['first_hour_volatility'].fillna(method='ffill')
        df['last_hour_volatility'] = df['last_hour_volatility'].fillna(method='ffill')
        
    except Exception as e:
        print(f"Hourly volatility error: {e}")
        df['first_hour_volatility'] = 0.01
        df['last_hour_volatility'] = 0.01

    # Weekend effect and week positioning
    df['start_of_week'] = (df['day_of_week'] <= 1).astype(int)  # Monday/Tuesday
    df['end_of_week'] = (df['day_of_week'] >= 3).astype(int)    # Thursday/Friday

    # Month-end effects (if applicable)
    try:
        df['is_month_end'] = (df['timestamp'].dt.day >= 28).astype(int)
        df['is_month_start'] = (df['timestamp'].dt.day <= 3).astype(int)
    except:
        df['is_month_end'] = 0
        df['is_month_start'] = 0

    # Remove timestamp column to prevent it from being used as a feature
    if 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)

    # Handle NaN values in time features
    time_features = [
        'hour_of_day', 'minute_of_hour', 'day_of_week', 'session_phase',
        'overnight_gap_pct', 'morning_return', 'afternoon_return',
        'volume_vs_hourly_avg', 'first_hour_volatility', 'last_hour_volatility'
    ]
    
    for feature in time_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(df[feature].median())

    return df
