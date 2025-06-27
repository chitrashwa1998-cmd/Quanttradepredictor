
import pandas as pd
import numpy as np

def add_direction_time_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time context features specifically for direction prediction."""
    df = df.copy()

    # Handle timestamp column if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime. Index type: {type(df.index)}. Error: {str(e)}")

    # Extract basic time features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month

    # Market session flags (for Indian market: 9:15 AM to 3:30 PM)
    opening_start = pd.to_datetime("09:15").time()
    opening_end = pd.to_datetime("10:00").time()
    mid_session_start = pd.to_datetime("10:00").time()
    mid_session_end = pd.to_datetime("14:00").time()
    closing_start = pd.to_datetime("14:00").time()
    closing_end = pd.to_datetime("15:30").time()
    
    df['is_opening_hour'] = [(t >= opening_start) & (t <= opening_end) for t in df.index.time]
    df['is_mid_session'] = [(t >= mid_session_start) & (t <= mid_session_end) for t in df.index.time]
    df['is_closing_hour'] = [(t >= closing_start) & (t <= closing_end) for t in df.index.time]

    # Pre-market and after-market flags
    pre_market_start = pd.to_datetime("09:00").time()
    pre_market_end = pd.to_datetime("09:15").time()
    df['is_pre_market'] = [(t >= pre_market_start) & (t < pre_market_end) for t in df.index.time]

    # Weekend flag
    df['is_weekend'] = df['day_of_week'] >= 5

    # Month-end and quarter-end effects
    df['is_month_end'] = df.index.is_month_end
    df['is_quarter_end'] = df.index.is_quarter_end
    df['is_year_end'] = (df.index.month == 12) & (df.index.day >= 25)

    # Intraday momentum patterns
    df['morning_momentum'] = df['is_opening_hour'].astype(int)
    df['afternoon_momentum'] = df['is_closing_hour'].astype(int)

    # Weekly patterns - different days might have different directional tendencies
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
    df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)
    df['is_thursday'] = (df['day_of_week'] == 3).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    # Expiry week effect (assuming weekly options expiry on Thursday)
    # This is a simplified version - you might want to use actual expiry dates
    week_of_month = (df.index.day - 1) // 7 + 1
    df['is_expiry_week'] = ((week_of_month == 4) | 
                           ((week_of_month == 5) & (df.index.day <= 7))).astype(int)

    # Seasonal effects
    df['is_summer'] = df.index.month.isin([4, 5, 6]).astype(int)
    df['is_monsoon'] = df.index.month.isin([7, 8, 9]).astype(int)
    df['is_festival_season'] = df.index.month.isin([10, 11]).astype(int)

    # Time-based volatility patterns
    df['high_activity_period'] = (df['is_opening_hour'] | df['is_closing_hour']).astype(int)
    df['low_activity_period'] = df['is_mid_session'].astype(int)

    return df
