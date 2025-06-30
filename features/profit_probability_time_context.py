
import pandas as pd
import numpy as np

def add_profit_probability_time_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time context features specifically for profit probability prediction."""
    df = df.copy()

    # Create timestamp from index if it doesn't exist
    if 'timestamp' not in df.columns:
        df['timestamp'] = df.index

    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday = 0, Friday = 4
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    # Trading session analysis (Indian market: 9:15 AM to 3:30 PM)
    # Opening session (9:15 - 10:30)
    df['is_opening_session'] = (((df['hour'] == 9) & (df['minute'] >= 15)) | 
                               ((df['hour'] == 10) & (df['minute'] <= 30))).astype(int)
    
    # Mid-morning session (10:30 - 12:00)
    df['is_mid_morning'] = (((df['hour'] == 10) & (df['minute'] > 30)) | 
                           ((df['hour'] == 11)) | 
                           ((df['hour'] == 12) & (df['minute'] == 0))).astype(int)
    
    # Lunch session (12:00 - 13:30)
    df['is_lunch_session'] = (((df['hour'] == 12) & (df['minute'] > 0)) | 
                              ((df['hour'] == 13) & (df['minute'] <= 30))).astype(int)
    
    # Closing session (13:30 - 15:30)
    df['is_closing_session'] = (((df['hour'] == 13) & (df['minute'] > 30)) | 
                               (df['hour'] == 14) | 
                               ((df['hour'] == 15) & (df['minute'] <= 30))).astype(int)

    # High volatility periods
    df['is_first_30_min'] = (((df['hour'] == 9) & (df['minute'] >= 15)) | 
                             ((df['hour'] == 9) & (df['minute'] <= 45))).astype(int)
    
    df['is_last_30_min'] = (((df['hour'] == 15) & (df['minute'] >= 0)) | 
                            ((df['hour'] == 15) & (df['minute'] <= 30))).astype(int)

    # Weekly patterns
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
    df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)
    df['is_thursday'] = (df['day_of_week'] == 3).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Beginning and end of week
    df['is_week_start'] = (df['day_of_week'] <= 1).astype(int)  # Monday, Tuesday
    df['is_week_end'] = (df['day_of_week'] >= 3).astype(int)    # Thursday, Friday

    # Monthly patterns
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_mid'] = ((df['day_of_month'] > 10) & (df['day_of_month'] <= 20)).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)

    # Intraday bar position
    # Calculate which bar of the day this is (assuming 5-minute bars)
    df['bar_number_in_day'] = df.groupby(df['timestamp'].dt.date).cumcount() + 1
    total_bars_per_day = 75  # Approximately 75 5-minute bars in a trading day
    df['bar_position_pct'] = df['bar_number_in_day'] / total_bars_per_day

    # Early, mid, late day classification
    df['is_early_day'] = (df['bar_position_pct'] <= 0.33).astype(int)
    df['is_mid_day'] = ((df['bar_position_pct'] > 0.33) & (df['bar_position_pct'] <= 0.67)).astype(int)
    df['is_late_day'] = (df['bar_position_pct'] > 0.67).astype(int)

    # Time-based momentum patterns
    # Check if current time typically sees trending moves
    hour_minute_key = df['hour'] * 100 + df['minute']
    df['time_key'] = hour_minute_key

    # Seasonal effects
    df['is_quarter_end'] = ((df['month'] == 3) | (df['month'] == 6) | 
                           (df['month'] == 9) | (df['month'] == 12)).astype(int)

    # Holiday proximity (simplified - can be enhanced with actual holiday calendar)
    # This is a basic implementation - you might want to add actual holiday detection
    df['is_potential_holiday_week'] = 0  # Placeholder for holiday detection

    # Cyclical time features (for capturing periodic patterns)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Time since market open (in minutes)
    market_open_time = pd.Timestamp('09:15:00').time()
    df['minutes_since_open'] = ((df['hour'] - 9) * 60 + df['minute'] - 15)
    df['minutes_since_open'] = df['minutes_since_open'].clip(lower=0)

    # Time until market close (in minutes)
    market_close_time = pd.Timestamp('15:30:00').time()
    df['minutes_until_close'] = ((15 - df['hour']) * 60 + (30 - df['minute']))
    df['minutes_until_close'] = df['minutes_until_close'].clip(lower=0)

    # Volatility time patterns
    # Typically higher volatility periods
    df['is_high_vol_time'] = ((df['is_first_30_min'] == 1) | 
                             (df['is_last_30_min'] == 1) | 
                             (df['is_lunch_session'] == 1)).astype(int)

    # Remove the timestamp column if it was created
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    return df
