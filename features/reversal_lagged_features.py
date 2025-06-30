
import pandas as pd
import numpy as np

def add_lagged_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Handle column name compatibility
    close_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'

    # Lagged price features
    df['lag_close_1'] = df[close_col].shift(1)
    df['lag_close_2'] = df[close_col].shift(2)
    df['lag_close_3'] = df[close_col].shift(3)
    df['lag_close_5'] = df[close_col].shift(5)
    df['lag_high_1'] = df[high_col].shift(1)
    df['lag_low_1'] = df[low_col].shift(1)

    # Lagged RSI (key reversal indicator)
    if 'rsi' in df.columns:
        df['lag_rsi_1'] = df['rsi'].shift(1)
        df['lag_rsi_2'] = df['rsi'].shift(2)
        df['lag_rsi_3'] = df['rsi'].shift(3)
        
        # RSI momentum
        df['rsi_change_1'] = df['rsi'] - df['lag_rsi_1']
        df['rsi_change_2'] = df['rsi'] - df['lag_rsi_2']
        
        # RSI divergence detection
        price_change_2 = df[close_col] - df['lag_close_2']
        rsi_change_2 = df['rsi'] - df['lag_rsi_2']
        df['rsi_price_divergence'] = (
            ((price_change_2 > 0) & (rsi_change_2 < 0)) |
            ((price_change_2 < 0) & (rsi_change_2 > 0))
        ).astype(int)

    # Lagged Williams %R
    if 'williams_r' in df.columns:
        df['lag_williams_r_1'] = df['williams_r'].shift(1)
        df['lag_williams_r_2'] = df['williams_r'].shift(2)
        df['williams_r_change'] = df['williams_r'] - df['lag_williams_r_1']

    # Lagged Stochastic
    if 'stoch_k' in df.columns:
        df['lag_stoch_k_1'] = df['stoch_k'].shift(1)
        df['lag_stoch_d_1'] = df['stoch_d'].shift(1) if 'stoch_d' in df.columns else None
        df['stoch_momentum'] = df['stoch_k'] - df['lag_stoch_k_1']

    # Lagged MACD
    if 'macd_histogram' in df.columns:
        df['lag_macd_histogram_1'] = df['macd_histogram'].shift(1)
        df['lag_macd_histogram_2'] = df['macd_histogram'].shift(2)
        df['macd_histogram_change'] = df['macd_histogram'] - df['lag_macd_histogram_1']
        
        # MACD histogram zero-line crosses
        df['macd_zero_cross'] = (
            (df['macd_histogram'] > 0) & (df['lag_macd_histogram_1'] <= 0)
        ).astype(int) + (
            (df['macd_histogram'] < 0) & (df['lag_macd_histogram_1'] >= 0)
        ).astype(int) * -1

    # Lagged Bollinger Band position
    if 'bb_position' in df.columns:
        df['lag_bb_position_1'] = df['bb_position'].shift(1)
        df['lag_bb_position_2'] = df['bb_position'].shift(2)
        df['bb_position_change'] = df['bb_position'] - df['lag_bb_position_1']

    # Lagged momentum features
    if 'momentum_3' in df.columns:
        df['lag_momentum_3_1'] = df['momentum_3'].shift(1)
        df['lag_momentum_5_1'] = df['momentum_5'].shift(1) if 'momentum_5' in df.columns else None
        df['momentum_persistence'] = (
            (df['momentum_3'] > 0) & (df['lag_momentum_3_1'] > 0)
        ).astype(int) + (
            (df['momentum_3'] < 0) & (df['lag_momentum_3_1'] < 0)
        ).astype(int) * -1

    # Lagged volatility features
    if 'volatility_5' in df.columns:
        df['lag_volatility_5_1'] = df['volatility_5'].shift(1)
        df['volatility_change'] = df['volatility_5'] - df['lag_volatility_5_1']

    # Lagged candle pattern features
    if 'hammer_score' in df.columns:
        df['lag_hammer_score_1'] = df['hammer_score'].shift(1)
        df['lag_shooting_star_score_1'] = df['shooting_star_score'].shift(1) if 'shooting_star_score' in df.columns else None
        df['lag_doji_score_1'] = df['doji_score'].shift(1) if 'doji_score' in df.columns else None

    # Lagged position features
    if 'position_in_10_range' in df.columns:
        df['lag_position_in_10_range_1'] = df['position_in_10_range'].shift(1)
        df['position_trend'] = df['position_in_10_range'] - df['lag_position_in_10_range_1']

    # Lagged near extremes
    if 'near_10_high' in df.columns:
        df['lag_near_10_high_1'] = df['near_10_high'].shift(1)
        df['lag_near_10_low_1'] = df['near_10_low'].shift(1) if 'near_10_low' in df.columns else None

    # Rolling statistics of key reversal indicators
    if 'rsi' in df.columns:
        df['rsi_mean_5'] = df['rsi'].rolling(5).mean()
        df['rsi_std_5'] = df['rsi'].rolling(5).std()
        df['rsi_min_5'] = df['rsi'].rolling(5).min()
        df['rsi_max_5'] = df['rsi'].rolling(5).max()
        
        # RSI extremes count
        df['rsi_oversold_count_5'] = (df['rsi'] < 30).rolling(5).sum()
        df['rsi_overbought_count_5'] = (df['rsi'] > 70).rolling(5).sum()

    # Price momentum consistency
    price_change_1 = df[close_col].pct_change(1)
    df['momentum_direction_1'] = np.sign(price_change_1)
    df['lag_momentum_direction_1'] = df['momentum_direction_1'].shift(1)
    df['lag_momentum_direction_2'] = df['momentum_direction_1'].shift(2)
    df['lag_momentum_direction_3'] = df['momentum_direction_1'].shift(3)
    
    # Momentum consistency score
    df['momentum_consistency_3'] = (
        df['momentum_direction_1'] + 
        df['lag_momentum_direction_1'] + 
        df['lag_momentum_direction_2']
    ) / 3

    # Trend change detection
    df['trend_change'] = (
        (df['momentum_direction_1'] != df['lag_momentum_direction_1'])
    ).astype(int)
    
    # Multiple timeframe trend change
    df['trend_change_multi'] = (
        df['trend_change'] & 
        (df['lag_momentum_direction_1'] == df['lag_momentum_direction_2']) &
        (df['lag_momentum_direction_2'] == df['lag_momentum_direction_3'])
    ).astype(int)

    # Lagged gaps
    if 'overnight_gap' in df.columns:
        df['lag_overnight_gap_1'] = df['overnight_gap'].shift(1)
        df['gap_follow_through'] = (
            (df['overnight_gap'] > 0.002) & (price_change_1 > 0)
        ).astype(int) + (
            (df['overnight_gap'] < -0.002) & (price_change_1 < 0)
        ).astype(int) * -1

    # Volume confirmation (if available)
    if 'volume_spike' in df.columns:
        df['lag_volume_spike_1'] = df['volume_spike'].shift(1)
        df['volume_reversal_confirm'] = (
            df['volume_spike'] & df['trend_change']
        ).astype(int)

    return df
