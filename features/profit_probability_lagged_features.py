
import pandas as pd
import numpy as np

def add_lagged_profit_probability_features(df):
    """Add lagged features specifically for profit probability prediction."""
    df = df.copy()

    # Ensure we have the right column names
    close_col = 'Close' if 'Close' in df.columns else 'close'

    # Lagged price features
    df['lag_close_1'] = df[close_col].shift(1)
    df['lag_close_3'] = df[close_col].shift(3)
    df['lag_close_5'] = df[close_col].shift(5)

    # Lagged returns
    df['lag_return_1'] = df[close_col].pct_change().shift(1)
    df['lag_return_3'] = df[close_col].pct_change().shift(3)
    df['lag_return_5'] = df[close_col].pct_change().shift(5)

    # Lagged MACD features (if available)
    if 'macd' in df.columns:
        df['lag_macd_1'] = df['macd'].shift(1)
        df['lag_macd_3'] = df['macd'].shift(3)
        df['lag_macd_histogram_1'] = df['macd_histogram'].shift(1) if 'macd_histogram' in df.columns else np.nan

    # Lagged RSI features (if available)
    if 'rsi' in df.columns:
        df['lag_rsi_1'] = df['rsi'].shift(1)
        df['lag_rsi_3'] = df['rsi'].shift(3)
        df['lag_rsi_5'] = df['rsi'].shift(5)

    # Lagged Stochastic features (if available)
    if 'stoch_k' in df.columns:
        df['lag_stoch_k_1'] = df['stoch_k'].shift(1)
        df['lag_stoch_k_3'] = df['stoch_k'].shift(3)

    if 'stoch_d' in df.columns:
        df['lag_stoch_d_1'] = df['stoch_d'].shift(1)

    # Lagged Bollinger Band position (if available)
    if 'bb_position' in df.columns:
        df['lag_bb_position_1'] = df['bb_position'].shift(1)
        df['lag_bb_position_3'] = df['bb_position'].shift(3)

    # Lagged EMA crossover (if available)
    if 'ema_crossover' in df.columns:
        df['lag_ema_crossover_1'] = df['ema_crossover'].shift(1)
        df['lag_ema_crossover_3'] = df['ema_crossover'].shift(3)

    # Lagged ATR features (if available)
    if 'atr' in df.columns:
        df['lag_atr_1'] = df['atr'].shift(1)
        df['lag_atr_3'] = df['atr'].shift(3)

    # Rolling statistics of lagged features
    df['rolling_return_mean_5'] = df[close_col].pct_change().rolling(5).mean().shift(1)
    df['rolling_return_std_5'] = df[close_col].pct_change().rolling(5).std().shift(1)
    df['rolling_return_skew_5'] = df[close_col].pct_change().rolling(5).skew().shift(1)

    # Momentum persistence features
    returns = df[close_col].pct_change()
    df['momentum_persistence_3'] = (returns.rolling(3).apply(lambda x: (x > 0).sum())).shift(1)
    df['momentum_persistence_5'] = (returns.rolling(5).apply(lambda x: (x > 0).sum())).shift(1)

    # Volatility regime classification
    if 'atr' in df.columns:
        atr_20 = df['atr'].rolling(20).mean()
        atr_std = df['atr'].rolling(20).std()
        
        conditions = [
            df['atr'] < (atr_20 - 0.5 * atr_std),
            df['atr'] > (atr_20 + 0.5 * atr_std)
        ]
        choices = [0, 2]  # 0=low_vol, 1=medium_vol, 2=high_vol
        df['volatility_regime'] = np.select(conditions, choices, default=1)
        df['lag_volatility_regime_1'] = df['volatility_regime'].shift(1)

    # Price pattern recognition (lagged)
    price_change = df[close_col].diff()
    df['consecutive_up'] = (price_change > 0).rolling(3).sum().shift(1)
    df['consecutive_down'] = (price_change < 0).rolling(3).sum().shift(1)

    # Lagged support/resistance features
    if 'near_resistance' in df.columns:
        df['lag_near_resistance_1'] = df['near_resistance'].shift(1)
    if 'near_support' in df.columns:
        df['lag_near_support_1'] = df['near_support'].shift(1)

    # Cross-feature interactions (lagged)
    if 'rsi' in df.columns and 'bb_position' in df.columns:
        df['lag_rsi_bb_interaction'] = (df['rsi'] * df['bb_position']).shift(1)

    if 'macd' in df.columns and 'rsi' in df.columns:
        df['lag_macd_rsi_interaction'] = (df['macd'] * df['rsi']).shift(1)

    # Sequential pattern features
    df['price_sequence_up_3'] = ((df[close_col] > df[close_col].shift(1)) & 
                                (df[close_col].shift(1) > df[close_col].shift(2)) & 
                                (df[close_col].shift(2) > df[close_col].shift(3))).astype(int)

    df['price_sequence_down_3'] = ((df[close_col] < df[close_col].shift(1)) & 
                                  (df[close_col].shift(1) < df[close_col].shift(2)) & 
                                  (df[close_col].shift(2) < df[close_col].shift(3))).astype(int)

    return df
