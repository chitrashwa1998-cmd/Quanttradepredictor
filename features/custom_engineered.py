import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import ta
import pandas_ta as pta
from finta import TA

class CustomEngineeredFeatures:
    """Generate custom engineered features using advanced mathematical and statistical methods."""

    @staticmethod
    def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Build comprehensive volatility features for volatility prediction model."""
        df = df.copy()

        # Ensure column names are lowercase for consistency
        column_mapping = {}
        for col in df.columns:
            if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                column_mapping[col] = col.lower()

        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Use Close if close doesn't exist
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        if 'open' not in df.columns and 'Open' in df.columns:
            df['open'] = df['Open']
        if 'high' not in df.columns and 'High' in df.columns:
            df['high'] = df['High']
        if 'low' not in df.columns and 'Low' in df.columns:
            df['low'] = df['Low']

        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Realized volatility over 10 bars
        df['realized_vol_10'] = df['log_return'].rolling(10).std()

        # Parkinson volatility
        df['parkinson_vol'] = np.sqrt((np.log(df['high'] / df['low'])**2) / (4 * np.log(2)))

        # High-low ratio
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']

        # Gap %
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Candle body ratio
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)

        # Wick-to-body ratio
        upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
        lower_wick = df[['close', 'open']].min(axis=1) - df['low']
        body = abs(df['close'] - df['open']) + 1e-6
        df['wick_body_ratio'] = (upper_wick + lower_wick) / body

        # Z-score of body size
        body_size = abs(df['close'] - df['open'])
        df['body_zscore'] = (body_size - body_size.rolling(20).mean()) / (body_size.rolling(20).std() + 1e-6)

        # Volatility spike flag using rolling BB width approximation
        bb_width = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close'].rolling(20).mean()
        df['vol_spike_flag'] = (bb_width > bb_width.quantile(0.95)).astype(int)

        # Momentum acceleration
        df['momentum'] = df['close'] - df['close'].shift(5)
        df['momentum_acceleration'] = df['momentum'] - df['momentum'].shift(1)

        # Inverted compression score
        df['compression_score'] = 1 / ((df['high'] - df['low']).rolling(5).mean() + 1e-6)

        return df

    @staticmethod
    def filter_intraday_window(df):
        """Filter data to keep only intraday trading window 10:00-15:15"""
        df['datetime'] = pd.to_datetime(df['datetime'])  # if not already datetime
        df['time'] = df['datetime'].dt.time

        market_start = pd.to_datetime("10:00").time()
        market_end = pd.to_datetime("15:15").time()

        # Keep rows between 10:00 and 15:15 only
        df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
        return df

    @staticmethod
    def generate_all_custom_features(df: pd.DataFrame, 
                                   market_data: Optional[Dict] = None,
                                   include_advanced_ta: bool = True) -> pd.DataFrame:
        """Generate all custom engineered features"""
        result_df = df.copy()

        # Apply volatility feature engineering
        result_df = CustomEngineeredFeatures.build_volatility_features(result_df)

        return result_df