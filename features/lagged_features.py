
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

class LaggedFeatures:
    """Generate lagged features and temporal patterns for trading analysis."""
    
    @staticmethod
    def create_price_lags(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """Create lagged price features"""
        result_df = df.copy()
        
        for lag in lags:
            result_df[f'close_lag_{lag}'] = result_df['Close'].shift(lag)
            result_df[f'open_lag_{lag}'] = result_df['Open'].shift(lag)
            result_df[f'high_lag_{lag}'] = result_df['High'].shift(lag)
            result_df[f'low_lag_{lag}'] = result_df['Low'].shift(lag)
            
            # Price ratios with lagged values
            result_df[f'close_ratio_lag_{lag}'] = result_df['Close'] / result_df[f'close_lag_{lag}']
            result_df[f'hl_ratio_lag_{lag}'] = result_df['High'] / result_df[f'low_lag_{lag}']
            
        return result_df
    
    @staticmethod
    def create_return_lags(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """Create lagged return features"""
        result_df = df.copy()
        
        # Calculate returns
        result_df['returns'] = result_df['Close'].pct_change()
        result_df['log_returns'] = np.log(result_df['Close'] / result_df['Close'].shift(1))
        
        for lag in lags:
            result_df[f'returns_lag_{lag}'] = result_df['returns'].shift(lag)
            result_df[f'log_returns_lag_{lag}'] = result_df['log_returns'].shift(lag)
            
            # Cumulative returns over different periods
            result_df[f'cum_returns_{lag}'] = result_df['returns'].rolling(lag).apply(
                lambda x: (1 + x).prod() - 1
            )
            
        return result_df
    
    @staticmethod
    def create_volume_lags(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """Create lagged volume features"""
        result_df = df.copy()
        
        if 'Volume' not in result_df.columns:
            result_df['Volume'] = 1  # Default volume if not available
        
        # Volume changes
        result_df['volume_change'] = result_df['Volume'].pct_change()
        result_df['volume_ma_ratio'] = result_df['Volume'] / result_df['Volume'].rolling(20).mean()
        
        for lag in lags:
            result_df[f'volume_lag_{lag}'] = result_df['Volume'].shift(lag)
            result_df[f'volume_change_lag_{lag}'] = result_df['volume_change'].shift(lag)
            result_df[f'volume_ratio_lag_{lag}'] = result_df['Volume'] / result_df[f'volume_lag_{lag}']
            
        return result_df
    
    @staticmethod
    def create_lagged_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive lagged volatility features for volatility prediction model."""
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
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['vol_rolling_10'] = df['log_return'].rolling(window=10).std()

        # Lagged volatility
        df['vol_lag_1'] = df['vol_rolling_10'].shift(1)
        df['vol_lag_3'] = df['vol_rolling_10'].shift(3)
        df['vol_lag_5'] = df['vol_rolling_10'].shift(5)

        # Lagged ATR and BB width (assume already calculated elsewhere)
        if 'atr' in df.columns:
            df['atr_lag_1'] = df['atr'].shift(1)
            df['atr_lag_3'] = df['atr'].shift(3)
        if 'bb_width' in df.columns:
            df['bb_width_lag_1'] = df['bb_width'].shift(1)
            df['bb_width_lag_5'] = df['bb_width'].shift(5)

        # Lagged log returns
        for i in range(1, 6):
            df[f'log_return_lag_{i}'] = df['log_return'].shift(i)

        # Volatility delta and slope
        df['vol_delta_1'] = df['vol_rolling_10'] - df['vol_lag_1']
        df['vol_rolling_10_slope'] = (
            df['vol_rolling_10'].rolling(window=5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True
            )
        )

        # Lagged regime (if created elsewhere)
        if 'volatility_regime' in df.columns:
            df['volatility_regime_lag'] = df['volatility_regime'].shift(1)

        return df

    @staticmethod
    def create_volatility_lags(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """Create lagged volatility features"""
        result_df = df.copy()
        
        # Calculate different volatility measures
        result_df['realized_vol'] = result_df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        result_df['garman_klass_vol'] = np.sqrt(
            252 * (
                0.5 * (np.log(result_df['High'] / result_df['Low'])) ** 2 -
                (2 * np.log(2) - 1) * (np.log(result_df['Close'] / result_df['Open'])) ** 2
            ).rolling(20).mean()
        )
        result_df['parkinson_vol'] = np.sqrt(
            252 * (np.log(result_df['High'] / result_df['Low']) ** 2).rolling(20).mean() / (4 * np.log(2))
        )
        
        for lag in lags:
            result_df[f'realized_vol_lag_{lag}'] = result_df['realized_vol'].shift(lag)
            result_df[f'garman_klass_vol_lag_{lag}'] = result_df['garman_klass_vol'].shift(lag)
            result_df[f'parkinson_vol_lag_{lag}'] = result_df['parkinson_vol'].shift(lag)
            
            # Volatility ratios
            result_df[f'vol_ratio_lag_{lag}'] = result_df['realized_vol'] / result_df[f'realized_vol_lag_{lag}']
            
        return result_df
    
    @staticmethod
    def create_technical_indicator_lags(df: pd.DataFrame, 
                                      indicators: List[str] = ['rsi', 'macd_histogram', 'bb_position'],
                                      lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged technical indicator features"""
        result_df = df.copy()
        
        for indicator in indicators:
            if indicator in result_df.columns:
                for lag in lags:
                    result_df[f'{indicator}_lag_{lag}'] = result_df[indicator].shift(lag)
                    
                    # Changes in indicators
                    result_df[f'{indicator}_change_lag_{lag}'] = (
                        result_df[indicator] - result_df[f'{indicator}_lag_{lag}']
                    )
                    
                    # Rate of change
                    result_df[f'{indicator}_roc_lag_{lag}'] = (
                        result_df[indicator] / result_df[f'{indicator}_lag_{lag}'] - 1
                    ) * 100
        
        return result_df
    
    @staticmethod
    def create_momentum_lags(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """Create lagged momentum features"""
        result_df = df.copy()
        
        # Calculate momentum indicators
        for period in [5, 10, 20]:
            result_df[f'momentum_{period}'] = result_df['Close'] / result_df['Close'].shift(period) - 1
            result_df[f'roc_{period}'] = result_df['Close'].pct_change(period) * 100
        
        # Create lags for momentum indicators
        momentum_cols = [col for col in result_df.columns if col.startswith(('momentum_', 'roc_'))]
        
        for col in momentum_cols:
            for lag in lags:
                result_df[f'{col}_lag_{lag}'] = result_df[col].shift(lag)
        
        return result_df
    
    @staticmethod
    def create_pattern_sequence_features(df: pd.DataFrame, sequence_length: int = 5) -> pd.DataFrame:
        """Create features based on recent price patterns"""
        result_df = df.copy()
        
        # Direction sequences (up/down patterns)
        result_df['price_direction'] = (result_df['Close'] > result_df['Close'].shift(1)).astype(int)
        
        # Create pattern sequences
        for i in range(1, sequence_length + 1):
            result_df[f'direction_seq_{i}'] = result_df['price_direction'].shift(i)
        
        # Pattern recognition
        result_df['consecutive_up'] = 0
        result_df['consecutive_down'] = 0
        
        for i in range(len(result_df)):
            if i >= sequence_length:
                # Count consecutive ups
                up_count = 0
                for j in range(sequence_length):
                    if result_df.iloc[i - j]['price_direction'] == 1:
                        up_count += 1
                    else:
                        break
                result_df.iloc[i, result_df.columns.get_loc('consecutive_up')] = up_count
                
                # Count consecutive downs
                down_count = 0
                for j in range(sequence_length):
                    if result_df.iloc[i - j]['price_direction'] == 0:
                        down_count += 1
                    else:
                        break
                result_df.iloc[i, result_df.columns.get_loc('consecutive_down')] = down_count
        
        return result_df
    
    @staticmethod
    def create_rolling_statistics_lags(df: pd.DataFrame, 
                                     windows: List[int] = [5, 10, 20],
                                     lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Create lagged rolling statistics"""
        result_df = df.copy()
        
        for window in windows:
            # Rolling statistics
            result_df[f'rolling_mean_{window}'] = result_df['Close'].rolling(window).mean()
            result_df[f'rolling_std_{window}'] = result_df['Close'].rolling(window).std()
            result_df[f'rolling_min_{window}'] = result_df['Close'].rolling(window).min()
            result_df[f'rolling_max_{window}'] = result_df['Close'].rolling(window).max()
            result_df[f'rolling_skew_{window}'] = result_df['Close'].rolling(window).skew()
            result_df[f'rolling_kurt_{window}'] = result_df['Close'].rolling(window).kurt()
            
            # Position in rolling window
            result_df[f'position_in_range_{window}'] = (
                (result_df['Close'] - result_df[f'rolling_min_{window}']) /
                (result_df[f'rolling_max_{window}'] - result_df[f'rolling_min_{window}'])
            )
            
            # Create lags for rolling statistics
            rolling_cols = [col for col in result_df.columns if col.startswith(f'rolling_') and f'_{window}' in col]
            rolling_cols.append(f'position_in_range_{window}')
            
            for col in rolling_cols:
                for lag in lags:
                    result_df[f'{col}_lag_{lag}'] = result_df[col].shift(lag)
        
        return result_df
    
    @staticmethod
    def generate_all_lagged_features(df: pd.DataFrame,
                                   price_lags: List[int] = [1, 2, 3, 5, 10],
                                   return_lags: List[int] = [1, 2, 3, 5, 10],
                                   volume_lags: List[int] = [1, 2, 3, 5],
                                   volatility_lags: List[int] = [1, 2, 3, 5],
                                   technical_lags: List[int] = [1, 2, 3, 5],
                                   momentum_lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Generate all lagged features with customizable lag periods"""
        result_df = df.copy()
        
        # Apply all lagged feature methods
        result_df = LaggedFeatures.create_lagged_volatility_features(result_df)
        result_df = LaggedFeatures.create_price_lags(result_df, price_lags)
        result_df = LaggedFeatures.create_return_lags(result_df, return_lags)
        result_df = LaggedFeatures.create_volume_lags(result_df, volume_lags)
        result_df = LaggedFeatures.create_volatility_lags(result_df, volatility_lags)
        result_df = LaggedFeatures.create_technical_indicator_lags(result_df, lags=technical_lags)
        result_df = LaggedFeatures.create_momentum_lags(result_df, momentum_lags)
        result_df = LaggedFeatures.create_pattern_sequence_features(result_df)
        result_df = LaggedFeatures.create_rolling_statistics_lags(result_df)
        
        return result_df
