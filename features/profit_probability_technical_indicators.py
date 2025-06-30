
import pandas as pd
import numpy as np
from typing import Dict, List

class ProfitProbabilityTechnicalIndicators:
    """Calculate technical indicators specifically for profit probability prediction."""

    @staticmethod
    def calculate_profit_probability_indicators(df):
        """Calculate indicators specifically for profit probability model"""
        df = df.copy()

        # Ensure we have the right column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'

        try:
            # MACD calculation
            ema_12 = df[close_col].ewm(span=12).mean()
            ema_26 = df[close_col].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # EMA fast and slow for crossover detection
            df['ema_fast'] = df[close_col].ewm(span=8).mean()
            df['ema_slow'] = df[close_col].ewm(span=21).mean()
            df['ema_crossover'] = (df['ema_fast'] > df['ema_slow']).astype(int)

            # RSI calculation
            delta = df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # ATR calculation
            tr1 = df[high_col] - df[low_col]
            tr2 = abs(df[high_col] - df[close_col].shift(1))
            tr3 = abs(df[low_col] - df[close_col].shift(1))
            true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = df[close_col].rolling(bb_period).mean()
            bb_std_dev = df[close_col].rolling(bb_period).std()
            df['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
            df['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
            df['bb_position'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # Stochastic Oscillator
            lowest_low = df[low_col].rolling(window=14).min()
            highest_high = df[high_col].rolling(window=14).max()
            df['stoch_k'] = 100 * (df[close_col] - lowest_low) / (highest_high - lowest_low)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

            # Williams %R
            df['williams_r'] = -100 * (highest_high - df[close_col]) / (highest_high - lowest_low)

            # Momentum indicators
            df['momentum_5'] = df[close_col] / df[close_col].shift(5) - 1
            df['momentum_10'] = df[close_col] / df[close_col].shift(10) - 1

            # Rate of Change
            df['roc_5'] = df[close_col].pct_change(5) * 100
            df['roc_10'] = df[close_col].pct_change(10) * 100

            # Replace inf and nan values
            numeric_cols = ['macd', 'macd_signal', 'macd_histogram', 'ema_fast', 'ema_slow', 
                           'ema_crossover', 'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_position',
                           'stoch_k', 'stoch_d', 'williams_r', 'momentum_5', 'momentum_10',
                           'roc_5', 'roc_10']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
        except Exception as e:
            print(f"Error calculating profit probability technical indicators: {e}")
            # Fallback calculations
            df['macd'] = df[close_col].ewm(span=12).mean() - df[close_col].ewm(span=26).mean()
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['rsi'] = 50.0
            df['atr'] = (df[high_col] - df[low_col]).rolling(14).mean()
            df['ema_fast'] = df[close_col].ewm(span=8).mean()
            df['ema_slow'] = df[close_col].ewm(span=21).mean()
            df['ema_crossover'] = 1
            df['bb_upper'] = df[close_col] * 1.02
            df['bb_lower'] = df[close_col] * 0.98
            df['bb_position'] = 0.5
            df['stoch_k'] = 50.0
            df['stoch_d'] = 50.0

        return df

    @staticmethod
    def calculate_all_profit_probability_indicators(df):
        """Calculate all profit probability technical indicators"""
        print("🔧 Calculating profit probability technical indicators...")

        # Validate input data
        from utils.data_processing import DataProcessor
        is_valid, message = DataProcessor.validate_ohlc_data(df)
        if not is_valid:
            raise ValueError(f"Invalid OHLC data provided: {message}")

        # Create a copy to avoid modifying original data
        result_df = df.copy()

        # Calculate profit probability indicators
        print("Step 1: Calculating basic profit probability indicators...")
        result_df = ProfitProbabilityTechnicalIndicators.calculate_profit_probability_indicators(result_df)
        basic_features = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"After basic indicators: {len(basic_features)} features")

        # Add custom engineered features for profit probability
        print("Step 2: Adding custom profit probability features...")
        from features.profit_probability_custom_engineered import add_custom_profit_probability_features
        try:
            result_df = add_custom_profit_probability_features(result_df)
            custom_features = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"After custom features: {len(custom_features)} features")
        except Exception as e:
            print(f"Error in custom features: {e}")

        # Add lagged features for profit probability
        print("Step 3: Adding lagged profit probability features...")
        from features.profit_probability_lagged_features import add_lagged_profit_probability_features
        try:
            result_df = add_lagged_profit_probability_features(result_df)
            lagged_features = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"After lagged features: {len(lagged_features)} features")
        except Exception as e:
            print(f"Error in lagged features: {e}")

        # Add time context features for profit probability
        print("Step 4: Adding time context features...")
        from features.profit_probability_time_context import add_profit_probability_time_context_features
        try:
            result_df = add_profit_probability_time_context_features(result_df)
            time_features = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"After time features: {len(time_features)} features")
        except Exception as e:
            print(f"Error in time features: {e}")

        # Final cleanup
        print("Step 5: Final cleanup...")
        print("Filling NaN values...")
        
        # Fill NaN values with appropriate defaults
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                result_df[col] = result_df[col].fillna(result_df[col].median())
        
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        # Count completely empty rows
        non_ohlc_cols = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        empty_rows = result_df[non_ohlc_cols].isna().all(axis=1).sum()
        
        # Drop completely empty rows
        result_df = result_df.dropna(subset=non_ohlc_cols, how='all')
        
        print(f"Data points after cleanup: {len(result_df)} (dropped {empty_rows} completely empty rows)")

        feature_cols = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✅ Final result: {len(feature_cols)} profit probability indicators")
        print(f"Profit probability features: {feature_cols}")

        return result_df
