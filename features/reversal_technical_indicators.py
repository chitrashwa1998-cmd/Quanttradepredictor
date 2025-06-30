
import pandas as pd
import numpy as np

class ReversalTechnicalIndicators:
    """Calculate reversal-specific technical indicators for trading analysis."""

    @staticmethod
    def calculate_reversal_indicators(df):
        """Calculate indicators specifically for reversal model"""
        df = df.copy()

        # Ensure we have the right column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'

        try:
            # RSI - key reversal indicator
            delta = df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Williams %R - momentum oscillator
            highest_high = df[high_col].rolling(14).max()
            lowest_low = df[low_col].rolling(14).min()
            df['williams_r'] = ((highest_high - df[close_col]) / (highest_high - lowest_low)) * -100

            # Stochastic Oscillator
            lowest_low_k = df[low_col].rolling(14).min()
            highest_high_k = df[high_col].rolling(14).max()
            df['stoch_k'] = ((df[close_col] - lowest_low_k) / (highest_high_k - lowest_low_k)) * 100
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()

            # Bollinger Bands for extremes
            bb_period = 20
            bb_std = 2
            bb_middle = df[close_col].rolling(bb_period).mean()
            bb_std_dev = df[close_col].rolling(bb_period).std()
            df['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
            df['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
            df['bb_position'] = (df[close_col] - bb_lower) / (df['bb_upper'] - bb_lower)
            
            # Bollinger Band hits (reversal signals)
            df['bb_upper_hit'] = (df[close_col] >= df['bb_upper']).astype(int)
            df['bb_lower_hit'] = (df[close_col] <= df['bb_lower']).astype(int)

            # MACD for momentum divergence
            ema_12 = df[close_col].ewm(span=12).mean()
            ema_26 = df[close_col].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # Money Flow Index (MFI) - volume-weighted RSI
            if 'Volume' in df.columns or 'volume' in df.columns:
                volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
                typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
                money_flow = typical_price * df[volume_col]
                
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
                money_ratio = positive_flow / negative_flow
                df['mfi'] = 100 - (100 / (1 + money_ratio))
            else:
                df['mfi'] = 50.0  # Neutral value

            # CCI - Commodity Channel Index
            typical_price_cci = (df[high_col] + df[low_col] + df[close_col]) / 3
            sma_tp = typical_price_cci.rolling(20).mean()
            mean_deviation = typical_price_cci.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df['cci'] = (typical_price_cci - sma_tp) / (0.015 * mean_deviation)

            # Ultimate Oscillator - multi-timeframe momentum
            def true_range_uo(high, low, close_prev):
                return np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))

            def buying_pressure(close, low, close_prev):
                return close - np.minimum(low, close_prev)

            close_prev = df[close_col].shift(1)
            tr = true_range_uo(df[high_col], df[low_col], close_prev)
            bp = buying_pressure(df[close_col], df[low_col], close_prev)

            avg7_bp = bp.rolling(7).sum()
            avg7_tr = tr.rolling(7).sum()
            avg14_bp = bp.rolling(14).sum()
            avg14_tr = tr.rolling(14).sum()
            avg28_bp = bp.rolling(28).sum()
            avg28_tr = tr.rolling(28).sum()

            uo_raw7 = avg7_bp / avg7_tr
            uo_raw14 = avg14_bp / avg14_tr
            uo_raw28 = avg28_bp / avg28_tr

            df['ultimate_oscillator'] = 100 * ((4 * uo_raw7) + (2 * uo_raw14) + uo_raw28) / 7

            # Price vs moving averages (trend strength)
            df['sma_5'] = df[close_col].rolling(5).mean()
            df['sma_10'] = df[close_col].rolling(10).mean()
            df['sma_20'] = df[close_col].rolling(20).mean()
            df['sma_50'] = df[close_col].rolling(50).mean()

            df['price_vs_sma_5'] = (df[close_col] / df['sma_5'] - 1) * 100
            df['price_vs_sma_10'] = (df[close_col] / df['sma_10'] - 1) * 100
            df['price_vs_sma_20'] = (df[close_col] / df['sma_20'] - 1) * 100
            df['price_vs_sma_50'] = (df[close_col] / df['sma_50'] - 1) * 100

            # Replace inf and nan values
            reversal_cols = ['rsi', 'williams_r', 'stoch_k', 'stoch_d', 'bb_position', 'bb_upper_hit', 'bb_lower_hit',
                           'macd', 'macd_signal', 'macd_histogram', 'mfi', 'cci', 'ultimate_oscillator',
                           'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20', 'price_vs_sma_50']
            
            for col in reversal_cols:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        except Exception as e:
            print(f"Error calculating reversal technical indicators: {e}")
            # Fallback calculations
            df['rsi'] = 50.0
            df['williams_r'] = -50.0
            df['stoch_k'] = 50.0
            df['stoch_d'] = 50.0
            df['bb_position'] = 0.5
            df['bb_upper_hit'] = 0
            df['bb_lower_hit'] = 0
            df['macd_histogram'] = 0.0
            df['mfi'] = 50.0
            df['cci'] = 0.0
            df['ultimate_oscillator'] = 50.0

        return df

    @staticmethod
    def calculate_all_reversal_indicators(df):
        """Calculate all reversal indicators for the dataset"""
        print("🔧 Calculating reversal-specific technical indicators...")

        # Validate input data
        from utils.data_processing import DataProcessor
        is_valid, message = DataProcessor.validate_ohlc_data(df)
        if not is_valid:
            raise ValueError(f"Invalid OHLC data provided: {message}")

        # Create a copy to avoid modifying original data
        result_df = df.copy()

        # Calculate reversal indicators
        result_df = ReversalTechnicalIndicators.calculate_reversal_indicators(result_df)

        # Add custom engineered features
        from features.reversal_custom_engineered import add_custom_reversal_features
        result_df = add_custom_reversal_features(result_df)

        # Add lagged features
        from features.reversal_lagged_features import add_lagged_reversal_features
        result_df = add_lagged_reversal_features(result_df)

        # Add time context features
        from features.reversal_time_context import add_time_context_features_reversal
        result_df = add_time_context_features_reversal(result_df)

        # Final cleanup
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.dropna()

        feature_cols = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✅ Calculated {len(feature_cols)} reversal indicators")
        print(f"Generated reversal features: {feature_cols}")

        return result_df
