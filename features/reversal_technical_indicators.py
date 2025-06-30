
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
            # RSI_14 - 14-period RSI
            delta = df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD Histogram
            ema_12 = df[close_col].ewm(span=12).mean()
            ema_26 = df[close_col].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            df['macd_histogram'] = macd - macd_signal

            # Stochastic K and D
            lowest_low_k = df[low_col].rolling(14).min()
            highest_high_k = df[high_col].rolling(14).max()
            df['stochastic_k'] = ((df[close_col] - lowest_low_k) / (highest_high_k - lowest_low_k)) * 100
            df['stochastic_d'] = df['stochastic_k'].rolling(3).mean()

            # CCI - Commodity Channel Index
            typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
            sma_tp = typical_price.rolling(20).mean()
            mean_deviation = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation)

            # Williams %R
            highest_high = df[high_col].rolling(14).max()
            lowest_low = df[low_col].rolling(14).min()
            df['williams_r'] = ((highest_high - df[close_col]) / (highest_high - lowest_low)) * -100

            # Bollinger Bands Percent B
            bb_period = 20
            bb_std = 2
            bb_middle = df[close_col].rolling(bb_period).mean()
            bb_std_dev = df[close_col].rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            df['bb_percent_b'] = (df[close_col] - bb_lower) / (bb_upper - bb_lower)

            # ATR - Average True Range
            tr1 = df[high_col] - df[low_col]
            tr2 = abs(df[high_col] - df[close_col].shift(1))
            tr3 = abs(df[low_col] - df[close_col].shift(1))
            true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()

            # Donchian Channel Width
            donchian_high = df[high_col].rolling(20).max()
            donchian_low = df[low_col].rolling(20).min()
            df['donchian_channel_width'] = (donchian_high - donchian_low) / df[close_col]

            # Parabolic SAR
            def calculate_parabolic_sar(df, af_start=0.02, af_increment=0.02, af_max=0.2):
                high = df[high_col].values
                low = df[low_col].values
                close = df[close_col].values
                
                length = len(df)
                psar = np.zeros(length)
                psarbull = np.zeros(length)
                psarbear = np.zeros(length)
                
                # Initialize
                psar[0] = close[0]
                psarbull[0] = low[0]
                psarbear[0] = high[0]
                
                bull = True
                af = af_start
                
                for i in range(1, length):
                    if bull:
                        psar[i] = psar[i-1] + af * (psarbull[i-1] - psar[i-1])
                        if high[i] > psarbull[i-1]:
                            psarbull[i] = high[i]
                            af = min(af + af_increment, af_max)
                        else:
                            psarbull[i] = psarbull[i-1]
                        
                        if low[i] <= psar[i]:
                            bull = False
                            psar[i] = psarbull[i-1]
                            psarbear[i] = low[i]
                            af = af_start
                        else:
                            psarbear[i] = psarbear[i-1]
                    else:
                        psar[i] = psar[i-1] + af * (psarbear[i-1] - psar[i-1])
                        if low[i] < psarbear[i-1]:
                            psarbear[i] = low[i]
                            af = min(af + af_increment, af_max)
                        else:
                            psarbear[i] = psarbear[i-1]
                        
                        if high[i] >= psar[i]:
                            bull = True
                            psar[i] = psarbear[i-1]
                            psarbull[i] = high[i]
                            af = af_start
                        else:
                            psarbull[i] = psarbull[i-1]
                
                return psar

            df['parabolic_sar'] = calculate_parabolic_sar(df)

            # Momentum ROC (Rate of Change)
            period = 10
            df['momentum_roc'] = ((df[close_col] - df[close_col].shift(period)) / df[close_col].shift(period)) * 100

            # Replace inf and nan values
            reversal_cols = ['rsi_14', 'macd_histogram', 'stochastic_k', 'stochastic_d', 'cci', 
                           'williams_r', 'bb_percent_b', 'atr', 'donchian_channel_width', 
                           'parabolic_sar', 'momentum_roc']
            
            for col in reversal_cols:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        except Exception as e:
            print(f"Error calculating reversal technical indicators: {e}")
            # Fallback calculations
            df['rsi_14'] = 50.0
            df['macd_histogram'] = 0.0
            df['stochastic_k'] = 50.0
            df['stochastic_d'] = 50.0
            df['cci'] = 0.0
            df['williams_r'] = -50.0
            df['bb_percent_b'] = 0.5
            df['atr'] = 0.0
            df['donchian_channel_width'] = 0.0
            df['parabolic_sar'] = df[close_col] if close_col in df.columns else 0.0
            df['momentum_roc'] = 0.0

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

def add_reversal_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper function to calculate reversal technical indicators"""
    return ReversalTechnicalIndicators.calculate_reversal_indicators(df)
