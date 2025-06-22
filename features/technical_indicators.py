
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class TechnicalIndicators:
    """Technical indicators calculator for trading strategies using pandas/numpy."""

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators optimized for 5-minute scalping."""

        if df.empty:
            raise ValueError("DataFrame is empty")

        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            raise ValueError("DataFrame must contain OHLC columns")

        # Work on a copy
        result_df = df.copy()

        # Ensure we have enough data for all indicators
        if len(result_df) < 50:
            raise ValueError("Need at least 50 data points for technical indicators")

        try:
            # 1. Simple Moving Averages (3 indicators)
            result_df['sma_5'] = result_df['Close'].rolling(window=5).mean()
            result_df['sma_10'] = result_df['Close'].rolling(window=10).mean()
            result_df['sma_20'] = result_df['Close'].rolling(window=20).mean()

            # 2. Exponential Moving Averages (3 indicators)
            result_df['ema_5'] = result_df['Close'].ewm(span=5).mean()
            result_df['ema_10'] = result_df['Close'].ewm(span=10).mean()
            result_df['ema_20'] = result_df['Close'].ewm(span=20).mean()

            # 3. RSI (1 indicator)
            delta = result_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df['rsi'] = 100 - (100 / (1 + rs))

            # 4. MACD (3 indicators)
            ema_12 = result_df['Close'].ewm(span=12).mean()
            ema_26 = result_df['Close'].ewm(span=26).mean()
            result_df['macd'] = ema_12 - ema_26
            result_df['macd_signal'] = result_df['macd'].ewm(span=9).mean()
            result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']

            # 5. Bollinger Bands (3 indicators)
            bb_middle = result_df['Close'].rolling(window=20).mean()
            bb_std = result_df['Close'].rolling(window=20).std()
            result_df['bb_upper'] = bb_middle + (bb_std * 2)
            result_df['bb_lower'] = bb_middle - (bb_std * 2)
            result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / bb_middle

            # 6. ATR (1 indicator)
            high_low = result_df['High'] - result_df['Low']
            high_close = np.abs(result_df['High'] - result_df['Close'].shift())
            low_close = np.abs(result_df['Low'] - result_df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            result_df['atr'] = true_range.rolling(window=14).mean()

            # 7. ADX (1 indicator)
            plus_dm = result_df['High'].diff()
            minus_dm = result_df['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = minus_dm.abs()
            
            tr_smooth = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr_smooth)
            dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            result_df['adx'] = dx.rolling(window=14).mean()

            # 8. CCI (1 indicator)
            tp = (result_df['High'] + result_df['Low'] + result_df['Close']) / 3
            tp_sma = tp.rolling(window=14).mean()
            mad = tp.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean())
            result_df['cci'] = (tp - tp_sma) / (0.015 * mad)

            # 9. Williams %R (1 indicator)
            highest_high = result_df['High'].rolling(window=14).max()
            lowest_low = result_df['Low'].rolling(window=14).min()
            result_df['williams_r'] = -100 * ((highest_high - result_df['Close']) / (highest_high - lowest_low))

            # 10. Stochastic (2 indicators)
            lowest_low_14 = result_df['Low'].rolling(window=14).min()
            highest_high_14 = result_df['High'].rolling(window=14).max()
            result_df['stoch_k'] = 100 * ((result_df['Close'] - lowest_low_14) / (highest_high_14 - lowest_low_14))
            result_df['stoch_d'] = result_df['stoch_k'].rolling(window=3).mean()

            # 11. Price Change (1 indicator)
            result_df['price_change'] = result_df['Close'].pct_change()

            # 12. Volume indicators (2 indicators)
            if 'Volume' in result_df.columns:
                result_df['volume_sma'] = result_df['Volume'].rolling(window=10).mean()
                result_df['volume_ratio'] = result_df['Volume'] / result_df['volume_sma']
            else:
                result_df['volume_sma'] = 1.0
                result_df['volume_ratio'] = 1.0

            # 13. Additional price-based indicators (3 indicators)
            result_df['high_low_ratio'] = (result_df['High'] - result_df['Low']) / result_df['Close']
            result_df['close_sma_ratio'] = result_df['Close'] / result_df['sma_20']
            result_df['volatility_5'] = result_df['price_change'].rolling(5).std()

            # Total: 23 indicators
            expected_indicators = [
                'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_width', 'atr', 'adx', 'cci', 'williams_r',
                'stoch_k', 'stoch_d', 'price_change', 'volume_sma', 'volume_ratio',
                'high_low_ratio', 'close_sma_ratio', 'volatility_5'
            ]

            # Verify all indicators were created
            missing_indicators = [ind for ind in expected_indicators if ind not in result_df.columns]
            if missing_indicators:
                print(f"Warning: Missing indicators: {missing_indicators}")

            # Fill NaN values using forward fill then backward fill
            for indicator in expected_indicators:
                if indicator in result_df.columns:
                    result_df[indicator] = result_df[indicator].fillna(method='ffill').fillna(method='bfill')

            # For any remaining NaN values, fill with reasonable defaults
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns
            result_df[numeric_columns] = result_df[numeric_columns].fillna(0)

            print(f"✅ Calculated 23 technical indicators successfully")
            return result_df

        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            # Return original dataframe if calculation fails
            return df.copy()

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all feature names."""
        return [
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_width', 'atr', 'adx', 'cci', 'williams_r',
            'stoch_k', 'stoch_d', 'price_change', 'volume_sma', 'volume_ratio',
            'high_low_ratio', 'close_sma_ratio', 'volatility_5'
        ]
