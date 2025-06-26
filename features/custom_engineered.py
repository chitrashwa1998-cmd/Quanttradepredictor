
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
    def create_fractal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create fractal and geometric pattern features"""
        result_df = df.copy()
        
        # Hurst Exponent (simplified calculation)
        def calculate_hurst(ts, max_lag=20):
            """Calculate Hurst exponent for time series"""
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            # Linear fit
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]
        
        # Calculate Hurst exponent over rolling windows
        result_df['hurst_50'] = result_df['Close'].rolling(50).apply(
            lambda x: calculate_hurst(x.values) if len(x.dropna()) >= 20 else 0.5
        )
        
        # Fractal dimension approximation
        result_df['fractal_dimension'] = 2 - result_df['hurst_50']
        
        # Self-similarity measure
        result_df['self_similarity'] = result_df['Close'].rolling(20).apply(
            lambda x: np.corrcoef(x[:-10], x[10:])[0, 1] if len(x) == 20 else 0
        )
        
        return result_df
    
    @staticmethod
    def create_regime_detection_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime detection features"""
        result_df = df.copy()
        
        # Variance regime detection
        returns = result_df['Close'].pct_change()
        result_df['variance_regime'] = returns.rolling(20).var().rolling(5).apply(
            lambda x: 1 if x.iloc[-1] > x.mean() + x.std() else 0
        )
        
        # Trend regime using multiple timeframes
        for window in [10, 20, 50]:
            result_df[f'trend_regime_{window}'] = (
                result_df['Close'] > result_df['Close'].rolling(window).mean()
            ).astype(int)
        
        # Combined regime score
        regime_cols = [col for col in result_df.columns if col.startswith('trend_regime_')]
        if regime_cols:
            result_df['combined_regime'] = result_df[regime_cols].mean(axis=1)
        
        # Regime transition detection
        result_df['regime_transition'] = result_df['combined_regime'].diff().abs()
        
        return result_df
    
    @staticmethod
    def create_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create information theory based features"""
        result_df = df.copy()
        
        def calculate_entropy(x, bins=10):
            """Calculate Shannon entropy"""
            if len(x) < bins:
                return 0
            
            hist, _ = np.histogram(x, bins=bins)
            hist = hist[hist > 0]  # Remove zero entries
            
            if len(hist) <= 1:
                return 0
            
            # Normalize
            probs = hist / hist.sum()
            return -np.sum(probs * np.log2(probs))
        
        # Price entropy over different windows
        for window in [10, 20, 50]:
            result_df[f'price_entropy_{window}'] = result_df['Close'].rolling(window).apply(
                lambda x: calculate_entropy(x.values)
            )
            
            # Return entropy
            returns = result_df['Close'].pct_change()
            result_df[f'return_entropy_{window}'] = returns.rolling(window).apply(
                lambda x: calculate_entropy(x.values)
            )
        
        # Entropy ratios (information content changes)
        result_df['entropy_ratio_short_long'] = (
            result_df['price_entropy_10'] / (result_df['price_entropy_50'] + 1e-8)
        )
        
        return result_df
    
    @staticmethod
    def create_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create Fourier transform based cyclical features"""
        result_df = df.copy()
        
        # Fast Fourier Transform features
        def extract_fft_features(x, n_components=5):
            """Extract dominant frequency components"""
            if len(x) < 10:
                return [0] * n_components
            
            fft = np.fft.fft(x)
            freqs = np.fft.fftfreq(len(x))
            
            # Get magnitude spectrum
            magnitude = np.abs(fft)
            
            # Find dominant frequencies (excluding DC component)
            dominant_indices = np.argsort(magnitude[1:len(magnitude)//2])[-n_components:]
            
            return magnitude[dominant_indices + 1].tolist()
        
        # Apply FFT to price data
        fft_window = 50
        for i in range(3):  # Extract top 3 frequency components
            result_df[f'fft_component_{i}'] = result_df['Close'].rolling(fft_window).apply(
                lambda x: extract_fft_features(x.values, 3)[i] if len(x) == fft_window else 0
            )
        
        return result_df
    
    @staticmethod
    def create_wavelet_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create wavelet-inspired features (simplified version)"""
        result_df = df.copy()
        
        # Discrete wavelet transform approximation using high/low pass filters
        def simple_wavelet_decomp(x):
            """Simplified wavelet decomposition"""
            if len(x) < 4:
                return 0, 0
            
            # Approximation (low-pass): moving average
            approx = np.mean(x)
            
            # Detail (high-pass): deviation from trend
            trend = np.linspace(x[0], x[-1], len(x))
            detail = np.std(x - trend)
            
            return approx, detail
        
        # Apply wavelet decomposition over different scales
        for window in [8, 16, 32]:
            wavelet_features = result_df['Close'].rolling(window).apply(
                lambda x: simple_wavelet_decomp(x.values)[0]  # Approximation
            )
            result_df[f'wavelet_approx_{window}'] = wavelet_features
            
            wavelet_details = result_df['Close'].rolling(window).apply(
                lambda x: simple_wavelet_decomp(x.values)[1]  # Detail
            )
            result_df[f'wavelet_detail_{window}'] = wavelet_details
        
        return result_df
    
    @staticmethod
    def create_statistical_arbitrage_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical arbitrage and mean reversion features"""
        result_df = df.copy()
        
        # Z-score features
        for window in [20, 50, 100]:
            mean = result_df['Close'].rolling(window).mean()
            std = result_df['Close'].rolling(window).std()
            result_df[f'zscore_{window}'] = (result_df['Close'] - mean) / (std + 1e-8)
            
            # Mean reversion probability
            result_df[f'mean_reversion_prob_{window}'] = 1 / (1 + np.exp(-np.abs(result_df[f'zscore_{window}'])))
        
        # Ornstein-Uhlenbeck process features
        # Half-life of mean reversion
        def calculate_half_life(x):
            """Calculate half-life of mean reversion"""
            if len(x) < 10:
                return 0
            
            y = x.diff().dropna()
            x_lag = x.shift(1).dropna()
            
            if len(y) != len(x_lag):
                return 0
            
            try:
                slope, _, _, _, _ = stats.linregress(x_lag, y)
                half_life = -np.log(2) / slope if slope < 0 else 0
                return max(0, min(half_life, 100))  # Cap at reasonable values
            except:
                return 0
        
        result_df['half_life'] = result_df['Close'].rolling(50).apply(
            lambda x: calculate_half_life(x)
        )
        
        return result_df
    
    @staticmethod
    def create_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        result_df = df.copy()
        
        # Price impact measures
        result_df['price_efficiency'] = result_df['Close'].pct_change().rolling(10).apply(
            lambda x: np.abs(x.autocorr(lag=1)) if len(x.dropna()) > 1 else 0
        )
        
        # Tick size effects (assuming minimum tick is 0.01)
        tick_size = 0.01
        result_df['ticks_from_round'] = (result_df['Close'] % (tick_size * 5)) / tick_size
        result_df['price_clustering'] = (result_df['Close'] % (tick_size * 10) == 0).astype(int)
        
        # Bid-ask spread proxy (using high-low range)
        result_df['implied_spread'] = (result_df['High'] - result_df['Low']) / result_df['Close']
        result_df['spread_volatility'] = result_df['implied_spread'].rolling(20).std()
        
        # Kyle's Lambda (price impact measure)
        if 'Volume' in result_df.columns:
            returns = result_df['Close'].pct_change()
            result_df['kyle_lambda'] = returns.rolling(20).apply(
                lambda x: np.abs(x.corr(result_df['Volume'].loc[x.index])) if len(x) > 10 else 0
            )
        else:
            result_df['kyle_lambda'] = 0
        
        return result_df
    
    @staticmethod
    def create_behavioral_finance_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral finance inspired features"""
        result_df = df.copy()
        
        # Anchoring bias proxy
        result_df['distance_from_52w_high'] = (
            result_df['High'].rolling(252).max() - result_df['Close']
        ) / result_df['High'].rolling(252).max()
        
        result_df['distance_from_52w_low'] = (
            result_df['Close'] - result_df['Low'].rolling(252).min()
        ) / result_df['Low'].rolling(252).min()
        
        # Momentum and reversal effects
        # Short-term reversal (1-week)
        result_df['short_reversal'] = -result_df['Close'].pct_change(5)
        
        # Medium-term momentum (1-month to 3-month)
        result_df['medium_momentum'] = result_df['Close'].pct_change(60) - result_df['Close'].pct_change(20)
        
        # Long-term reversal (1-year)
        result_df['long_reversal'] = -result_df['Close'].pct_change(252)
        
        # Overreaction measures
        result_df['overreaction_indicator'] = (
            np.abs(result_df['Close'].pct_change()) > 
            result_df['Close'].pct_change().rolling(50).std() * 2
        ).astype(int)
        
        return result_df
    
    @staticmethod
    def create_cross_asset_features(df: pd.DataFrame, market_data: Optional[Dict] = None) -> pd.DataFrame:
        """Create cross-asset and correlation features"""
        result_df = df.copy()
        
        # If market data is provided, calculate correlations
        if market_data:
            for asset_name, asset_data in market_data.items():
                if len(asset_data) == len(result_df):
                    # Rolling correlation
                    result_df[f'corr_{asset_name}'] = result_df['Close'].rolling(20).corr(
                        pd.Series(asset_data, index=result_df.index)
                    )
                    
                    # Beta calculation
                    returns = result_df['Close'].pct_change()
                    market_returns = pd.Series(asset_data, index=result_df.index).pct_change()
                    
                    result_df[f'beta_{asset_name}'] = returns.rolling(50).apply(
                        lambda x: np.cov(x, market_returns.loc[x.index])[0, 1] / 
                                 np.var(market_returns.loc[x.index]) if len(x) > 10 else 1.0
                    )
        else:
            # Create synthetic market features
            result_df['synthetic_market_corr'] = result_df['Close'].pct_change().rolling(20).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )
        
        return result_df
    
    @staticmethod
    def create_machine_learning_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically designed for ML models"""
        result_df = df.copy()
        
        # Interaction features
        result_df['price_volume_interaction'] = result_df['Close'] * result_df.get('Volume', 1)
        result_df['high_low_range_interaction'] = (result_df['High'] - result_df['Low']) * result_df['Close']
        
        # Polynomial features (degree 2)
        returns = result_df['Close'].pct_change()
        result_df['returns_squared'] = returns ** 2
        result_df['returns_cubed'] = returns ** 3
        
        # Rolling rank features
        for window in [10, 20, 50]:
            result_df[f'price_rank_{window}'] = result_df['Close'].rolling(window).rank(pct=True)
            result_df[f'volume_rank_{window}'] = result_df.get('Volume', pd.Series(1, index=result_df.index)).rolling(window).rank(pct=True)
        
        # Quantile features
        for quantile in [0.1, 0.25, 0.75, 0.9]:
            for window in [20, 50]:
                result_df[f'price_quantile_{int(quantile*100)}_{window}'] = (
                    result_df['Close'] <= result_df['Close'].rolling(window).quantile(quantile)
                ).astype(int)
        
        return result_df
    
    @staticmethod
    def create_advanced_ta_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical analysis features using ta, pandas-ta, and finta"""
        result_df = df.copy()
        
        try:
            # Using ta library
            # Momentum indicators
            result_df['ta_rsi'] = ta.momentum.rsi(result_df['Close'])
            result_df['ta_stoch'] = ta.momentum.stoch(result_df['High'], result_df['Low'], result_df['Close'])
            result_df['ta_williams_r'] = ta.momentum.williams_r(result_df['High'], result_df['Low'], result_df['Close'])
            
            # Volatility indicators
            result_df['ta_atr'] = ta.volatility.average_true_range(result_df['High'], result_df['Low'], result_df['Close'])
            result_df['ta_keltner_upper'] = ta.volatility.keltner_channel_hband(result_df['High'], result_df['Low'], result_df['Close'])
            result_df['ta_keltner_lower'] = ta.volatility.keltner_channel_lband(result_df['High'], result_df['Low'], result_df['Close'])
            
            # Volume indicators (if volume available)
            if 'Volume' in result_df.columns:
                result_df['ta_obv'] = ta.volume.on_balance_volume(result_df['Close'], result_df['Volume'])
                result_df['ta_cmf'] = ta.volume.chaikin_money_flow(result_df['High'], result_df['Low'], result_df['Close'], result_df['Volume'])
                result_df['ta_vpt'] = ta.volume.volume_price_trend(result_df['Close'], result_df['Volume'])
            
            # Using pandas-ta
            # Create a temporary DataFrame for pandas-ta
            temp_df = pd.DataFrame({
                'open': result_df['Open'],
                'high': result_df['High'],
                'low': result_df['Low'],
                'close': result_df['Close'],
                'volume': result_df.get('Volume', 1)
            })
            
            # Add pandas-ta indicators
            result_df['pta_supertrend'] = pta.supertrend(temp_df['high'], temp_df['low'], temp_df['close'])['SUPERT_7_3.0']
            result_df['pta_vwap'] = pta.vwap(temp_df['high'], temp_df['low'], temp_df['close'], temp_df['volume'])
            result_df['pta_squeeze'] = pta.squeeze(temp_df['high'], temp_df['low'], temp_df['close'])['SQZ_20_2.0_20_1.5']
            
            # Using finta
            # Convert to required format for finta
            finta_df = pd.DataFrame({
                'open': result_df['Open'],
                'high': result_df['High'],
                'low': result_df['Low'],
                'close': result_df['Close'],
                'volume': result_df.get('Volume', 1)
            })
            
            # Finta indicators
            result_df['finta_ichimoku_a'] = TA.ICHIMOKU(finta_df)['SENKOU']
            result_df['finta_vama'] = TA.VAMA(finta_df)
            result_df['finta_smma'] = TA.SMMA(finta_df)
            
        except Exception as e:
            print(f"Warning: Some advanced TA features could not be calculated: {e}")
            # Fill with zeros if calculation fails
            advanced_ta_cols = [
                'ta_rsi', 'ta_stoch', 'ta_williams_r', 'ta_atr', 'ta_keltner_upper', 'ta_keltner_lower',
                'ta_obv', 'ta_cmf', 'ta_vpt', 'pta_supertrend', 'pta_vwap', 'pta_squeeze',
                'finta_ichimoku_a', 'finta_vama', 'finta_smma'
            ]
            for col in advanced_ta_cols:
                if col not in result_df.columns:
                    result_df[col] = 0
        
        return result_df
    
    @staticmethod
    def generate_all_custom_features(df: pd.DataFrame, 
                                   market_data: Optional[Dict] = None,
                                   include_advanced_ta: bool = True) -> pd.DataFrame:
        """Generate all custom engineered features"""
        result_df = df.copy()
        
        # Apply all custom feature engineering methods
        result_df = CustomEngineeredFeatures.create_fractal_features(result_df)
        result_df = CustomEngineeredFeatures.create_regime_detection_features(result_df)
        result_df = CustomEngineeredFeatures.create_entropy_features(result_df)
        result_df = CustomEngineeredFeatures.create_fourier_features(result_df)
        result_df = CustomEngineeredFeatures.create_wavelet_features(result_df)
        result_df = CustomEngineeredFeatures.create_statistical_arbitrage_features(result_df)
        result_df = CustomEngineeredFeatures.create_microstructure_features(result_df)
        result_df = CustomEngineeredFeatures.create_behavioral_finance_features(result_df)
        result_df = CustomEngineeredFeatures.create_cross_asset_features(result_df, market_data)
        result_df = CustomEngineeredFeatures.create_machine_learning_features(result_df)
        
        if include_advanced_ta:
            result_df = CustomEngineeredFeatures.create_advanced_ta_features(result_df)
        
        return result_df
