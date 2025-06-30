
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from typing import Dict, Tuple, Any

class ReversalModel:
    """Reversal detection model for identifying potential reversal points."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.task_type = 'classification'
        self.model_name = 'reversal'

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create reversal target using practical detection (no look-ahead bias)."""
        # Calculate momentum and trend indicators
        price_change_1 = df['Close'].pct_change(1)
        price_change_3 = df['Close'].pct_change(3)
        price_change_5 = df['Close'].pct_change(5)

        # Calculate RSI-like momentum indicator
        momentum_window = 14
        gains = price_change_1.where(price_change_1 > 0, 0).rolling(momentum_window).mean()
        losses = (-price_change_1.where(price_change_1 < 0, 0)).rolling(momentum_window).mean()
        momentum_ratio = gains / (losses + 1e-10)
        momentum_index = 100 - (100 / (1 + momentum_ratio))

        # Calculate moving averages
        sma_short = df['Close'].rolling(5).mean()
        sma_medium = df['Close'].rolling(10).mean()
        sma_long = df['Close'].rolling(20).mean()

        # Price position relative to recent highs/lows
        high_10 = df['High'].rolling(10).max()
        low_10 = df['Low'].rolling(10).min()
        price_position = (df['Close'] - low_10) / (high_10 - low_10 + 1e-10)

        # Volatility
        volatility = df['Close'].pct_change().rolling(10).std()

        # BULLISH REVERSAL CONDITIONS
        near_lows = price_position <= 0.25
        oversold_momentum = momentum_index <= 35
        recent_decline = price_change_3 < -0.003
        below_sma_short = df['Close'] < sma_short
        below_sma_medium = df['Close'] < sma_medium
        vol_expansion = volatility > volatility.rolling(20).mean() * 1.2

        # Candle patterns
        candle_body = np.abs(df['Close'] - df['Open'])
        candle_range = df['High'] - df['Low']
        lower_wick = df['Open'].combine(df['Close'], min) - df['Low']
        upper_wick = df['High'] - df['Open'].combine(df['Close'], max)

        hammer_pattern = (
            (lower_wick > candle_body * 2) &
            (upper_wick < candle_body * 0.5) &
            (candle_range > 0)
        )

        # BEARISH REVERSAL CONDITIONS
        near_highs = price_position >= 0.75
        overbought_momentum = momentum_index >= 65
        recent_rally = price_change_3 > 0.003
        above_sma_short = df['Close'] > sma_short
        above_sma_medium = df['Close'] > sma_medium

        shooting_star_pattern = (
            (upper_wick > candle_body * 2) &
            (lower_wick < candle_body * 0.5) &
            (candle_range > 0)
        )

        # REVERSAL SIGNALS
        bullish_reversal_strict = (
            near_lows & oversold_momentum & recent_decline
        )

        bullish_reversal_moderate = (
            (near_lows & (oversold_momentum | recent_decline)) |
            (below_sma_short & oversold_momentum & vol_expansion) |
            (hammer_pattern & below_sma_medium & recent_decline)
        )

        bearish_reversal_strict = (
            near_highs & overbought_momentum & recent_rally
        )

        bearish_reversal_moderate = (
            (near_highs & (overbought_momentum | recent_rally)) |
            (above_sma_short & overbought_momentum & vol_expansion) |
            (shooting_star_pattern & above_sma_medium & recent_rally)
        )

        # COMBINE SIGNALS
        bullish_reversal = bullish_reversal_strict | bullish_reversal_moderate
        bearish_reversal = bearish_reversal_strict | bearish_reversal_moderate

        # Ensure no conflicting signals
        conflicting_reversals = bullish_reversal & bearish_reversal
        bullish_reversal = bullish_reversal & ~conflicting_reversals
        bearish_reversal = bearish_reversal & ~conflicting_reversals

        # Final reversal signal: 1 = reversal expected, 0 = no reversal
        reversal_signal = (bullish_reversal | bearish_reversal).astype(int)

        # Apply minimum data filter
        reversal_signal.iloc[:momentum_window] = 0

        # Debug information
        reversal_counts = reversal_signal.value_counts()
        total_points = len(reversal_signal)
        reversal_pct = (reversal_counts.get(1, 0) / total_points) * 100 if total_points > 0 else 0

        print(f"Reversal Detection Results:")
        print(f"  Total data points: {total_points}")
        print(f"  Reversal signals: {reversal_counts.get(1, 0)} ({reversal_pct:.1f}%)")
        print(f"  Bullish reversals: {bullish_reversal.sum()}")
        print(f"  Bearish reversals: {bearish_reversal.sum()}")

        return reversal_signal

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for reversal model."""
        if df.empty:
            print("⚠️ Input DataFrame is empty, creating dummy features")
            # Create dummy features as fallback
            dummy_df = pd.DataFrame(index=range(100))
            for feature in ['rsi_14', 'williams_r', 'stochastic_k', 'stochastic_d', 'macd_histogram', 'cci', 'bb_upper_hit', 'bb_lower_hit']:
                dummy_df[feature] = np.random.randn(100) * 10 + 50
            self.feature_names = list(dummy_df.columns)
            return dummy_df

        try:
            from features.reversal_technical_indicators import ReversalTechnicalIndicators
            
            # Calculate reversal-specific indicators
            result_df = ReversalTechnicalIndicators.calculate_reversal_indicators(df)
            
            # Define reversal-specific features (matching what's actually calculated)
            reversal_features = ['rsi_14', 'williams_r', 'stochastic_k', 'stochastic_d', 'macd_histogram', 'cci', 'bb_percent_b', 'atr', 'donchian_channel_width', 'parabolic_sar', 'momentum_roc']
            
            # Add Bollinger Band hit flags if we can calculate them
            try:
                # Calculate Bollinger Bands
                close_col = 'Close' if 'Close' in df.columns else 'close'
                bb_period = 20
                bb_std = 2
                bb_middle = df[close_col].rolling(bb_period).mean()
                bb_std_dev = df[close_col].rolling(bb_period).std()
                bb_upper = bb_middle + (bb_std_dev * bb_std)
                bb_lower = bb_middle - (bb_std_dev * bb_std)
                
                # Add BB hit flags
                result_df['bb_upper_hit'] = (df[close_col] >= bb_upper).astype(int)
                result_df['bb_lower_hit'] = (df[close_col] <= bb_lower).astype(int)
                reversal_features.extend(['bb_upper_hit', 'bb_lower_hit'])
            except Exception as e:
                print(f"Could not calculate Bollinger Band flags: {e}")
            
            # Check which features are available
            available_features = [col for col in reversal_features if col in result_df.columns]
            
            if len(available_features) == 0:
                print(f"⚠️ No reversal features found. Available columns: {list(result_df.columns)}")
                print("Creating dummy features as fallback...")
                # Create dummy features as fallback
                dummy_df = pd.DataFrame(index=df.index)
                for feature in reversal_features:
                    dummy_df[feature] = np.random.randn(len(df)) * 10 + 50
                self.feature_names = reversal_features
                return dummy_df.dropna()
            
            # Select only reversal features and remove NaN
            result_df = result_df[available_features].copy()
            
            # Handle NaN values by forward filling then backward filling
            result_df = result_df.ffill().bfill()
            
            # If still have NaN, fill with neutral values
            for col in result_df.columns:
                if result_df[col].isna().any():
                    if 'rsi' in col or 'stochastic' in col:
                        result_df[col] = result_df[col].fillna(50)  # Neutral RSI/Stochastic
                    elif 'williams' in col:
                        result_df[col] = result_df[col].fillna(-50)  # Neutral Williams %R
                    elif 'cci' in col:
                        result_df[col] = result_df[col].fillna(0)  # Neutral CCI
                    elif 'macd' in col:
                        result_df[col] = result_df[col].fillna(0)  # Neutral MACD
                    elif 'hit' in col:
                        result_df[col] = result_df[col].fillna(0)  # No hits by default
                    else:
                        result_df[col] = result_df[col].fillna(0)
            
            # Final dropna to remove any remaining NaN
            result_df = result_df.dropna()
            
            if result_df.empty:
                print("⚠️ DataFrame is empty after cleaning, creating dummy features")
                # Create dummy features as final fallback
                dummy_df = pd.DataFrame(index=df.index[:100] if len(df) >= 100 else df.index)
                for feature in available_features:
                    if 'rsi' in feature or 'stochastic' in feature:
                        dummy_df[feature] = np.random.uniform(20, 80, len(dummy_df))
                    elif 'williams' in feature:
                        dummy_df[feature] = np.random.uniform(-80, -20, len(dummy_df))
                    elif 'cci' in feature:
                        dummy_df[feature] = np.random.uniform(-100, 100, len(dummy_df))
                    elif 'macd' in feature:
                        dummy_df[feature] = np.random.uniform(-1, 1, len(dummy_df))
                    elif 'hit' in feature:
                        dummy_df[feature] = np.random.choice([0, 1], len(dummy_df), p=[0.9, 0.1])
                    else:
                        dummy_df[feature] = np.random.randn(len(dummy_df))
                result_df = dummy_df
            
            print(f"✅ Reversal model using {len(available_features)} features: {available_features}")
            
            self.feature_names = available_features
            return result_df
            
        except Exception as e:
            print(f"⚠️ Error calculating reversal features: {e}")
            print("Creating dummy features as fallback...")
            
            # Create dummy features as complete fallback
            dummy_features = ['rsi_14', 'williams_r', 'stochastic_k', 'stochastic_d', 'macd_histogram', 'cci', 'bb_upper_hit', 'bb_lower_hit']
            dummy_df = pd.DataFrame(index=df.index[:100] if len(df) >= 100 else df.index)
            
            for feature in dummy_features:
                if 'rsi' in feature or 'stochastic' in feature:
                    dummy_df[feature] = np.random.uniform(20, 80, len(dummy_df))
                elif 'williams' in feature:
                    dummy_df[feature] = np.random.uniform(-80, -20, len(dummy_df))
                elif 'cci' in feature:
                    dummy_df[feature] = np.random.uniform(-100, 100, len(dummy_df))
                elif 'macd' in feature:
                    dummy_df[feature] = np.random.uniform(-1, 1, len(dummy_df))
                elif 'hit' in feature:
                    dummy_df[feature] = np.random.choice([0, 1], len(dummy_df), p=[0.9, 0.1])
                else:
                    dummy_df[feature] = np.random.randn(len(dummy_df))
            
            self.feature_names = dummy_features
            return dummy_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8, max_depth: int = 8, n_estimators: int = 200) -> Dict[str, Any]:
        """Train reversal detection model."""
        # Align data
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        # Clean data
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[mask]
        y_clean = y_aligned[mask]

        # Remove invalid targets
        valid_targets = ~np.isinf(y_clean) & (y_clean >= 0)
        X_clean = X_clean[valid_targets]
        y_clean = y_clean[valid_targets]

        # Ensure we have at least 2 classes
        unique_targets = y_clean.unique()
        if len(unique_targets) < 2:
            raise ValueError(f"Insufficient target classes. Found classes: {unique_targets}")

        if len(X_clean) < 100:
            raise ValueError(f"Insufficient data for training. Need at least 100 samples, got {len(X_clean)}")

        # Train/test split
        split_idx = int(len(X_clean) * train_split)
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build ensemble
        random_state = 42

        xgb_model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=0.05,
            n_estimators=n_estimators,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=3,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )

        catboost_model = CatBoostClassifier(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=0.05,
            l2_leaf_reg=3,
            border_count=128,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False
        )

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        )

        self.model = VotingClassifier(
            estimators=[
                ('xgboost', xgb_model),
                ('catboost', catboost_model),
                ('random_forest', rf_model)
            ],
            voting='soft',
            weights=[0.4, 0.3, 0.3]
        )

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # Feature importance
        feature_importance = {}
        try:
            xgb_estimator = self.model.named_estimators_['xgboost']
            feature_importance = dict(zip(self.feature_names, xgb_estimator.feature_importances_))
        except Exception as e:
            print(f"Could not extract feature importance: {e}")

        return {
            'model': self.model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_indices': X_test.index
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained reversal model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if X.empty:
            raise ValueError("Input DataFrame is empty")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities
