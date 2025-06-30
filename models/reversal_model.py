
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
        """Prepare comprehensive reversal features including technical indicators, custom features, lagged features, and time context."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        print("🔧 Calculating comprehensive reversal features...")
        
        # Start with a copy of the input data
        result_df = df.copy()
        
        # Step 1: Calculate reversal technical indicators
        print("  - Computing reversal technical indicators...")
        from features.reversal_technical_indicators import ReversalTechnicalIndicators
        result_df = ReversalTechnicalIndicators.calculate_reversal_indicators(result_df)
        
        # Step 2: Add custom reversal features
        print("  - Adding custom reversal features...")
        from features.reversal_custom_engineered import add_custom_reversal_features
        result_df = add_custom_reversal_features(result_df)
        
        # Step 3: Add lagged reversal features
        print("  - Adding lagged reversal features...")
        from features.reversal_lagged_features import add_lagged_reversal_features
        result_df = add_lagged_reversal_features(result_df)
        
        # Step 4: Add time context features
        print("  - Adding time context features...")
        from features.reversal_time_context import add_time_context_features_reversal
        result_df = add_time_context_features_reversal(result_df)
        
        # Step 5: Select only feature columns (exclude OHLC columns)
        ohlc_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in result_df.columns if col not in ohlc_columns]
        
        if len(feature_columns) == 0:
            raise ValueError(f"No reversal features were generated. Available columns: {list(result_df.columns)}")
        
        # Select only feature columns
        features_df = result_df[feature_columns].copy()
        
        # Step 6: Handle missing values
        print(f"  - Cleaning {len(feature_columns)} features...")
        
        # Replace infinite values with NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill
        features_df = features_df.ffill().bfill()
        
        # Fill remaining NaN with appropriate neutral values
        for col in features_df.columns:
            if features_df[col].isna().any():
                if any(term in col.lower() for term in ['rsi', 'stochastic']):
                    features_df[col] = features_df[col].fillna(50)  # Neutral RSI/Stochastic
                elif 'williams' in col.lower():
                    features_df[col] = features_df[col].fillna(-50)  # Neutral Williams %R
                elif 'cci' in col.lower():
                    features_df[col] = features_df[col].fillna(0)  # Neutral CCI
                elif 'macd' in col.lower():
                    features_df[col] = features_df[col].fillna(0)  # Neutral MACD
                elif any(term in col.lower() for term in ['hit', 'flag', 'signal']):
                    features_df[col] = features_df[col].fillna(0)  # No signals by default
                elif any(term in col.lower() for term in ['ratio', 'percent']):
                    features_df[col] = features_df[col].fillna(0.5)  # Neutral ratio
                else:
                    features_df[col] = features_df[col].fillna(0)  # Default to 0
        
        # Final cleanup - remove any rows that still have NaN
        initial_rows = len(features_df)
        features_df = features_df.dropna()
        final_rows = len(features_df)
        
        if final_rows < initial_rows:
            print(f"  - Removed {initial_rows - final_rows} rows with missing values")
        
        if features_df.empty:
            raise ValueError("All data was removed during cleaning. Check your input data quality.")
        
        # Store feature names for later use
        self.feature_names = list(features_df.columns)
        
        print(f"✅ Generated {len(self.feature_names)} reversal features: {self.feature_names}")
        print(f"✅ Final feature dataset: {features_df.shape[0]} samples × {features_df.shape[1]} features")
        
        return features_df

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
