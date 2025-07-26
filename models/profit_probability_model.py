import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from typing import Dict, Tuple, Any

class ProfitProbabilityModel:
    """Profit probability prediction model for predicting likelihood of profit."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.task_type = 'classification'
        self.model_name = 'profit_prob'

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create profit probability target based on next 5 periods."""
        # More realistic profit threshold for 5-min scalping
        base_profit_threshold = 0.001  # 0.1% minimum profit target

        # Look ahead only 5 candles (25 minutes for 5-min data)
        future_returns_list = []
        for i in range(5):
            future_return = df['Close'].shift(-i-1) / df['Close'] - 1
            future_returns_list.append(future_return)

        # Get maximum return within 5 periods
        future_returns_df = pd.concat(future_returns_list, axis=1)
        max_future_return = future_returns_df.max(axis=1)

        # Use adaptive threshold based on actual data distribution
        profit_threshold = np.percentile(max_future_return.dropna(), 65)  # Top 35% as profit opportunities
        profit_threshold = max(profit_threshold, base_profit_threshold)

        target = (max_future_return > profit_threshold).astype(int)

        # Ensure target has the same index as the input dataframe
        target.index = df.index

        # Debug information
        profit_prob_stats = target.value_counts()
        print(f"Profit Probability Target Distribution: {profit_prob_stats.to_dict()}")
        print(f"Profit threshold used: {profit_threshold:.4f}")
        print(f"Target index range: {target.index.min()} to {target.index.max()}")

        return target

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train profit probability model from raw OHLC data"""
        try:
            print(f"🚀 Training profit probability model with {len(data)} data points")
            
            # Prepare features and target
            features_df = self.prepare_features(data)
            target = self.create_target(data)
            
            # Train the model
            result = self.train(features_df, target)
            
            # Save model to database
            from utils.database_adapter import DatabaseAdapter
            db = DatabaseAdapter()
            db.save_trained_model('profit_probability', self.model, self.scaler, self.feature_names)
            
            return {
                'success': True,
                'model_type': 'profit_probability',
                'accuracy': result.get('test_accuracy', 0.0),
                'training_samples': len(features_df),
                'message': 'Profit probability model trained successfully'
            }
            
        except Exception as e:
            print(f"Error training profit probability model: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_type': 'profit_probability'
            }

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive profit probability features including technical indicators, custom features, lagged features, and time context."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        print("🔧 Calculating comprehensive profit probability features...")
        
        # Start with a copy of the input data
        result_df = df.copy()
        
        # Step 1: Calculate profit probability technical indicators
        print("  - Computing profit probability technical indicators...")
        from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
        calc = ProfitProbabilityTechnicalIndicators()
        result_df = calc.calculate_profit_probability_features(result_df)
        
        # Step 2: Add custom profit probability features
        print("  - Adding custom profit probability features...")
        from features.profit_probability_custom_engineered import add_custom_profit_features
        result_df = add_custom_profit_features(result_df)
        
        # Step 3: Add lagged profit probability features
        print("  - Adding lagged profit probability features...")
        from features.profit_probability_lagged_features import add_lagged_features_profit_prob
        result_df = add_lagged_features_profit_prob(result_df)
        
        # Step 4: Add time context features
        print("  - Adding time context features...")
        from features.profit_probability_time_context import add_time_context_features_profit_prob
        result_df = add_time_context_features_profit_prob(result_df)

        # Step 5: Select only feature columns (exclude OHLC columns and non-numeric columns)
        ohlc_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'open', 'high', 'low', 'close', 'volume']
        exclude_columns = ohlc_columns + ['date']  # Exclude datetime columns
        feature_columns = [col for col in result_df.columns if col not in exclude_columns]
        
        if len(feature_columns) == 0:
            raise ValueError(f"No profit probability features were generated. Available columns: {list(result_df.columns)}")
        
        # Select only feature columns
        features_df = result_df[feature_columns].copy()
        
        # Step 6: Handle missing values  
        print(f"  - Cleaning {len(feature_columns)} features...")
        print(f"  - Feature columns before cleaning: {feature_columns[:10]}...")
        
        # Replace infinite values with NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Count initial NaN values
        initial_nan_count = features_df.isna().sum().sum()
        print(f"  - Initial NaN values: {initial_nan_count}")
        
        # Forward fill then backward fill
        features_df = features_df.ffill().bfill()
        
        # Fill remaining NaN with appropriate neutral values
        for col in features_df.columns:
            if features_df[col].isna().any():
                # Use median for most features, 0 for binary features, 50 for RSI-like features
                if 'rsi' in col.lower() or 'stoch' in col.lower():
                    features_df[col] = features_df[col].fillna(50)
                elif any(x in col.lower() for x in ['_above_', '_below_', '_bullish', '_bearish', '_up', '_down']):
                    features_df[col] = features_df[col].fillna(0)
                else:
                    median_val = features_df[col].median()
                    if pd.isna(median_val):
                        features_df[col] = features_df[col].fillna(0)  # Use 0 if median is NaN
                    else:
                        features_df[col] = features_df[col].fillna(median_val)
        
        # Ensure all features are numeric
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Only remove rows that are completely NaN (keep rows with some valid features)
        features_df = features_df.dropna(how='all')
        
        # If still all features are NaN, fill with zeros as fallback
        if features_df.isna().all().any():
            features_df = features_df.fillna(0)
        
        print(f"  - Features after cleaning: {len(features_df)} rows × {len(features_df.columns)} columns")
        
        if features_df.empty:
            raise ValueError("No valid profit probability features remain after cleaning")
        
        print(f"✅ Generated {len(feature_columns)} profit probability features: {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
        print(f"✅ Final feature dataset: {len(features_df)} samples × {len(feature_columns)} features")
        
        # Ensure features_df has the same index as input df for proper alignment with targets
        # Keep the original datetime index from input data
        if len(features_df) == len(df):
            features_df.index = df.index
        else:
            # If lengths differ due to feature engineering, use the last N indices from original df
            features_df.index = df.index[-len(features_df):]
        
        self.feature_names = feature_columns
        return features_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8) -> Dict[str, Any]:
        """Train profit probability prediction model."""
        print(f"Training data input: X shape={X.shape}, y shape={y.shape}")
        print(f"X index range: {X.index.min()} to {X.index.max()}")
        print(f"y index range: {y.index.min()} to {y.index.max()}")
        
        # Align data
        common_index = X.index.intersection(y.index)
        print(f"Common index size: {len(common_index)}")
        
        if len(common_index) == 0:
            raise ValueError("No common indices between features and targets")
        
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        print(f"After alignment: X shape={X_aligned.shape}, y shape={y_aligned.shape}")

        # Check target distribution before cleaning
        print(f"Target distribution before cleaning: {y_aligned.value_counts().to_dict()}")

        # Clean data
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[mask]
        y_clean = y_aligned[mask]
        print(f"After NaN removal: X shape={X_clean.shape}, y shape={y_clean.shape}")

        # Remove invalid targets
        valid_targets = ~np.isinf(y_clean) & (y_clean >= 0)
        X_clean = X_clean[valid_targets]
        y_clean = y_clean[valid_targets]
        print(f"After invalid target removal: X shape={X_clean.shape}, y shape={y_clean.shape}")

        # Ensure we have at least 2 classes
        unique_targets = y_clean.unique()
        print(f"Final unique targets: {unique_targets}")
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

        # Final safety check: remove any datetime columns that might have slipped through
        datetime_cols = []
        for col in X_train.columns:
            if pd.api.types.is_datetime64_any_dtype(X_train[col]) or 'timestamp' in col.lower() or 'date' in col.lower():
                datetime_cols.append(col)

        if datetime_cols:
            print(f"Removing datetime columns before scaling: {datetime_cols}")
            X_train = X_train.drop(datetime_cols, axis=1)
            X_test = X_test.drop(datetime_cols, axis=1)
            # Update feature names - preserve the existing feature names from prepare_features
            self.feature_names = [fn for fn in self.feature_names if fn not in datetime_cols]
        
        # Ensure feature names match the final training columns
        if len(self.feature_names) == 0:
            self.feature_names = list(X_train.columns)
            print(f"Feature names were empty, setting to training columns: {len(self.feature_names)} features")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build ensemble
        random_state = 42

        xgb_model = xgb.XGBClassifier(
            max_depth=8,
            learning_rate=0.05,
            n_estimators=200,
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
            iterations=200,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            border_count=128,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False
        )

        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
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

        # Store exact feature names for consistency - this locks in the 61 features
        # self.feature_names was already set in prepare_features method

        print(f"✅ Profit probability model trained successfully")
        print(f"Exact features locked in: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")
        print(f"Model performance - Accuracy: {accuracy:.3f}")

        return {
            'model': self.model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_indices': X_test.index,
            'feature_count': len(self.feature_names)
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained profit probability model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if X.empty:
            raise ValueError("Input DataFrame is empty")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities