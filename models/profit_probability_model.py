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
        returns = df['Close'].pct_change().dropna()

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

        # Debug information
        profit_prob_stats = target.value_counts()
        print(f"Profit Probability Target Distribution: {profit_prob_stats.to_dict()}")
        print(f"Profit threshold used: {profit_threshold:.4f}")

        return target

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for profit probability model."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators

        # Calculate all profit probability-specific indicators
        result_df = ProfitProbabilityTechnicalIndicators.calculate_all_profit_probability_indicators(df)

        # Get all non-OHLC features, but exclude non-numeric columns
        excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp', 'date', 'Timestamp', 'Date', 'DateTime']

        # First pass: exclude known non-numeric columns
        feature_columns = [col for col in result_df.columns if col not in excluded_cols]

        # Second pass: check each remaining column for actual numeric content
        numeric_columns = []
        for col in feature_columns:
            try:
                # Check if column is numeric and not a datetime type
                if pd.api.types.is_numeric_dtype(result_df[col]) and not pd.api.types.is_datetime64_any_dtype(result_df[col]):
                    # Additional check: ensure all values can be converted to float
                    test_values = result_df[col].dropna().head(10)
                    if len(test_values) > 0:
                        pd.to_numeric(test_values, errors='raise')
                    numeric_columns.append(col)
                else:
                    print(f"Excluding non-numeric or datetime column: {col}")
            except (ValueError, TypeError) as e:
                print(f"Excluding column {col} due to conversion error: {e}")

        feature_columns = numeric_columns

        if len(feature_columns) == 0:
            raise ValueError(f"No numeric features found. Available columns: {list(result_df.columns)}")

        # Select only numeric features and remove NaN
        result_df = result_df[feature_columns].dropna()

        if result_df.empty:
            raise ValueError("DataFrame is empty after removing NaN values")

        print(f"Profit probability model using {len(feature_columns)} features: {feature_columns}")

        self.feature_names = feature_columns
        return result_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8) -> Dict[str, Any]:
        """Train profit probability prediction model."""
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

        result = {
            'model': self.model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_indices': X_test.index,
            'feature_count': len(self.feature_names),
            'scaler': self.scaler
        }
        
        # Auto-save to database
        self._save_model_to_database(result)
        
        return result

    def _save_model_to_database(self, training_result):
        """Save trained profit probability model to database for persistence."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()
            
            models_to_save = {
                'profit_probability': {
                    'ensemble': training_result['model'],
                    'scaler': training_result['scaler'],
                    'feature_names': training_result['feature_names'],
                    'task_type': training_result['task_type'],
                    'metrics': training_result['metrics'],
                    'feature_importance': training_result['feature_importance']
                }
            }
            
            success = db.save_trained_models(models_to_save)
            if success:
                print("✅ Profit probability model auto-saved to database")
            else:
                print("❌ Failed to auto-save profit probability model to database")
                
        except Exception as e:
            print(f"❌ Error auto-saving profit probability model: {str(e)}")

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