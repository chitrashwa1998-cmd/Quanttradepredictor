
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, VotingRegressor, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class UltimateModel:
    """Ultimate ensemble model that combines predictions from all individual models."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.task_type = 'classification'  # Can be 'classification' or 'regression'
        self.model_name = 'ultimate'
        
        # Store individual model predictions
        self.individual_models = {}
        self.prediction_history = pd.DataFrame()

    def prepare_meta_features(self, df: pd.DataFrame, individual_predictions: Dict[str, Any]) -> pd.DataFrame:
        """Prepare meta-features from individual model predictions and market data."""
        
        meta_features = pd.DataFrame(index=df.index)
        
        # 1. Individual Model Predictions
        if 'volatility' in individual_predictions:
            vol_pred, _ = individual_predictions['volatility']
            meta_features['volatility_pred'] = vol_pred
            meta_features['volatility_regime'] = pd.cut(vol_pred, 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']).cat.codes
        
        if 'direction' in individual_predictions:
            dir_pred, dir_prob = individual_predictions['direction']
            meta_features['direction_pred'] = dir_pred
            meta_features['direction_confidence'] = np.max(dir_prob, axis=1) if dir_prob is not None else 0.5
            meta_features['direction_prob_bullish'] = dir_prob[:, 1] if dir_prob is not None and dir_prob.shape[1] > 1 else 0.5
        
        if 'profit_prob' in individual_predictions:
            profit_pred, profit_prob = individual_predictions['profit_prob']
            meta_features['profit_pred'] = profit_pred
            meta_features['profit_confidence'] = np.max(profit_prob, axis=1) if profit_prob is not None else 0.5
        
        if 'reversal' in individual_predictions:
            rev_pred, rev_prob = individual_predictions['reversal']
            meta_features['reversal_pred'] = rev_pred
            meta_features['reversal_confidence'] = np.max(rev_prob, axis=1) if rev_prob is not None else 0.5
        
        if 'trend_sideways' in individual_predictions:
            trend_pred, trend_prob = individual_predictions['trend_sideways']
            meta_features['trend_pred'] = trend_pred
            meta_features['trend_confidence'] = np.max(trend_prob, axis=1) if trend_prob is not None else 0.5
        
        # 2. Model Agreement Features
        if 'direction_pred' in meta_features.columns and 'profit_pred' in meta_features.columns:
            meta_features['dir_profit_agreement'] = (meta_features['direction_pred'] == meta_features['profit_pred']).astype(int)
        
        if 'direction_pred' in meta_features.columns and 'trend_pred' in meta_features.columns:
            meta_features['dir_trend_agreement'] = (meta_features['direction_pred'] == meta_features['trend_pred']).astype(int)
        
        # 3. Confidence-based Features
        confidence_cols = [col for col in meta_features.columns if 'confidence' in col]
        if confidence_cols:
            meta_features['avg_confidence'] = meta_features[confidence_cols].mean(axis=1)
            meta_features['min_confidence'] = meta_features[confidence_cols].min(axis=1)
            meta_features['max_confidence'] = meta_features[confidence_cols].max(axis=1)
            meta_features['confidence_std'] = meta_features[confidence_cols].std(axis=1)
        
        # 4. Market Context Features
        meta_features['price'] = df['Close']
        meta_features['volume'] = df.get('Volume', 0)
        meta_features['price_change'] = df['Close'].pct_change()
        meta_features['price_volatility'] = df['Close'].pct_change().rolling(10).std()
        
        # 5. Time-based Features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            meta_features['hour'] = df['timestamp'].dt.hour
            meta_features['day_of_week'] = df['timestamp'].dt.dayofweek
            meta_features['is_market_open'] = ((df['timestamp'].dt.hour >= 9) & (df['timestamp'].dt.hour <= 16)).astype(int)
        
        # 6. Weighted Prediction Features
        if 'volatility_pred' in meta_features.columns and 'direction_confidence' in meta_features.columns:
            meta_features['vol_weighted_direction'] = meta_features['volatility_pred'] * meta_features['direction_confidence']
        
        # Remove NaN values
        meta_features = meta_features.fillna(0)
        
        self.feature_names = list(meta_features.columns)
        print(f"Ultimate model using {len(self.feature_names)} meta-features: {self.feature_names}")
        
        return meta_features

    def create_target(self, df: pd.DataFrame, target_type: str = 'direction') -> pd.Series:
        """Create target variable for the ultimate model."""
        
        if target_type == 'direction':
            # Predict next period direction
            future_return = df['Close'].shift(-1) / df['Close'] - 1
            target = (future_return > 0).astype(int)
            self.task_type = 'classification'
            
        elif target_type == 'return':
            # Predict next period return
            target = df['Close'].shift(-1) / df['Close'] - 1
            self.task_type = 'regression'
            
        elif target_type == 'volatility':
            # Predict next period volatility
            returns = df['Close'].pct_change()
            target = returns.rolling(5).std().shift(-1)
            self.task_type = 'regression'
            
        elif target_type == 'profit_signal':
            # Complex profit signal based on multiple criteria
            future_return = df['Close'].shift(-1) / df['Close'] - 1
            future_high = df['High'].shift(-1) / df['Close'] - 1
            
            # Profitable if we can capture at least 0.1% return
            profit_threshold = 0.001
            target = ((future_return > profit_threshold) | (future_high > profit_threshold)).astype(int)
            self.task_type = 'classification'
            
        else:
            # Default to direction prediction
            future_return = df['Close'].shift(-1) / df['Close'] - 1
            target = (future_return > 0).astype(int)
            self.task_type = 'classification'
        
        # Remove NaN values
        target = target.dropna()
        
        print(f"Ultimate model target ({target_type}): {len(target)} samples")
        if self.task_type == 'classification':
            print(f"Target distribution: {target.value_counts().to_dict()}")
        else:
            print(f"Target statistics: min={target.min():.4f}, max={target.max():.4f}, mean={target.mean():.4f}")
        
        return target

    def train(self, meta_features: pd.DataFrame, target: pd.Series, train_split: float = 0.8) -> Dict[str, Any]:
        """Train the ultimate ensemble model."""
        
        # Align data
        common_index = meta_features.index.intersection(target.index)
        X_aligned = meta_features.loc[common_index]
        y_aligned = target.loc[common_index]
        
        # Clean data
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[mask]
        y_clean = y_aligned[mask]
        
        if self.task_type == 'regression':
            # Remove invalid targets for regression
            valid_targets = np.isfinite(y_clean)
            X_clean = X_clean[valid_targets]
            y_clean = y_clean[valid_targets]
        else:
            # Remove invalid targets for classification
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
        
        print(f"Ultimate model training on {len(X_train)} samples with {X_train.shape[1]} features")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build ensemble model based on task type
        random_state = 42
        
        if self.task_type == 'classification':
            # Classification ensemble
            xgb_model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            catboost_model = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False
            )
            
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
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
            
        else:
            # Regression ensemble
            xgb_model = xgb.XGBRegressor(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1
            )
            
            catboost_model = CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False
            )
            
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=random_state,
                n_jobs=-1
            )
            
            self.model = VotingRegressor(
                estimators=[
                    ('xgboost', xgb_model),
                    ('catboost', catboost_model),
                    ('random_forest', rf_model)
                ]
            )
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        if self.task_type == 'classification':
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"Ultimate model accuracy: {accuracy:.4f}")
            
        else:
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = None
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = np.mean(np.abs(y_test - y_pred))
            
            metrics = {
                'rmse': rmse,
                'mae': mae
            }
            
            print(f"Ultimate model RMSE: {rmse:.6f}, MAE: {mae:.6f}")
        
        # Feature importance
        feature_importance = {}
        try:
            if hasattr(self.model, 'named_estimators_'):
                xgb_estimator = self.model.named_estimators_['xgboost']
                feature_importance = dict(zip(self.feature_names, xgb_estimator.feature_importances_))
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_indices': X_test.index
        }

    def predict(self, meta_features: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the trained ultimate model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if meta_features.empty:
            raise ValueError("Input DataFrame is empty")
        
        X_scaled = self.scaler.transform(meta_features)
        predictions = self.model.predict(X_scaled)
        
        if self.task_type == 'classification':
            probabilities = self.model.predict_proba(X_scaled)
        else:
            probabilities = None
        
        return predictions, probabilities

    def create_unified_prediction_table(self, df: pd.DataFrame, individual_predictions: Dict[str, Any], 
                                      ultimate_predictions: Tuple[np.ndarray, Optional[np.ndarray]]) -> pd.DataFrame:
        """Create a unified data table with all model predictions."""
        
        ultimate_pred, ultimate_prob = ultimate_predictions
        
        # Start with base market data
        unified_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Add individual model predictions
        if 'volatility' in individual_predictions:
            vol_pred, _ = individual_predictions['volatility']
            unified_df['Volatility_Pred'] = vol_pred
            unified_df['Vol_Regime'] = pd.cut(vol_pred, 5, labels=['Very Low', 'Low', 'Med', 'High', 'Very High'])
        
        if 'direction' in individual_predictions:
            dir_pred, dir_prob = individual_predictions['direction']
            unified_df['Direction_Pred'] = ['Bullish' if x == 1 else 'Bearish' for x in dir_pred]
            unified_df['Direction_Confidence'] = np.max(dir_prob, axis=1) if dir_prob is not None else 0.5
        
        if 'profit_prob' in individual_predictions:
            profit_pred, profit_prob = individual_predictions['profit_prob']
            unified_df['Profit_Signal'] = ['High' if x == 1 else 'Low' for x in profit_pred]
            unified_df['Profit_Confidence'] = np.max(profit_prob, axis=1) if profit_prob is not None else 0.5
        
        if 'reversal' in individual_predictions:
            rev_pred, rev_prob = individual_predictions['reversal']
            unified_df['Reversal_Signal'] = ['Yes' if x == 1 else 'No' for x in rev_pred]
            unified_df['Reversal_Confidence'] = np.max(rev_prob, axis=1) if rev_prob is not None else 0.5
        
        if 'trend_sideways' in individual_predictions:
            trend_pred, trend_prob = individual_predictions['trend_sideways']
            unified_df['Market_Regime'] = ['Trending' if x == 1 else 'Sideways' for x in trend_pred]
            unified_df['Regime_Confidence'] = np.max(trend_prob, axis=1) if trend_prob is not None else 0.5
        
        # Add ultimate model predictions
        if self.task_type == 'classification':
            unified_df['Ultimate_Signal'] = ['Bullish' if x == 1 else 'Bearish' for x in ultimate_pred]
            if ultimate_prob is not None:
                unified_df['Ultimate_Confidence'] = np.max(ultimate_prob, axis=1)
        else:
            unified_df['Ultimate_Pred'] = ultimate_pred
        
        # Add consensus features
        signal_cols = [col for col in unified_df.columns if 'Signal' in col or 'Pred' in col]
        if len(signal_cols) > 1:
            # Model agreement score
            bullish_signals = 0
            total_signals = 0
            
            for col in ['Direction_Pred', 'Profit_Signal', 'Ultimate_Signal']:
                if col in unified_df.columns:
                    bullish_signals += (unified_df[col].isin(['Bullish', 'High'])).astype(int)
                    total_signals += 1
            
            if total_signals > 0:
                unified_df['Consensus_Score'] = bullish_signals / total_signals
                unified_df['Consensus_Strength'] = pd.cut(
                    unified_df['Consensus_Score'], 
                    bins=[0, 0.3, 0.7, 1.0], 
                    labels=['Bearish', 'Neutral', 'Bullish']
                )
        
        # Add market context
        unified_df['Price_Change_1'] = df['Close'].pct_change()
        unified_df['Price_Change_5'] = df['Close'].pct_change(5)
        unified_df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        # Add timestamp if available
        if 'timestamp' in df.columns:
            unified_df['Timestamp'] = df['timestamp']
            unified_df.insert(0, 'Timestamp', unified_df.pop('Timestamp'))
        
        return unified_df
