import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Any
import streamlit as st

class QuantTradingModels:
    """XGBoost models for quantitative trading predictions."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training."""
        # Remove any rows with NaN values
        df_clean = df.dropna()
        
        # Select feature columns (exclude OHLC and target columns)
        feature_cols = [col for col in df_clean.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        feature_cols = [col for col in feature_cols if not col.startswith(('target_', 'future_'))]
        
        self.feature_names = feature_cols
        return df_clean[feature_cols]
    
    def create_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create target variables for different prediction tasks."""
        targets = {}
        
        # 1. Direction prediction (up/down)
        future_return = df['Close'].shift(-1) / df['Close'] - 1
        targets['direction'] = (future_return > 0).astype(int)
        
        # 2. Magnitude of move (percentage change)
        targets['magnitude'] = np.abs(future_return) * 100
        
        # 3. Probability of profit (based on next 5 periods)
        future_returns_5 = []
        for i in range(5):
            future_returns_5.append(df['Close'].shift(-i-1) / df['Close'] - 1)
        max_future_return = pd.concat(future_returns_5, axis=1).max(axis=1)
        
        # Use a more adaptive threshold based on data volatility
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std()
        profit_threshold = min(0.005, volatility)  # Use 0.5% or data volatility, whichever is smaller
        
        targets['profit_prob'] = (max_future_return > profit_threshold).astype(int)
        
        # 4. Volatility forecasting (next period volatility)
        volatility_window = 10
        current_vol = df['Close'].rolling(volatility_window).std()
        future_vol = current_vol.shift(-1)
        targets['volatility'] = future_vol
        
        # 5. Trend vs sideways
        price_change_5 = df['Close'].shift(-5) / df['Close'] - 1
        trend_threshold = 0.02  # 2% threshold
        # Handle NaN values and create valid binary targets
        trend_sideways = np.where(np.abs(price_change_5) > trend_threshold, 1, 0)
        targets['trend_sideways'] = pd.Series(trend_sideways, index=df.index)
        
        # 6. Reversal points
        # Look for price reversals in next 3 periods
        high_3 = df['High'].rolling(3).max().shift(-3)
        low_3 = df['Low'].rolling(3).min().shift(-3)
        current_high = df['High'].rolling(3).max()
        current_low = df['Low'].rolling(3).min()
        
        reversal_up = (df['Close'] <= current_low * 1.01) & (high_3 > df['Close'] * 1.02)
        reversal_down = (df['Close'] >= current_high * 0.99) & (low_3 < df['Close'] * 0.98)
        targets['reversal'] = (reversal_up | reversal_down).astype(int)
        
        # 7. Buy/Sell/Hold signals
        # Simplified signal generation to avoid issues with missing indicators
        # Use only price-based signals for reliability
        
        # Create more robust signals based on price momentum
        price_momentum_1 = df['Close'].shift(-1) / df['Close'] - 1
        price_momentum_3 = df['Close'].shift(-3) / df['Close'] - 1
        
        # Simple signal logic: Buy if strong positive momentum, Sell if strong negative, Hold otherwise
        buy_threshold = 0.015   # 1.5% positive momentum
        sell_threshold = -0.015 # 1.5% negative momentum
        
        buy_signal = (price_momentum_1 > buy_threshold) | (price_momentum_3 > buy_threshold * 2)
        sell_signal = (price_momentum_1 < sell_threshold) | (price_momentum_3 < sell_threshold * 2)
        
        signals = np.where(buy_signal, 2, np.where(sell_signal, 0, 1))  # 2=Buy, 1=Hold, 0=Sell
        targets['trading_signal'] = pd.Series(signals, index=df.index)
        
        # Debug information for profit_prob
        if 'profit_prob' in targets:
            profit_prob_stats = targets['profit_prob'].value_counts()
            print(f"Profit Probability Target Distribution: {profit_prob_stats.to_dict()}")
            print(f"Profit threshold used: {profit_threshold:.4f}")
            print(f"Max future return range: {max_future_return.min():.4f} to {max_future_return.max():.4f}")
        
        return targets
    
    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> Dict[str, Any]:
        """Train XGBoost model for specific task."""
        
        # Remove NaN values and ensure we have valid targets
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Additional validation for target values
        if task_type == 'classification':
            # Remove any invalid target values
            valid_targets = ~np.isinf(y_clean) & (y_clean >= 0)
            X_clean = X_clean[valid_targets]
            y_clean = y_clean[valid_targets]
            
            # Ensure we have at least 2 classes
            unique_targets = y_clean.unique()
            if len(unique_targets) < 2:
                raise ValueError(f"Insufficient target classes for {model_name}. Found classes: {unique_targets}")
        
        if len(X_clean) < 100:
            raise ValueError(f"Insufficient data for training {model_name}. Need at least 100 samples, got {len(X_clean)}")
        
        # Split data using time series split to avoid look-ahead bias
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X_clean))
        train_idx, test_idx = splits[-1]  # Use the last split
        
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        
        # Scale features for regression tasks
        if task_type == 'regression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Set up XGBoost parameters
        if task_type == 'classification':
            if len(np.unique(y_train)) > 2:
                objective = 'multi:softprob'
                num_class = len(np.unique(y_train))
            else:
                objective = 'binary:logistic'
                num_class = None
        else:
            objective = 'reg:squarederror'
            num_class = None
        
        params = {
            'objective': objective,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if num_class:
            params['num_class'] = num_class
        
        # Train model
        if task_type == 'classification':
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        if task_type == 'classification':
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = None
        
        # Calculate metrics
        if task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
        
        # Store model
        self.models[model_name] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(self.feature_names, model.feature_importances_)),
            'task_type': task_type,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_indices': test_idx
        }
        
        return self.models[model_name]
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all trading models."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare features
        status_text.text("Preparing features...")
        X = self.prepare_features(df)
        
        # Create targets
        status_text.text("Creating target variables...")
        targets = self.create_targets(df)
        
        models_config = [
            ('direction', 'classification'),
            ('magnitude', 'regression'),
            ('profit_prob', 'classification'),
            ('volatility', 'regression'),
            ('trend_sideways', 'classification'),
            ('reversal', 'classification'),
            ('trading_signal', 'classification')
        ]
        
        results = {}
        total_models = len(models_config)
        
        for i, (model_name, task_type) in enumerate(models_config):
            status_text.text(f"Training {model_name} model...")
            
            try:
                if model_name in targets:
                    result = self.train_model(model_name, X, targets[model_name], task_type)
                    results[model_name] = result
                    st.success(f"✅ {model_name} model trained successfully")
                else:
                    st.warning(f"⚠️ Target {model_name} not found")
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                results[model_name] = None
            
            progress_bar.progress((i + 1) / total_models)
        
        status_text.text("All models trained!")
        return results
    
    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Prepare features
        X_features = X[self.feature_names]
        
        # Scale if needed
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X_features)
        else:
            X_scaled = X_features.values
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Get probabilities for classification tasks
        if model_info['task_type'] == 'classification':
            probabilities = model.predict_proba(X_scaled)
        else:
            probabilities = None
        
        return predictions, probabilities
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for a specific model."""
        if model_name not in self.models:
            return {}
        
        return self.models[model_name]['feature_importance']
