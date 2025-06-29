
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import streamlit as st

from .ultimate_model import UltimateModel
from .model_manager import ModelManager
from .direction_model import DirectionModel

class UltimateModelManager:
    """Manager for coordinating all individual models and the ultimate ensemble model."""
    
    def __init__(self):
        self.ultimate_model = UltimateModel()
        self.model_manager = ModelManager()  # For volatility
        self.direction_model = DirectionModel()
        
        # Store individual model instances and their trained states
        self.individual_models = {
            'volatility': None,
            'direction': None,
            'profit_prob': None,
            'reversal': None,
            'trend_sideways': None
        }
        
        self.trained_models = {}
        self.ultimate_trained = False

    def load_all_models(self):
        """Load all trained individual models."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()
            
            # Load volatility model
            loaded_models = db.load_trained_models()
            if loaded_models and 'volatility' in loaded_models:
                self.model_manager.trained_models['volatility'] = loaded_models['volatility']
                print("✅ Loaded volatility model")
            
            # Load direction model
            direction_results = db.load_model_results('direction')
            if direction_results:
                self.trained_models['direction'] = direction_results
                print("✅ Loaded direction model")
                
            # Load other models if available
            for model_name in ['profit_prob', 'reversal', 'trend_sideways']:
                model_results = db.load_model_results(model_name)
                if model_results:
                    self.trained_models[model_name] = model_results
                    print(f"✅ Loaded {model_name} model")
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def get_all_individual_predictions(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Get predictions from all available individual models."""
        predictions = {}
        
        try:
            # Volatility predictions
            if self.model_manager.is_model_trained('volatility'):
                vol_pred, _ = self.model_manager.predict('volatility', df)
                predictions['volatility'] = (vol_pred, None)
                print(f"✅ Got volatility predictions: {len(vol_pred)} samples")
        except Exception as e:
            print(f"❌ Error getting volatility predictions: {str(e)}")
        
        try:
            # Direction predictions
            if 'direction' in self.trained_models:
                # Prepare direction features
                from features.direction_technical_indicators import DirectionTechnicalIndicators
                direction_features = DirectionTechnicalIndicators.calculate_all_direction_indicators(df)
                
                # Set up direction model with trained data
                model_data = self.trained_models['direction']
                self.direction_model.model = model_data.get('model')
                self.direction_model.scaler = model_data.get('scaler')
                self.direction_model.selected_features = model_data.get('feature_names', [])
                
                if self.direction_model.model is not None:
                    dir_pred, dir_prob = self.direction_model.predict(direction_features)
                    predictions['direction'] = (dir_pred, dir_prob)
                    print(f"✅ Got direction predictions: {len(dir_pred)} samples")
        except Exception as e:
            print(f"❌ Error getting direction predictions: {str(e)}")
        
        # Add other models as they become available
        # TODO: Add profit_prob, reversal, trend_sideways predictions
        
        return predictions

    def train_ultimate_model(self, df: pd.DataFrame, target_type: str = 'direction') -> Dict[str, Any]:
        """Train the ultimate ensemble model using individual model predictions."""
        
        # Get individual model predictions
        individual_predictions = self.get_all_individual_predictions(df)
        
        if not individual_predictions:
            raise ValueError("No individual model predictions available. Train individual models first.")
        
        print(f"Training ultimate model with {len(individual_predictions)} individual model predictions")
        
        # Prepare meta-features
        meta_features = self.ultimate_model.prepare_meta_features(df, individual_predictions)
        
        # Create target
        target = self.ultimate_model.create_target(df, target_type)
        
        # Train the ultimate model
        result = self.ultimate_model.train(meta_features, target)
        
        # Store the result
        self.trained_models['ultimate'] = result
        self.ultimate_trained = True
        
        return result

    def predict_ultimate(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the ultimate model."""
        if not self.ultimate_trained:
            raise ValueError("Ultimate model not trained. Call train_ultimate_model() first.")
        
        # Get individual model predictions
        individual_predictions = self.get_all_individual_predictions(df)
        
        # Prepare meta-features
        meta_features = self.ultimate_model.prepare_meta_features(df, individual_predictions)
        
        # Make ultimate prediction
        return self.ultimate_model.predict(meta_features)

    def create_unified_prediction_table(self, df: pd.DataFrame, limit_rows: Optional[int] = None) -> pd.DataFrame:
        """Create a unified table with all model predictions."""
        
        # Limit data if specified to prevent memory issues
        if limit_rows and len(df) > limit_rows:
            df_subset = df.tail(limit_rows).copy()
        else:
            df_subset = df.copy()
        
        # Get individual model predictions
        individual_predictions = self.get_all_individual_predictions(df_subset)
        
        if not individual_predictions:
            raise ValueError("No individual model predictions available")
        
        # Get ultimate model predictions if available
        ultimate_predictions = None
        if self.ultimate_trained:
            try:
                ultimate_predictions = self.predict_ultimate(df_subset)
            except Exception as e:
                print(f"Warning: Could not get ultimate predictions: {str(e)}")
        
        # Create unified table
        if ultimate_predictions:
            unified_df = self.ultimate_model.create_unified_prediction_table(
                df_subset, individual_predictions, ultimate_predictions
            )
        else:
            # Create table without ultimate predictions
            unified_df = self._create_basic_unified_table(df_subset, individual_predictions)
        
        return unified_df

    def _create_basic_unified_table(self, df: pd.DataFrame, individual_predictions: Dict[str, Any]) -> pd.DataFrame:
        """Create unified table without ultimate model predictions."""
        
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
            if dir_prob is not None:
                unified_df['Direction_Confidence'] = np.max(dir_prob, axis=1)
        
        # Add consensus if multiple predictions available
        if len(individual_predictions) > 1:
            bullish_count = 0
            total_count = 0
            
            if 'direction' in individual_predictions:
                dir_pred, _ = individual_predictions['direction']
                bullish_count += np.sum(dir_pred == 1)
                total_count += len(dir_pred)
            
            if total_count > 0:
                unified_df['Consensus_Bullish_Pct'] = (bullish_count / total_count) * 100
        
        # Add market context
        unified_df['Price_Change_1'] = df['Close'].pct_change()
        unified_df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        # Add timestamp if available
        if 'timestamp' in df.columns:
            unified_df['Timestamp'] = df['timestamp']
            unified_df.insert(0, 'Timestamp', unified_df.pop('Timestamp'))
        
        return unified_df

    def get_model_summary(self) -> Dict[str, str]:
        """Get summary of all model states."""
        summary = {}
        
        # Check volatility model
        summary['Volatility'] = "✅ Trained" if self.model_manager.is_model_trained('volatility') else "❌ Not Trained"
        
        # Check direction model
        summary['Direction'] = "✅ Trained" if 'direction' in self.trained_models else "❌ Not Trained"
        
        # Check other models
        for model_name in ['profit_prob', 'reversal', 'trend_sideways']:
            summary[model_name.replace('_', ' ').title()] = "✅ Trained" if model_name in self.trained_models else "❌ Not Trained"
        
        # Check ultimate model
        summary['Ultimate'] = "✅ Trained" if self.ultimate_trained else "❌ Not Trained"
        
        return summary
