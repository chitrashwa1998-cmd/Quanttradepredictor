
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import streamlit as st

class RealtimePredictor:
    """Optimized predictor for live trading with sub-5 second response times."""
    
    def __init__(self):
        self.model_manager = None
        self.last_features = None
        self.feature_cache = {}
        self.models_loaded = False
        
    def initialize_models(self):
        """Pre-load all models for faster predictions."""
        try:
            from models.model_manager import ModelManager
            self.model_manager = ModelManager()
            
            # Check which models are available
            available_models = []
            for model_name in ['volatility', 'direction', 'profit_probability', 'reversal']:
                if self.model_manager.is_model_trained(model_name):
                    available_models.append(model_name)
            
            self.models_loaded = len(available_models) > 0
            print(f"⚡ Loaded {len(available_models)} models for real-time prediction: {available_models}")
            return available_models
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return []
    
    def get_fast_predictions(self, latest_data: pd.DataFrame, models_to_run: List[str] = None) -> Dict:
        """Get predictions optimized for speed (target: <5 seconds)."""
        start_time = time.time()
        
        if not self.models_loaded:
            available_models = self.initialize_models()
            if not available_models:
                return {"error": "No trained models available"}
        
        if models_to_run is None:
            models_to_run = ['volatility', 'direction', 'profit_probability', 'reversal']
        
        # Filter to only available models
        models_to_run = [m for m in models_to_run if self.model_manager.is_model_trained(m)]
        
        predictions = {}
        
        for model_name in models_to_run:
            try:
                model_start = time.time()
                
                # Calculate features specific to this model
                if model_name == 'volatility':
                    features = self._get_volatility_features(latest_data)
                elif model_name == 'direction':
                    features = self._get_direction_features(latest_data)
                elif model_name == 'profit_probability':
                    features = self._get_profit_features(latest_data)
                elif model_name == 'reversal':
                    features = self._get_reversal_features(latest_data)
                
                if features is not None and len(features) > 0:
                    # Get prediction for latest data point
                    preds, probs = self.model_manager.predict(model_name, features.tail(1))
                    
                    predictions[model_name] = {
                        'prediction': preds[0] if len(preds) > 0 else None,
                        'confidence': np.max(probs[0]) if probs is not None and len(probs) > 0 else None,
                        'timestamp': datetime.now(),
                        'processing_time': time.time() - model_start
                    }
                else:
                    predictions[model_name] = {'error': 'Feature calculation failed'}
                    
            except Exception as e:
                predictions[model_name] = {'error': str(e)}
        
        total_time = time.time() - start_time
        predictions['meta'] = {
            'total_processing_time': total_time,
            'models_processed': len([p for p in predictions.values() if 'error' not in p]),
            'timestamp': datetime.now()
        }
        
        print(f"⚡ Real-time predictions completed in {total_time:.2f} seconds")
        return predictions
    
    def _get_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get volatility-specific features quickly."""
        try:
            from features.technical_indicators import TechnicalIndicators
            return TechnicalIndicators.calculate_realtime_indicators(df, last_n_rows=50)
        except:
            from features.technical_indicators import TechnicalIndicators
            return TechnicalIndicators.calculate_all_indicators(df.tail(50))
    
    def _get_direction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get direction-specific features quickly."""
        try:
            from features.direction_technical_indicators import DirectionTechnicalIndicators
            return DirectionTechnicalIndicators.calculate_all_direction_indicators(df.tail(50))
        except Exception as e:
            print(f"Direction features error: {e}")
            return None
    
    def _get_profit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get profit probability features quickly."""
        try:
            from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
            return ProfitProbabilityTechnicalIndicators.calculate_all_profit_probability_indicators(df.tail(50))
        except Exception as e:
            print(f"Profit probability features error: {e}")
            return None
    
    def _get_reversal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get reversal-specific features quickly."""
        try:
            from models.reversal_model import ReversalModel
            reversal_model = ReversalModel()
            return reversal_model.prepare_features(df.tail(50))
        except Exception as e:
            print(f"Reversal features error: {e}")
            return None
    
    def get_trading_signal(self, latest_data: pd.DataFrame) -> Dict:
        """Get consolidated trading signal for immediate decision making."""
        predictions = self.get_fast_predictions(latest_data)
        
        if 'error' in predictions:
            return predictions
        
        # Consolidate signals into trading recommendation
        signal = {
            'action': 'HOLD',  # Default
            'confidence': 0.0,
            'reasoning': [],
            'individual_signals': {},
            'processing_time': predictions.get('meta', {}).get('total_processing_time', 0)
        }
        
        # Extract individual signals
        if 'direction' in predictions and 'error' not in predictions['direction']:
            direction_pred = predictions['direction']
            signal['individual_signals']['direction'] = {
                'signal': 'BULLISH' if direction_pred['prediction'] == 1 else 'BEARISH',
                'confidence': direction_pred.get('confidence', 0)
            }
        
        if 'volatility' in predictions and 'error' not in predictions['volatility']:
            vol_pred = predictions['volatility']
            signal['individual_signals']['volatility'] = {
                'level': 'HIGH' if vol_pred['prediction'] > 0.02 else 'LOW',
                'value': vol_pred['prediction']
            }
        
        if 'profit_probability' in predictions and 'error' not in predictions['profit_probability']:
            profit_pred = predictions['profit_probability']
            signal['individual_signals']['profit_probability'] = {
                'signal': 'HIGH_PROFIT' if profit_pred['prediction'] == 1 else 'LOW_PROFIT',
                'confidence': profit_pred.get('confidence', 0)
            }
        
        if 'reversal' in predictions and 'error' not in predictions['reversal']:
            reversal_pred = predictions['reversal']
            signal['individual_signals']['reversal'] = {
                'signal': 'REVERSAL' if reversal_pred['prediction'] == 1 else 'CONTINUATION',
                'confidence': reversal_pred.get('confidence', 0)
            }
        
        # Generate trading action based on signals
        if len(signal['individual_signals']) >= 2:
            # Simple consensus logic
            bullish_signals = 0
            bearish_signals = 0
            total_confidence = 0
            
            for model, sig in signal['individual_signals'].items():
                conf = sig.get('confidence', 0.5)
                total_confidence += conf
                
                if model == 'direction':
                    if sig['signal'] == 'BULLISH':
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
                elif model == 'profit_probability':
                    if sig['signal'] == 'HIGH_PROFIT':
                        bullish_signals += 0.5
                
            avg_confidence = total_confidence / len(signal['individual_signals'])
            
            if bullish_signals > bearish_signals and avg_confidence > 0.6:
                signal['action'] = 'BUY'
                signal['confidence'] = avg_confidence
                signal['reasoning'].append(f"Bullish consensus with {avg_confidence:.1%} confidence")
            elif bearish_signals > bullish_signals and avg_confidence > 0.6:
                signal['action'] = 'SELL'  
                signal['confidence'] = avg_confidence
                signal['reasoning'].append(f"Bearish consensus with {avg_confidence:.1%} confidence")
            else:
                signal['action'] = 'HOLD'
                signal['reasoning'].append("Mixed signals or low confidence")
        
        return signal

# Global instance for reuse
realtime_predictor = RealtimePredictor()
