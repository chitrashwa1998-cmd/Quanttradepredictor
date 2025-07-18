import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time
from collections import deque
from utils.live_data_manager import LiveDataManager
from models.model_manager import ModelManager
from features.direction_technical_indicators import DirectionTechnicalIndicators
import pytz
import streamlit as st

class LivePredictionPipeline:
    """Pipeline to process live OHLC data through direction model for real-time predictions."""

    def __init__(self, access_token: str, api_key: str):
        """Initialize live prediction pipeline."""
        self.live_data_manager = LiveDataManager(access_token, api_key)
        self.model_manager = ModelManager()

        # Prediction storage
        self.live_predictions = {}  # Store predictions for each instrument
        self.prediction_history = {}  # Store prediction history
        self.max_history = 500  # Maximum predictions to store per instrument

        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.update_interval = 10  # Check for new candles every 10 seconds

        # Minimum data requirements
        self.min_ohlc_rows = 100  # Minimum OHLC rows needed for predictions
        
        # Candle completion tracking
        self.last_candle_timestamps = {}  # Track last processed candle for each instrument

    def start_pipeline(self) -> bool:
        """Start the live prediction pipeline."""
        try:
            # Connect to live data feed
            if not self.live_data_manager.connect():
                print("❌ Failed to connect to live data feed")
                return False

            # Force refresh of model manager to ensure latest models are loaded
            print(f"🔄 Refreshing ModelManager to load latest trained models...")
            self.model_manager._load_existing_models()
            
            # Check which models are trained
            available_models = []
            model_names = ['direction', 'volatility', 'profit_probability', 'reversal']

            print(f"🔍 Checking model availability after refresh...")
            print(f"🔍 ModelManager.trained_models keys: {list(self.model_manager.trained_models.keys())}")
            
            # Also check database directly
            try:
                from utils.database_adapter import get_trading_database
                db = get_trading_database()
                db_models = db.load_trained_models()
                print(f"🔍 Direct database check - available models: {list(db_models.keys()) if db_models else 'None'}")
            except Exception as e:
                print(f"❌ Could not check database directly: {e}")
            
            for model_name in model_names:
                is_trained = self.model_manager.is_model_trained(model_name)
                if is_trained:
                    available_models.append(model_name)
                    print(f"✅ {model_name} model ready for live predictions")
                    
                    # Check if model has all required components
                    model_data = self.model_manager.trained_models.get(model_name, {})
                    has_model = 'model' in model_data or 'ensemble' in model_data
                    has_scaler = 'scaler' in model_data
                    has_features = 'feature_names' in model_data
                    print(f"   - Has model: {has_model}, Has scaler: {has_scaler}, Has features: {has_features}")
                else:
                    print(f"⚠️ {model_name} model not trained")
                    
                    # Check what's in session state for this model
                    if hasattr(st, 'session_state'):
                        # Check main trained_models
                        if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
                            if model_name in st.session_state.trained_models:
                                print(f"   - Found {model_name} in session_state.trained_models")
                        
                        # Check individual model session states
                        if hasattr(st.session_state, f'{model_name}_trained_models'):
                            session_models = getattr(st.session_state, f'{model_name}_trained_models', {})
                            print(f"   - Found in {model_name}_trained_models: {list(session_models.keys())}")
                        else:
                            print(f"   - No session state found for {model_name}_trained_models")
            
            print(f"🎯 Total available models: {len(available_models)} out of {len(model_names)}")

            if not available_models:
                print("❌ No trained models available. Please train at least one model first.")
                print("💡 Hint: Go to Model Training page and train at least one model, then try reconnecting.")
                return False

            print(f"🎯 Starting live prediction pipeline with {len(available_models)} models: {available_models}")

            # Start processing thread
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()

            print("✅ Live prediction pipeline started successfully")
            return True

        except Exception as e:
            print(f"❌ Error starting pipeline: {e}")
            return False

    def stop_pipeline(self):
        """Stop the live prediction pipeline."""
        self.is_processing = False
        self.live_data_manager.disconnect()

        if self.processing_thread:
            self.processing_thread.join(timeout=5)

        # Reset candle tracking so predictions can restart on reconnection
        self.last_candle_timestamps = {}
        self.live_predictions = {}
        self.prediction_history = {}

        print("🔌 Live prediction pipeline stopped")

    def subscribe_instruments(self, instrument_keys: List[str]) -> bool:
        """Subscribe to instruments for live predictions."""
        return self.live_data_manager.subscribe_instruments(instrument_keys)

    def _processing_loop(self):
        """Main processing loop for generating live predictions."""
        print(f"🚀 Prediction processing loop started - will check every {self.update_interval} seconds")
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_processing:
            try:
                # Check if connection is still alive
                connection_status = self.live_data_manager.get_connection_status()

                if not connection_status['connected']:
                    print("🔄 Connection lost, waiting for reconnection...")
                    time.sleep(10)
                    continue

                # Get tick statistics to see which instruments have data
                tick_stats = self.live_data_manager.get_tick_statistics()

                if tick_stats:
                    for instrument_key, stats in tick_stats.items():
                        # Debug: Show processing attempt
                        if not hasattr(self, '_debug_counter'):
                            self._debug_counter = 0
                        self._debug_counter += 1
                        
                        # Only show debug every 20 iterations to avoid spam
                        if self._debug_counter % 20 == 0:
                            print(f"🔍 Processing loop #{self._debug_counter} - checking {instrument_key}")
                        
                        # Check if a new candle has closed before processing predictions
                        if self._has_new_candle_closed(instrument_key):
                            print(f"🎯 New candle detected for {instrument_key}, processing predictions...")
                            self._process_instrument_predictions(instrument_key)

                # Reset error counter on successful processing
                consecutive_errors = 0

                # Wait before next processing cycle
                time.sleep(self.update_interval)

            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Error in processing loop ({consecutive_errors}/{max_consecutive_errors}): {e}")

                if consecutive_errors >= max_consecutive_errors:
                    print("❌ Too many consecutive errors, stopping pipeline")
                    self.is_processing = False
                    break

                time.sleep(min(30, 5 * consecutive_errors))  # Progressive backoff

    def _has_new_candle_closed(self, instrument_key: str) -> bool:
        """Check if a new 5-minute candle has closed for the instrument."""
        try:
            # Get latest OHLC data
            ohlc_data = self.live_data_manager.get_live_ohlc(instrument_key, rows=200)
            
            if ohlc_data is None or len(ohlc_data) < 1:
                return False
            
            # Get the latest candle timestamp
            latest_candle_timestamp = ohlc_data.index[-1]
            
            # Check if this is a new candle compared to last processed
            if instrument_key not in self.last_candle_timestamps:
                # First time processing this instrument
                seeding_status = self.live_data_manager.get_seeding_status()
                if instrument_key in seeding_status.get('instruments_seeded', []):
                    print(f"🎯 First prediction for {instrument_key} with pre-seeded data")
                    print(f"📊 Available OHLC data: {len(ohlc_data)} rows")
                    self.last_candle_timestamps[instrument_key] = latest_candle_timestamp
                    return True
                elif len(ohlc_data) >= self.min_ohlc_rows:
                    self.last_candle_timestamps[instrument_key] = latest_candle_timestamp
                    print(f"🎯 First prediction for {instrument_key} - sufficient data available ({len(ohlc_data)} rows)")
                    return True
                return False
            
            # Check if we have a new candle timestamp (this means a new candle was formed)
            if latest_candle_timestamp > self.last_candle_timestamps[instrument_key]:
                print(f"🕐 NEW 5-MINUTE CANDLE CLOSED for {instrument_key}!")
                print(f"   Previous candle: {self.last_candle_timestamps[instrument_key]}")
                print(f"   New candle: {latest_candle_timestamp}")
                print(f"   Processing prediction immediately...")
                
                # Update our tracking and trigger prediction
                self.last_candle_timestamps[instrument_key] = latest_candle_timestamp
                return True
            
            # No new candle detected
            return False
            
        except Exception as e:
            print(f"❌ Error checking candle completion for {instrument_key}: {e}")
            return False

    def _process_instrument_predictions(self, instrument_key: str):
        """Process predictions for a specific instrument using all available models."""
        try:
            instrument_display = instrument_key  # Set default display name

            # Check if we have enough OHLC data for predictions
            ohlc_data = self.live_data_manager.get_live_ohlc(instrument_key, rows=200)

            # Check if we have pre-seeded data
            seeding_status = self.live_data_manager.get_seeding_status()
            is_seeded = instrument_key in seeding_status.get('instruments_seeded', [])
            
            if ohlc_data is None or len(ohlc_data) < 1:
                print(f"📊 No OHLC data available for {instrument_display}")
                return
                
            # If we have pre-seeded data, we can proceed with any amount of data
            if not is_seeded and len(ohlc_data) < self.min_ohlc_rows:
                current_rows = len(ohlc_data) if ohlc_data is not None else 0
                print(f"📊 Building OHLC data for {instrument_display}: {current_rows}/{self.min_ohlc_rows} rows needed")
                return

            # Generate predictions from all available models
            all_predictions = {}
            timestamp = ohlc_data.index[-1]
            ohlc_row = ohlc_data.iloc[-1]
            
            # Debug: Show which models are detected as trained
            trained_models = self.model_manager.get_trained_models()
            print(f"🎯 Starting prediction generation for {instrument_key}")
            print(f"📊 Data: {len(ohlc_data)} OHLC rows available")
            print(f"🤖 Trained models detected: {trained_models}")
            print(f"🔍 Model Manager trained models: {list(self.model_manager.trained_models.keys())}")

            # Check all 4 models explicitly
            for model_name in ['direction', 'volatility', 'profit_probability', 'reversal']:
                is_trained = self.model_manager.is_model_trained(model_name)
                print(f"🔍 {model_name} model trained status: {is_trained}")

            # Direction Model
            direction_trained = self.model_manager.is_model_trained('direction')
            
            if direction_trained:
                print(f"🔧 Calculating direction features for {instrument_key}...")
                direction_features = self._calculate_direction_features(ohlc_data)
                if direction_features is not None and len(direction_features) > 0:
                    print(f"✅ Direction features calculated: {direction_features.shape}")
                    try:
                        predictions, probabilities = self.model_manager.predict('direction', direction_features)
                        if predictions is not None:
                            direction = 'Bullish' if predictions[-1] == 1 else 'Bearish'
                            confidence = np.max(probabilities[-1]) if probabilities is not None else 0.5
                            all_predictions['direction'] = {
                                'prediction': direction,
                                'confidence': confidence,
                                'value': int(predictions[-1])
                            }
                            print(f"✅ Direction prediction successful: {direction}")
                        else:
                            print(f"❌ Direction model returned None predictions")
                    except Exception as e:
                        print(f"❌ Direction prediction error for {instrument_key}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"❌ Direction features calculation failed")
            else:
                print(f"⚠️ Direction model not detected as trained")

            # Volatility Model
            volatility_trained = self.model_manager.is_model_trained('volatility')
            print(f"🔍 Volatility model trained status: {volatility_trained}")
            
            if volatility_trained:
                print(f"🔧 Calculating volatility features for {instrument_key}...")
                volatility_features = self._calculate_volatility_features(ohlc_data)
                
                if volatility_features is not None and len(volatility_features) > 0:
                    print(f"✅ Volatility features calculated: {volatility_features.shape}")
                    try:
                        print(f"🎯 Making volatility prediction...")
                        predictions, _ = self.model_manager.predict('volatility', volatility_features)
                        if predictions is not None:
                            volatility_level = self._categorize_volatility(predictions[-1])
                            all_predictions['volatility'] = {
                                'prediction': volatility_level,
                                'value': float(predictions[-1])
                            }
                            print(f"✅ Volatility prediction successful: {volatility_level}")
                        else:
                            print(f"❌ Volatility model returned None predictions")
                    except Exception as e:
                        print(f"❌ Volatility prediction error for {instrument_key}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"❌ Volatility features calculation failed or returned empty: {volatility_features}")
            else:
                print(f"⚠️ Volatility model not detected as trained")

            # Profit Probability Model
            profit_trained = self.model_manager.is_model_trained('profit_probability')
            print(f"🔍 Profit probability model trained status: {profit_trained}")
            
            if profit_trained:
                print(f"🔧 Calculating profit probability features for {instrument_key}...")
                profit_features = self._calculate_profit_probability_features(ohlc_data)
                if profit_features is not None and len(profit_features) > 0:
                    print(f"✅ Profit probability features calculated: {profit_features.shape}")
                    try:
                        predictions, probabilities = self.model_manager.predict('profit_probability', profit_features)
                        if predictions is not None:
                            profit_likely = 'High' if predictions[-1] == 1 else 'Low'
                            confidence = np.max(probabilities[-1]) if probabilities is not None else 0.5
                            all_predictions['profit_probability'] = {
                                'prediction': profit_likely,
                                'confidence': confidence,
                                'value': int(predictions[-1])
                            }
                            print(f"✅ Profit probability prediction successful: {profit_likely}")
                        else:
                            print(f"❌ Profit probability model returned None predictions")
                    except Exception as e:
                        print(f"❌ Profit probability prediction error for {instrument_key}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"❌ Profit probability features calculation failed")

            # Reversal Model
            reversal_trained = self.model_manager.is_model_trained('reversal')
            print(f"🔍 Reversal model trained status: {reversal_trained}")
            
            if reversal_trained:
                print(f"🔧 Calculating reversal features for {instrument_key}...")
                reversal_features = self._calculate_reversal_features(ohlc_data)
                if reversal_features is not None and len(reversal_features) > 0:
                    print(f"✅ Reversal features calculated: {reversal_features.shape}")
                    try:
                        predictions, probabilities = self.model_manager.predict('reversal', reversal_features)
                        if predictions is not None:
                            reversal_expected = 'Yes' if predictions[-1] == 1 else 'No'
                            confidence = np.max(probabilities[-1]) if probabilities is not None else 0.5
                            all_predictions['reversal'] = {
                                'prediction': reversal_expected,
                                'confidence': confidence,
                                'value': int(predictions[-1])
                            }
                            print(f"✅ Reversal prediction successful: {reversal_expected}")
                        else:
                            print(f"❌ Reversal model returned None predictions")
                    except Exception as e:
                        print(f"❌ Reversal prediction error for {instrument_key}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"❌ Reversal features calculation failed")

            if not all_predictions:
                print(f"❌ No predictions generated for {instrument_key}")
                return

            # Format comprehensive prediction
            latest_prediction = self._format_comprehensive_prediction(
                instrument_key, timestamp, all_predictions, ohlc_row
            )

            self.live_predictions[instrument_key] = latest_prediction

            # Store in history
            if instrument_key not in self.prediction_history:
                self.prediction_history[instrument_key] = deque(maxlen=self.max_history)

            self.prediction_history[instrument_key].append(latest_prediction)

            # Log summary with candle completion info
            model_count = len(all_predictions)
            model_names = list(all_predictions.keys())
            print(f"✅ Generated {model_count} live predictions for {instrument_key} on CANDLE CLOSE: {model_names}")
            print(f"🕐 Prediction timestamp: {timestamp} (Complete 5-minute candle)")
            
            # Show clear indication that this is a candle-close prediction
            seeding_status = self.live_data_manager.get_seeding_status()
            if instrument_key in seeding_status.get('instruments_seeded', []):
                current_rows = len(ohlc_data)
                seed_count = seeding_status['seeding_details'][instrument_key]['seed_count']
                live_count = current_rows - seed_count
                print(f"📊 Total data: {current_rows} rows ({seed_count} seeded + {live_count} live candles)")

        except Exception as e:
            print(f"❌ Error processing predictions for {instrument_key}: {e}")

    def _calculate_direction_features(self, ohlc_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate direction-specific features from OHLC data."""
        try:
            # Use direction model's prepare_features method to ensure proper feature preparation
            features = self.model_manager.models['direction'].prepare_features(ohlc_data)
            return features
        except Exception as e:
            print(f"❌ Error calculating direction features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_volatility_features(self, ohlc_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate volatility-specific features from OHLC data."""
        try:
            # Use volatility model's prepare_features method directly
            features = self.model_manager.models['volatility'].prepare_features(ohlc_data)
            return features
        except Exception as e:
            print(f"❌ Error calculating volatility features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_profit_probability_features(self, ohlc_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate profit probability features from OHLC data."""
        try:
            # Use profit probability model's prepare_features method directly
            features = self.model_manager.models['profit_probability'].prepare_features(ohlc_data)
            return features
        except Exception as e:
            print(f"❌ Error calculating profit probability features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_reversal_features(self, ohlc_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate reversal detection features from OHLC data."""
        try:
            # Use reversal model's prepare_features method which includes all feature types
            features = self.model_manager.models['reversal'].prepare_features(ohlc_data)
            return features
        except Exception as e:
            print(f"❌ Error calculating reversal features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _categorize_volatility(self, volatility_value: float) -> str:
        """Categorize volatility value into readable format."""
        if volatility_value < 0.01:
            return 'Low'
        elif volatility_value < 0.02:
            return 'Medium'
        elif volatility_value < 0.03:
            return 'High'
        else:
            return 'Very High'

    def _format_comprehensive_prediction(self, instrument_key: str, timestamp: pd.Timestamp, 
                                        all_predictions: Dict, ohlc_row: pd.Series) -> Dict:
        """Format comprehensive prediction data from all models."""
        # Ensure timestamp is timezone-naive
        if hasattr(timestamp, 'tz_localize'):
            timestamp = timestamp.tz_localize(None)

        # Base prediction structure
        formatted_prediction = {
            'instrument': instrument_key,
            'timestamp': timestamp,
            'current_price': float(ohlc_row['Close']),
            'volume': int(ohlc_row['Volume']) if 'Volume' in ohlc_row else 0,
            'generated_at': datetime.now(pytz.timezone('Asia/Kolkata')).replace(tzinfo=None),
            'models_used': list(all_predictions.keys()),
            'candle_close_prediction': True,  # Flag to indicate this is a candle-close prediction
            'candle_period': '5T',  # 5-minute timeframe
            'prediction_type': 'candle_completion'
        }

        # Add predictions from each model
        for model_name, prediction_data in all_predictions.items():
            formatted_prediction[model_name] = prediction_data

        # Legacy support - use direction as primary if available
        if 'direction' in all_predictions:
            formatted_prediction['direction'] = all_predictions['direction']['prediction']
            formatted_prediction['confidence'] = all_predictions['direction']['confidence']
            formatted_prediction['prediction_value'] = all_predictions['direction']['value']

        return formatted_prediction

    def get_latest_predictions(self) -> Dict:
        """Get latest predictions for all instruments."""
        return self.live_predictions.copy()

    def get_prediction_history(self, instrument_key: str, count: int = 50) -> List[Dict]:
        """Get prediction history for a specific instrument."""
        if instrument_key not in self.prediction_history:
            return []

        history = list(self.prediction_history[instrument_key])
        return history[-count:] if count > 0 else history

    def get_pipeline_status(self) -> Dict:
        """Get pipeline status and statistics."""
        connection_status = self.live_data_manager.get_connection_status()

        # Check status of all models
        model_status = {}
        model_names = ['direction', 'volatility', 'profit_probability', 'reversal']

        for model_name in model_names:
            model_status[f'{model_name}_ready'] = self.model_manager.is_model_trained(model_name)

        trained_models = [name for name in model_names if self.model_manager.is_model_trained(name)]

        return {
            'pipeline_active': self.is_processing,
            'data_connected': connection_status['connected'],
            'subscribed_instruments': connection_status['subscribed_instruments'],
            'instruments_with_predictions': len(self.live_predictions),
            'total_ticks_received': connection_status['total_ticks_received'],
            'last_prediction_time': max(
                [pred['generated_at'] for pred in self.live_predictions.values()]
            ) if self.live_predictions else None,
            'trained_models': trained_models,
            'total_trained_models': len(trained_models),
            **model_status
        }

    def get_instrument_summary(self, instrument_key: str) -> Optional[Dict]:
        """Get comprehensive summary for a specific instrument."""
        if instrument_key not in self.live_predictions:
            return None

        latest = self.live_predictions[instrument_key]
        history = self.get_prediction_history(instrument_key, 20)

        if not history:
            return latest

        # Calculate statistics for all models
        stats = {
            'total_predictions': len(history),
            'models_used': latest.get('models_used', [])
        }

        # Direction model statistics
        if 'direction' in latest:
            recent_directions = []
            for p in history:
                if isinstance(p, dict) and 'direction' in p:
                    direction_data = p.get('direction', {})
                    if isinstance(direction_data, dict):
                        prediction = direction_data.get('prediction', 'Unknown')
                    else:
                        prediction = str(direction_data)
                    recent_directions.append(prediction)
            
            bullish_count = recent_directions.count('Bullish')
            bearish_count = recent_directions.count('Bearish')

            stats['direction_stats'] = {
                'bullish_signals': bullish_count,
                'bearish_signals': bearish_count,
                'bullish_percentage': (bullish_count / len(recent_directions)) * 100 if recent_directions else 0
            }

        # Volatility model statistics
        if 'volatility' in latest:
            recent_volatility = []
            for p in history:
                if isinstance(p, dict) and 'volatility' in p:
                    volatility_data = p.get('volatility', {})
                    if isinstance(volatility_data, dict):
                        prediction = volatility_data.get('prediction', 'Unknown')
                    else:
                        prediction = str(volatility_data)
                    recent_volatility.append(prediction)
            
            if recent_volatility:
                high_vol_count = sum(1 for v in recent_volatility if v in ['High', 'Very High'])
                stats['volatility_stats'] = {
                    'high_volatility_percentage': (high_vol_count / len(recent_volatility)) * 100
                }

        # Profit probability statistics
        if 'profit_probability' in latest:
            recent_profit = []
            for p in history:
                if isinstance(p, dict) and 'profit_probability' in p:
                    profit_data = p.get('profit_probability', {})
                    if isinstance(profit_data, dict):
                        prediction = profit_data.get('prediction', 'Unknown')
                    else:
                        prediction = str(profit_data)
                    recent_profit.append(prediction)
            
            if recent_profit:
                high_profit_count = recent_profit.count('High')
                stats['profit_stats'] = {
                    'high_profit_percentage': (high_profit_count / len(recent_profit)) * 100
                }

        # Reversal statistics
        if 'reversal' in latest:
            recent_reversal = [p.get('reversal', {}).get('prediction', 'Unknown') 
                             for p in history if 'reversal' in p]
            if recent_reversal:
                reversal_count = recent_reversal.count('Yes')
                stats['reversal_stats'] = {
                    'reversal_signals': reversal_count,
                    'reversal_percentage': (reversal_count / len(recent_reversal)) * 100
                }

        return {
            **latest,
            'comprehensive_stats': stats
        }