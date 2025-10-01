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
from utils.obi_cvd_confirmation import OBICVDConfirmation
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
        self.update_interval = 1  # Check for new candles every 1 second

        # Black-Scholes continuous calculation
        self.bs_thread = None
        self.bs_update_interval = 1  # Update Black-Scholes every 1 second with live tick data
        self.latest_volatility_predictions = {}  # Store latest volatility for each instrument

        # Minimum data requirements
        self.min_ohlc_rows = 100  # Minimum OHLC rows needed for predictions

        # Candle completion tracking
        self.last_candle_timestamps = {}  # Track last processed candle for each instrument

        # Dedicated instrument routing for specialized analysis
        # ML Models + Black-Scholes: Use spot index for accurate pricing models
        self.ml_models_instrument = "NSE_INDEX|Nifty 50"  # Spot price for ML models + Black-Scholes
        self.obi_cvd_instrument = "NSE_FO|52168"  # Active futures contract for OBI+CVD

        # OBI+CVD Confirmation
        self.obi_cvd_confirmation = OBICVDConfirmation(cvd_reset_minutes=30, obi_window_seconds=60)

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

            # Start continuous Black-Scholes calculation thread
            self.bs_thread = threading.Thread(target=self._black_scholes_continuous_loop, daemon=True)
            self.bs_thread.start()

            print("✅ Live prediction pipeline started successfully")
            print("🔧 Continuous Black-Scholes calculator started")
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

        if self.bs_thread:
            self.bs_thread.join(timeout=5)

        # Reset candle tracking so predictions can restart on reconnection
        self.last_candle_timestamps = {}
        self.live_predictions = {}
        self.prediction_history = {}
        self.latest_volatility_predictions = {}

        # Reset OBI+CVD confirmation data
        self.obi_cvd_confirmation = OBICVDConfirmation(cvd_reset_minutes=30, obi_window_seconds=60)

        print("🔌 Live prediction pipeline stopped")
        print("🔧 Continuous Black-Scholes calculator stopped")

    def subscribe_instruments(self, instrument_keys: List[str]) -> bool:
        """Subscribe to instruments for live predictions with dedicated routing."""
        # Always subscribe to both dedicated instruments regardless of user selection
        dedicated_instruments = [self.ml_models_instrument, self.obi_cvd_instrument]

        # Combine with user selected instruments but avoid duplicates
        all_instruments = list(set(instrument_keys + dedicated_instruments))

        print(f"🎯 Subscribing with dedicated instrument routing:")
        print(f"   ML Models + BSM: {self.ml_models_instrument}")
        print(f"   OBI+CVD: {self.obi_cvd_instrument}")
        print(f"   Additional instruments: {[inst for inst in instrument_keys if inst not in dedicated_instruments]}")

        # Configure mixed mode subscription
        mixed_mode_config = {
            "NSE_INDEX|Nifty 50": "ltpc",  # Index uses LTPC mode
            "NSE_FO|52168": "full"         # Futures use full mode
        }

        # Subscribe to instruments with mixed mode configuration (this includes seeding)
        if self.live_data_manager.subscribe(all_instruments, mixed_mode_config):
            print(f"✅ Subscribed to {len(all_instruments)} instruments with mixed mode configuration")
            return True
        else:
            print("❌ Failed to subscribe to instruments")
            return False

    def _processing_loop(self):
        """Main processing loop for generating live predictions."""
        print(f"🚀 Prediction processing loop started - will check every {self.update_interval} second")
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
                    # Process ML models predictions only on dedicated ML instrument
                    if self.ml_models_instrument in tick_stats:
                        # Debug: Show processing attempt
                        if not hasattr(self, '_debug_counter'):
                            self._debug_counter = 0
                        self._debug_counter += 1

                        # Only show debug every 10 iterations to avoid spam
                        if self._debug_counter % 10 == 0:
                            print(f"🔍 Processing loop #{self._debug_counter} - checking ML models on {self.ml_models_instrument}")

                        # Check if a new candle has closed before processing predictions
                        if self._has_new_candle_closed(self.ml_models_instrument):
                            print(f"🎯 New candle detected for ML models on {self.ml_models_instrument}, processing predictions...")
                            self._process_instrument_predictions(self.ml_models_instrument)

                    # Process any additional user-selected instruments (legacy support)
                    for instrument_key, stats in tick_stats.items():
                        if instrument_key not in [self.ml_models_instrument, self.obi_cvd_instrument]:
                            if self._has_new_candle_closed(instrument_key):
                                print(f"🎯 New candle detected for additional instrument {instrument_key}, processing predictions...")
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
                            volatility_value = float(predictions[-1])

                            all_predictions['volatility'] = {
                                'prediction': volatility_level,
                                'value': volatility_value
                            }
                            print(f"✅ Volatility prediction successful: {volatility_level}")

                            # Store latest volatility for continuous Black-Scholes calculations
                            self.latest_volatility_predictions[instrument_key] = {
                                'volatility_value': volatility_value,
                                'volatility_level': volatility_level,
                                'timestamp': timestamp,
                                'valid_until': timestamp + pd.Timedelta(minutes=5)  # Valid for next 5 minutes
                            }
                            print(f"📊 Volatility stored for continuous Black-Scholes: {volatility_value:.4f}")

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

            # Format comprehensive prediction (without OBI+CVD confirmation)
            latest_prediction = self._format_comprehensive_prediction(
                instrument_key, timestamp, all_predictions, ohlc_row, None
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
            # Get expected features from trained model
            model_data = self.model_manager.trained_models.get('volatility', {})
            expected_features = model_data.get('feature_names', [])

            if not expected_features:
                print(f"❌ No feature names found for volatility model")
                return None

            # Use volatility model's prepare_features method directly
            features = self.model_manager.models['volatility'].prepare_features(ohlc_data)

            if features is None:
                return None

            # Ensure we have all expected features
            missing_features = [col for col in expected_features if col not in features.columns]
            available_features = [col for col in expected_features if col in features.columns]

            if missing_features:
                print(f"⚠️ Missing features for volatility: {missing_features}")

                # Add missing OHLC features if they exist in original data
                for missing_col in missing_features:
                    if missing_col in ['Open', 'High', 'Low', 'Close', 'Volume'] and missing_col in ohlc_data.columns:
                        features[missing_col] = ohlc_data[missing_col]
                        available_features.append(missing_col)
                        print(f"✅ Added missing OHLC feature: {missing_col}")

            # Use only available features that match training
            final_features = [col for col in expected_features if col in features.columns]

            if len(final_features) < len(expected_features) * 0.8:
                print(f"❌ Too many missing features for volatility. Available: {len(final_features)}, Expected: {len(expected_features)}")
                return None

            return features[final_features]

        except Exception as e:
            print(f"❌ Error calculating volatility features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_profit_probability_features(self, ohlc_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate profit probability features from OHLC data."""
        try:
            # Get expected features from trained model
            model_data = self.model_manager.trained_models.get('profit_probability', {})
            expected_features = model_data.get('feature_names', [])

            if not expected_features:
                print(f"❌ No feature names found for profit probability model")
                return None

            # Use profit probability model's prepare_features method directly
            features = self.model_manager.models['profit_probability'].prepare_features(ohlc_data)

            if features is None:
                return None

            # Ensure we have all expected features
            missing_features = [col for col in expected_features if col not in features.columns]
            available_features = [col for col in expected_features if col in features.columns]

            if missing_features:
                print(f"⚠️ Missing features for profit probability: {missing_features}")

                # Add missing OHLC features if they exist in original data
                for missing_col in missing_features:
                    if missing_col in ['Open', 'High', 'Low', 'Close', 'Volume'] and missing_col in ohlc_data.columns:
                        features[missing_col] = ohlc_data[missing_col]
                        available_features.append(missing_col)
                        print(f"✅ Added missing OHLC feature: {missing_col}")

            # Use only available features that match training
            final_features = [col for col in expected_features if col in features.columns]

            if len(final_features) < len(expected_features) * 0.8:
                print(f"❌ Too many missing features for profit probability. Available: {len(final_features)}, Expected: {len(expected_features)}")
                return None

            return features[final_features]

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

    def _black_scholes_continuous_loop(self):
        """Continuous loop for Black-Scholes calculations using live tick data."""
        print(f"🚀 Black-Scholes continuous loop started - updating every {self.bs_update_interval} second")
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_processing:
            try:
                # Check if connection is still alive
                connection_status = self.live_data_manager.get_connection_status()

                if not connection_status['connected']:
                    time.sleep(10)
                    continue

                # Process Black-Scholes only for ML models instrument (where volatility is predicted)
                if self.ml_models_instrument in self.latest_volatility_predictions:
                    instrument_key = self.ml_models_instrument
                    vol_data = self.latest_volatility_predictions[instrument_key]

                    try:
                        # Check if volatility prediction is still valid (within 5 minutes)
                        current_time = pd.Timestamp.now()
                        if current_time > vol_data['valid_until']:
                            print(f"⚠️ Volatility prediction expired for {instrument_key}, waiting for new prediction...")
                            time.sleep(self.bs_update_interval)
                            continue

                        # Get latest tick data from ML models instrument
                        latest_tick = self.live_data_manager.ws_client.get_latest_tick(instrument_key)

                        # Update OBI+CVD ONLY with dedicated 53001 futures instrument data - NO FALLBACK
                        if self.obi_cvd_instrument in self.live_data_manager.ws_client.last_tick_data:
                            obi_cvd_tick = self.live_data_manager.ws_client.get_latest_tick(self.obi_cvd_instrument)
                            if obi_cvd_tick and '53001' in str(self.obi_cvd_instrument):
                                # Strict validation - only process 53001 data
                                if '53001' in str(obi_cvd_tick.get('instrument_token', '')):
                                    self.obi_cvd_confirmation.update_confirmation(self.obi_cvd_instrument, obi_cvd_tick)

                                    # Generate trade signal every 10 ticks to avoid excessive computation
                                    if not hasattr(self, '_trade_signal_counter'):
                                        self._trade_signal_counter = 0
                                    self._trade_signal_counter += 1

                                    if self._trade_signal_counter % 10 == 0:
                                        trade_signal = self.obi_cvd_confirmation.generate_trade_signal(self.obi_cvd_instrument)
                                        signal = trade_signal.get('signal', 'NEUTRAL')
                                        confidence = trade_signal.get('confidence', 0.0)
                                        score = trade_signal.get('score', 0.0)

                                        # Store latest trade signal for UI access
                                        if not hasattr(self, 'latest_trade_signals'):
                                            self.latest_trade_signals = {}
                                        self.latest_trade_signals[self.obi_cvd_instrument] = trade_signal

                                        print(f"🎯 Trade Signal (53001): {signal} | Score: {score:.3f} | Confidence: {confidence:.1f}%")

                                    print(f"✅ OBI+CVD updated with 53001 data")
                                else:
                                    print(f"⚠️ Skipping OBI+CVD update - tick not from 53001: {obi_cvd_tick.get('instrument_token', 'unknown')}")
                            else:
                                print(f"⚠️ Waiting for 53001 tick data for OBI+CVD...")

                        if latest_tick and 'ltp' in latest_tick:
                            current_price = float(latest_tick['ltp'])
                            volatility_value = vol_data['volatility_value']

                            # Calculate Black-Scholes with current tick price from ML models instrument
                            bs_results = self._calculate_black_scholes_fair_values(current_price, volatility_value)

                            if bs_results.get('calculation_successful', False):
                                # Update the live predictions with fresh Black-Scholes data
                                if instrument_key in self.live_predictions:
                                    self.live_predictions[instrument_key]['black_scholes'] = bs_results
                                    self.live_predictions[instrument_key]['bs_current_price'] = current_price
                                    self.live_predictions[instrument_key]['bs_volatility_5min'] = volatility_value
                                    self.live_predictions[instrument_key]['bs_volatility_annualized'] = bs_results.get('annualized_volatility', volatility_value)
                                    self.live_predictions[instrument_key]['bs_last_update'] = current_time

                                    # Show update every 1 iteration for 1-second refresh
                                    if not hasattr(self, '_bs_counter'):
                                        self._bs_counter = 0
                                    self._bs_counter += 1

                                    if self._bs_counter % 1 == 0:
                                        display_name = instrument_key.split('|')[-1] if '|' in instrument_key else instrument_key
                                        annualized_vol = bs_results.get('annualized_volatility', volatility_value)

                                        # Get OBI+CVD signal from futures instrument
                                        obi_cvd_signal = 'Unknown'
                                        if self.obi_cvd_instrument in self.obi_cvd_confirmation.instrument_data:
                                            obi_cvd_status = self.obi_cvd_confirmation.get_confirmation_status(self.obi_cvd_instrument)
                                            if obi_cvd_status:
                                                obi_cvd_signal = obi_cvd_status.get('combined_confirmation', 'Unknown')

                                        futures_name = self.obi_cvd_instrument.split('|')[-1] if '|' in self.obi_cvd_instrument else self.obi_cvd_instrument
                                        print(f"🔧 Live Update ML+BSM ({display_name}): ₹{current_price:.2f} | Vol: {volatility_value:.4f}→{annualized_vol:.2f}")
                                        print(f"📊 OBI+CVD (53001 ONLY): {obi_cvd_signal}")

                    except Exception as e:
                        print(f"❌ Error calculating Black-Scholes for {instrument_key}: {e}")
                        continue

                # Reset error counter on successful processing
                consecutive_errors = 0

                # Wait before next Black-Scholes update
                time.sleep(self.bs_update_interval)

            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Error in Black-Scholes loop ({consecutive_errors}/{max_consecutive_errors}): {e}")

                if consecutive_errors >= max_consecutive_errors:
                    print("❌ Too many consecutive errors in Black-Scholes loop, stopping")
                    break

                time.sleep(min(30, 5 * consecutive_errors))  # Progressive backoff

    def _calculate_black_scholes_fair_values(self, current_price: float, volatility_prediction: float) -> Dict:
        """Calculate Black-Scholes fair values for options and index."""
        try:
            from utils.black_scholes import BlackScholesCalculator

            # Convert 5-minute volatility to annualized volatility
            # Trading periods: 75 periods per day (5-min candles in 6.25-hour trading day)
            # Trading days: 250 per year
            # Total periods per year: 75 × 250 = 18,750
            periods_per_year = 75 * 250  # 18,750
            annualized_volatility = volatility_prediction * (periods_per_year ** 0.5)

            print(f"📊 Volatility conversion: 5-min={volatility_prediction:.6f} → Annualized={annualized_volatility:.4f}")

            # Initialize Black-Scholes calculator with RBI repo rate 5.50% and dividend yield 1.2%
            bs_calculator = BlackScholesCalculator(risk_free_rate=0.055, dividend_yield=0.012)

            # Calculate index fair value using annualized volatility
            index_fair_value = bs_calculator.calculate_index_fair_value(current_price, annualized_volatility)

            # Calculate options fair values for different strikes using annualized volatility
            options_fair_values = bs_calculator.calculate_options_fair_values(current_price, annualized_volatility)

            # Get quick summary for display using annualized volatility
            quick_summary = bs_calculator.get_quick_summary(current_price, annualized_volatility)

            return {
                'index_fair_value': index_fair_value,
                'options_fair_values': options_fair_values,
                'quick_summary': quick_summary,
                'raw_volatility_5min': volatility_prediction,
                'annualized_volatility': annualized_volatility,
                'conversion_factor': periods_per_year ** 0.5,
                'calculation_successful': True
            }

        except Exception as e:
            print(f"❌ Error calculating Black-Scholes fair values: {e}")
            return {
                'error': str(e),
                'calculation_successful': False
            }

    def _format_comprehensive_prediction(self, instrument_key: str, timestamp: pd.Timestamp,
                                        all_predictions: Dict, ohlc_row: pd.Series, obi_cvd_status: Optional[Dict] = None) -> Dict:
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
            'prediction_type': 'candle_completion',
            'has_volatility_for_bs': instrument_key in self.latest_volatility_predictions,
            'bs_continuous_active': instrument_key in self.latest_volatility_predictions
        }

        # Add predictions from each model
        for model_name, prediction_data in all_predictions.items():
            formatted_prediction[model_name] = prediction_data

        # Add OBI+CVD confirmation if available
        if obi_cvd_status:
            formatted_prediction['obi_cvd_confirmation'] = obi_cvd_status

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
            'instruments_with_volatility': len(self.latest_volatility_predictions),
            'black_scholes_active': len(self.latest_volatility_predictions) > 0,
            'obi_cvd_active': len(self.obi_cvd_confirmation.instrument_data) > 0,
            'instruments_with_obi_cvd': len(self.obi_cvd_confirmation.instrument_data),
            'total_ticks_received': connection_status['total_ticks_received'],
            'last_prediction_time': max(
                [pred['generated_at'] for pred in self.live_predictions.values()]
            ) if self.live_predictions else None,
            'trained_models': trained_models,
            'total_trained_models': len(trained_models),
            **model_status
        }

    def get_independent_obi_cvd_status(self, instrument_key: str = None) -> Optional[Dict]:
        """Get independent OBI+CVD confirmation status from dedicated futures instrument."""
        # Always use dedicated OBI+CVD instrument regardless of input
        target_instrument = self.obi_cvd_instrument
        return self.obi_cvd_confirmation.get_confirmation_status(target_instrument)

    def get_all_independent_obi_cvd_status(self) -> Dict:
        """Get independent OBI+CVD status for dedicated futures instrument."""
        obi_cvd_status = {}
        # Only return status for the dedicated OBI+CVD instrument
        status = self.get_independent_obi_cvd_status()
        if status:
            obi_cvd_status[self.obi_cvd_instrument] = status
        return obi_cvd_status

    def get_latest_trade_signal(self, instrument_key: str = None) -> Optional[Dict]:
        """Get the latest trade signal for an instrument (defaults to OBI+CVD instrument)."""
        if instrument_key is None:
            instrument_key = self.obi_cvd_instrument

        if hasattr(self, 'latest_trade_signals') and instrument_key in self.latest_trade_signals:
            return self.latest_trade_signals[instrument_key]

        # Generate fresh signal if none cached
        try:
            trade_signal = self.obi_cvd_confirmation.generate_trade_signal(instrument_key)
            return trade_signal
        except Exception as e:
            print(f"❌ Error getting trade signal for {instrument_key}: {e}")
            return None

    def get_all_trade_signals(self) -> Dict:
        """Get all available trade signals."""
        if hasattr(self, 'latest_trade_signals'):
            return self.latest_trade_signals.copy()
        return {}

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