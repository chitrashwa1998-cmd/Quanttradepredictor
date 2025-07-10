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
        self.update_interval = 30  # Process predictions every 30 seconds

        # Minimum data requirements (reduced for faster live predictions)
        self.min_ohlc_rows = 50  # Minimum OHLC rows needed for reliable predictions

    def start_pipeline(self) -> bool:
        """Start the live prediction pipeline."""
        try:
            # Connect to live data feed
            if not self.live_data_manager.connect():
                print("❌ Failed to connect to live data feed")
                return False

            # Check if direction model is trained
            if not self.model_manager.is_model_trained('direction'):
                print("❌ Direction model not trained. Please train the model first.")
                return False

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

        print("🔌 Live prediction pipeline stopped")

    def subscribe_instruments(self, instrument_keys: List[str]) -> bool:
        """Subscribe to instruments for live predictions."""
        return self.live_data_manager.subscribe_instruments(instrument_keys)

    def _processing_loop(self):
        """Main processing loop for generating live predictions."""
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
                        # Bootstrap OHLC data if we have enough ticks but insufficient OHLC
                        if stats['tick_count'] >= 20 and stats['ohlc_rows'] < self.min_ohlc_rows:
                            print(f"🚀 Bootstrapping OHLC data for {instrument_key}")
                            self.live_data_manager.bootstrap_ohlc_from_ticks(instrument_key)

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

    def _process_instrument_predictions(self, instrument_key: str):
        """Process predictions for a specific instrument."""
        try:
            # Get live OHLC data
            ohlc_data = self.live_data_manager.get_live_ohlc(instrument_key, rows=200)

            if ohlc_data is None or len(ohlc_data) < self.min_ohlc_rows:
                if ohlc_data is not None:
                    # Check if data is seeded from historical database
                    seeding_status = self.live_data_manager.get_seeding_status()
                    if seeding_status['is_seeded'] and len(ohlc_data) >= 20:
                        print(f"🌱 Using seeded historical data for {instrument_key}: {len(ohlc_data)} rows (sufficient for predictions)")
                    else:
                        print(f"📊 Building OHLC data for {instrument_key}: {len(ohlc_data)}/{self.min_ohlc_rows} rows needed")
                        return
                else:
                    print(f"📊 No OHLC data available for {instrument_key}")
                    return

            # Calculate direction features
            features = self._calculate_direction_features(ohlc_data)

            if features is None or len(features) == 0:
                print(f"❌ Failed to calculate features for {instrument_key}")
                return

            # Make predictions
            predictions, probabilities = self.model_manager.predict('direction', features)

            if predictions is None:
                print(f"❌ Failed to generate predictions for {instrument_key}")
                return

            # Store latest prediction
            latest_prediction = self._format_prediction(
                instrument_key, 
                features.index[-1], 
                predictions[-1], 
                probabilities[-1] if probabilities is not None else None,
                ohlc_data.iloc[-1]
            )

            self.live_predictions[instrument_key] = latest_prediction

            # Store in history
            if instrument_key not in self.prediction_history:
                self.prediction_history[instrument_key] = deque(maxlen=self.max_history)

            self.prediction_history[instrument_key].append(latest_prediction)

            print(f"✅ Generated live prediction for {instrument_key}: {latest_prediction['direction']} ({latest_prediction['confidence']:.3f})")

        except Exception as e:
            print(f"❌ Error processing predictions for {instrument_key}: {e}")

    def _calculate_direction_features(self, ohlc_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate direction-specific features from OHLC data."""
        try:
            # Use the same feature calculation as the trained model
            dti = DirectionTechnicalIndicators()
            features = dti.calculate_all_direction_indicators(ohlc_data)

            # Remove rows with NaN values
            features = features.dropna()

            return features

        except Exception as e:
            print(f"❌ Error calculating direction features: {e}")
            return None

    def _format_prediction(self, instrument_key: str, timestamp: pd.Timestamp, 
                          prediction: int, probability: Optional[np.ndarray],
                          ohlc_row: pd.Series) -> Dict:
        """Format prediction data for storage and display."""
        direction = 'Bullish' if prediction == 1 else 'Bearish'
        confidence = np.max(probability) if probability is not None else 0.5

        return {
            'instrument': instrument_key,
            'timestamp': timestamp,
            'direction': direction,
            'confidence': confidence,
            'prediction_value': int(prediction),
            'current_price': float(ohlc_row['Close']),
            'volume': int(ohlc_row['Volume']) if 'Volume' in ohlc_row else 0,
            'generated_at': datetime.now()
        }

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

        return {
            'pipeline_active': self.is_processing,
            'data_connected': connection_status['connected'],
            'subscribed_instruments': connection_status['subscribed_instruments'],
            'instruments_with_predictions': len(self.live_predictions),
            'total_ticks_received': connection_status['total_ticks_received'],
            'last_prediction_time': max(
                [pred['generated_at'] for pred in self.live_predictions.values()]
            ) if self.live_predictions else None,
            'model_ready': self.model_manager.is_model_trained('direction')
        }

    def get_instrument_summary(self, instrument_key: str) -> Optional[Dict]:
        """Get summary for a specific instrument."""
        if instrument_key not in self.live_predictions:
            return None

        latest = self.live_predictions[instrument_key]
        history = self.get_prediction_history(instrument_key, 20)

        if not history:
            return latest

        # Calculate recent statistics
        recent_directions = [p['direction'] for p in history]
        bullish_count = recent_directions.count('Bullish')
        bearish_count = recent_directions.count('Bearish')

        recent_confidences = [p['confidence'] for p in history]
        avg_confidence = np.mean(recent_confidences)

        return {
            **latest,
            'recent_stats': {
                'total_predictions': len(history),
                'bullish_signals': bullish_count,
                'bearish_signals': bearish_count,
                'bullish_percentage': (bullish_count / len(history)) * 100,
                'average_confidence': avg_confidence,
                'confidence_trend': 'improving' if recent_confidences[-1] > avg_confidence else 'declining'
            }
        }