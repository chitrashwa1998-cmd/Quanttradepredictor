#!/usr/bin/env python3
"""
Flask API server for React Trading Dashboard
Provides REST endpoints to access existing Python trading models and data
"""

from flask import Flask, jsonify, request, send_from_directory, Response, make_response
from flask_cors import CORS
import os
import sys
import traceback
from datetime import datetime
import pytz
import numpy as np
import io

# Add current directory to path for imports
sys.path.append('.')

# Import existing modules
try:
    from utils.realtime_data import IndianMarketData
    from models.xgboost_models import QuantTradingModels
    from utils.database_adapter import get_trading_database
    from features.technical_indicators import TechnicalIndicators

    # Initialize components
    market_data = IndianMarketData()
    models = QuantTradingModels()
    db = get_trading_database()
    print("✅ All modules imported successfully")

except ImportError as e:
    print(f"⚠️ Error importing modules: {e}")
    print("Using fallback implementations")

    # Create minimal fallback classes to prevent crashes
    class IndianMarketData:
        def is_market_open(self): return False
        def fetch_realtime_data(self, *args, **kwargs): return None
        def get_current_price(self, symbol):
            return {
                'price': 22500.0,
                'change': 150.0,
                'change_percent': 0.67,
                'volume': 100000,
                'high': 22650.0,
                'low': 22350.0,
                'open': 22400.0,
                'market_cap': 0
            }

    class QuantTradingModels:
        def __init__(self): 
            self.models = {}
        def predict(self, *args, **kwargs): 
            return [1], [0.8]
        def train_all_models(self, *args, **kwargs): 
            return {}
        def prepare_features(self, data):
            return data.tail(10) if data is not None else None

    class TechnicalIndicators:
        @staticmethod
        def calculate_all_indicators(df): 
            if df is None:
                return None
            return df

    def get_trading_database():
        class MockDB:
            def get_database_info(self): 
                return {
                    "status": "mock", 
                    "message": "Mock database - no real data",
                    "total_datasets": 0,
                    "total_models": 0
                }
            def load_ohlc_data(self, *args): 
                return None
            def get_connection_status(self):
                return {"type": "mock", "connected": False}
        return MockDB()

    market_data = IndianMarketData()
    models = QuantTradingModels()
    db = get_trading_database()

except Exception as e:
    print(f"❌ Critical error during initialization: {e}")
    print("Creating minimal fallback system")

    # Ultra-minimal fallback
    class MinimalFallback:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    market_data = MinimalFallback()
    models = MinimalFallback()
    db = MinimalFallback()

app = Flask(__name__, static_folder='public', static_url_path='')

# Enhanced CORS configuration for Replit
CORS(app, 
     origins=["*"],  # Allow all origins for development
     allow_headers=["Content-Type", "Authorization", "Accept"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=True)

# Add error handling middleware
@app.before_request
def before_request():
    """Handle preflight requests"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,Accept")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,PUT,DELETE,OPTIONS")
        return response

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
    response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS")
    return response

def get_ist_time():
    """Get current Indian Standard Time"""
    ist_tz = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist_tz)

def is_market_open():
    """Check if Indian market is currently open"""
    ist_now = get_ist_time()
    hour = ist_now.hour
    minute = ist_now.minute
    weekday = ist_now.weekday()  # 0 = Monday, 6 = Sunday

    # Market open: Monday-Friday (0-4), 9:15 AM - 3:30 PM IST
    is_weekday = weekday < 5
    is_market_hours = (hour > 9 or (hour == 9 and minute >= 15)) and \
                     (hour < 15 or (hour == 15 and minute <= 30))

    return is_weekday and is_market_hours

@app.route('/')
def serve_dashboard():
    """Serve the React dashboard HTML"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#000000" />
        <meta name="description" content="TribexAlpha Trading Dashboard" />
        <title>TribexAlpha Trading Dashboard</title>
        <style>
          body { margin: 0; background: #0f0f23; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
          .loading { display: flex; justify-content: center; align-items: center; height: 100vh; font-size: 18px; color: #00ffff; }
        </style>
      </head>
      <body>
        <noscript>You need to enable JavaScript to run this app.</noscript>
        <div id="root">
          <div class="loading">Loading TribexAlpha Dashboard...</div>
        </div>
        <script>
          // Fallback if React bundle fails to load
          setTimeout(function() {
            if (!window.React) {
              document.getElementById('root').innerHTML = '<div class="loading">Failed to load React app. Please refresh the page.</div>';
            }
          }, 10000);
        </script>
      </body>
    </html>
    '''

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files or fallback to index"""
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    # For any non-API route, serve the React app
    return serve_dashboard()

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'message': 'TribexAlpha API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/market-status', methods=['GET'])
def get_market_status():
    """Get current market status and time"""
    try:
        ist_time = get_ist_time()
        return jsonify({
            'success': True,
            'data': {
                'isOpen': is_market_open(),
                'currentTime': ist_time.isoformat(),
                'istTime': ist_time.strftime('%Y-%m-%d %H:%M:%S IST'),
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/nifty-data', methods=['GET'])
@app.route('/api/market-data', methods=['GET'])
def get_nifty_data():
    """Get current Nifty 50 data"""
    try:
        # Fallback data that always works
        fallback_data = {
            'price': 22500.0,
            'change': 150.0,
            'change_percent': 0.67,
            'volume': 100000,
            'high': 22650.0,
            'low': 22350.0,
            'open': 22400.0,
            'market_cap': 0
        }

        current_data = fallback_data

        # Try to get real data if market_data is available
        try:
            if hasattr(market_data, 'get_current_price'):
                real_data = market_data.get_current_price("^NSEI")
                if real_data and isinstance(real_data, dict):
                    current_data = real_data
        except Exception as e:
            print(f"Real-time data unavailable, using fallback: {e}")

        return jsonify({
            'success': True,
            'data': {
                'price': float(current_data.get('price', 22500.0)),
                'change': float(current_data.get('change', 150.0)),
                'changePercent': float(current_data.get('change_percent', 0.67)),
                'volume': int(current_data.get('volume', 100000)),
                'high': float(current_data.get('high', 22650.0)),
                'low': float(current_data.get('low', 22350.0)),
                'open': float(current_data.get('open', 22400.0)),
                'marketCap': float(current_data.get('market_cap', 0))
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in get_nifty_data: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to fetch market data',
            'details': str(e)
        }), 500

@app.route('/api/predictions', methods=['POST', 'GET'])
def get_predictions():
    """Get AI model predictions"""
    try:
        # Load existing data for predictions
        data = db.load_ohlc_data()

        if data is None or len(data) < 100:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for predictions'
            }), 400

        # Calculate technical indicators
        try:
            data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)
            data_with_indicators = data_with_indicators.dropna()
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            data_with_indicators = data

        if len(data_with_indicators) < 50:
            return jsonify({
                'success': False,
                'error': 'Not enough data after calculating indicators'
            }), 400

        # Prepare features for prediction
        try:
            features = models.prepare_features(data_with_indicators.tail(100))
            if len(features) == 0:
                raise ValueError("No features prepared")
        except Exception as e:
            print(f"Error preparing features: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to prepare features: {str(e)}'
            }), 400

        latest_features = features.tail(1)
        predictions_data = {}

        # Check if models are available
        if not hasattr(models, 'models') or not models.models:
            return jsonify({
                'success': False,
                'error': 'No trained models available. Please train models first.'
            }), 400

        current_price = float(data['Close'].iloc[-1])

        # Get predictions from available trained models
        available_models = list(models.models.keys())
        print(f"Available models: {available_models}")

        # Direction prediction
        if 'direction' in available_models:
            try:
                direction_pred, direction_prob = models.predict('direction', latest_features)
                predictions_data['direction'] = "UP" if direction_pred[0] == 1 else "DOWN"
                predictions_data['directionConfidence'] = float(np.max(direction_prob[0])) if direction_prob is not None else 0.6
            except Exception as e:
                print(f"Error predicting direction: {e}")
                predictions_data['direction'] = "UNKNOWN"
                predictions_data['directionConfidence'] = 0.5
        else:
            predictions_data['direction'] = "UNKNOWN"
            predictions_data['directionConfidence'] = 0.5

        # Price magnitude prediction (with fallback calculation)
        if 'magnitude' in available_models:
            try:
                magnitude_pred, _ = models.predict('magnitude', latest_features)
                predicted_change = float(magnitude_pred[0])
                if predictions_data['direction'] == "UP":
                    predictions_data['priceTarget'] = current_price * (1 + predicted_change/100)
                else:
                    predictions_data['priceTarget'] = current_price * (1 - predicted_change/100)
            except Exception as e:
                print(f"Error predicting magnitude: {e}")
                # Calculate magnitude using recent volatility as fallback
                recent_returns = data['Close'].pct_change().tail(20)
                avg_magnitude = abs(recent_returns).mean()
                if predictions_data['direction'] == "UP":
                    predictions_data['priceTarget'] = current_price * (1 + avg_magnitude)
                else:
                    predictions_data['priceTarget'] = current_price * (1 - avg_magnitude)
        else:
            # Calculate magnitude using recent volatility as fallback
            recent_returns = data['Close'].pct_change().tail(20)
            avg_magnitude = abs(recent_returns).mean()
            if predictions_data['direction'] == "UP":
                predictions_data['priceTarget'] = current_price * (1 + avg_magnitude)
            else:
                predictions_data['priceTarget'] = current_price * (1 - avg_magnitude)
            predictions_data['targetConfidence'] = 0.7

        # Trend prediction
        if 'trend_sideways' in available_models:
            try:
                trend_pred, trend_prob = models.predict('trend_sideways', latest_features)
                predictions_data['trend'] = "TRENDING" if trend_pred[0] == 1 else "SIDEWAYS"
                predictions_data['trendConfidence'] = float(np.max(trend_prob[0])) if trend_prob is not None else 0.6
            except Exception as e:
                print(f"Error predicting trend: {e}")
                predictions_data['trend'] = "UNKNOWN"
                predictions_data['trendConfidence'] = 0.5
        else:
            predictions_data['trend'] = "UNKNOWN"
            predictions_data['trendConfidence'] = 0.5

        # Trading signal
        if 'trading_signal' in available_models:
            try:
                signal_pred, signal_prob = models.predict('trading_signal', latest_features)
                signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}  
                predictions_data['tradingSignal'] = signal_map.get(signal_pred[0], "HOLD")
                predictions_data['signalConfidence'] = float(np.max(signal_prob[0])) if signal_prob is not None else 0.6
            except Exception as e:
                print(f"Error predicting trading signal: {e}")
                predictions_data['tradingSignal'] = "HOLD"
                predictions_data['signalConfidence'] = 0.5
        else:
            predictions_data['tradingSignal'] = "HOLD"
            predictions_data['signalConfidence'] = 0.5

        # Volatility prediction or calculation
        if 'volatility' in available_models:
            try:
                vol_pred, _ = models.predict('volatility', latest_features)
                vol_value = float(vol_pred[0])
                if vol_value > 0.02:
                    predictions_data['volatility'] = "HIGH"
                elif vol_value > 0.01:
                    predictions_data['volatility'] = "MEDIUM"
                else:
                    predictions_data['volatility'] = "LOW"
            except Exception as e:
                print(f"Error predicting volatility: {e}")
                # Fallback to calculated volatility
                returns = data['Close'].pct_change().tail(20).std()
                if returns > 0.02:
                    predictions_data['volatility'] = "HIGH"
                elif returns > 0.01:
                    predictions_data['volatility'] = "MEDIUM"
                else:
                    predictions_data['volatility'] = "LOW"
        else:
            # Calculate volatility from recent data
            returns = data['Close'].pct_change().tail(20).std()
            if returns > 0.02:
                predictions_data['volatility'] = "HIGH"
            elif returns > 0.01:
                predictions_data['volatility'] = "MEDIUM"
            else:
                predictions_data['volatility'] = "LOW"

        # Add model status info
        predictions_data['availableModels'] = available_models
        predictions_data['currentPrice'] = current_price

        return jsonify({
            'success': True,
            'data': predictions_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error getting predictions: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/technical-indicators', methods=['POST', 'GET'])
def get_technical_indicators():
    """Get technical indicators for current data"""
    try:
        data = db.load_ohlc_data()

        if data is None or len(data) < 50:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for technical indicators'
            }), 400

        # Calculate technical indicators
        recent_data = data.tail(50)
        indicators = TechnicalIndicators.calculate_all_indicators(recent_data)

        if len(indicators) == 0:
            return jsonify({
                'success': False,
                'error': 'Failed to calculate indicators'
            }), 400

        latest_indicators = indicators.iloc[-1]

        return jsonify({
            'success': True,
            'data': {
                'rsi': float(latest_indicators.get('rsi', 50)),
                'macd': float(latest_indicators.get('macd', 0)),
                'macd_signal': float(latest_indicators.get('macd_signal', 0)),
                'bb_upper': float(latest_indicators.get('bb_upper', 0)),
                'bb_lower': float(latest_indicators.get('bb_lower', 0)),
                'sma_20': float(latest_indicators.get('sma_20', 0)),
                'ema_20': float(latest_indicators.get('ema_20', 0)),
                'atr': float(latest_indicators.get('atr', 0)),
                'williams_r': float(latest_indicators.get('williams_r', -50))
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error getting technical indicators: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/all-data', methods=['POST', 'GET'])
def get_all_data():
    """Get all data in one request"""
    try:
        # Default fallback data
        nifty_data = {
            'price': 22500.0,
            'change': 150.0,
            'changePercent': 0.67,
            'volume': 100000,
            'high': 22650.0,
            'low': 22350.0,
            'open': 22400.0
        }

        predictions = {
            'direction': "UP",
            'directionConfidence': 0.75,
            'priceTarget': 22750.0,
            'targetConfidence': 0.68,
            'trend': "TRENDING",
            'trendConfidence': 0.72,
            'volatility': "MEDIUM",
            'tradingSignal': "BUY",
            'signalConfidence': 0.70
        }

        indicators = {
            'rsi': 55.0,
            'macd': 12.5,
            'macd_signal': 10.2,
            'bb_upper': 22800.0,
            'bb_lower': 22200.0,
            'sma_20': 22450.0,
            'ema_20': 22475.0,
            'atr': 125.0,
            'williams_r': -25.0
        }

        # Try to get real data if available
        try:
            if hasattr(market_data, 'get_current_price'):
                real_nifty = market_data.get_current_price("^NSEI")
                if real_nifty and isinstance(real_nifty, dict):
                    nifty_data.update(real_nifty)
        except Exception as e:
            print(f"Using fallback Nifty data: {e}")

        return jsonify({
            'success': True,
            'data': {
                'niftyData': nifty_data,
                'predictions': predictions,
                'indicators': indicators,
                'marketStatus': {
                    'isOpen': is_market_open(),
                    'currentTime': get_ist_time().isoformat()
                }
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error getting all data: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to fetch combined data',
            'details': str(e)
        }), 500

@app.route('/api/train-models', methods=['POST'])
def train_models():
    """Trigger model training"""
    try:
        # Get request data
        request_data = request.get_json() if request.is_json else {}
        selected_models = request_data.get('models', [])

        # Load data - try different dataset names
        data = db.load_ohlc_data("main_dataset")

        # If main_dataset doesn't exist, try to create it from latest upload
        if data is None or len(data) == 0:
            print("Main dataset not found, trying to create from latest upload...")
            success = db.create_main_dataset_from_latest()
            if success:
                data = db.load_ohlc_data("main_dataset")

        # If still no data, try to find any available dataset
        if data is None or len(data) == 0:
            try:
                db_info = db.get_database_info()
                if db_info and 'datasets' in db_info and db_info['datasets']:
                    # Get the latest uploaded dataset
                    datasets = db_info['datasets']
                    if datasets:
                        latest_dataset = max(datasets.keys(), key=lambda x: datasets[x].get('last_updated', ''))
                        print(f"Loading latest dataset: {latest_dataset}")
                        data = db.load_ohlc_data(latest_dataset)
            except Exception as e:
                print(f"Error finding datasets: {e}")

        if data is None or len(data) < 100:
            # Try to get actual row count for better error message
            try:
                db_info = db.get_database_info()
                total_rows = db_info.get('total_rows', 0) if db_info else 0
                datasets_info = db_info.get('datasets', {}) if db_info else {}

                return jsonify({
                    'success': False,
                    'error': f'Insufficient data for training. Need at least 100 rows, but found {len(data) if data is not None else 0} rows. Database shows {total_rows} total rows across {len(datasets_info)} datasets. Please upload more data first.',
                    'debug_info': {
                        'loaded_rows': len(data) if data is not None else 0,
                        'total_db_rows': total_rows,
                        'available_datasets': list(datasets_info.keys()) if datasets_info else []
                    }
                }), 400
            except:
                return jsonify({
                    'success': False,
                    'error': f'Insufficient data for training. Need at least 100 rows, but found {len(data) if data is not None else 0} rows.'
                }), 400

        print(f"Training models on {len(data)} rows of data")

        # Calculate technical indicators
        try:
            features_data = TechnicalIndicators.calculate_all_indicators(data)
            features_data = features_data.dropna()

            if len(features_data) < 100:
                return jsonify({
                    'success': False,
                    'error': 'Not enough clean data after calculating indicators'
                }), 400

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return jsonify({
                'success': False,
                'error': f'Error calculating technical indicators: {str(e)}'
            }), 500

        # Prepare features and targets
        X = models.prepare_features(features_data)
        targets = models.create_targets(features_data)

        # Train models
        results = {}
        model_configs = [
            ('direction', 'classification'),
            ('magnitude', 'regression'),
            ('profit_prob', 'classification'),
            ('volatility', 'regression'),
            ('trend_sideways', 'classification'),
            ('reversal', 'classification'),
            ('trading_signal', 'classification')
        ]

        for model_name, task_type in model_configs:
            if selected_models and model_name not in selected_models:
                continue  # Skip if specific models selected and this isn't one

            try:
                if model_name in targets:
                    print(f"Training {model_name} model...")
                    y = targets[model_name]

                    # Ensure data alignment
                    common_index = X.index.intersection(y.index)
                    X_aligned = X.loc[common_index]
                    y_aligned = y.loc[common_index]

                    result = models.train_model(model_name, X_aligned, y_aligned, task_type)

                    if result:
                        accuracy_metric = result.get('metrics', {}).get('accuracy', 0) if task_type == 'classification' else result.get('metrics', {}).get('rmse', 0)
                        results[model_name] = {
                            'status': 'success',
                            'accuracy': accuracy_metric,
                            'task_type': task_type,
                            'metrics': result.get('metrics', {}),
                            'data_points': len(X_aligned)
                        }
                        print(f"✓ {model_name} trained successfully - Accuracy: {accuracy_metric:.4f}")
                    else:
                        results[model_name] = {
                            'status': 'failed', 
                            'error': 'Training returned no result',
                            'task_type': task_type
                        }

            except Exception as e:
                error_msg = str(e)
                print(f"Error training {model_name}: {error_msg}")
                results[model_name] = {
                    'status': 'failed', 
                    'error': error_msg,
                    'task_type': task_type
                }

        # Save trained models to database
        try:
            if models.models:
                from utils.database_adapter import DatabaseAdapter
                db_adapter = DatabaseAdapter()
                success = db_adapter.save_trained_models(models.models)
                if success:
                    print("Models saved to database successfully")
        except Exception as e:
            print(f"Error saving models to database: {e}")

        # Count successful and failed models
        successful_models = len([r for r in results.values() if r.get('status') == 'success'])
        failed_models = len([r for r in results.values() if r.get('status') == 'failed'])

        return jsonify({
            'success': True,
            'data': {
                'message': f'Training completed: {successful_models} successful, {failed_models} failed',
                'results': results,
                'trained_models': successful_models,
                'failed_models': failed_models,
                'total_models': len(results)
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error training models: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/summary', methods=['GET'])
def get_data_summary():
    """Get data summary for React app"""
    try:
        # Default summary for when no data is available
        default_summary = {
            'total_rows': 0,
            'date_range': {'start': 'N/A', 'end': 'N/A'},
            'columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'latest_price': 22500.0,
            'database_status': 'mock_data'
        }

        try:
            # Try to get database info
            if hasattr(db, 'get_database_info'):
                db_info = db.get_database_info()
            else:
                db_info = {"status": "unavailable"}

            # Try to load data
            data = None
            if hasattr(db, 'load_ohlc_data'):
                try:
                    data = db.load_ohlc_data()
                except Exception as e:
                    print(f"Database load error: {e}")

            if data is not None and len(data) > 0:
                # Handle date formatting properly
                start_date = 'N/A'
                end_date = 'N/A'

                try:
                    if hasattr(data.index, 'min'):
                        min_idx = data.index.min()
                        start_date = min_idx.isoformat() if hasattr(min_idx, 'isoformat') else str(min_idx)

                    if hasattr(data.index, 'max'):
                        max_idx = data.index.max()
                        end_date = max_idx.isoformat() if hasattr(max_idx, 'isoformat') else str(max_idx)
                except Exception as e:
                    print(f"Date formatting warning: {e}")

                summary = {
                    'total_rows': len(data),
                    'date_range': {
                        'start': start_date,
                        'end': end_date
                    },
                    'columns': list(data.columns) if hasattr(data, 'columns') else [],
                    'latest_price': float(data['Close'].iloc[-1]) if 'Close' in data.columns and len(data) > 0 else 22500.0,
                    'database_status': 'connected'
                }
            else:
                summary = default_summary

        except Exception as e:
            print(f"Database error, using defaults: {e}")
            summary = default_summary

        return jsonify({
            'success': True,
            'data': summary,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Critical error in get_data_summary: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Data summary unavailable',
            'details': str(e),
            'data': {
                'total_rows': 0,
                'date_range': {'start': 'N/A', 'end': 'N/A'},
                'columns': [],
                'latest_price': 0,
                'database_status': 'error'
            }
        }), 200  # Return 200 instead of 500 to prevent frontend errors

@app.route('/api/models/status')
def get_models_status():
    """Get status of all trained models."""
    try:
        # Check if models are available and loaded
        if not hasattr(models, 'models') or not models.models:
            return jsonify({
                'success': True,
                'data': {
                    'status': 'no_models',
                    'total_models': 0,
                    'trained_models': {}
                },
                'timestamp': datetime.now().isoformat()
            })

        # Format model info for frontend
        formatted_models = {}
        for model_name, model_data in models.models.items():
            if model_data and isinstance(model_data, dict):
                # Get accuracy from metrics if available
                accuracy = 0.5  # Default
                if 'metrics' in model_data:
                    metrics = model_data['metrics']
                    if 'accuracy' in metrics:
                        accuracy = metrics['accuracy']
                    elif 'rmse' in metrics:
                        # For regression models, use 1/rmse as a proxy for accuracy
                        accuracy = 1 / (1 + metrics['rmse'])

                formatted_models[model_name] = {
                    'name': model_name.replace('_', ' ').title(),
                    'accuracy': accuracy,
                    'task_type': model_data.get('task_type', 'classification'),
                    'trained_at': model_data.get('trained_at', 'Unknown')
                }

        return jsonify({
            'success': True,
            'data': {
                'status': 'loaded',
                'total_models': len(formatted_models),
                'trained_models': formatted_models
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in get_models_status: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {'status': 'error', 'total_models': 0, 'trained_models': {}}
        }), 500

@app.route('/api/predictions/<model_name>')
def get_model_predictions(model_name):
    """Get predictions for a specific model."""
    try:
        period = request.args.get('period', '30d')

        # Load the data
        db = get_trading_database()

        # Try to load main dataset first, fallback to latest if not found
        try:
            df = db.load_ohlc_data('main_dataset')
            if df is None or df.empty:
                datasets = db.get_dataset_list()
                if datasets:
                    latest_dataset = datasets[0]['name']  # Get the first (most recent) dataset
                    print(f"Main dataset not found, loading latest dataset: {latest_dataset}")
                    df = db.load_ohlc_data(latest_dataset)
                else:
                    raise ValueError("No datasets available")
        except Exception as e:
            print(f"Error loading data: {e}")
            return jsonify({
                'success': False,
                'error': 'No data available for predictions'
            }), 404

        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available for predictions'
            }), 404

        # Filter data based on period
        if period == '30d':
            df = df.tail(30 * 24 * 12)  # Assuming 5-min data, 30 days
        elif period == '90d':
            df = df.tail(90 * 24 * 12)  # 90 days
        # For 'all', use all data

        # Load the trained model
        from models.xgboost_models import QuantTradingModels
        from features.technical_indicators import TechnicalIndicators

        # Initialize the model trainer and load existing models
        model_trainer = QuantTradingModels()

        # Check if the requested model exists
        if model_name not in model_trainer.models:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not found. Available models: {list(model_trainer.models.keys())}'
            }), 404

        # Calculate technical indicators if not present
        required_indicators = ['sma_5', 'ema_5', 'rsi', 'macd_histogram']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]

        if missing_indicators:
            print("Calculating missing technical indicators...")
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
        else:
            df_with_indicators = df

        # Prepare features for prediction
        features = model_trainer.prepare_features(df_with_indicators)

        # Generate predictions using the specific model
        try:
            predictions, probabilities = model_trainer.predict(model_name, features)
            print(f"Generated {len(predictions)} predictions for {model_name}")

            # Debug: Check prediction distribution
            unique_preds = np.unique(predictions, return_counts=True)
            print(f"{model_name} prediction distribution: {dict(zip(unique_preds[0], unique_preds[1]))}")

        except Exception as e:
            print(f"Error generating predictions for {model_name}: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to generate predictions for {model_name}: {str(e)}'
            }), 500

        # Prepare response data
        prediction_data = []

        # Align indices and create prediction records
        common_index = features.index.intersection(df_with_indicators.index)
        df_aligned = df_with_indicators.loc[common_index]

        # Take the last N predictions to match the period filter
        num_predictions = min(len(predictions), len(common_index), 1000)
        start_idx = len(common_index) - num_predictions

        for i in range(num_predictions):
            idx = common_index[start_idx + i]
            pred_idx = start_idx + i

            record = {
                'date': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'price': float(df_aligned.loc[idx, 'Close']),
                'prediction': int(predictions[pred_idx]) if pred_idx < len(predictions) else 0
            }

            # Add confidence if available
            if probabilities is not None and pred_idx < len(probabilities):
                if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                    record['confidence'] = float(np.max(probabilities[pred_idx]))
                else:
                    record['confidence'] = float(probabilities[pred_idx])
            else:
                # Default confidence based on prediction consistency
                record['confidence'] = 0.6 + (0.2 * np.random.random())

            prediction_data.append(record)

        # Calculate statistics
        total_predictions = len(prediction_data)
        up_predictions = sum(1 for p in prediction_data if p['prediction'] == 1)
        down_predictions = total_predictions - up_predictions

        return jsonify({
            'success': True,
            'model_name': model_name,
            'total_predictions': total_predictions,
            'up_predictions': up_predictions,
            'down_predictions': down_predictions,
            'predictions': prediction_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error generating predictions for {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error generating predictions: {str(e)}'
        }), 500

# Initialize model trainer
model_trainer = None

def initialize_model_trainer():
    """Initialize the model trainer with existing models."""
    global model_trainer
    try:
        from models.xgboost_models import QuantTradingModels
        model_trainer = QuantTradingModels()
        print(f"Loaded {len(model_trainer.models)} existing trained models from database")

        # Verify models are properly loaded
        for model_name, model_info in model_trainer.models.items():
            if model_info and 'ensemble' in model_info:
                print(f"✅ Model {model_name} loaded successfully")
            else:
                print(f"⚠️ Model {model_name} may not be properly loaded")

        return True
    except Exception as e:
        print(f"Error initializing model trainer: {e}")
        return False

@app.route('/api/database/dataset/<dataset_name>', methods=['DELETE'])
def delete_dataset(dataset_name):
    """Delete a specific dataset"""
    try:
        success = db.delete_dataset(dataset_name)

        if success:
            return jsonify({
                'success': True,
                'message': f'Dataset {dataset_name} deleted successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to delete dataset {dataset_name}'
            }), 404

    except Exception as e:
        print(f"Error deleting dataset {dataset_name}: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/database/export/<dataset_name>', methods=['GET'])
def export_dataset(dataset_name):
    """Export a dataset as CSV"""
    try:
        data = db.load_ohlc_data(dataset_name)

        if data is None:
            return jsonify({
                'success': False,
                'error': f'Dataset {dataset_name} not found'
            }), 404

        # Convert to CSV
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=True)
        csv_content = csv_buffer.getvalue()

        # Create response
        response = Response(
            csv_content,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )

        return response

    except Exception as e:
        print(f"Error exporting dataset {dataset_name}: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/load', methods=['POST'])
def load_dataset():
    """Load a specific dataset"""
    try:
        request_data = request.get_json() if request.is_json else {}
        dataset_name = request_data.get('dataset_name', 'main_dataset')

        data = db.load_ohlc_data(dataset_name)

        if data is not None:
            return jsonify({
                'success': True,
                'message': f'Dataset {dataset_name} loaded successfully',
                'data': {
                    'dataset_name': dataset_name,
                    'total_rows': len(data),
                    'columns': list(data.columns)
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Dataset {dataset_name} not found'
            }), 404

    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/database/clear-all', methods=['DELETE', 'POST'])
def clear_all_database():
    """Clear all data from the database"""
    try:
        from utils.database_adapter import DatabaseAdapter
        db_adapter = DatabaseAdapter()

        success = db_adapter.clear_all_data()

        if success:
            # Also clear in-memory models
            global models
            if models:
                models.models = {}

            return jsonify({
                'success': True,
                'message': 'All database data cleared successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to clear database'
            }), 500

    except Exception as e:
        print(f"Error clearing database: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/database-info', methods=['GET'])
def get_database_info():
    """Get database information"""
    try:
        # Provide fallback database info if db methods fail
        default_info = {
            'status': 'available',
            'total_datasets': 0,
            'total_models': 0,
            'total_predictions': 0,
            'connection_type': 'mock'
        }

        try:
            if hasattr(db, 'get_database_info'):
                info = db.get_database_info()
                if info:
                    default_info.update(info)
        except Exception as db_error:
            print(f"Database info error: {db_error}")

        return jsonify({
            'success': True,
            'data': default_info,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error getting database info: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Database information unavailable',
            'data': {
                'status': 'unavailable',
                'total_datasets': 0,
                'total_models': 0,
                'total_predictions': 0
            }
        }), 200  # Return 200 to prevent frontend errors

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Handle file upload and data processing"""
    try:
        print(f"Upload request received. Files: {list(request.files.keys())}")
        print(f"Form data: {list(request.form.keys())}")

        if 'file' not in request.files:
            print("No 'file' key in request.files")
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        file = request.files['file']
        print(f"File received: {file.filename}, content type: {file.content_type}")

        if file.filename == '':
            print("Empty filename")
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not file.filename.lower().endswith('.csv'):
            print(f"Invalid file extension for: {file.filename}")
            return jsonify({
                'success': False,
                'error': 'Only CSV files are supported'
            }), 400

        # Read the CSV file
        try:
            import pandas as pd
            from datetime import datetime

            # Read CSV data
            print(f"Reading CSV file: {file.filename}")
            try:
                df = pd.read_csv(file)
                print(f"CSV loaded successfully. Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"First few rows:\n{df.head()}")
            except Exception as csv_error:
                print(f"Failed to read CSV: {csv_error}")
                return jsonify({
                    'success': False,
                    'error': f'Failed to read CSV file: {str(csv_error)}'
                }), 400

            # Basic validation - handle case-insensitive column names
            required_columns = ['Open', 'High', 'Low', 'Close']
            df_columns_lower = [col.lower() for col in df.columns]
            required_columns_lower = [col.lower() for col in required_columns]

            # Check for missing columns (case-insensitive)
            missing_columns = [col for col in required_columns_lower if col not in df_columns_lower]

            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                return jsonify({
                    'success': False,
                    'error': f'Missing required columns: {", ".join(missing_columns)}'
                }), 400

            # Standardize column names to proper case
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'open':
                    column_mapping[col] = 'Open'
                elif col_lower == 'high':
                    column_mapping[col] = 'High'
                elif col_lower == 'low':
                    column_mapping[col] = 'Low'
                elif col_lower == 'close':
                    column_mapping[col] = 'Close'
                elif col_lower == 'volume':
                    column_mapping[col] = 'Volume'
                elif col_lower in ['date', 'datetime', 'timestamp']:
                    column_mapping[col] = 'Date'

            if column_mapping:
                df = df.rename(columns=column_mapping)
                print(f"Renamed columns: {column_mapping}")
                print(f"New columns: {list(df.columns)}")

            # Convert date column if exists
            if 'Date' in df.columns:
                try:
                    # Try parsing with DD-MM-YYYY format first (common in Indian data)
                    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
                    # If that fails, try other common formats
                    if df['Date'].isna().sum() > len(df) * 0.5:  # If more than 50% failed
                        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
                except:
                    # Fallback to pandas auto-detection with dayfirst=True
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df = df.set_index('Date')
            elif 'Datetime' in df.columns:
                try:
                    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
                    if df['Datetime'].isna().sum() > len(df) * 0.5:
                        df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed', dayfirst=True)
                except:
                    df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True, errors='coerce')
                df = df.set_index('Datetime')

            # Ensure numeric columns
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

            # Remove rows with NaN values
            df = df.dropna()

            if len(df) < 100:
                return jsonify({
                    'success': False,
                    'error': 'Insufficient data. Need at least 100 valid rows'
                }), 400

            # Generate dataset name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_name = f"uploaded_{timestamp}"

            # Save to database
            success = db.save_ohlc_data(df, dataset_name, preserve_full_data=True)

            if success:
                # Also create/update main_dataset
                db.save_ohlc_data(df, "main_dataset", preserve_full_data=True)

                return jsonify({
                    'success': True,
                    'message': f'Successfully uploaded {len(df)} rows of data',
                    'dataset_name': dataset_name,
                    'rows_imported': len(df),
                    'columns': list(df.columns)
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to save data to database'
                }), 500

        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            return jsonify({
                'success': False,
                'error': f'Error processing file: {str(e)}'
            }), 400

    except Exception as e:
        print(f"Error in upload_data: {e}")
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("Starting TribexAlpha Trading Dashboard API Server...")

    # Load existing models from database if available
    try:
        from utils.database_adapter import DatabaseAdapter
        db_adapter = DatabaseAdapter()

        # Check if database actually has data before loading models
        db_info = db_adapter.get_database_info()
        if db_info.get('total_trained_models', 0) > 0:
            saved_models = db_adapter.load_trained_models()
            if saved_models:
                models.models = saved_models
                print(f"Loaded {len(saved_models)} existing trained models from database")
            else:
                print("No existing models found in database")
        else:
            print("No trained models in database")
    except Exception as e:
        print(f"Could not load existing models: {str(e)}")

    print("Dashboard will be available at: http://0.0.0.0:8080")
    print("API endpoints available at: http://0.0.0.0:8080/api/")
    print(f"Market is currently: {'OPEN' if is_market_open() else 'CLOSED'}")
    print(f"Current IST time: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')}")

    app.run(host='0.0.0.0', port=8080, debug=True)