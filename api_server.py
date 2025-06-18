#!/usr/bin/env python3
"""
Flask API server for React Trading Dashboard
Provides REST endpoints to access existing Python trading models and data
"""

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import os
import sys
import traceback
from datetime import datetime
import pytz
import numpy as np

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
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are available")
    
    # Create minimal fallback classes to prevent crashes
    class IndianMarketData:
        def is_market_open(self): return False
        def fetch_realtime_data(self, *args, **kwargs): return None
    
    class QuantTradingModels:
        def __init__(self): self.models = {}
        def predict(self, *args, **kwargs): return [], []
        def train_all_models(self, *args, **kwargs): return {}
    
    class TechnicalIndicators:
        @staticmethod
        def calculate_all_indicators(df): return df
    
    def get_trading_database():
        class MockDB:
            def get_database_info(self): return {"status": "error", "message": "Database not connected"}
            def load_ohlc_data(self): return None
        return MockDB()
    
    market_data = IndianMarketData()
    models = QuantTradingModels()
    db = get_trading_database()

app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app)

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
    """Serve the React dashboard"""
    return app.send_static_file('simple.html')

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

@app.route('/api/nifty-data', methods=['POST', 'GET'])
def get_nifty_data():
    """Get current Nifty 50 data"""
    try:
        # Get current price data
        symbol = "^NSEI"  # Nifty 50 symbol
        current_data = market_data.get_current_price(symbol)
        
        if not current_data:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch Nifty data'
            }), 500
        
        return jsonify({
            'success': True,
            'data': {
                'price': current_data.get('price', 0),
                'change': current_data.get('change', 0),
                'changePercent': current_data.get('change_percent', 0),
                'volume': current_data.get('volume', 0),
                'high': current_data.get('high', 0),
                'low': current_data.get('low', 0),
                'open': current_data.get('open', 0),
                'marketCap': current_data.get('market_cap', 0)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error fetching Nifty data: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
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
        
        # Price magnitude prediction
        if 'magnitude' in available_models:
            try:
                magnitude_pred, _ = models.predict('magnitude', latest_features)
                predicted_change = float(magnitude_pred[0])
                if predictions_data['direction'] == "UP":
                    predictions_data['priceTarget'] = current_price * (1 + predicted_change/100)
                else:
                    predictions_data['priceTarget'] = current_price * (1 - predicted_change/100)
                predictions_data['targetConfidence'] = 0.7
            except Exception as e:
                print(f"Error predicting magnitude: {e}")
                predictions_data['priceTarget'] = current_price
                predictions_data['targetConfidence'] = 0.5
        else:
            predictions_data['priceTarget'] = current_price
            predictions_data['targetConfidence'] = 0.5
        
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
        # Fetch all data components
        nifty_data = None
        predictions = None
        indicators = None
        
        # Get Nifty data
        try:
            current_data = market_data.get_current_price("^NSEI")
            if current_data:
                nifty_data = {
                    'price': current_data.get('price', 0),
                    'change': current_data.get('change', 0),
                    'changePercent': current_data.get('change_percent', 0),
                    'volume': current_data.get('volume', 0),
                    'high': current_data.get('high', 0),
                    'low': current_data.get('low', 0),
                    'open': current_data.get('open', 0)
                }
        except Exception as e:
            print(f"Error fetching Nifty data in all-data: {e}")
        
        # Get predictions
        try:
            data = db.load_ohlc_data()
            if data is not None and len(data) >= 100:
                features = models.prepare_features(data.tail(100))
                if len(features) > 0:
                    latest_features = features.tail(1)
                    
                    direction_pred, direction_conf = models.predict('direction', latest_features)
                    price_pred, price_conf = models.predict('price_target', latest_features)
                    trend_pred, trend_conf = models.predict('trend', latest_features)
                    
                    predictions = {
                        'direction': "UP" if direction_pred[0] > 0.5 else "DOWN",
                        'directionConfidence': float(direction_conf[0]),
                        'priceTarget': float(price_pred[0]),
                        'targetConfidence': float(price_conf[0]),
                        'trend': "TRENDING" if trend_pred[0] > 0.5 else "SIDEWAYS",
                        'trendConfidence': float(trend_conf[0]),
                        'volatility': "MEDIUM"
                    }
        except Exception as e:
            print(f"Error fetching predictions in all-data: {e}")
        
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
            'error': str(e)
        }), 500

@app.route('/api/train-models', methods=['POST'])
def train_models():
    """Trigger model training"""
    try:
        # Get request data
        request_data = request.get_json() if request.is_json else {}
        selected_models = request_data.get('models', [])
        
        # Load data
        data = db.load_ohlc_data()
        if data is None or len(data) < 1000:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for training. Need at least 1000 rows.'
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
                        results[model_name] = {
                            'status': 'success',
                            'accuracy': result.get('metrics', {}).get('accuracy', 0) if task_type == 'classification' else result.get('metrics', {}).get('rmse', 0),
                            'task_type': task_type
                        }
                        print(f"✓ {model_name} trained successfully")
                    else:
                        results[model_name] = {'status': 'failed', 'error': 'Training returned no result'}
                        
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {'status': 'failed', 'error': str(e)}
        
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
        
        return jsonify({
            'success': True,
            'data': {
                'message': f'Training completed for {len(results)} models',
                'results': results,
                'trained_models': len([r for r in results.values() if r.get('status') == 'success'])
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
        # Get database info
        db_info = db.get_database_info()
        
        # Load data to get summary stats
        data = db.load_ohlc_data()
        
        if data is not None:
            summary = {
                'total_rows': len(data),
                'date_range': {
                    'start': data.index.min().isoformat() if hasattr(data.index, 'min') else 'N/A',
                    'end': data.index.max().isoformat() if hasattr(data.index, 'max') else 'N/A'
                },
                'columns': list(data.columns),
                'latest_price': float(data['Close'].iloc[-1]) if 'Close' in data.columns else 0,
                'database_status': 'connected'
            }
        else:
            summary = {
                'total_rows': 0,
                'date_range': {'start': 'N/A', 'end': 'N/A'},
                'columns': [],
                'latest_price': 0,
                'database_status': 'no_data'
            }
        
        return jsonify({
            'success': True,
            'data': summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error getting data summary: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/status', methods=['GET'])
def get_models_status():
    """Get trained models status"""
    try:
        # Get trained models from the models object
        trained_models = {}
        
        if hasattr(models, 'models') and models.models:
            for model_name, model_info in models.models.items():
                trained_models[model_name] = {
                    'name': model_name.replace('_', ' ').title(),
                    'task_type': model_info.get('task_type', 'unknown'),
                    'trained_at': model_info.get('trained_at', 'Unknown'),
                    'accuracy': model_info.get('metrics', {}).get('accuracy', 0) if model_info.get('task_type') == 'classification' else model_info.get('metrics', {}).get('rmse', 0)
                }
        
        # Also check database for saved models
        try:
            from utils.database_adapter import DatabaseAdapter
            db_adapter = DatabaseAdapter()
            saved_models = db_adapter.load_trained_models()
            
            if saved_models:
                for model_name, model_data in saved_models.items():
                    if model_name not in trained_models:
                        trained_models[model_name] = {
                            'name': model_name.replace('_', ' ').title(),
                            'task_type': model_data.get('task_type', 'unknown'),
                            'trained_at': model_data.get('trained_at', 'Unknown'),
                            'accuracy': 0.8  # Default value
                        }
        except Exception as e:
            print(f"Could not load models from database: {e}")
        
        return jsonify({
            'success': True,
            'data': {
                'trained_models': trained_models,
                'total_models': len(trained_models),
                'status': 'loaded' if trained_models else 'no_models'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error getting models status: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/database-info', methods=['GET'])
def get_database_info():
    """Get database information"""
    try:
        info = db.get_database_info()
        
        return jsonify({
            'success': True,
            'data': info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error getting database info: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
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
    print("Dashboard will be available at: http://0.0.0.0:8080")
    print("API endpoints available at: http://0.0.0.0:8080/api/")
    print(f"Market is currently: {'OPEN' if is_market_open() else 'CLOSED'}")
    print(f"Current IST time: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')}")
    
    app.run(host='0.0.0.0', port=8080, debug=True)