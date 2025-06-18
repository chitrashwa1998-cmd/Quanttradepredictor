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

# Add current directory to path for imports
sys.path.append('.')

# Import existing modules
try:
    from utils.realtime_data import IndianMarketData
    from models.xgboost_models import QuantTradingModels
    from utils.database_adapter import get_trading_database
    from features.technical_indicators import TechnicalIndicators
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are available")

app = Flask(__name__)
CORS(app)

# Initialize components
market_data = IndianMarketData()
models = QuantTradingModels()
db = get_trading_database()

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
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content, mimetype='text/html')
    except FileNotFoundError:
        return jsonify({
            'error': 'Dashboard not found',
            'message': 'React dashboard HTML file is missing'
        }), 404

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
        
        # Prepare features for prediction
        features = models.prepare_features(data.tail(100))
        
        if len(features) == 0:
            return jsonify({
                'success': False,
                'error': 'Failed to prepare features'
            }), 400
        
        latest_features = features.tail(1)
        predictions_data = {}
        
        # Get predictions from available models
        try:
            direction_pred, direction_conf = models.predict('direction', latest_features)
            predictions_data['direction'] = "UP" if direction_pred[0] > 0.5 else "DOWN"
            predictions_data['directionConfidence'] = float(direction_conf[0])
        except:
            predictions_data['direction'] = "UNKNOWN"
            predictions_data['directionConfidence'] = 0.5
        
        try:
            price_pred, price_conf = models.predict('price_target', latest_features)
            predictions_data['priceTarget'] = float(price_pred[0])
            predictions_data['targetConfidence'] = float(price_conf[0])
        except:
            current_price = data['Close'].iloc[-1]
            predictions_data['priceTarget'] = float(current_price)
            predictions_data['targetConfidence'] = 0.5
        
        try:
            trend_pred, trend_conf = models.predict('trend', latest_features)
            predictions_data['trend'] = "TRENDING" if trend_pred[0] > 0.5 else "SIDEWAYS"
            predictions_data['trendConfidence'] = float(trend_conf[0])
        except:
            predictions_data['trend'] = "UNKNOWN"
            predictions_data['trendConfidence'] = 0.5
        
        # Calculate volatility from recent data
        try:
            returns = data['Close'].pct_change().tail(20).std()
            if returns > 0.02:
                volatility = "HIGH"
            elif returns > 0.01:
                volatility = "MEDIUM"
            else:
                volatility = "LOW"
            predictions_data['volatility'] = volatility
        except:
            predictions_data['volatility'] = "MEDIUM"
        
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
        data = db.load_ohlc_data()
        if data is None or len(data) < 1000:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for training'
            }), 400
        
        # Train models
        results = models.train_all_models(data)
        
        return jsonify({
            'success': True,
            'data': {
                'message': 'Models trained successfully',
                'results': results
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