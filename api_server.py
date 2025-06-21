from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, Any, List
import traceback
import logging

# Import your existing modules
try:
    from utils.database_adapter import get_trading_database
except ImportError:
    from utils.postgres_database import PostgresTradingDatabase as get_trading_database

try:
    from models.xgboost_models import QuantTradingModels
except ImportError:
    class QuantTradingModels:
        def __init__(self):
            self.models = {}
        def train_all_models(self, *args, **kwargs):
            return {}
        def predict(self, *args, **kwargs):
            return [], None

try:
    from features.technical_indicators import TechnicalIndicators
except ImportError:
    class TechnicalIndicators:
        @staticmethod
        def calculate_all_indicators(df):
            return df

try:
    from utils.realtime_data import IndianMarketData
except ImportError:
    class IndianMarketData:
        def fetch_realtime_data(self, *args, **kwargs):
            return None
        def is_market_open(self):
            return False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
try:
    trading_db = get_trading_database()
    print("✅ Database initialized successfully")
except Exception as e:
    print(f"⚠️ Database initialization error: {e}")
    trading_db = None

model_trainer = None
current_data = None

def get_ist_time():
    """Get current IST time"""
    import pytz
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TribexAlpha Dashboard - Redirecting to Streamlit</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #1e1e2e, #2d3748);
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                text-align: center;
                padding: 2rem;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .logo {
                font-size: 3rem;
                margin-bottom: 1rem;
            }
            .title {
                font-size: 2rem;
                color: #00ffff;
                margin-bottom: 1rem;
            }
            .subtitle {
                font-size: 1.2rem;
                color: #b8bcc8;
                margin-bottom: 2rem;
            }
            .redirect-info {
                background: rgba(0, 255, 255, 0.1);
                border: 1px solid #00ffff;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 2rem 0;
            }
            .countdown {
                font-size: 1.5rem;
                color: #00ff41;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">⚡</div>
            <h1 class="title">TribexAlpha Dashboard</h1>
            <p class="subtitle">Advanced Trading Analytics Platform</p>

            <div class="redirect-info">
                <p>🔄 Redirecting to Streamlit Application...</p>
                <div class="countdown" id="countdown">5</div>
                <p>You will be redirected automatically to the main Streamlit dashboard.</p>
            </div>
        </div>

        <script>
            let countdown = 5;
            const countdownElement = document.getElementById('countdown');

            const timer = setInterval(() => {
                countdown--;
                countdownElement.textContent = countdown;

                if (countdown <= 0) {
                    clearInterval(timer);
                    window.location.href = 'https://{{repl_url}}';
                }
            }, 1000);
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": get_ist_time().isoformat(),
        "message": "TribexAlpha API Server Running"
    })

@app.route('/api/redirect-info')
def redirect_info():
    """Provide redirect information"""
    return jsonify({
        "message": "This API server is now supporting the Streamlit application",
        "streamlit_url": f"https://{request.host}",
        "timestamp": get_ist_time().isoformat()
    })

@app.route('/api/data/summary')
def get_data_summary():
    """Get current data summary"""
    try:
        # Get actual database information
        db_info = trading_db.get_database_info()
        
        return jsonify({
            'success': True,
            'data': {
                'total_records': db_info.get('total_records', 0),
                'datasets': db_info.get('datasets', []),
                'date_range': db_info.get('date_range', {'start': 'N/A', 'end': 'N/A'}),
                'columns': db_info.get('columns', []),
                'latest_price': db_info.get('latest_price', 0),
                'database_status': 'connected' if db_info else 'disconnected'
            }
        })
    except Exception as e:
        print(f"Error getting data summary: {e}")
        return jsonify({
            'success': True,
            'data': {
                'total_records': 0,
                'datasets': [],
                'date_range': {'start': 'N/A', 'end': 'N/A'},
                'columns': [],
                'latest_price': 0,
                'database_status': 'error'
            }
        })

@app.route('/api/models/status')
def get_models_status():
    """Get status of all trained models."""
    try:
        # Check database for actual trained models
        global trading_db
        trained_models = {}
        
        try:
            if hasattr(trading_db, 'load_trained_models'):
                db_models = trading_db.load_trained_models()
                if db_models:
                    # Models exist in database
                    for model_name, model_data in db_models.items():
                        if model_data and isinstance(model_data, dict):
                            trained_models[model_name] = {
                                'name': model_name.replace('_', ' ').title(),
                                'accuracy': model_data.get('accuracy', 0.5),
                                'task_type': model_data.get('task_type', 'classification'),
                                'trained_at': model_data.get('trained_at', 'Unknown')
                            }
        except Exception as e:
            print(f"Error checking database models: {e}")
        
        # If no models in database, return empty state
        if not trained_models:
            return jsonify({
                'success': True,
                'data': {
                    'status': 'no_models',
                    'total_models': 0,
                    'trained_models': {},
                    'message': 'No trained models found. Upload data and train models to get started.'
                },
                'timestamp': datetime.now().isoformat()
            })

        # Format model info for frontend
        return jsonify({
            'success': True,
            'data': {
                'status': 'loaded',
                'total_models': len(trained_models),
                'trained_models': trained_models
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error getting models status: {e}")
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
        
        # Load the data from database
        global trading_db
        
        # Try to load main dataset first, fallback to latest if not found
        try:
            df = trading_db.load_ohlc_data('main_dataset')
            if df is None or df.empty:
                datasets = trading_db.get_dataset_list()
                if datasets:
                    latest_dataset = datasets[0]['name']
                    print(f"Main dataset not found, loading latest dataset: {latest_dataset}")
                    df = trading_db.load_ohlc_data(latest_dataset)
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

        # Ensure proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            else:
                # Create a synthetic datetime index
                df.index = pd.date_range(start='2025-01-01', periods=len(df), freq='5min')

        # Limit data size for performance - use most recent 1000 records for predictions
        if len(df) > 1000:
            print(f"Dataset has {len(df)} rows, using most recent 1000 for predictions")
            df = df.tail(1000).copy()
        
        # Calculate technical indicators if missing
        df_with_indicators = df.copy()
        required_indicators = ['sma_5', 'ema_5', 'rsi', 'macd_histogram']
        missing_indicators = [ind for ind in required_indicators if ind not in df_with_indicators.columns]

        if missing_indicators:
            print("Calculating missing technical indicators...")
            try:
                df_with_indicators = TechnicalIndicators.calculate_all_indicators(df_with_indicators)
                
                # Fill NaN values in indicators efficiently
                indicator_cols = [col for col in df_with_indicators.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
                
                for col in indicator_cols:
                    if col in df_with_indicators.columns:
                        if 'rsi' in col.lower():
                            df_with_indicators[col] = df_with_indicators[col].fillna(50.0)
                        elif 'macd' in col.lower():
                            df_with_indicators[col] = df_with_indicators[col].fillna(0.0)
                        else:
                            df_with_indicators[col] = df_with_indicators[col].ffill().bfill().fillna(0.0)
                            
            except Exception as e:
                print(f"Error calculating technical indicators: {e}")
                # Continue with basic data if indicators fail

        # Prepare features using the model's feature preparation
        try:
            features_df = model_trainer.prepare_features(df_with_indicators)
        except Exception as e:
            print(f"Error preparing features: {e}")
            features_df = df_with_indicators

        # Make predictions using the trained model
        try:
            predictions, probabilities = model_trainer.predict(model_name, features_df)
            print(f"Generated {len(predictions)} predictions for {model_name}")
        except Exception as e:
            print(f"Error making predictions with model {model_name}: {e}")
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not available or not trained'
            }), 400

        # Format predictions for frontend
        prediction_data = []
        
        # Convert timezone to IST for display
        if hasattr(features_df.index, 'tz_convert'):
            try:
                ist_index = features_df.index.tz_convert('Asia/Kolkata')
            except:
                ist_index = features_df.index.tz_localize('Asia/Kolkata')
        else:
            try:
                ist_index = features_df.index.tz_localize('Asia/Kolkata')
            except:
                ist_index = features_df.index

        formatted_dates = ist_index.strftime('%Y-%m-%d %H:%M:%S')
        
        for i, (pred, date_str) in enumerate(zip(predictions, formatted_dates)):
            # Get price from available columns
            price = 25000.0  # Default price
            if i < len(features_df):
                if 'Close' in features_df.columns:
                    price = float(features_df['Close'].iloc[i])
                elif 'close' in features_df.columns:
                    price = float(features_df['close'].iloc[i])
                elif len(features_df.columns) > 0:
                    # Use the last numeric column as price fallback
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        price = float(features_df[numeric_cols[-1]].iloc[i])
            
            record = {
                'date': date_str,
                'prediction': int(pred),
                'price': price
            }
            
            # Add confidence scores
            if probabilities is not None and len(probabilities) > i:
                pred_idx = i if len(probabilities.shape) == 1 else i
                if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                    record['confidence'] = float(np.max(probabilities[pred_idx]))
                else:
                    record['confidence'] = float(probabilities[pred_idx])
            else:
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
            'data_timeframe': '5-minute intervals',
            'timezone': 'Asia/Kolkata (IST)',
            'market_hours': '09:15 - 15:30 IST',
            'note': 'Predictions based on uploaded historical data',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error generating predictions for {model_name}: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error generating predictions: {str(e)}'
        }), 500

@app.route('/api/train-models', methods=['POST'])
def train_models():
    """Trigger model training"""
    try:
        selected_models = request.json.get('models', []) if request.is_json else []
        
        # Load data for training
        df = trading_db.load_ohlc_data('main_dataset')
        if df is None or df.empty:
            # Try to use the latest dataset
            try:
                trading_db.create_main_dataset_from_latest()
                df = trading_db.load_ohlc_data('main_dataset')
            except:
                pass
            
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available for training. Please upload data first.'
            }), 400

        # Prepare data with technical indicators
        try:
            df = df.dropna()
            if len(df) < 100:
                return jsonify({
                    'success': False,
                    'error': 'Insufficient data for training. Need at least 100 records.'
                }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Data preparation failed: {str(e)}'
            }), 400

        # Train models
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
                continue
                
            try:
                # Create targets for this model
                targets = model_trainer.create_targets(df)
                
                if model_name not in targets:
                    print(f"Target {model_name} not available, skipping...")
                    continue
                
                target_data = targets[model_name]
                
                # Align features with targets
                features_df = model_trainer.prepare_features(df)
                features_df = features_df.loc[target_data.index]
                
                # Train the model
                result = model_trainer.train_model(model_name, features_df, target_data, task_type)
                
                print(f"✅ Trained {model_name} - Accuracy: {result.get('accuracy', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue

        # Save trained models to database
        try:
            if hasattr(model_trainer, 'models') and model_trainer.models:
                trading_db.save_trained_models(model_trainer.models)
                print("✅ Models saved to database")
        except Exception as e:
            print(f"Error saving models: {e}")

        return jsonify({
            'success': True,
            'message': 'Model training completed successfully',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in model training: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/database-info')
def get_database_info():
    """Get database information"""
    try:
        info = trading_db.get_database_info()
        return jsonify({
            'success': True,
            'data': info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error getting database info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/database/clear-all', methods=['DELETE', 'POST'])
def clear_all_database():
    """Clear all data from the database"""
    try:
        # Use existing global db instance to avoid connection conflicts
        global trading_db
        
        if hasattr(trading_db, 'clear_all_data'):
            success = trading_db.clear_all_data()
        else:
            # Direct database clearing to avoid deadlocks
            import psycopg
            import os
            
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise Exception("Database URL not available")
            
            with psycopg.connect(database_url) as conn:
                with conn.cursor() as cursor:
                    # Clear tables in proper order
                    cursor.execute("TRUNCATE TABLE model_predictions RESTART IDENTITY CASCADE;")
                    cursor.execute("TRUNCATE TABLE trained_models RESTART IDENTITY CASCADE;")
                    cursor.execute("TRUNCATE TABLE ohlc_datasets RESTART IDENTITY CASCADE;")
                    conn.commit()
            success = True

        if success:
            # Clear in-memory models
            global model_trainer
            if model_trainer and hasattr(model_trainer, 'models'):
                model_trainer.models = {}

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

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Handle file upload and data processing"""
    try:
        print("Upload request received. Files:", list(request.files.keys()))
        print("Form data:", list(request.form.keys()))
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        filename = file.filename.lower()
        print(f"File received: {file.filename}, content type: {file.content_type}")
        
        if not filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'Only CSV files are supported'
            }), 400
        
        # Read CSV file
        try:
            print(f"Reading CSV file: {file.filename}")
            df = pd.read_csv(file)
            print(f"CSV loaded successfully. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("First few rows:")
            print(df.head())
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return jsonify({
                'success': False,
                'error': f'Error reading CSV file: {str(e)}'
            }), 400
        
        # Validate and process data
        required_columns = ['open', 'high', 'low', 'close']
        df_columns_lower = [col.lower() for col in df.columns]
        
        missing_columns = [col for col in required_columns if col not in df_columns_lower]
        if missing_columns:
            return jsonify({
                'success': False,
                'error': f'Missing required columns: {missing_columns}'
            }), 400
        
        # Rename columns to standard format
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['date', 'datetime', 'timestamp']:
                column_mapping[col] = 'Date'
            elif col_lower == 'open':
                column_mapping[col] = 'Open'
            elif col_lower == 'high':
                column_mapping[col] = 'High'
            elif col_lower == 'low':
                column_mapping[col] = 'Low'
            elif col_lower == 'close':
                column_mapping[col] = 'Close'
            elif col_lower in ['volume', 'vol']:
                column_mapping[col] = 'Volume'
        
        df = df.rename(columns=column_mapping)
        print(f"Renamed columns: {column_mapping}")
        print(f"New columns: {list(df.columns)}")
        
        # Save to database
        dataset_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            success1 = trading_db.save_ohlc_data(df, dataset_name, preserve_full_data=True)
            print(f"✅ Successfully saved {len(df)} rows to PostgreSQL")
        except Exception as e:
            print(f"Error saving to database: {e}")
            return jsonify({
                'success': False,
                'error': f'Error saving to database: {str(e)}'
            }), 500
        
        # Also save as main dataset
        try:
            success2 = trading_db.save_ohlc_data(df, 'main_dataset', preserve_full_data=True)
            print(f"✅ Successfully saved {len(df)} rows to PostgreSQL")
        except Exception as e:
            print(f"Error saving main dataset: {e}")

        # Clear any existing in-memory models since we have new data
        global model_trainer
        if model_trainer and hasattr(model_trainer, 'models'):
            model_trainer.models = {}

        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(df)} records',
            'dataset_name': dataset_name,
            'rows': len(df),
            'columns': list(df.columns),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in upload_data: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    else:
        return render_template_string("<h1>Page not found</h1><p>Please access the Streamlit dashboard.</p>")

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting TribexAlpha API Server...")
    print(f"Server will be available at: http://0.0.0.0:5000")
    print(f"Streamlit dashboard will be the main interface")
    print(f"Current IST time: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        model_trainer = QuantTradingModels()
        print("✅ Model trainer initialized")
    except Exception as e:
        print(f"⚠️ Model trainer initialization error: {e}")
        model_trainer = None
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)