
import os
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
import pickle
import base64

class DatabaseAdapter:
    """Database adapter supporting both PostgreSQL and Replit Key-Value store"""
    
    def __init__(self):
        self.backend = self._detect_backend()
        self.setup_tables()
    
    def _detect_backend(self):
        """Detect and initialize the appropriate database backend"""
        if os.environ.get('DATABASE_URL'):
            print("Using PostgreSQL backend")
            return PostgreSQLBackend()
        else:
            print("Using Replit Key-Value backend")
            from replit import db
            return ReplitKVBackend(db)
    
    def setup_tables(self):
        """Setup required tables/collections"""
        self.backend.setup_tables()
    
    # OHLC Data Operations
    def save_ohlc_data(self, data: pd.DataFrame, dataset_name: str = "main_dataset") -> bool:
        return self.backend.save_ohlc_data(data, dataset_name)
    
    def load_ohlc_data(self, dataset_name: str = "main_dataset") -> Optional[pd.DataFrame]:
        return self.backend.load_ohlc_data(dataset_name)
    
    def get_dataset_list(self) -> List[str]:
        return self.backend.get_dataset_list()
    
    def delete_dataset(self, dataset_name: str) -> bool:
        return self.backend.delete_dataset(dataset_name)
    
    # Model Operations
    def save_trained_models(self, models_dict: Dict[str, Any]) -> bool:
        return self.backend.save_trained_models(models_dict)
    
    def load_trained_models(self) -> Optional[Dict[str, Any]]:
        return self.backend.load_trained_models()
    
    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        return self.backend.save_model_results(model_name, results)
    
    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self.backend.load_model_results(model_name)
    
    # Predictions Operations
    def save_predictions(self, predictions: pd.DataFrame, model_name: str) -> bool:
        return self.backend.save_predictions(predictions, model_name)
    
    def load_predictions(self, model_name: str) -> Optional[pd.DataFrame]:
        return self.backend.load_predictions(model_name)
    
    # Database Info
    def get_database_info(self) -> Dict[str, Any]:
        return self.backend.get_database_info()
    
    def clear_all_data(self) -> bool:
        return self.backend.clear_all_data()


class PostgreSQLBackend:
    """PostgreSQL backend implementation"""
    
    def __init__(self):
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        self.database_url = os.environ['DATABASE_URL']
        # Use connection pooling for better performance
        self.pooled_url = self.database_url.replace('.us-east-2', '-pooler.us-east-2')
        self.psycopg2 = psycopg2
        self.RealDictCursor = RealDictCursor
    
    def get_connection(self):
        """Get database connection"""
        return self.psycopg2.connect(self.pooled_url)
    
    def setup_tables(self):
        """Create required tables if they don't exist"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # OHLC Data table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ohlc_data (
                    id SERIAL PRIMARY KEY,
                    dataset_name VARCHAR(255) NOT NULL,
                    date_time TIMESTAMP NOT NULL,
                    open_price DECIMAL(15,6) NOT NULL,
                    high_price DECIMAL(15,6) NOT NULL,
                    low_price DECIMAL(15,6) NOT NULL,
                    close_price DECIMAL(15,6) NOT NULL,
                    volume BIGINT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(dataset_name, date_time)
                );
            """)
            
            # Dataset metadata table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    dataset_name VARCHAR(255) PRIMARY KEY,
                    total_rows INTEGER NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL,
                    columns_info JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Trained models table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trained_models (
                    model_name VARCHAR(255) PRIMARY KEY,
                    model_data BYTEA NOT NULL,
                    feature_names JSONB,
                    task_type VARCHAR(50),
                    model_type VARCHAR(100),
                    accuracy DECIMAL(10,6),
                    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Model results table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_results (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    results_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Predictions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    date_time TIMESTAMP NOT NULL,
                    prediction_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, date_time)
                );
            """)
            
            # Create indexes for better performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_dataset_date ON ohlc_data(dataset_name, date_time);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model_date ON predictions(model_name, date_time);")
            
            conn.commit()
            print("✅ PostgreSQL tables created successfully")
            
        except Exception as e:
            print(f"Error setting up PostgreSQL tables: {str(e)}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()
    
    def save_ohlc_data(self, data: pd.DataFrame, dataset_name: str = "main_dataset") -> bool:
        """Save OHLC data to PostgreSQL"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Clear existing data for this dataset
            cur.execute("DELETE FROM ohlc_data WHERE dataset_name = %s", (dataset_name,))
            
            # Prepare data for bulk insert
            records = []
            for idx, row in data.iterrows():
                record = (
                    dataset_name,
                    idx,  # datetime index
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row.get('Volume', 0)) if 'Volume' in row and pd.notna(row['Volume']) else 0
                )
                records.append(record)
            
            # Bulk insert
            cur.executemany("""
                INSERT INTO ohlc_data (dataset_name, date_time, open_price, high_price, low_price, close_price, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (dataset_name, date_time) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume
            """, records)
            
            # Save metadata
            metadata = {
                'columns': data.columns.tolist(),
                'data_types': data.dtypes.astype(str).to_dict()
            }
            
            cur.execute("""
                INSERT INTO dataset_metadata (dataset_name, total_rows, start_date, end_date, columns_info)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (dataset_name) DO UPDATE SET
                    total_rows = EXCLUDED.total_rows,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    columns_info = EXCLUDED.columns_info,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                dataset_name,
                len(data),
                data.index.min(),
                data.index.max(),
                json.dumps(metadata)
            ))
            
            conn.commit()
            print(f"✅ Saved {len(data)} rows to PostgreSQL")
            return True
            
        except Exception as e:
            print(f"Error saving OHLC data to PostgreSQL: {str(e)}")
            conn.rollback()
            return False
        finally:
            cur.close()
            conn.close()
    
    def load_ohlc_data(self, dataset_name: str = "main_dataset") -> Optional[pd.DataFrame]:
        """Load OHLC data from PostgreSQL"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=self.RealDictCursor)
            
            cur.execute("""
                SELECT date_time, open_price, high_price, low_price, close_price, volume
                FROM ohlc_data 
                WHERE dataset_name = %s 
                ORDER BY date_time ASC
            """, (dataset_name,))
            
            rows = cur.fetchall()
            
            if not rows:
                return None
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append({
                    'Open': float(row['open_price']),
                    'High': float(row['high_price']),
                    'Low': float(row['low_price']),
                    'Close': float(row['close_price']),
                    'Volume': int(row['volume']) if row['volume'] else 0
                })
            
            df = pd.DataFrame(data)
            df.index = pd.to_datetime([row['date_time'] for row in rows])
            df.index.name = 'Date'
            
            print(f"✅ Loaded {len(df)} rows from PostgreSQL")
            return df
            
        except Exception as e:
            print(f"Error loading OHLC data from PostgreSQL: {str(e)}")
            return None
        finally:
            cur.close()
            conn.close()
    
    def get_dataset_list(self) -> List[str]:
        """Get list of available datasets"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("SELECT DISTINCT dataset_name FROM dataset_metadata ORDER BY dataset_name")
            rows = cur.fetchall()
            
            return [row[0] for row in rows]
            
        except Exception as e:
            print(f"Error getting dataset list: {str(e)}")
            return []
        finally:
            cur.close()
            conn.close()
    
    def save_trained_models(self, models_dict: Dict[str, Any]) -> bool:
        """Save trained models to PostgreSQL"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            for model_name, model_data in models_dict.items():
                # Serialize model
                model_bytes = pickle.dumps(model_data['ensemble'])
                
                cur.execute("""
                    INSERT INTO trained_models (model_name, model_data, feature_names, task_type, model_type, accuracy)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_name) DO UPDATE SET
                        model_data = EXCLUDED.model_data,
                        feature_names = EXCLUDED.feature_names,
                        task_type = EXCLUDED.task_type,
                        model_type = EXCLUDED.model_type,
                        accuracy = EXCLUDED.accuracy,
                        trained_at = CURRENT_TIMESTAMP
                """, (
                    model_name,
                    model_bytes,
                    json.dumps(model_data.get('feature_names', [])),
                    model_data.get('task_type', 'classification'),
                    str(type(model_data['ensemble']).__name__),
                    model_data.get('accuracy', None)
                ))
            
            conn.commit()
            print(f"✅ Saved {len(models_dict)} models to PostgreSQL")
            return True
            
        except Exception as e:
            print(f"Error saving models to PostgreSQL: {str(e)}")
            conn.rollback()
            return False
        finally:
            cur.close()
            conn.close()
    
    def load_trained_models(self) -> Optional[Dict[str, Any]]:
        """Load trained models from PostgreSQL"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=self.RealDictCursor)
            
            cur.execute("SELECT * FROM trained_models")
            rows = cur.fetchall()
            
            if not rows:
                return None
            
            models = {}
            for row in rows:
                ensemble = pickle.loads(row['model_data'])
                models[row['model_name']] = {
                    'ensemble': ensemble,
                    'feature_names': json.loads(row['feature_names']) if row['feature_names'] else [],
                    'task_type': row['task_type'],
                    'model_type': row['model_type'],
                    'trained_at': row['trained_at'].strftime('%Y-%m-%d %H:%M:%S'),
                    'accuracy': float(row['accuracy']) if row['accuracy'] else None
                }
            
            print(f"✅ Loaded {len(models)} models from PostgreSQL")
            return models
            
        except Exception as e:
            print(f"Error loading models from PostgreSQL: {str(e)}")
            return None
        finally:
            cur.close()
            conn.close()
    
    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        """Save model training results"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO model_results (model_name, results_data)
                VALUES (%s, %s)
            """, (model_name, json.dumps(results)))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error saving model results: {str(e)}")
            conn.rollback()
            return False
        finally:
            cur.close()
            conn.close()
    
    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model training results"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT results_data FROM model_results 
                WHERE model_name = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (model_name,))
            
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
            return None
            
        except Exception as e:
            print(f"Error loading model results: {str(e)}")
            return None
        finally:
            cur.close()
            conn.close()
    
    def save_predictions(self, predictions: pd.DataFrame, model_name: str) -> bool:
        """Save predictions to PostgreSQL"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Clear existing predictions for this model
            cur.execute("DELETE FROM predictions WHERE model_name = %s", (model_name,))
            
            # Prepare records
            records = []
            for idx, row in predictions.iterrows():
                prediction_data = row.to_dict()
                records.append((model_name, idx, json.dumps(prediction_data)))
            
            # Bulk insert
            cur.executemany("""
                INSERT INTO predictions (model_name, date_time, prediction_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (model_name, date_time) DO UPDATE SET
                    prediction_data = EXCLUDED.prediction_data,
                    created_at = CURRENT_TIMESTAMP
            """, records)
            
            conn.commit()
            print(f"✅ Saved {len(predictions)} predictions to PostgreSQL")
            return True
            
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            conn.rollback()
            return False
        finally:
            cur.close()
            conn.close()
    
    def load_predictions(self, model_name: str) -> Optional[pd.DataFrame]:
        """Load predictions from PostgreSQL"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=self.RealDictCursor)
            
            cur.execute("""
                SELECT date_time, prediction_data
                FROM predictions 
                WHERE model_name = %s 
                ORDER BY date_time ASC
            """, (model_name,))
            
            rows = cur.fetchall()
            
            if not rows:
                return None
            
            # Convert to DataFrame
            data = []
            dates = []
            for row in rows:
                prediction_data = json.loads(row['prediction_data'])
                data.append(prediction_data)
                dates.append(row['date_time'])
            
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(dates)
            df.index.name = 'Date'
            
            return df
            
        except Exception as e:
            print(f"Error loading predictions: {str(e)}")
            return None
        finally:
            cur.close()
            conn.close()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Get dataset count
            cur.execute("SELECT COUNT(*) FROM dataset_metadata")
            dataset_count = cur.fetchone()[0]
            
            # Get total records
            cur.execute("SELECT COUNT(*) FROM ohlc_data")
            total_records = cur.fetchone()[0]
            
            # Get model count
            cur.execute("SELECT COUNT(*) FROM trained_models")
            model_count = cur.fetchone()[0]
            
            # Get prediction count
            cur.execute("SELECT COUNT(*) FROM predictions")
            prediction_count = cur.fetchone()[0]
            
            return {
                'backend': 'PostgreSQL',
                'total_datasets': dataset_count,
                'total_records': total_records,
                'total_models': model_count,
                'total_predictions': prediction_count,
                'database_url': self.database_url.split('@')[1] if '@' in self.database_url else 'Hidden'
            }
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            cur.close()
            conn.close()
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete dataset from PostgreSQL"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("DELETE FROM ohlc_data WHERE dataset_name = %s", (dataset_name,))
            cur.execute("DELETE FROM dataset_metadata WHERE dataset_name = %s", (dataset_name,))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error deleting dataset: {str(e)}")
            conn.rollback()
            return False
        finally:
            cur.close()
            conn.close()
    
    def clear_all_data(self) -> bool:
        """Clear all data from PostgreSQL"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("TRUNCATE TABLE predictions, model_results, trained_models, ohlc_data, dataset_metadata RESTART IDENTITY CASCADE")
            
            conn.commit()
            print("✅ Cleared all data from PostgreSQL")
            return True
            
        except Exception as e:
            print(f"Error clearing PostgreSQL data: {str(e)}")
            conn.rollback()
            return False
        finally:
            cur.close()
            conn.close()


class ReplitKVBackend:
    """Replit Key-Value store backend (fallback)"""
    
    def __init__(self, db):
        self.db = db
    
    def setup_tables(self):
        """No setup needed for Key-Value store"""
        pass
    
    # Implement all the same methods as your current TradingDatabase class
    # This maintains compatibility with existing code
    
    def save_ohlc_data(self, data: pd.DataFrame, dataset_name: str = "main_dataset") -> bool:
        # Your existing save_ohlc_data implementation
        try:
            data_records = []
            for idx, row in data.iterrows():
                record = {
                    'date': idx.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close'])
                }
                if 'Volume' in row and pd.notna(row['Volume']):
                    record['volume'] = float(row['Volume'])
                data_records.append(record)
            
            metadata = {
                'rows': len(data),
                'start_date': data.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': data.index.max().strftime('%Y-%m-%d %H:%M:%S'),
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'columns': data.columns.tolist()
            }
            
            data_dict = {
                'data': data_records,
                'metadata': metadata
            }
            self.db[f"ohlc_data_{dataset_name}"] = data_dict
            
            # Update dataset list
            existing_datasets = self.get_dataset_list()
            if dataset_name not in existing_datasets:
                existing_datasets.append(dataset_name)
                self.db["dataset_list"] = existing_datasets
            
            return True
        except Exception as e:
            print(f"Error saving to Key-Value store: {str(e)}")
            return False
    
    def load_ohlc_data(self, dataset_name: str = "main_dataset") -> Optional[pd.DataFrame]:
        # Your existing load_ohlc_data implementation
        try:
            key = f"ohlc_data_{dataset_name}"
            if key not in self.db:
                return None
            
            data_dict = self.db[key]
            if 'data' in data_dict and isinstance(data_dict['data'], list):
                records = data_dict['data']
                df_data = []
                for record in records:
                    row_data = {
                        'Open': record['open'],
                        'High': record['high'],
                        'Low': record['low'],
                        'Close': record['close']
                    }
                    if 'volume' in record:
                        row_data['Volume'] = record['volume']
                    df_data.append(row_data)
                
                df = pd.DataFrame(df_data)
                dates = [record['date'] for record in records]
                df.index = pd.to_datetime(dates)
                df.index.name = 'Date'
                return df
            return None
        except Exception as e:
            print(f"Error loading from Key-Value store: {str(e)}")
            return None
    
    def get_dataset_list(self) -> List[str]:
        try:
            return self.db.get("dataset_list", [])
        except:
            return []
    
    def delete_dataset(self, dataset_name: str) -> bool:
        try:
            key = f"ohlc_data_{dataset_name}"
            if key in self.db:
                del self.db[key]
            
            existing_datasets = self.get_dataset_list()
            if dataset_name in existing_datasets:
                existing_datasets.remove(dataset_name)
                self.db["dataset_list"] = existing_datasets
            
            return True
        except Exception as e:
            print(f"Error deleting dataset: {str(e)}")
            return False
    
    def save_trained_models(self, models_dict: Dict[str, Any]) -> bool:
        try:
            serialized_models = {}
            for model_name, model_data in models_dict.items():
                model_bytes = pickle.dumps(model_data['ensemble'])
                model_b64 = base64.b64encode(model_bytes).decode('utf-8')
                
                serialized_models[model_name] = {
                    'ensemble': model_b64,
                    'feature_names': model_data.get('feature_names', []),
                    'task_type': model_data.get('task_type', 'classification'),
                    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_type': str(type(model_data['ensemble']).__name__)
                }
            
            if serialized_models:
                self.db['trained_models'] = serialized_models
                return True
            return False
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            return False
    
    def load_trained_models(self) -> Optional[Dict[str, Any]]:
        try:
            if 'trained_models' not in self.db:
                return None
            
            serialized_models = self.db['trained_models']
            loaded_models = {}
            
            for model_name, model_data in serialized_models.items():
                model_b64 = model_data['ensemble']
                model_bytes = base64.b64decode(model_b64.encode('utf-8'))
                ensemble = pickle.loads(model_bytes)
                
                loaded_models[model_name] = {
                    'ensemble': ensemble,
                    'feature_names': model_data.get('feature_names', []),
                    'task_type': model_data.get('task_type', 'classification'),
                    'trained_at': model_data.get('trained_at', ''),
                    'model_type': model_data.get('model_type', 'Unknown')
                }
            
            return loaded_models if loaded_models else None
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return None
    
    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        try:
            key = f"model_results_{model_name}"
            self.db[key] = {
                'results': results,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return True
        except Exception as e:
            print(f"Error saving model results: {str(e)}")
            return False
    
    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        try:
            key = f"model_results_{model_name}"
            if key not in self.db:
                return None
            return self.db[key]['results']
        except Exception as e:
            print(f"Error loading model results: {str(e)}")
            return None
    
    def save_predictions(self, predictions: pd.DataFrame, model_name: str) -> bool:
        try:
            predictions_dict = {
                'data': predictions.to_dict('records'),
                'index': predictions.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'columns': predictions.columns.tolist(),
                'model_name': model_name,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            key = f"predictions_{model_name}"
            self.db[key] = predictions_dict
            return True
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return False
    
    def load_predictions(self, model_name: str) -> Optional[pd.DataFrame]:
        try:
            key = f"predictions_{model_name}"
            if key not in self.db:
                return None
            
            pred_dict = self.db[key]
            df = pd.DataFrame(pred_dict['data'])
            df.index = pd.to_datetime(pred_dict['index'])
            
            if 'columns' in pred_dict:
                df = df[pred_dict['columns']]
            
            return df
        except Exception as e:
            print(f"Error loading predictions: {str(e)}")
            return None
    
    def get_database_info(self) -> Dict[str, Any]:
        try:
            datasets = self.get_dataset_list()
            return {
                'backend': 'Replit Key-Value Store',
                'total_datasets': len(datasets),
                'total_keys': len(list(self.db.keys())),
                'available_keys': list(self.db.keys())
            }
        except Exception as e:
            return {'error': str(e)}
    
    def clear_all_data(self) -> bool:
        try:
            keys_to_delete = list(self.db.keys())
            for key in keys_to_delete:
                del self.db[key]
            return True
        except Exception as e:
            print(f"Error clearing data: {str(e)}")
            return False
