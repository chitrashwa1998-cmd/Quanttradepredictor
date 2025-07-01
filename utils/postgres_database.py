
"""
PostgreSQL database implementation for TribexAlpha trading app
Uses psycopg (version 3) for database operations
"""
import os
import json
import pickle
import base64
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

class PostgresTradingDatabase:
    """PostgreSQL database using psycopg for storing trading data, models, and predictions."""

    def __init__(self):
        """Initialize PostgreSQL connection with psycopg."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self._create_tables()

    def _get_connection(self):
        """Get database connection."""
        return psycopg.connect(self.database_url)

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # OHLC datasets table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS ohlc_datasets (
                            id SERIAL PRIMARY KEY,
                            dataset_name VARCHAR(255) UNIQUE NOT NULL,
                            data_json TEXT NOT NULL,
                            metadata_json TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Model results table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS model_results (
                            id SERIAL PRIMARY KEY,
                            model_name VARCHAR(255) UNIQUE NOT NULL,
                            results_json TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Trained models table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS trained_models (
                            id SERIAL PRIMARY KEY,
                            model_name VARCHAR(255) UNIQUE NOT NULL,
                            model_data TEXT NOT NULL,
                            task_type VARCHAR(100),
                            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Predictions table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS predictions (
                            id SERIAL PRIMARY KEY,
                            model_name VARCHAR(255) NOT NULL,
                            predictions_json TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Create indexes
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_dataset_name ON ohlc_datasets(dataset_name);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON model_results(model_name);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trained_model_name ON trained_models(model_name);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);")
                    
                    conn.commit()
            
            print("✅ PostgreSQL tables created successfully")
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
            raise e

    def test_connection(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1;")
                    result = cursor.fetchone()
                    return result[0] == 1
        except Exception as e:
            print(f"PostgreSQL connection test failed: {str(e)}")
            return False

    def save_ohlc_data(self, data: pd.DataFrame, dataset_name: str = "main_dataset", preserve_full_data: bool = False) -> bool:
        """Save OHLC dataframe to PostgreSQL database."""
        try:
            # Preserve the index by resetting it to a column before JSON conversion
            data_with_index = data.reset_index()
            
            # Convert DataFrame to JSON for storage
            data_json = data_with_index.to_json(orient='records', date_format='iso')

            # Create metadata with index information
            metadata = {
                'rows': len(data),
                'columns': list(data.columns),
                'index_name': data.index.name or 'index',
                'index_dtype': str(data.index.dtype),
                'has_datetime_index': pd.api.types.is_datetime64_any_dtype(data.index),
                'start_date': str(data.index[0]) if not data.empty else None,
                'end_date': str(data.index[-1]) if not data.empty else None,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'preserve_full_data': preserve_full_data
            }
            metadata_json = json.dumps(metadata)

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check if dataset exists
                    cursor.execute("SELECT id FROM ohlc_datasets WHERE dataset_name = %s;", (dataset_name,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        cursor.execute("""
                            UPDATE ohlc_datasets 
                            SET data_json = %s, metadata_json = %s, updated_at = CURRENT_TIMESTAMP 
                            WHERE dataset_name = %s;
                        """, (data_json, metadata_json, dataset_name))
                    else:
                        cursor.execute("""
                            INSERT INTO ohlc_datasets (dataset_name, data_json, metadata_json) 
                            VALUES (%s, %s, %s);
                        """, (dataset_name, data_json, metadata_json))
                    
                    conn.commit()

            print(f"✅ Successfully saved {len(data)} rows to PostgreSQL")
            return True

        except Exception as e:
            print(f"Error saving OHLC data: {str(e)}")
            return False

    def load_ohlc_data(self, dataset_name: str = "main_dataset") -> Optional[pd.DataFrame]:
        """Load OHLC dataframe from PostgreSQL database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute("SELECT data_json, metadata_json FROM ohlc_datasets WHERE dataset_name = %s;", (dataset_name,))
                    result = cursor.fetchone()
                    
                    if result:
                        df = pd.read_json(result['data_json'], orient='records')
                        
                        # Get metadata to restore index properly
                        metadata = json.loads(result['metadata_json']) if result['metadata_json'] else {}
                        index_name = metadata.get('index_name', 'index')
                        has_datetime_index = metadata.get('has_datetime_index', False)

                        # Restore the original index
                        if index_name in df.columns:
                            if has_datetime_index:
                                df[index_name] = pd.to_datetime(df[index_name])
                            df.set_index(index_name, inplace=True)
                            print(f"✅ Restored datetime index: {index_name}")
                        else:
                            # Fallback: try common datetime column names
                            datetime_cols = ['timestamp', 'Datetime', 'Date', 'date', 'time']
                            for col in datetime_cols:
                                if col in df.columns:
                                    df[col] = pd.to_datetime(df[col])
                                    df.set_index(col, inplace=True)
                                    print(f"✅ Set datetime index from column: {col}")
                                    break

                        print(f"✅ Loaded {len(df)} rows from PostgreSQL with proper datetime index")
                        return df

            return None

        except Exception as e:
            print(f"Error loading OHLC data: {str(e)}")
            return None

    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """Get list of saved datasets."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute("SELECT * FROM ohlc_datasets ORDER BY updated_at DESC;")
                    datasets_query = cursor.fetchall()

            datasets = []
            for dataset in datasets_query:
                metadata = json.loads(dataset['metadata_json']) if dataset['metadata_json'] else {}
                datasets.append({
                    'name': dataset['dataset_name'],
                    'rows': metadata.get('rows', 0),
                    'columns': metadata.get('columns', []),
                    'start_date': metadata.get('start_date'),
                    'end_date': metadata.get('end_date'),
                    'created_at': dataset['created_at'],
                    'updated_at': dataset['updated_at']
                })

            return datasets

        except Exception as e:
            print(f"Error getting dataset list: {str(e)}")
            return []

    def get_dataset_metadata(self, dataset_name: str = "main_dataset") -> Optional[Dict[str, Any]]:
        """Get metadata for a dataset."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute("SELECT metadata_json FROM ohlc_datasets WHERE dataset_name = %s;", (dataset_name,))
                    result = cursor.fetchone()
                    
                    if result and result['metadata_json']:
                        return json.loads(result['metadata_json'])

            return None

        except Exception as e:
            print(f"Error getting dataset metadata: {str(e)}")
            return None

    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a dataset from database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM ohlc_datasets WHERE dataset_name = %s;", (dataset_name,))
                    conn.commit()
                    return cursor.rowcount > 0

        except Exception as e:
            print(f"Error deleting dataset: {str(e)}")
            return False

    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        """Save model training results."""
        try:
            results_json = json.dumps(results, default=str)

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id FROM model_results WHERE model_name = %s;", (model_name,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        cursor.execute("""
                            UPDATE model_results 
                            SET results_json = %s, updated_at = CURRENT_TIMESTAMP 
                            WHERE model_name = %s;
                        """, (results_json, model_name))
                    else:
                        cursor.execute("""
                            INSERT INTO model_results (model_name, results_json) 
                            VALUES (%s, %s);
                        """, (model_name, results_json))
                    
                    conn.commit()

            return True

        except Exception as e:
            print(f"Error saving model results: {str(e)}")
            return False

    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model training results."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute("SELECT results_json FROM model_results WHERE model_name = %s;", (model_name,))
                    result = cursor.fetchone()
                    
                    if result:
                        return json.loads(result['results_json'])

            return None

        except Exception as e:
            print(f"Error loading model results: {str(e)}")
            return None

    def save_trained_models(self, models_dict: Dict[str, Any]) -> bool:
        """Save trained model objects for persistence."""
        try:
            success_count = 0

            for model_name, model_data in models_dict.items():
                try:
                    # Serialize model data using pickle and base64 encoding
                    model_pickle = pickle.dumps(model_data)
                    model_b64 = base64.b64encode(model_pickle).decode('utf-8')

                    task_type = model_data.get('task_type', 'unknown') if isinstance(model_data, dict) else 'unknown'

                    with self._get_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT id FROM trained_models WHERE model_name = %s;", (model_name,))
                            existing = cursor.fetchone()
                            
                            if existing:
                                cursor.execute("""
                                    UPDATE trained_models 
                                    SET model_data = %s, task_type = %s, updated_at = CURRENT_TIMESTAMP 
                                    WHERE model_name = %s;
                                """, (model_b64, task_type, model_name))
                            else:
                                cursor.execute("""
                                    INSERT INTO trained_models (model_name, model_data, task_type) 
                                    VALUES (%s, %s, %s);
                                """, (model_name, model_b64, task_type))
                            
                            conn.commit()

                    success_count += 1
                    print(f"Serialized model: {model_name}")

                except Exception as e:
                    print(f"Error saving model {model_name}: {str(e)}")
                    continue

            return success_count > 0

        except Exception as e:
            print(f"Error saving trained models: {str(e)}")
            return False

    def load_trained_models(self) -> Optional[Dict[str, Any]]:
        """Load trained model objects from database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute("SELECT model_name, model_data FROM trained_models;")
                    models_query = cursor.fetchall()

            if not models_query:
                return None

            models_dict = {}
            for model in models_query:
                try:
                    # Deserialize model data
                    model_pickle = base64.b64decode(model['model_data'].encode('utf-8'))
                    model_data = pickle.loads(model_pickle)
                    models_dict[model['model_name']] = model_data
                except Exception as e:
                    print(f"Error deserializing model {model['model_name']}: {str(e)}")
                    continue

            return models_dict if models_dict else None

        except Exception as e:
            print(f"Error loading trained models: {str(e)}")
            return None

    def save_predictions(self, predictions: pd.DataFrame, model_name: str) -> bool:
        """Save model predictions."""
        try:
            predictions_json = predictions.to_json(orient='records', date_format='iso')

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO predictions (model_name, predictions_json) 
                        VALUES (%s, %s);
                    """, (model_name, predictions_json))
                    conn.commit()

            return True

        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return False

    def load_predictions(self, model_name: str) -> Optional[pd.DataFrame]:
        """Load model predictions."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute("""
                        SELECT predictions_json FROM predictions 
                        WHERE model_name = %s 
                        ORDER BY created_at DESC LIMIT 1;
                    """, (model_name,))
                    result = cursor.fetchone()
                    
                    if result:
                        return pd.read_json(result['predictions_json'], orient='records')

            return None

        except Exception as e:
            print(f"Error loading predictions: {str(e)}")
            return None

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about stored data."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cursor:
                    # Count datasets
                    cursor.execute("SELECT COUNT(*) as count FROM ohlc_datasets;")
                    dataset_count = cursor.fetchone()['count']
                    
                    # Count model results
                    cursor.execute("SELECT COUNT(*) as count FROM model_results;")
                    model_count = cursor.fetchone()['count']
                    
                    # Count trained models
                    cursor.execute("SELECT COUNT(*) as count FROM trained_models;")
                    trained_model_count = cursor.fetchone()['count']
                    
                    # Count predictions
                    cursor.execute("SELECT COUNT(*) as count FROM predictions;")
                    prediction_count = cursor.fetchone()['count']
                    
                    # Get dataset info
                    cursor.execute("SELECT dataset_name, metadata_json FROM ohlc_datasets;")
                    datasets_query = cursor.fetchall()

            datasets = []
            for dataset in datasets_query:
                metadata = json.loads(dataset['metadata_json']) if dataset['metadata_json'] else {}
                datasets.append({
                    'name': dataset['dataset_name'],
                    'rows': metadata.get('rows', 0),
                    'columns': metadata.get('columns', [])
                })

            return {
                'total_datasets': dataset_count,
                'total_models': model_count,
                'total_trained_models': trained_model_count,
                'total_predictions': prediction_count,
                'total_keys': dataset_count + model_count + trained_model_count + prediction_count,
                'datasets': datasets,
                'available_keys': self.get_available_keys(),
                'database_type': 'PostgreSQL'
            }

        except Exception as e:
            print(f"Error getting database info: {str(e)}")
            return {'error': str(e), 'database_type': 'PostgreSQL'}

    def recover_data(self) -> Optional[pd.DataFrame]:
        """Try to recover any available OHLC data from database."""
        try:
            datasets = self.get_dataset_list()
            if not datasets:
                return None

            # Get the most recent dataset
            latest_dataset = datasets[0]  # Already sorted by updated_at DESC
            return self.load_ohlc_data(latest_dataset['name'])

        except Exception as e:
            print(f"Error recovering data: {str(e)}")
            return None

    def get_available_keys(self) -> List[str]:
        """Get all available keys from database tables for compatibility with key-value interface."""
        keys = []
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cursor:
                    # Get OHLC dataset keys
                    cursor.execute("SELECT dataset_name FROM ohlc_datasets;")
                    datasets = cursor.fetchall()
                    for dataset in datasets:
                        keys.append(f"ohlc_{dataset['dataset_name']}")
                    
                    # Get model results keys
                    cursor.execute("SELECT model_name FROM model_results;")
                    models = cursor.fetchall()
                    for model in models:
                        keys.append(f"model_results_{model['model_name']}")
                    
                    # Get trained models keys
                    cursor.execute("SELECT model_name FROM trained_models;")
                    trained = cursor.fetchall()
                    for model in trained:
                        keys.append(f"trained_models_{model['model_name']}")
                    
                    # Get predictions keys
                    cursor.execute("SELECT DISTINCT model_name FROM predictions;")
                    preds = cursor.fetchall()
                    for pred in preds:
                        keys.append(f"predictions_{pred['model_name']}")
            
            return keys
        except Exception as e:
            print(f"Error getting available keys: {str(e)}")
            return []

    def clear_all_data(self) -> bool:
        """Clear all data from database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    print("Clearing predictions...")
                    cursor.execute("DELETE FROM predictions;")
                    rows_deleted = cursor.rowcount
                    print(f"Deleted {rows_deleted} prediction records")
                    
                    print("Clearing trained models...")
                    cursor.execute("DELETE FROM trained_models;")
                    rows_deleted = cursor.rowcount
                    print(f"Deleted {rows_deleted} trained model records")
                    
                    print("Clearing model results...")
                    cursor.execute("DELETE FROM model_results;")
                    rows_deleted = cursor.rowcount
                    print(f"Deleted {rows_deleted} model result records")
                    
                    print("Clearing OHLC datasets...")
                    cursor.execute("DELETE FROM ohlc_datasets;")
                    rows_deleted = cursor.rowcount
                    print(f"Deleted {rows_deleted} dataset records")
                    
                    # Reset sequences to start from 1
                    cursor.execute("ALTER SEQUENCE IF EXISTS predictions_id_seq RESTART WITH 1;")
                    cursor.execute("ALTER SEQUENCE IF EXISTS trained_models_id_seq RESTART WITH 1;")
                    cursor.execute("ALTER SEQUENCE IF EXISTS model_results_id_seq RESTART WITH 1;")
                    cursor.execute("ALTER SEQUENCE IF EXISTS ohlc_datasets_id_seq RESTART WITH 1;")
                    
                    conn.commit()
                    print("✅ Database cleared successfully")

            # Verify all data is cleared
            verification_info = self.get_database_info()
            total_remaining = (verification_info.get('total_datasets', 0) + 
                             verification_info.get('total_models', 0) + 
                             verification_info.get('total_trained_models', 0) + 
                             verification_info.get('total_predictions', 0))
            
            if total_remaining == 0:
                print("✅ Verification: All data successfully cleared")
                return True
            else:
                print(f"⚠️ Warning: {total_remaining} records still remain")
                return False

        except Exception as e:
            print(f"Error clearing database: {str(e)}")
            return False

    def delete_model_results(self, model_name: str) -> bool:
        """Delete model results for a specific model."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM model_results WHERE model_name = %s;", (model_name,))
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting model results: {str(e)}")
            return False

    def delete_trained_model(self, model_name: str) -> bool:
        """Delete a trained model."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM trained_models WHERE model_name = %s;", (model_name,))
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting trained model: {str(e)}")
            return False

    def delete_predictions(self, model_name: str) -> bool:
        """Delete predictions for a specific model."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM predictions WHERE model_name = %s;", (model_name,))
                    conn.commit()  
                    return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting predictions: {str(e)}")
            return False
