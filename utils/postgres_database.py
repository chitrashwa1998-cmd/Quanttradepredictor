
import os
import pickle
import json
import pandas as pd
import psycopg
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

__all__ = ['PostgresTradingDatabase']

class PostgresTradingDatabase:
    """PostgreSQL implementation for trading database operations."""
    
    def __init__(self):
        """Initialize PostgreSQL connection."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        try:
            self.conn = psycopg.connect(self.database_url)
            self.conn.autocommit = True
            self._create_tables()
        except Exception as e:
            print(f"PostgreSQL connection failed: {str(e)}")
            raise e
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            with self.conn.cursor() as cursor:
                # OHLC datasets table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlc_datasets (
                    name VARCHAR(255) PRIMARY KEY,
                    data BYTEA NOT NULL,
                    rows INTEGER,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Model results table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_results (
                    model_name VARCHAR(255) PRIMARY KEY,
                    results JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Trained models table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trained_models (
                    id SERIAL PRIMARY KEY,
                    models_data BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Predictions table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    model_name VARCHAR(255) PRIMARY KEY,
                    predictions_data BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
        except Exception as e:
            print(f"Table creation failed: {str(e)}")
            raise e
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception:
            return False
    
    def save_ohlc_data(self, data: pd.DataFrame, dataset_name: str = "main_dataset", preserve_full_data: bool = False) -> bool:
        """Save OHLC dataframe to database."""
        try:
            # Serialize the dataframe
            serialized_data = pickle.dumps(data)
            
            # Get metadata
            rows = len(data)
            start_date = data.index.min() if len(data) > 0 else None
            end_date = data.index.max() if len(data) > 0 else None
            
            with self.conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO ohlc_datasets (name, data, rows, start_date, end_date, updated_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (name) DO UPDATE SET
                    data = EXCLUDED.data,
                    rows = EXCLUDED.rows,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    updated_at = CURRENT_TIMESTAMP
                """, (dataset_name, serialized_data, rows, start_date, end_date))
            
            return True
        except Exception as e:
            print(f"Failed to save OHLC data: {str(e)}")
            return False
    
    def load_ohlc_data(self, dataset_name: str = "main_dataset") -> Optional[pd.DataFrame]:
        """Load OHLC dataframe from database."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT data FROM ohlc_datasets WHERE name = %s", (dataset_name,))
                result = cursor.fetchone()
                
                if result:
                    return pickle.loads(result[0])
                return None
        except Exception as e:
            print(f"Failed to load OHLC data: {str(e)}")
            return None
    
    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """Get list of saved datasets."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                SELECT name, rows, start_date, end_date, created_at, updated_at 
                FROM ohlc_datasets ORDER BY updated_at DESC
                """)
                results = cursor.fetchall()
                
                datasets = []
                for row in results:
                    datasets.append({
                        'name': row[0],
                        'rows': row[1],
                        'start_date': row[2].strftime('%Y-%m-%d') if row[2] else None,
                        'end_date': row[3].strftime('%Y-%m-%d') if row[3] else None,
                        'created_at': row[4].strftime('%Y-%m-%d %H:%M:%S') if row[4] else None,
                        'updated_at': row[5].strftime('%Y-%m-%d %H:%M:%S') if row[5] else None
                    })
                
                return datasets
        except Exception as e:
            print(f"Failed to get dataset list: {str(e)}")
            return []
    
    def get_dataset_metadata(self, dataset_name: str = "main_dataset") -> Optional[Dict[str, Any]]:
        """Get metadata for a dataset."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                SELECT name, rows, start_date, end_date, created_at, updated_at 
                FROM ohlc_datasets WHERE name = %s
                """, (dataset_name,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        'name': result[0],
                        'rows': result[1],
                        'start_date': result[2].strftime('%Y-%m-%d') if result[2] else None,
                        'end_date': result[3].strftime('%Y-%m-%d') if result[3] else None,
                        'created_at': result[4].strftime('%Y-%m-%d %H:%M:%S') if result[4] else None,
                        'updated_at': result[5].strftime('%Y-%m-%d %H:%M:%S') if result[5] else None
                    }
                return None
        except Exception as e:
            print(f"Failed to get dataset metadata: {str(e)}")
            return None
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a dataset from database."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM ohlc_datasets WHERE name = %s", (dataset_name,))
                return True
        except Exception as e:
            print(f"Failed to delete dataset: {str(e)}")
            return False
    
    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        """Save model training results."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO model_results (model_name, results, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (model_name) DO UPDATE SET
                    results = EXCLUDED.results,
                    updated_at = CURRENT_TIMESTAMP
                """, (model_name, json.dumps(results)))
            return True
        except Exception as e:
            print(f"Failed to save model results: {str(e)}")
            return False
    
    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model training results."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT results FROM model_results WHERE model_name = %s", (model_name,))
                result = cursor.fetchone()
                
                if result:
                    return json.loads(result[0])
                return None
        except Exception as e:
            print(f"Failed to load model results: {str(e)}")
            return None
    
    def save_trained_models(self, models_dict: Dict[str, Any]) -> bool:
        """Save trained model objects for persistence."""
        try:
            serialized_models = pickle.dumps(models_dict)
            
            with self.conn.cursor() as cursor:
                # Clear existing models and insert new ones
                cursor.execute("DELETE FROM trained_models")
                cursor.execute("""
                INSERT INTO trained_models (models_data, updated_at)
                VALUES (%s, CURRENT_TIMESTAMP)
                """, (serialized_models,))
            return True
        except Exception as e:
            print(f"Failed to save trained models: {str(e)}")
            return False
    
    def load_trained_models(self) -> Optional[Dict[str, Any]]:
        """Load trained model objects from database."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT models_data FROM trained_models ORDER BY updated_at DESC LIMIT 1")
                result = cursor.fetchone()
                
                if result:
                    return pickle.loads(result[0])
                return None
        except Exception as e:
            print(f"Failed to load trained models: {str(e)}")
            return None
    
    def save_predictions(self, predictions: pd.DataFrame, model_name: str) -> bool:
        """Save model predictions."""
        try:
            serialized_predictions = pickle.dumps(predictions)
            
            with self.conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO predictions (model_name, predictions_data, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (model_name) DO UPDATE SET
                    predictions_data = EXCLUDED.predictions_data,
                    updated_at = CURRENT_TIMESTAMP
                """, (model_name, serialized_predictions))
            return True
        except Exception as e:
            print(f"Failed to save predictions: {str(e)}")
            return False
    
    def load_predictions(self, model_name: str) -> Optional[pd.DataFrame]:
        """Load model predictions."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT predictions_data FROM predictions WHERE model_name = %s", (model_name,))
                result = cursor.fetchone()
                
                if result:
                    return pickle.loads(result[0])
                return None
        except Exception as e:
            print(f"Failed to load predictions: {str(e)}")
            return None
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about stored data."""
        try:
            with self.conn.cursor() as cursor:
                # Count datasets
                cursor.execute("SELECT COUNT(*) FROM ohlc_datasets")
                dataset_count = cursor.fetchone()[0]
                
                # Count models
                cursor.execute("SELECT COUNT(*) FROM model_results")
                model_count = cursor.fetchone()[0]
                
                # Count trained models
                cursor.execute("SELECT COUNT(*) FROM trained_models")
                trained_model_count = cursor.fetchone()[0]
                
                # Count predictions
                cursor.execute("SELECT COUNT(*) FROM predictions")
                prediction_count = cursor.fetchone()[0]
                
                # Get datasets
                datasets = self.get_dataset_list()
                
                # Get available keys (simulate key-value behavior)
                available_keys = []
                cursor.execute("SELECT model_name FROM model_results")
                for row in cursor.fetchall():
                    available_keys.append(f"model_results_{row[0]}")
                
                cursor.execute("SELECT model_name FROM predictions")
                for row in cursor.fetchall():
                    available_keys.append(f"predictions_{row[0]}")
                
                return {
                    'database_type': 'postgresql',
                    'total_datasets': dataset_count,
                    'total_models': model_count,
                    'total_trained_models': trained_model_count,
                    'total_predictions': prediction_count,
                    'total_keys': len(available_keys),
                    'total_records': dataset_count + model_count + trained_model_count + prediction_count,
                    'datasets': datasets,
                    'available_keys': available_keys,
                    'backend': 'PostgreSQL'
                }
        except Exception as e:
            print(f"Failed to get database info: {str(e)}")
            return {
                'database_type': 'postgresql',
                'total_datasets': 0,
                'total_models': 0,
                'total_trained_models': 0,
                'total_predictions': 0,
                'total_keys': 0,
                'total_records': 0,
                'datasets': [],
                'available_keys': [],
                'backend': 'PostgreSQL'
            }
    
    def recover_data(self) -> Optional[pd.DataFrame]:
        """Try to recover any available OHLC data from database."""
        try:
            datasets = self.get_dataset_list()
            if datasets:
                # Return the most recently updated dataset
                return self.load_ohlc_data(datasets[0]['name'])
            return None
        except Exception as e:
            print(f"Failed to recover data: {str(e)}")
            return None
    
    def clear_all_data(self) -> bool:
        """Clear all data from database."""
        try:
            with self.conn.cursor() as cursor:
                print("Clearing predictions...")
                cursor.execute("DELETE FROM predictions")
                cursor.execute("SELECT COUNT(*) FROM predictions")
                pred_count = cursor.fetchone()[0]
                print(f"Deleted {cursor.rowcount} prediction records")
                
                print("Clearing trained models...")
                cursor.execute("DELETE FROM trained_models")
                cursor.execute("SELECT COUNT(*) FROM trained_models")
                model_count = cursor.fetchone()[0]
                print(f"Deleted {cursor.rowcount} trained model records")
                
                print("Clearing model results...")
                cursor.execute("DELETE FROM model_results")
                cursor.execute("SELECT COUNT(*) FROM model_results")
                result_count = cursor.fetchone()[0]
                print(f"Deleted {cursor.rowcount} model result records")
                
                print("Clearing OHLC datasets...")
                cursor.execute("DELETE FROM ohlc_datasets")
                cursor.execute("SELECT COUNT(*) FROM ohlc_datasets")
                dataset_count = cursor.fetchone()[0]
                print(f"Deleted {cursor.rowcount} dataset records")
                
                print("✅ Database cleared successfully")
                
                # Verify all tables are empty
                cursor.execute("SELECT COUNT(*) FROM predictions")
                pred_verify = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM trained_models")
                model_verify = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM model_results")
                result_verify = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM ohlc_datasets")
                dataset_verify = cursor.fetchone()[0]
                
                if pred_verify == 0 and model_verify == 0 and result_verify == 0 and dataset_verify == 0:
                    print("✅ Verification: All data successfully cleared")
                    return True
                else:
                    print(f"⚠️ Warning: Some data may remain - Predictions: {pred_verify}, Models: {model_verify}, Results: {result_verify}, Datasets: {dataset_verify}")
                    return False
                    
        except Exception as e:
            print(f"Failed to clear database: {str(e)}")
            return False
    
    def delete_model_results(self, model_name: str) -> bool:
        """Delete model results for a specific model."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM model_results WHERE model_name = %s", (model_name,))
                return True
        except Exception as e:
            print(f"Failed to delete model results: {str(e)}")
            return False
    
    def delete_predictions(self, model_name: str) -> bool:
        """Delete predictions for a specific model."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM predictions WHERE model_name = %s", (model_name,))
                return True
        except Exception as e:
            print(f"Failed to delete predictions: {str(e)}")
            return False
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
