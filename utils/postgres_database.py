"""
PostgreSQL database implementation for TribexAlpha trading app
Replaces the key-value store with proper relational database
"""
import os
import json
import pickle
import base64
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

class PostgresTradingDatabase:
    """PostgreSQL database for storing trading data, models, and predictions."""

    def __init__(self):
        """Initialize PostgreSQL connection."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        self._create_tables()

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.database_url)

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

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        create_tables_sql = """
        -- OHLC data storage
        CREATE TABLE IF NOT EXISTS ohlc_datasets (
            id SERIAL PRIMARY KEY,
            dataset_name VARCHAR(255) UNIQUE NOT NULL,
            data_json TEXT NOT NULL,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Model results storage
        CREATE TABLE IF NOT EXISTS model_results (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) UNIQUE NOT NULL,
            results_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Trained models storage (serialized)
        CREATE TABLE IF NOT EXISTS trained_models (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) UNIQUE NOT NULL,
            model_data TEXT NOT NULL,
            task_type VARCHAR(100),
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Predictions storage
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            predictions_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_ohlc_dataset_name ON ohlc_datasets(dataset_name);
        CREATE INDEX IF NOT EXISTS idx_model_name ON model_results(model_name);
        CREATE INDEX IF NOT EXISTS idx_trained_model_name ON trained_models(model_name);
        CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_tables_sql)
                conn.commit()
            print("✅ PostgreSQL tables created successfully")
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
            raise e

    def save_ohlc_data(self, data: pd.DataFrame, dataset_name: str = "main_dataset", preserve_full_data: bool = False) -> bool:
        """Save OHLC dataframe to PostgreSQL database."""
        try:
            # Convert DataFrame to JSON for storage
            data_json = data.to_json(orient='records', date_format='iso')

            # Create metadata
            metadata = {
                'rows': len(data),
                'columns': list(data.columns),
                'start_date': str(data.index[0]) if not data.empty else None,
                'end_date': str(data.index[-1]) if not data.empty else None,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'preserve_full_data': preserve_full_data
            }
            metadata_json = json.dumps(metadata)

            # Insert or update data
            sql = """
            INSERT INTO ohlc_datasets (dataset_name, data_json, metadata_json, updated_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (dataset_name) 
            DO UPDATE SET 
                data_json = EXCLUDED.data_json,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = CURRENT_TIMESTAMP;
            """

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (dataset_name, data_json, metadata_json))
                conn.commit()

            print(f"✅ Successfully saved {len(data)} rows to PostgreSQL")
            return True

        except Exception as e:
            print(f"Error saving OHLC data: {str(e)}")
            return False

    def load_ohlc_data(self, dataset_name: str = "main_dataset") -> Optional[pd.DataFrame]:
        """Load OHLC dataframe from PostgreSQL database."""
        try:
            sql = "SELECT data_json FROM ohlc_datasets WHERE dataset_name = %s;"

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (dataset_name,))
                    result = cursor.fetchone()

            if result:
                data_json = result[0]
                df = pd.read_json(data_json, orient='records')

                # Set datetime index if present
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif 'Datetime' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                    df.set_index('Datetime', inplace=True)

                print(f"✅ Loaded {len(df)} rows from PostgreSQL")
                return df

            return None

        except Exception as e:
            print(f"Error loading OHLC data: {str(e)}")
            return None

    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """Get list of saved datasets."""
        try:
            sql = """
            SELECT dataset_name, metadata_json, created_at, updated_at 
            FROM ohlc_datasets 
            ORDER BY updated_at DESC;
            """

            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql)
                    results = cursor.fetchall()

            datasets = []
            for row in results:
                metadata = json.loads(row['metadata_json']) if row['metadata_json'] else {}
                datasets.append({
                    'name': row['dataset_name'],
                    'rows': metadata.get('rows', 0),
                    'columns': metadata.get('columns', []),
                    'start_date': metadata.get('start_date'),
                    'end_date': metadata.get('end_date'),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                })

            return datasets

        except Exception as e:
            print(f"Error getting dataset list: {str(e)}")
            return []

    def get_dataset_metadata(self, dataset_name: str = "main_dataset") -> Optional[Dict[str, Any]]:
        """Get metadata for a dataset."""
        try:
            sql = "SELECT metadata_json FROM ohlc_datasets WHERE dataset_name = %s;"

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (dataset_name,))
                    result = cursor.fetchone()

            if result and result[0]:
                return json.loads(result[0])

            return None

        except Exception as e:
            print(f"Error getting dataset metadata: {str(e)}")
            return None

    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a dataset from database."""
        try:
            sql = "DELETE FROM ohlc_datasets WHERE dataset_name = %s;"

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (dataset_name,))
                    rows_affected = cursor.rowcount
                conn.commit()

            return rows_affected > 0

        except Exception as e:
            print(f"Error deleting dataset: {str(e)}")
            return False

    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        """Save model training results."""
        try:
            results_json = json.dumps(results, default=str)

            sql = """
            INSERT INTO model_results (model_name, results_json, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (model_name) 
            DO UPDATE SET 
                results_json = EXCLUDED.results_json,
                updated_at = CURRENT_TIMESTAMP;
            """

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (model_name, results_json))
                conn.commit()

            return True

        except Exception as e:
            print(f"Error saving model results: {str(e)}")
            return False

    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model training results."""
        try:
            sql = "SELECT results_json FROM model_results WHERE model_name = %s;"

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (model_name,))
                    result = cursor.fetchone()

            if result:
                return json.loads(result[0])

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

                    sql = """
                    INSERT INTO trained_models (model_name, model_data, task_type, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (model_name) 
                    DO UPDATE SET 
                        model_data = EXCLUDED.model_data,
                        task_type = EXCLUDED.task_type,
                        updated_at = CURRENT_TIMESTAMP;
                    """

                    with self._get_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute(sql, (model_name, model_b64, task_type))
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
            sql = "SELECT model_name, model_data, task_type FROM trained_models;"

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    results = cursor.fetchall()

            if not results:
                return None

            models_dict = {}
            for row in results:
                model_name, model_b64, task_type = row
                try:
                    # Deserialize model data
                    model_pickle = base64.b64decode(model_b64.encode('utf-8'))
                    model_data = pickle.loads(model_pickle)
                    models_dict[model_name] = model_data
                except Exception as e:
                    print(f"Error deserializing model {model_name}: {str(e)}")
                    continue

            return models_dict if models_dict else None

        except Exception as e:
            print(f"Error loading trained models: {str(e)}")
            return None

    def save_predictions(self, predictions: pd.DataFrame, model_name: str) -> bool:
        """Save model predictions."""
        try:
            predictions_json = predictions.to_json(orient='records', date_format='iso')

            sql = """
            INSERT INTO predictions (model_name, predictions_json, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP);
            """

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (model_name, predictions_json))
                conn.commit()

            return True

        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return False

    def load_predictions(self, model_name: str) -> Optional[pd.DataFrame]:
        """Load model predictions."""
        try:
            sql = """
            SELECT predictions_json FROM predictions 
            WHERE model_name = %s 
            ORDER BY created_at DESC 
            LIMIT 1;
            """

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (model_name,))
                    result = cursor.fetchone()

            if result:
                return pd.read_json(result[0], orient='records')

            return None

        except Exception as e:
            print(f"Error loading predictions: {str(e)}")
            return None

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about stored data."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Count datasets
                    cursor.execute("SELECT COUNT(*) FROM ohlc_datasets;")
                    dataset_count = cursor.fetchone()[0]

                    # Count model results
                    cursor.execute("SELECT COUNT(*) FROM model_results;")
                    model_count = cursor.fetchone()[0]

                    # Count trained models
                    cursor.execute("SELECT COUNT(*) FROM trained_models;")
                    trained_model_count = cursor.fetchone()[0]

                    # Count predictions
                    cursor.execute("SELECT COUNT(*) FROM predictions;")
                    prediction_count = cursor.fetchone()[0]

                    # Get dataset info
                    cursor.execute("SELECT dataset_name, metadata_json FROM ohlc_datasets;")
                    dataset_results = cursor.fetchall()

            datasets = []
            for name, metadata_json in dataset_results:
                metadata = json.loads(metadata_json) if metadata_json else {}
                datasets.append({
                    'name': name,
                    'rows': metadata.get('rows', 0),
                    'columns': metadata.get('columns', [])
                })

            return {
                'total_datasets': dataset_count,
                'total_models': model_count,
                'total_trained_models': trained_model_count,
                'total_predictions': prediction_count,
                'datasets': datasets,
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

    def clear_all_data(self) -> bool:
        """Clear all data from database."""
        try:
            clear_sql = """
            DELETE FROM predictions;
            DELETE FROM trained_models;
            DELETE FROM model_results;
            DELETE FROM ohlc_datasets;
            """

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(clear_sql)
                conn.commit()

            return True

        except Exception as e:
            print(f"Error clearing database: {str(e)}")
            return False