import os
import json
import pandas as pd
import psycopg
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

__all__ = ['RowBasedPostgresDatabase']

class RowBasedPostgresDatabase:
    """Row-based PostgreSQL implementation for trading database operations."""

    def __init__(self):
        """Initialize PostgreSQL connection."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")

        try:
            self.conn = psycopg.connect(self.database_url)
            self.conn.autocommit = True
            self._create_row_based_tables()
        except Exception as e:
            print(f"PostgreSQL connection failed: {str(e)}")
            raise e

    def _create_row_based_tables(self):
        """Create row-based tables for efficient data operations."""
        try:
            with self.conn.cursor() as cursor:
                # Row-based OHLC data table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlc_data (
                    id SERIAL PRIMARY KEY,
                    dataset_name VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DECIMAL(12,4) NOT NULL,
                    high DECIMAL(12,4) NOT NULL,
                    low DECIMAL(12,4) NOT NULL,
                    close DECIMAL(12,4) NOT NULL,
                    volume BIGINT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(dataset_name, timestamp)
                )
                """)

                # Create indexes for performance
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlc_dataset_time 
                ON ohlc_data(dataset_name, timestamp)
                """)

                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlc_timestamp 
                ON ohlc_data(timestamp)
                """)

                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlc_dataset 
                ON ohlc_data(dataset_name)
                """)

                # Dataset metadata table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    dataset_name VARCHAR(255) PRIMARY KEY,
                    total_rows INTEGER DEFAULT 0,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

                print("✅ Row-based database tables initialized successfully")

        except Exception as e:
            print(f"Row-based table creation failed: {str(e)}")
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
        """Save OHLC dataframe to row-based storage with append capability."""
        try:
            if data is None or len(data) == 0:
                return False

            # Prepare data for insertion
            data_copy = data.copy()

            # Map common column name variations to standard names
            column_mapping = {
                'open': 'Open', 'OPEN': 'Open', 'o': 'Open',
                'high': 'High', 'HIGH': 'High', 'h': 'High',
                'low': 'Low', 'LOW': 'Low', 'l': 'Low',
                'close': 'Close', 'CLOSE': 'Close', 'c': 'Close',
                'volume': 'Volume', 'VOLUME': 'Volume', 'v': 'Volume', 'vol': 'Volume'
            }

            # Apply column mapping
            for old_name, new_name in column_mapping.items():
                if old_name in data_copy.columns and new_name not in data_copy.columns:
                    data_copy = data_copy.rename(columns={old_name: new_name})

            # Check for required columns and create missing ones
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data_copy.columns]

            if missing_columns:
                print(f"Missing columns: {missing_columns}")

                # Try to create missing columns from available data
                if 'Close' in data_copy.columns:
                    # Use Close price as fallback for missing OHLC values
                    for col in ['Open', 'High', 'Low']:
                        if col not in data_copy.columns:
                            data_copy[col] = data_copy['Close']
                            print(f"✅ Created {col} column using Close price")

                # Always ensure Volume exists
                if 'Volume' not in data_copy.columns:
                    data_copy['Volume'] = 0
                    print("✅ Created Volume column with default value 0")

                # Final check
                still_missing = [col for col in required_columns if col not in data_copy.columns]
                if still_missing:
                    print(f"❌ Cannot proceed - still missing: {still_missing}")
                    return False

            # Ensure Volume column exists and has proper values
            if 'Volume' not in data_copy.columns:
                data_copy['Volume'] = 0

            # Convert timestamps to proper format
            if not isinstance(data_copy.index, pd.DatetimeIndex):
                data_copy.index = pd.to_datetime(data_copy.index)

            with self.conn.cursor() as cursor:
                # Insert data row by row using ON CONFLICT for upsert
                insert_count = 0
                update_count = 0

                for timestamp, row in data_copy.iterrows():
                    cursor.execute("""
                    INSERT INTO ohlc_data 
                    (dataset_name, timestamp, open, high, low, close, volume, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (dataset_name, timestamp) 
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        created_at = CURRENT_TIMESTAMP
                    """, (
                        dataset_name,
                        timestamp,
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']) if pd.notna(row['Volume']) else 0
                    ))

                    if cursor.rowcount > 0:
                        insert_count += 1

                # Update metadata
                self._update_dataset_metadata(dataset_name)

                print(f"✅ Saved {insert_count} rows to dataset '{dataset_name}'")
                return True

        except Exception as e:
            print(f"Failed to save OHLC data: {str(e)}")
            print(f"Data shape: {data_copy.shape if 'data_copy' in locals() else 'Unknown'}")
            print(f"Data columns: {list(data_copy.columns) if 'data_copy' in locals() else 'Unknown'}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
            return False

    def append_ohlc_data(self, new_data: pd.DataFrame, dataset_name: str = "main_dataset") -> bool:
        """Append new OHLC data to existing dataset (true append operation)."""
        try:
            if new_data is None or len(new_data) == 0:
                return False

            # This uses the same save_ohlc_data method which handles conflicts automatically
            return self.save_ohlc_data(new_data, dataset_name, preserve_full_data=True)

        except Exception as e:
            print(f"Failed to append OHLC data: {str(e)}")
            return False

    def load_ohlc_data(self, dataset_name: str = "main_dataset", limit: Optional[int] = None, 
                       start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load OHLC dataframe from row-based storage with optional filtering."""
        try:
            with self.conn.cursor() as cursor:
                # Build query with optional filters
                query = """
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlc_data 
                WHERE dataset_name = %s
                """
                params = [dataset_name]

                # Add date filters if provided
                if start_date:
                    query += " AND timestamp >= %s"
                    params.append(start_date)

                if end_date:
                    query += " AND timestamp <= %s"
                    params.append(end_date)

                # Order by timestamp
                query += " ORDER BY timestamp ASC"

                # Add limit if specified
                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query, params)
                results = cursor.fetchall()

                if results:
                    # Convert to DataFrame
                    df = pd.DataFrame(results, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')

                    # Convert to proper data types
                    for col in ['Open', 'High', 'Low', 'Close']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)

                    return df

                return None

        except Exception as e:
            print(f"Failed to load OHLC data: {str(e)}")
            return None

    def get_latest_rows(self, dataset_name: str = "main_dataset", count: int = 250) -> Optional[pd.DataFrame]:
        """Get the latest N rows for a dataset (useful for seeding live data)."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlc_data 
                WHERE dataset_name = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """, (dataset_name, count))

                results = cursor.fetchall()

                if results:
                    # Convert to DataFrame
                    df = pd.DataFrame(results, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')

                    # Convert to proper data types
                    for col in ['Open', 'High', 'Low', 'Close']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)

                    # Sort ascending (since we got descending from query)
                    df = df.sort_index()

                    return df

                return None

        except Exception as e:
            print(f"Failed to get latest rows: {str(e)}")
            return None

    def _update_dataset_metadata(self, dataset_name: str):
        """Update metadata for a dataset."""
        try:
            with self.conn.cursor() as cursor:
                # Get dataset statistics
                cursor.execute("""
                SELECT 
                    COUNT(*) as total_rows,
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date
                FROM ohlc_data 
                WHERE dataset_name = %s
                """, (dataset_name,))

                result = cursor.fetchone()
                if result:
                    total_rows, start_date, end_date = result

                    # Update or insert metadata
                    cursor.execute("""
                    INSERT INTO dataset_metadata 
                    (dataset_name, total_rows, start_date, end_date, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (dataset_name) 
                    DO UPDATE SET
                        total_rows = EXCLUDED.total_rows,
                        start_date = EXCLUDED.start_date,
                        end_date = EXCLUDED.end_date,
                        updated_at = CURRENT_TIMESTAMP
                    """, (dataset_name, total_rows, start_date, end_date))

        except Exception as e:
            print(f"Failed to update metadata: {str(e)}")

    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """Get list of datasets with metadata."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                SELECT 
                    dataset_name, 
                    total_rows, 
                    start_date, 
                    end_date, 
                    created_at, 
                    updated_at 
                FROM dataset_metadata 
                ORDER BY updated_at DESC
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

    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a dataset and its metadata."""
        try:
            with self.conn.cursor() as cursor:
                # Delete data rows
                cursor.execute("DELETE FROM ohlc_data WHERE dataset_name = %s", (dataset_name,))
                data_deleted = cursor.rowcount

                # Delete metadata
                cursor.execute("DELETE FROM dataset_metadata WHERE dataset_name = %s", (dataset_name,))
                meta_deleted = cursor.rowcount

                print(f"✅ Deleted {data_deleted} data rows and metadata for '{dataset_name}'")
                return True

        except Exception as e:
            print(f"Failed to delete dataset: {str(e)}")
            return False

    def migrate_from_blob_storage(self, blob_db, dataset_mapping: Dict[str, str] = None) -> Dict[str, bool]:
        """Migrate data from blob-based storage to row-based storage."""
        try:
            if dataset_mapping is None:
                dataset_mapping = {"main_dataset": "main_dataset"}

            migration_results = {}

            for blob_name, row_name in dataset_mapping.items():
                try:
                    print(f"🔄 Migrating {blob_name} to {row_name}...")

                    # Load data from blob storage
                    blob_data = blob_db.load_ohlc_data(blob_name)

                    if blob_data is not None and len(blob_data) > 0:
                        # Save to row-based storage
                        success = self.save_ohlc_data(blob_data, row_name)
                        migration_results[blob_name] = success

                        if success:
                            print(f"✅ Migrated {len(blob_data)} rows from {blob_name} to {row_name}")
                        else:
                            print(f"❌ Failed to migrate {blob_name}")
                    else:
                        print(f"⚠️ No data found in {blob_name}")
                        migration_results[blob_name] = False

                except Exception as e:
                    print(f"❌ Error migrating {blob_name}: {str(e)}")
                    migration_results[blob_name] = False

            return migration_results

        except Exception as e:
            print(f"Migration failed: {str(e)}")
            return {}

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about stored data."""
        try:
            with self.conn.cursor() as cursor:
                # Force refresh connection to avoid stale data
                self.conn.commit()

                # Get dataset count and details
                cursor.execute("SELECT COUNT(*) FROM dataset_metadata")
                result = cursor.fetchone()
                dataset_count = result[0] if result else 0

                # Get total record count from ohlc_data
                cursor.execute("SELECT COUNT(*) FROM ohlc_data")
                result = cursor.fetchone()
                total_records = result[0] if result else 0

                # Get model counts
                cursor.execute("SELECT COUNT(*) FROM model_results")
                result = cursor.fetchone()
                model_count = result[0] if result else 0

                cursor.execute("SELECT COUNT(*) FROM trained_models")
                result = cursor.fetchone()
                trained_model_count = result[0] if result else 0

                cursor.execute("SELECT COUNT(*) FROM predictions")
                result = cursor.fetchone()
                prediction_count = result[0] if result else 0

                # Get datasets list
                datasets = self.get_dataset_list()

                # Debug print
                print(f"Database info - Datasets: {dataset_count}, Records: {total_records}, Models: {model_count}")

                return {
                    'database_type': 'postgresql_row_based',
                    'total_datasets': dataset_count,
                    'total_records': total_records,
                    'total_models': model_count,
                    'total_trained_models': trained_model_count,
                    'total_predictions': prediction_count,
                    'datasets': datasets,
                    'backend': 'PostgreSQL (Row-Based)',
                    'storage_type': 'Row-Based',
                    'supports_append': True,
                    'supports_range_queries': True
                }
        except Exception as e:
            print(f"Failed to get database info: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {
                'database_type': 'postgresql_row_based',
                'total_datasets': 0,
                'total_records': 0,
                'total_models': 0,
                'total_trained_models': 0,
                'total_predictions': 0,
                'datasets': [],
                'backend': 'PostgreSQL (Row-Based)',
                'storage_type': 'Row-Based',
                'supports_append': True,
                'supports_range_queries': True
            }

    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        """Save model training results."""
        try:
            with self.conn.cursor() as cursor:
                # Create table if it doesn't exist
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_results (
                    model_name VARCHAR(255) PRIMARY KEY,
                    results JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

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
            import pickle
            serialized_models = pickle.dumps(models_dict)

            with self.conn.cursor() as cursor:
                # Create table if it doesn't exist
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trained_models (
                    id SERIAL PRIMARY KEY,
                    models_data BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

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
            import pickle
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
            import pickle
            serialized_predictions = pickle.dumps(predictions)

            with self.conn.cursor() as cursor:
                # Create table if it doesn't exist
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    model_name VARCHAR(255) PRIMARY KEY,
                    predictions_data BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

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
            import pickle
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT predictions_data FROM predictions WHERE model_name = %s", (model_name,))
                result = cursor.fetchone()

                if result:
                    return pickle.loads(result[0])
                return None
        except Exception as e:
            print(f"Failed to load predictions: {str(e)}")
            return None

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
                print("Clearing all data from database...")

                # Get counts before clearing
                cursor.execute("SELECT COUNT(*) FROM dataset_metadata")
                old_dataset_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM ohlc_data")
                old_record_count = cursor.fetchone()[0]

                print(f"Before clearing: {old_dataset_count} datasets, {old_record_count} records")

                print("Clearing predictions...")
                cursor.execute("DROP TABLE IF EXISTS predictions CASCADE")

                print("Clearing trained models...")
                cursor.execute("DROP TABLE IF EXISTS trained_models CASCADE")

                print("Clearing model results...")
                cursor.execute("DROP TABLE IF EXISTS model_results CASCADE")

                print("Clearing OHLC data...")
                cursor.execute("DROP TABLE IF EXISTS ohlc_data CASCADE")

                print("Clearing dataset metadata...")
                cursor.execute("DROP TABLE IF EXISTS dataset_metadata CASCADE")

                # Recreate tables
                self._create_row_based_tables()

                # Verify tables are empty
                cursor.execute("SELECT COUNT(*) FROM dataset_metadata")
                new_dataset_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM ohlc_data")
                new_record_count = cursor.fetchone()[0]

                print(f"After clearing: {new_dataset_count} datasets, {new_record_count} records")

                if new_dataset_count == 0 and new_record_count == 0:
                    print("✅ Database cleared and recreated successfully")
                    return True
                else:
                    print(f"⚠️ Warning: Database may not be fully cleared - {new_dataset_count} datasets, {new_record_count} records remain")
                    return False

        except Exception as e:
            print(f"Failed to clear database: {str(e)}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
            return False

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()