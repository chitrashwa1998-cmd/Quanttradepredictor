"""
Database adapter for PostgreSQL only
Provides a unified interface for the trading application using PostgreSQL exclusively
"""
import os
from typing import Dict, List, Optional, Any

class DatabaseAdapter:
    """PostgreSQL database interface with support for both blob-based and row-based storage."""

    def __init__(self, use_row_based: bool = True):
        """Initialize database adapter with PostgreSQL."""
        self.use_row_based = use_row_based
        self.db_type = "postgresql_row_based" if use_row_based else "postgresql"

        # Check for PostgreSQL environment variable
        if not os.getenv('DATABASE_URL'):
            raise ValueError("DATABASE_URL environment variable not set. Please create a PostgreSQL database in Replit first.")

        try:
            if use_row_based:
                from utils.row_based_database import RowBasedPostgresDatabase
                self.db = RowBasedPostgresDatabase()
                storage_type = "Row-Based PostgreSQL"
            else:
                from utils.postgres_database import PostgresTradingDatabase
                self.db = PostgresTradingDatabase()
                storage_type = "Blob-Based PostgreSQL"

            if self._test_connection():
                print(f"✅ Using {storage_type} database")
            else:
                raise ConnectionError("Failed to connect to PostgreSQL")
        except Exception as e:
            error_str = str(e).lower()
            if "adminshutdown" in error_str or "terminating connection" in error_str:
                raise ConnectionError("Database connection was terminated. This is normal for idle connections. Please refresh the page to reconnect.")
            print(f"❌ PostgreSQL initialization failed: {str(e)}")
            raise e

    def _test_connection(self) -> bool:
        """Test database connection."""
        try:
            return self.db.test_connection()
        except Exception as e:
            print(f"Database connection test failed: {str(e)}")
            return False

    def save_ohlc_data(self, data, dataset_name: str = "main_dataset", preserve_full_data: bool = False, data_only_mode: bool = False, dataset_purpose: str = "training") -> bool:
        """Save OHLC dataframe to database."""
        if hasattr(self.db, 'save_ohlc_data'):
            # Check if the method supports dataset_purpose parameter
            import inspect
            sig = inspect.signature(self.db.save_ohlc_data)
            if 'data_only_mode' in sig.parameters and 'dataset_purpose' in sig.parameters:
                return self.db.save_ohlc_data(data, dataset_name, preserve_full_data, data_only_mode, dataset_purpose)
            elif 'dataset_purpose' in sig.parameters:
                return self.db.save_ohlc_data(data, dataset_name, preserve_full_data, dataset_purpose=dataset_purpose)
            else:
                return self.db.save_ohlc_data(data, dataset_name, preserve_full_data)
        return False

    def append_ohlc_data(self, new_data, dataset_name: str = "main_dataset") -> bool:
        """Append new OHLC data to existing dataset (only available in row-based storage)."""
        if self.use_row_based and hasattr(self.db, 'append_ohlc_data'):
            return self.db.append_ohlc_data(new_data, dataset_name)
        else:
            # Fallback to save_ohlc_data for blob-based storage
            return self.save_ohlc_data(new_data, dataset_name, preserve_full_data=True)

    def get_latest_rows(self, dataset_name: str = "main_dataset", count: int = 250):
        """Get latest N rows (only available in row-based storage)."""
        if self.use_row_based and hasattr(self.db, 'get_latest_rows'):
            return self.db.get_latest_rows(dataset_name, count)
        else:
            # Fallback: load all data and get tail
            data = self.load_ohlc_data(dataset_name)
            return data.tail(count) if data is not None else None

    def load_ohlc_data_range(self, dataset_name: str = "main_dataset", start_date: str = None, end_date: str = None, limit: int = None):
        """Load OHLC data with date range filtering (only available in row-based storage)."""
        if self.use_row_based and hasattr(self.db, 'load_ohlc_data'):
            return self.db.load_ohlc_data(dataset_name, limit=limit, start_date=start_date, end_date=end_date)
        else:
            # Fallback: load all data and filter
            data = self.load_ohlc_data(dataset_name)
            if data is not None and (start_date or end_date):
                if start_date:
                    data = data[data.index >= start_date]
                if end_date:
                    data = data[data.index <= end_date]
                if limit:
                    data = data.tail(limit)
            return data

    def load_ohlc_data(self, dataset_name: str = "main_dataset"):
        """Load OHLC dataframe from database."""
        return self.db.load_ohlc_data(dataset_name)

    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """Get list of saved datasets."""
        return self.db.get_dataset_list()

    def get_datasets_by_purpose(self, purpose: str = None) -> List[Dict[str, Any]]:
        """Get datasets filtered by purpose."""
        if hasattr(self.db, 'get_datasets_by_purpose'):
            return self.db.get_datasets_by_purpose(purpose)
        else:
            # Fallback for databases without purpose support
            return self.get_dataset_list()

    def get_training_dataset(self) -> str:
        """Get the primary training dataset name."""
        if hasattr(self.db, 'get_training_dataset'):
            return self.db.get_training_dataset()
        return "main_dataset"

    def get_pre_seed_dataset(self) -> str:
        """Get the pre-seed dataset name."""
        if hasattr(self.db, 'get_pre_seed_dataset'):
            return self.db.get_pre_seed_dataset()
        return None

    def get_dataset_metadata(self, dataset_name: str = "main_dataset") -> Optional[Dict[str, Any]]:
        """Get metadata for a dataset."""
        return self.db.get_dataset_metadata(dataset_name)

    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a dataset from database."""
        return self.db.delete_dataset(dataset_name)

    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        """Save model training results."""
        return self.db.save_model_results(model_name, results)

    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model training results."""
        return self.db.load_model_results(model_name)

    def save_trained_models(self, models_dict: Dict[str, Any]) -> bool:
        """Save trained model objects for persistence."""
        return self.db.save_trained_models(models_dict)

    def load_trained_models(self) -> Optional[Dict[str, Any]]:
        """Load trained model objects from database."""
        return self.db.load_trained_models()

    def save_predictions(self, predictions, model_name: str) -> bool:
        """Save model predictions."""
        return self.db.save_predictions(predictions, model_name)

    def load_predictions(self, model_name: str):
        """Load model predictions."""
        return self.db.load_predictions(model_name)

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            if hasattr(self.db, 'get_database_info'):
                info = self.db.get_database_info()

                # Debug logging
                print(f"Database info retrieved: {info}")

                # Ensure datasets list is populated
                if 'datasets' not in info or not info['datasets']:
                    print("No datasets in info, trying direct dataset list...")
                    if hasattr(self.db, 'get_dataset_list'):
                        datasets = self.db.get_dataset_list()
                        info['datasets'] = datasets
                        info['total_datasets'] = len(datasets)
                        print(f"Direct dataset list: {datasets}")

                return info
            else:
                # Fallback for databases without this method
                return {
                    'total_datasets': 0,
                    'datasets': [],
                    'backend': 'Unknown'
                }
        except Exception as e:
            print(f"Error getting database info: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {
                'total_datasets': 0,
                'datasets': [],
                'backend': 'Error'
            }

    def recover_data(self):
        """Try to recover any available OHLC data from database."""
        return self.db.recover_data()

    def clear_all_data(self) -> bool:
        """Clear all data from database."""
        return self.db.clear_all_data()

    def get_connection_status(self) -> Dict[str, Any]:
        """Get database connection status."""
        return {
            'type': self.db_type,
            'connected': self._test_connection(),
            'has_data': len(self.get_dataset_list()) > 0
        }

# Create a global instance
def get_trading_database():
    """Get the trading database instance."""
    return DatabaseAdapter()