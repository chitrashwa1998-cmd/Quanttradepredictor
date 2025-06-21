"""
Database adapter for PostgreSQL only
Provides a unified interface for the trading application using PostgreSQL exclusively
"""
import os
from typing import Dict, List, Optional, Any

class DatabaseAdapter:
    """PostgreSQL-only database interface for the trading application."""

    def __init__(self):
        """Initialize database adapter with PostgreSQL only."""
        self.db_type = "postgresql"

        # Check for PostgreSQL environment variable
        if not os.getenv('DATABASE_URL'):
            raise ValueError("DATABASE_URL environment variable not set. Please create a PostgreSQL database in Replit first.")

        try:
            from utils.postgres_database import PostgresTradingDatabase
            self.db = PostgresTradingDatabase()
            if self._test_connection():
                print("✅ Using PostgreSQL database")
            else:
                raise ConnectionError("Failed to connect to PostgreSQL")
        except Exception as e:
            print(f"❌ PostgreSQL initialization failed: {str(e)}")
            raise e

    def _test_connection(self) -> bool:
        """Test database connection."""
        try:
            return self.db.test_connection()
        except Exception as e:
            print(f"Database connection test failed: {str(e)}")
            return False

    def save_ohlc_data(self, data, dataset_name: str = "main_dataset", preserve_full_data: bool = False) -> bool:
        """Save OHLC dataframe to database."""
        return self.db.save_ohlc_data(data, dataset_name, preserve_full_data)

    def load_ohlc_data(self, dataset_name: str = "main_dataset"):
        """Load OHLC data from database"""
        try:
            # Since we're using PostgreSQL exclusively, call load_ohlc_data directly
            data = self.db.load_ohlc_data(dataset_name)

            # If no data found and looking for main_dataset, try to find any available dataset
            if (data is None or len(data) == 0) and dataset_name == "main_dataset":
                # Get list of available datasets
                db_info = self.db.get_database_info()
                if db_info and 'datasets' in db_info and db_info['datasets']:
                    # Use the most recent uploaded dataset
                    available_datasets = db_info['datasets']
                    if available_datasets:
                        latest_dataset = available_datasets[0]['name']  # datasets are sorted by updated_at DESC
                        print(f"Main dataset not found, loading latest dataset: {latest_dataset}")
                        data = self.db.load_ohlc_data(latest_dataset)

            return data
        except Exception as e:
            print(f"Error loading OHLC data: {e}")
            return None

    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """Get list of saved datasets."""
        return self.db.get_dataset_list()

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
        """Get information about stored data."""
        info = self.db.get_database_info()
        info['adapter_type'] = self.db_type
        return info

    def recover_data(self):
        """Try to recover any available OHLC data from database."""
        return self.db.recover_data()

    def clear_all_data(self) -> bool:
        """Clear all data from database"""
        try:
            # Since we're using PostgreSQL exclusively, call clear_all_data directly
            return self.db.clear_all_data()
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False

    def create_main_dataset_from_latest(self):
        """Create main_dataset from the latest uploaded dataset"""
        try:
            # Get latest uploaded dataset
            datasets = self.get_dataset_list()
            if datasets:
                latest_dataset = datasets[0]['name']  # datasets are sorted by updated_at DESC
                
                # Load the latest dataset
                data = self.db.load_ohlc_data(latest_dataset)
                if data is not None:
                    # Save it as main_dataset
                    success = self.db.save_ohlc_data(data, "main_dataset")
                    if success:
                        print(f"Created main_dataset from {latest_dataset}")
                        return True
            return False
        except Exception as e:
            print(f"Error creating main_dataset: {e}")
            return False

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