"""
Database adapter to switch between PostgreSQL and key-value store
Provides a unified interface for the trading application
"""
import os
from typing import Dict, List, Optional, Any
from utils.postgres_database import PostgresTradingDatabase

class DatabaseAdapter:
    """Unified database interface that automatically selects PostgreSQL or fallback."""
    
    def __init__(self):
        """Initialize database adapter with PostgreSQL."""
        self.db_type = "postgresql"
        self.db = PostgresTradingDatabase()
        
        # Test PostgreSQL connection
        if not self._test_connection():
            print("PostgreSQL connection failed, check database configuration")
            self.db_type = "error"
    
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
        """Load OHLC dataframe from database."""
        return self.db.load_ohlc_data(dataset_name)
    
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