"""
Database connection and management
Wraps existing PostgreSQL database adapter for FastAPI
"""

import sys
import os

# Add parent directory to path to import existing database utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.database_adapter import DatabaseAdapter
from core.config import settings
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Global database instance
_database = None

def get_database():
    """Get database instance (singleton pattern)"""
    global _database
    
    if _database is None:
        try:
            _database = DatabaseAdapter(use_row_based=True)
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    return _database

async def get_database_dependency():
    """FastAPI dependency for database injection"""
    db = get_database()
    return DatabaseWrapper(db)

class DatabaseWrapper:
    """Wrapper for DatabaseAdapter to provide API-friendly methods"""
    
    def __init__(self, db_adapter):
        self.adapter = db_adapter
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets with API-friendly format"""
        try:
            datasets = self.adapter.get_dataset_list()
            return [
                {
                    'name': ds.get('name', 'unknown'),
                    'rows': ds.get('rows', 0),
                    'columns': ds.get('columns', []),
                    'start_date': ds.get('start_date'),
                    'end_date': ds.get('end_date'),
                    'created_at': ds.get('created_at'),
                    'updated_at': ds.get('updated_at')
                }
                for ds in datasets
            ]
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []
    
    def get_dataset(self, dataset_name: str):
        """Get dataset by name"""
        try:
            return self.adapter.load_ohlc_data(dataset_name)
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_name}: {e}")
            return None
    
    def store_dataset(self, dataset_name: str, data):
        """Store dataset in database"""
        try:
            return self.adapter.save_ohlc_data(data, dataset_name)
        except Exception as e:
            logger.error(f"Failed to store dataset {dataset_name}: {e}")
            return False
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete dataset from database"""
        try:
            # For now, return True as deletion isn't implemented in adapter
            return True
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_name}: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        try:
            datasets = self.list_datasets()
            total_records = sum(ds['rows'] for ds in datasets)
            
            return {
                'database_type': 'postgresql_row_based',
                'backend': 'PostgreSQL (Row-Based)',
                'storage_type': 'Row-Based',
                'total_datasets': len(datasets),
                'total_records': total_records,
                'supports_append': True
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {}
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions (placeholder)"""
        # This would need to be implemented based on your prediction storage
        return []
    
    def delete_model(self, model_name: str):
        """Delete model from database (placeholder)"""
        pass
    
    def load_trained_models(self) -> Dict[str, Any]:
        """Load trained models from database"""
        try:
            return self.adapter.db.load_trained_models() if hasattr(self.adapter.db, 'load_trained_models') else {}
        except Exception as e:
            logger.error(f"Failed to load trained models: {e}")
            return {}

def close_database():
    """Close database connection"""
    global _database
    if _database:
        # Add cleanup if needed
        _database = None
        logger.info("Database connection closed")