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
    return get_database()

def close_database():
    """Close database connection"""
    global _database
    if _database:
        # Add cleanup if needed
        _database = None
        logger.info("Database connection closed")