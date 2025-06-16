"""
Clear all data from TribexAlpha database
"""

from utils.database import TradingDatabase

def clear_all_data():
    """Clear all data from the database"""
    try:
        db = TradingDatabase()
        success = db.clear_all_data()
        if success:
            print("✓ All data cleared successfully from database")
        else:
            print("✗ Failed to clear data")
        return success
    except Exception as e:
        print(f"Error clearing data: {str(e)}")
        return False

if __name__ == "__main__":
    clear_all_data()