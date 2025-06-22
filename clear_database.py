
"""
Clear all data from TribexAlpha database
"""

from utils.database_adapter import DatabaseAdapter

def clear_all_data():
    """Clear all data from the database with detailed logging"""
    try:
        db = DatabaseAdapter()
        
        # Show what's in the database before clearing
        print("=== DATABASE CONTENTS BEFORE CLEARING ===")
        db_info = db.get_database_info()
        
        print(f"Total datasets: {db_info.get('total_datasets', 0)}")
        print(f"Total model results: {db_info.get('total_models', 0)}")
        print(f"Total trained models: {db_info.get('total_trained_models', 0)}")
        print(f"Total predictions: {db_info.get('total_predictions', 0)}")
        
        if 'available_keys' in db_info:
            keys = db_info['available_keys']
            print(f"Total keys found: {len(keys)}")
            
            # Group keys by type for better understanding
            key_types = {}
            for key in keys:
                if key.startswith('ohlc_'):
                    key_type = 'OHLC Data'
                elif key.startswith('model_results_'):
                    key_type = 'Model Results'
                elif key.startswith('predictions_'):
                    key_type = 'Predictions'
                elif key.startswith('trained_models_'):
                    key_type = 'Trained Models'
                else:
                    key_type = 'Other'
                
                if key_type not in key_types:
                    key_types[key_type] = []
                key_types[key_type].append(key)
            
            for key_type, key_list in key_types.items():
                print(f"{key_type}: {len(key_list)} keys")
                for key in key_list:
                    print(f"  - {key}")
        else:
            print("No keys found or unable to retrieve key list")
        
        print("\n=== CLEARING DATABASE ===")
        
        # Clear all data
        success = db.clear_all_data()
        
        if success:
            print("✓ Database clearing method executed successfully")
            
            # Verify clearing by checking database info again
            final_db_info = db.get_database_info()
            final_keys = final_db_info.get('available_keys', [])
            
            if final_keys:
                print(f"✗ Warning: {len(final_keys)} keys still remain:")
                for key in final_keys:
                    print(f"  - {key}")
            else:
                print("✓ All data cleared successfully from database")
        else:
            print("✗ Failed to clear data")
        
        # Final verification
        final_db_info = db.get_database_info()
        final_keys = final_db_info.get('available_keys', [])
        print(f"\n=== FINAL STATUS ===")
        print(f"Keys remaining: {len(final_keys)}")
        if final_keys:
            print("Remaining keys:", final_keys)
        
        return len(final_keys) == 0
        
    except Exception as e:
        print(f"Error clearing data: {str(e)}")
        return False

if __name__ == "__main__":
    clear_all_data()
