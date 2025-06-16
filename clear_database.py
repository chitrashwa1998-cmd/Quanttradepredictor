
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
        keys = list(db.db.keys())
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
            elif key.startswith('trained_models'):
                key_type = 'Trained Models'
            elif key == 'dataset_list':
                key_type = 'Dataset List'
            else:
                key_type = 'Other'
            
            if key_type not in key_types:
                key_types[key_type] = []
            key_types[key_type].append(key)
        
        for key_type, key_list in key_types.items():
            print(f"{key_type}: {len(key_list)} keys")
            for key in key_list:
                print(f"  - {key}")
        
        print("\n=== CLEARING DATABASE ===")
        
        # Clear all data
        success = db.clear_all_data()
        
        if success:
            # Verify clearing
            remaining_keys = list(db.db.keys())
            if remaining_keys:
                print(f"✗ Warning: {len(remaining_keys)} keys still remain:")
                for key in remaining_keys:
                    print(f"  - {key}")
                    try:
                        del db.db[key]
                        print(f"    → Manually deleted {key}")
                    except Exception as e:
                        print(f"    → Failed to delete {key}: {str(e)}")
            else:
                print("✓ All data cleared successfully from database")
        else:
            print("✗ Failed to clear data")
        
        # Final verification
        final_keys = list(db.db.keys())
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
