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
        print(f"Total records: {db_info.get('total_records', 0)}")
        print(f"Total model results: {db_info.get('total_models', 0)}")
        print(f"Total trained models: {db_info.get('total_trained_models', 0)}")
        print(f"Total predictions: {db_info.get('total_predictions', 0)}")

        datasets = db_info.get('datasets', [])
        if datasets:
            print(f"Datasets found: {len(datasets)}")
            for dataset in datasets:
                print(f"  - {dataset['name']}: {dataset['rows']} rows")
        else:
            print("No datasets found in metadata")

        print("\n=== CLEARING DATABASE ===")

        # Clear all data
        success = db.clear_all_data()

        if success:
            print("✓ Database clearing method executed successfully")

            # Verify clearing by checking database info again
            print("\n=== VERIFYING CLEAR OPERATION ===")
            final_db_info = db.get_database_info()

            print(f"Final datasets: {final_db_info.get('total_datasets', 0)}")
            print(f"Final records: {final_db_info.get('total_records', 0)}")
            print(f"Final models: {final_db_info.get('total_models', 0)}")
            print(f"Final trained models: {final_db_info.get('total_trained_models', 0)}")
            print(f"Final predictions: {final_db_info.get('total_predictions', 0)}")

            final_datasets = final_db_info.get('datasets', [])
            if final_datasets:
                print(f"⚠️ Warning: {len(final_datasets)} datasets still remain:")
                for dataset in final_datasets:
                    print(f"  - {dataset['name']}: {dataset['rows']} rows")
                return False
            else:
                print("✅ All data cleared successfully from database")
                return True
        else:
            print("✗ Failed to clear data")
            return False

    except Exception as e:
        print(f"Error clearing data: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    result = clear_all_data()
    if result:
        print("\n🎉 Database successfully cleared!")
    else:
        print("\n❌ Database clearing failed or incomplete!")