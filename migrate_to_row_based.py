
#!/usr/bin/env python3
"""
Migration script from blob-based to row-based PostgreSQL storage
"""

import os
import sys
from datetime import datetime
from utils.postgres_database import PostgresTradingDatabase
from utils.row_based_database import RowBasedPostgresDatabase

def migrate_database():
    """Migrate from blob-based to row-based storage."""
    
    print("🚀 Starting Database Migration")
    print("=" * 50)
    
    try:
        # Initialize both databases
        print("🔄 Initializing databases...")
        blob_db = PostgresTradingDatabase()
        row_db = RowBasedPostgresDatabase()
        
        print("✅ Both databases initialized successfully")
        
        # Get existing datasets from blob storage
        print("\n📊 Analyzing existing data...")
        existing_datasets = blob_db.get_dataset_list()
        
        if not existing_datasets:
            print("⚠️ No datasets found in blob storage")
            return True
        
        print(f"Found {len(existing_datasets)} datasets to migrate:")
        for dataset in existing_datasets:
            print(f"  • {dataset['name']}: {dataset['rows']} rows")
        
        # Create migration mapping
        dataset_mapping = {}
        for dataset in existing_datasets:
            old_name = dataset['name']
            new_name = old_name  # Keep same names, but you can customize this
            dataset_mapping[old_name] = new_name
        
        print(f"\n🔄 Starting migration of {len(dataset_mapping)} datasets...")
        
        # Perform migration
        migration_results = row_db.migrate_from_blob_storage(blob_db, dataset_mapping)
        
        # Report results
        successful_migrations = sum(1 for success in migration_results.values() if success)
        failed_migrations = len(migration_results) - successful_migrations
        
        print(f"\n📊 Migration Results:")
        print(f"✅ Successful: {successful_migrations}")
        print(f"❌ Failed: {failed_migrations}")
        
        if failed_migrations > 0:
            print("\n❌ Failed migrations:")
            for dataset, success in migration_results.items():
                if not success:
                    print(f"  • {dataset}")
        
        # Verify migration
        print(f"\n🔍 Verifying migration...")
        row_datasets = row_db.get_dataset_list()
        
        print(f"Row-based storage now contains {len(row_datasets)} datasets:")
        for dataset in row_datasets:
            print(f"  • {dataset['name']}: {dataset['rows']} rows")
        
        # Show next steps
        if successful_migrations > 0:
            print(f"\n✅ Migration completed successfully!")
            print(f"\n📋 Next Steps:")
            print(f"1. Test the new row-based system")
            print(f"2. Update your application to use row-based storage")
            print(f"3. Keep blob tables as backup until you're confident")
            print(f"4. Enjoy true append operations and better performance!")
            
            print(f"\n🎯 Benefits of row-based storage:")
            print(f"• True append operations (no more dataset replacement)")
            print(f"• Efficient range queries")
            print(f"• Better memory usage")
            print(f"• Concurrent access support")
            print(f"• Partial data loading capabilities")
        
        return successful_migrations > 0
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_row_based_operations():
    """Test basic row-based operations."""
    
    print("\n🧪 Testing Row-Based Operations")
    print("=" * 40)
    
    try:
        row_db = RowBasedPostgresDatabase()
        
        # Test 1: Load data
        print("📥 Test 1: Loading data...")
        test_data = row_db.load_ohlc_data("main_dataset")
        if test_data is not None:
            print(f"✅ Loaded {len(test_data)} rows")
        else:
            print("❌ No data found")
            return False
        
        # Test 2: Get latest rows (for seeding)
        print("🌱 Test 2: Getting latest rows for seeding...")
        latest_rows = row_db.get_latest_rows("main_dataset", 10)
        if latest_rows is not None:
            print(f"✅ Retrieved {len(latest_rows)} latest rows")
        else:
            print("❌ Failed to get latest rows")
        
        # Test 3: Range query
        print("📅 Test 3: Range query...")
        if len(test_data) > 0:
            mid_date = test_data.index[len(test_data)//2]
            end_date = test_data.index[-1]
            
            range_data = row_db.load_ohlc_data(
                "main_dataset", 
                start_date=mid_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if range_data is not None:
                print(f"✅ Range query returned {len(range_data)} rows")
            else:
                print("❌ Range query failed")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"🕐 Migration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run migration
    migration_success = migrate_database()
    
    if migration_success:
        # Run tests
        test_success = test_row_based_operations()
        
        if test_success:
            print(f"\n🎉 Migration and testing completed successfully!")
            print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"\n⚠️ Migration completed but testing failed")
    else:
        print(f"\n❌ Migration failed")
        sys.exit(1)
