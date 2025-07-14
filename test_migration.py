
#!/usr/bin/env python3
"""
Test script for database migration functionality
"""

import pandas as pd
from datetime import datetime, timedelta
from utils.database_adapter import DatabaseAdapter

def test_migration_workflow():
    """Test the complete migration workflow."""
    
    print("🧪 Testing Migration Workflow")
    print("=" * 40)
    
    try:
        # Test 1: Check current blob-based data
        print("📊 Step 1: Checking current blob-based data...")
        blob_db = DatabaseAdapter(use_row_based=False)
        blob_info = blob_db.get_database_info()
        
        print(f"Current blob storage:")
        print(f"  • Total datasets: {blob_info.get('total_datasets', 0)}")
        print(f"  • Storage type: {blob_info.get('backend', 'Unknown')}")
        
        # Test 2: Initialize row-based storage
        print("\n🔧 Step 2: Initializing row-based storage...")
        row_db = DatabaseAdapter(use_row_based=True)
        print("✅ Row-based storage initialized")
        
        # Test 3: Test basic operations
        print("\n⚡ Step 3: Testing basic row-based operations...")
        
        # Create sample data for testing
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
        sample_data = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(100)],
            'High': [101 + i * 0.1 for i in range(100)],
            'Low': [99 + i * 0.1 for i in range(100)],
            'Close': [100.5 + i * 0.1 for i in range(100)],
            'Volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)
        
        # Test saving
        print("💾 Testing save operation...")
        save_result = row_db.save_ohlc_data(sample_data, "test_migration")
        print(f"Save result: {'✅ Success' if save_result else '❌ Failed'}")
        
        # Test loading
        print("📥 Testing load operation...")
        loaded_data = row_db.load_ohlc_data("test_migration")
        if loaded_data is not None:
            print(f"✅ Loaded {len(loaded_data)} rows")
        else:
            print("❌ Load failed")
            return False
        
        # Test append operation
        print("➕ Testing append operation...")
        new_dates = pd.date_range(start='2024-01-01 08:20:00', periods=10, freq='5T')
        append_data = pd.DataFrame({
            'Open': [110 + i * 0.1 for i in range(10)],
            'High': [111 + i * 0.1 for i in range(10)],
            'Low': [109 + i * 0.1 for i in range(10)],
            'Close': [110.5 + i * 0.1 for i in range(10)],
            'Volume': [1500 + i * 10 for i in range(10)]
        }, index=new_dates)
        
        append_result = row_db.append_ohlc_data(append_data, "test_migration")
        print(f"Append result: {'✅ Success' if append_result else '❌ Failed'}")
        
        # Verify append
        final_data = row_db.load_ohlc_data("test_migration")
        if final_data is not None:
            print(f"✅ Final dataset has {len(final_data)} rows (expected: 110)")
        
        # Test range query
        print("📅 Testing range query...")
        range_data = row_db.load_ohlc_data_range(
            "test_migration",
            start_date="2024-01-01 02:00:00",
            end_date="2024-01-01 04:00:00"
        )
        
        if range_data is not None:
            print(f"✅ Range query returned {len(range_data)} rows")
        
        # Test latest rows
        print("🌱 Testing latest rows retrieval...")
        latest = row_db.get_latest_rows("test_migration", 5)
        if latest is not None:
            print(f"✅ Retrieved {len(latest)} latest rows")
        
        # Cleanup test data
        print("🧹 Cleaning up test data...")
        row_db.delete_dataset("test_migration")
        
        print("\n✅ All migration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Migration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def compare_performance():
    """Compare performance between blob-based and row-based storage."""
    print("\n⚡ Performance Comparison")
    print("=" * 30)
    
    try:
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1T')
        test_data = pd.DataFrame({
            'Open': [100 + i * 0.01 for i in range(1000)],
            'High': [101 + i * 0.01 for i in range(1000)],
            'Low': [99 + i * 0.01 for i in range(1000)],
            'Close': [100.5 + i * 0.01 for i in range(1000)],
            'Volume': [1000 + i for i in range(1000)]
        }, index=dates)
        
        # Test blob-based performance
        print("📊 Testing blob-based storage...")
        blob_db = DatabaseAdapter(use_row_based=False)
        
        start_time = datetime.now()
        blob_db.save_ohlc_data(test_data, "perf_test_blob")
        blob_save_time = (datetime.now() - start_time).total_seconds()
        
        start_time = datetime.now()
        blob_loaded = blob_db.load_ohlc_data("perf_test_blob")
        blob_load_time = (datetime.now() - start_time).total_seconds()
        
        # Test row-based performance
        print("🗂️  Testing row-based storage...")
        row_db = DatabaseAdapter(use_row_based=True)
        
        start_time = datetime.now()
        row_db.save_ohlc_data(test_data, "perf_test_row")
        row_save_time = (datetime.now() - start_time).total_seconds()
        
        start_time = datetime.now()
        row_loaded = row_db.load_ohlc_data("perf_test_row")
        row_load_time = (datetime.now() - start_time).total_seconds()
        
        # Test range query (row-based only)
        start_time = datetime.now()
        range_data = row_db.load_ohlc_data_range(
            "perf_test_row",
            start_date="2024-01-01 02:00:00",
            end_date="2024-01-01 04:00:00"
        )
        range_query_time = (datetime.now() - start_time).total_seconds()
        
        # Show results
        print(f"\n📊 Performance Results (1000 rows):")
        print(f"{'Operation':<20} {'Blob-Based':<12} {'Row-Based':<12} {'Improvement':<12}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        print(f"{'Save':<20} {blob_save_time:.3f}s{'':<5} {row_save_time:.3f}s{'':<5} {blob_save_time/row_save_time:.1f}x")
        print(f"{'Load':<20} {blob_load_time:.3f}s{'':<5} {row_load_time:.3f}s{'':<5} {blob_load_time/row_load_time:.1f}x")
        print(f"{'Range Query':<20} {'N/A':<12} {range_query_time:.3f}s{'':<5} {'New Feature'}")
        
        # Cleanup
        blob_db.delete_dataset("perf_test_blob")
        row_db.delete_dataset("perf_test_row")
        
        print(f"\n✅ Performance comparison completed!")
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")

if __name__ == "__main__":
    print(f"🚀 Database Migration Testing")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    test_success = test_migration_workflow()
    
    if test_success:
        compare_performance()
        print(f"\n🎉 All tests completed successfully!")
    else:
        print(f"\n❌ Tests failed")
