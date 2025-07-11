
import os
import psycopg
from datetime import datetime

def inspect_database():
    """Inspect PostgreSQL database to show all datasets and identify conflicts."""
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL environment variable not found")
        return
    
    try:
        conn = psycopg.connect(database_url)
        conn.autocommit = True
        
        print("🔍 DATABASE INSPECTION REPORT")
        print("=" * 50)
        
        with conn.cursor() as cursor:
            # Get all datasets
            cursor.execute("""
            SELECT name, rows, start_date, end_date, created_at, updated_at 
            FROM ohlc_datasets 
            ORDER BY updated_at DESC
            """)
            datasets = cursor.fetchall()
            
            print(f"\n📊 TOTAL DATASETS FOUND: {len(datasets)}")
            print("-" * 50)
            
            if not datasets:
                print("❌ No datasets found in database")
                return
            
            # Analyze each dataset
            main_datasets = []
            live_datasets = []
            other_datasets = []
            
            for i, (name, rows, start_date, end_date, created_at, updated_at) in enumerate(datasets, 1):
                print(f"\n{i}. DATASET: '{name}'")
                print(f"   📈 Rows: {rows}")
                print(f"   📅 Date Range: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}")
                print(f"   🕒 Created: {created_at.strftime('%Y-%m-%d %H:%M:%S') if created_at else 'N/A'}")
                print(f"   🔄 Updated: {updated_at.strftime('%Y-%m-%d %H:%M:%S') if updated_at else 'N/A'}")
                
                # Categorize datasets
                if name == "main_dataset":
                    main_datasets.append((name, rows))
                    print(f"   🏷️  TYPE: MAIN DATASET")
                elif "live" in name.lower() or "nse" in name.lower():
                    live_datasets.append((name, rows))
                    print(f"   🏷️  TYPE: LIVE DATASET")
                else:
                    other_datasets.append((name, rows))
                    print(f"   🏷️  TYPE: OTHER/CUSTOM DATASET")
            
            # Summary analysis
            print("\n" + "=" * 50)
            print("📋 DATASET SUMMARY")
            print("=" * 50)
            
            print(f"\n🎯 MAIN DATASETS ({len(main_datasets)}):")
            for name, rows in main_datasets:
                print(f"   - {name}: {rows} rows")
            
            print(f"\n🔴 LIVE DATASETS ({len(live_datasets)}):")
            for name, rows in live_datasets:
                print(f"   - {name}: {rows} rows")
            
            print(f"\n📁 OTHER DATASETS ({len(other_datasets)}):")
            for name, rows in other_datasets:
                print(f"   - {name}: {rows} rows")
            
            # Conflict analysis
            print("\n" + "=" * 50)
            print("⚠️  CONFLICT ANALYSIS")
            print("=" * 50)
            
            # Check for duplicate names (should be impossible with PRIMARY KEY)
            cursor.execute("SELECT name, COUNT(*) FROM ohlc_datasets GROUP BY name HAVING COUNT(*) > 1")
            duplicates = cursor.fetchall()
            
            if duplicates:
                print("❌ DUPLICATE NAMES FOUND (THIS SHOULD NOT HAPPEN):")
                for name, count in duplicates:
                    print(f"   - '{name}': {count} entries")
            else:
                print("✅ NO DUPLICATE DATASET NAMES")
            
            # Check for similar names that might cause confusion
            all_names = [name for name, _, _, _, _, _ in datasets]
            similar_pairs = []
            
            for i, name1 in enumerate(all_names):
                for j, name2 in enumerate(all_names[i+1:], i+1):
                    # Check for similar names
                    if (name1.lower().replace('_', '').replace('-', '') == 
                        name2.lower().replace('_', '').replace('-', '')):
                        similar_pairs.append((name1, name2))
            
            if similar_pairs:
                print("⚠️  SIMILAR DATASET NAMES (potential confusion):")
                for name1, name2 in similar_pairs:
                    print(f"   - '{name1}' vs '{name2}'")
            else:
                print("✅ NO CONFUSINGLY SIMILAR NAMES")
            
            # Expected datasets for live trading
            expected_live_name = "live_NSE_INDEX_Nifty_50"
            has_expected_live = any(name == expected_live_name for name, _, _, _, _, _ in datasets)
            
            print(f"\n🎯 LIVE TRADING COMPATIBILITY:")
            print(f"   Expected live dataset name: '{expected_live_name}'")
            if has_expected_live:
                live_data = next((rows for name, rows, _, _, _, _ in datasets if name == expected_live_name), 0)
                print(f"   ✅ Found: {live_data} rows")
                if live_data >= 100:
                    print(f"   ✅ Sufficient data for continuation trading")
                else:
                    print(f"   ⚠️  May need more data for optimal continuation")
            else:
                print(f"   ❌ NOT FOUND - Live trading will start from 0 rows")
                
                # Check if main_dataset exists to suggest rename
                main_data = next((rows for name, rows, _, _, _, _ in datasets if name == "main_dataset"), 0)
                if main_data > 0:
                    print(f"   💡 SUGGESTION: Rename 'main_dataset' ({main_data} rows) to '{expected_live_name}'")
            
            print("\n" + "=" * 50)
            print("✅ DATABASE INSPECTION COMPLETE")
            print("=" * 50)
            
    except Exception as e:
        print(f"❌ Database inspection failed: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    inspect_database()
