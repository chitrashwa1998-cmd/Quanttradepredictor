
#!/usr/bin/env python3

import os
from utils.database_adapter import DatabaseAdapter

def test_postgresql_connection():
    """Test PostgreSQL connection and setup"""
    
    # Check if DATABASE_URL exists
    if not os.environ.get('DATABASE_URL'):
        print("❌ DATABASE_URL environment variable not found!")
        print("Please create a PostgreSQL database in Replit first.")
        return False
    
    print("✅ DATABASE_URL found")
    print(f"Database URL: {os.environ['DATABASE_URL'].split('@')[1] if '@' in os.environ['DATABASE_URL'] else 'Hidden'}")
    
    try:
        # Initialize database adapter
        print("\n🔄 Initializing DatabaseAdapter...")
        db = DatabaseAdapter()
        
        # Test database info
        print("\n📊 Getting database info...")
        info = db.get_database_info()
        print(f"Backend: {info.get('backend', 'Unknown')}")
        print(f"Total datasets: {info.get('total_datasets', 0)}")
        print(f"Total records: {info.get('total_records', 0)}")
        
        print("\n✅ PostgreSQL connection successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ PostgreSQL connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_postgresql_connection()
