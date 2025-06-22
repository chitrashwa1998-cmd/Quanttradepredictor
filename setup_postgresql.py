#!/usr/bin/env python3
"""
PostgreSQL setup verification for TribexAlpha
"""

import os
import sys
from utils.database_adapter import DatabaseAdapter

def check_postgresql_setup():
    """Check if PostgreSQL is properly set up"""

    print("🔍 Checking PostgreSQL setup...")

    # Check DATABASE_URL
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL environment variable not found!")
        print("\n📝 To set up PostgreSQL in Replit:")
        print("1. Open a new tab and type 'Database'")
        print("2. Click 'Create a database'")
        print("3. Choose PostgreSQL")
        print("4. The DATABASE_URL will be automatically set")
        return False

    print("✅ DATABASE_URL found")

    try:
        # Test connection
        print("🔄 Testing database connection...")
        db = DatabaseAdapter()

        print("✅ PostgreSQL connection successful!")

        # Get database info
        info = db.get_database_info()
        print(f"📊 Database info:")
        print(f"   - Type: {info.get('database_type', 'Unknown')}")
        print(f"   - Datasets: {info.get('total_datasets', 0)}")
        print(f"   - Models: {info.get('total_models', 0)}")
        print(f"   - Trained Models: {info.get('total_trained_models', 0)}")

        return True

    except Exception as e:
        print(f"❌ PostgreSQL setup failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🚀 TribexAlpha PostgreSQL Setup")
    print("=" * 40)

    if check_postgresql_setup():
        print("\n✅ PostgreSQL is ready to use!")
        print("You can now run your Streamlit app.")
    else:
        print("\n❌ PostgreSQL setup incomplete.")
        print("Please follow the instructions above to set up PostgreSQL.")
        sys.exit(1)

if __name__ == "__main__":
    main()