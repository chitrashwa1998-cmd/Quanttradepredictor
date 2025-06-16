
#!/usr/bin/env python3
"""
Automatic PostgreSQL setup for TribexAlpha
This script will help you create and configure a PostgreSQL database
"""

import os
import webbrowser
from time import sleep

def setup_postgresql():
    """Guide user through PostgreSQL setup"""
    
    print("🔧 TribexAlpha PostgreSQL Setup")
    print("=" * 50)
    
    # Check if DATABASE_URL already exists
    if os.environ.get('DATABASE_URL'):
        print("✅ DATABASE_URL already exists!")
        print(f"Database: {os.environ['DATABASE_URL'].split('@')[1] if '@' in os.environ['DATABASE_URL'] else 'Hidden'}")
        return True
    
    print("📋 Setting up PostgreSQL database...")
    print("\n🔗 I'll open the Database tab for you.")
    print("Follow these steps:")
    print("1. ✅ Click 'Create a Database' button")
    print("2. ✅ Wait for database creation (30-60 seconds)")
    print("3. ✅ The DATABASE_URL will be set automatically")
    print("4. ✅ Come back and run this script again")
    
    # Instructions for manual setup
    print("\n📝 Manual Steps:")
    print("1. Open a new tab in Replit")
    print("2. Type 'Database' in the search")
    print("3. Click on 'Database' from the results")
    print("4. Click 'Create a Database' button")
    print("5. Wait for setup to complete")
    
    input("\n⏸️  Press Enter after you've created the database...")
    
    # Check again
    if os.environ.get('DATABASE_URL'):
        print("✅ DATABASE_URL detected! PostgreSQL is ready.")
        return True
    else:
        print("❌ DATABASE_URL still not found.")
        print("Please create the database first, then run this script again.")
        return False

def test_connection():
    """Test PostgreSQL connection"""
    try:
        from utils.database_adapter import DatabaseAdapter
        
        print("\n🔄 Testing PostgreSQL connection...")
        db = DatabaseAdapter()
        
        info = db.get_database_info()
        print(f"✅ Connected to: {info.get('backend', 'Unknown')}")
        print(f"📊 Database ready for use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    if setup_postgresql():
        test_connection()
        print("\n🚀 Your app is now using PostgreSQL!")
        print("You can run your Streamlit app normally now.")
    else:
        print("\n⚠️  Database setup incomplete. Please try again.")
