#!/usr/bin/env python3
'''
Database Restore Script for TribexAlpha
Generated on 2025-07-27 09:26:03
'''

import os
import json
import pickle
import pandas as pd
from utils.database_adapter import DatabaseAdapter

def restore_database():
    '''Restore database from exported files'''
    print("🔄 Starting database restoration...")
    
    try:
        # Initialize database
        db = DatabaseAdapter()
        
        # Restore datasets
        print("📊 Restoring datasets...")

        print("  📥 Restoring training_dataset...")
        data = pd.read_csv("datasets/training_dataset.csv", index_col=0, parse_dates=True)
        if db.save_ohlc_data(data, "training_dataset"):
            print("    ✅ Restored training_dataset")
        else:
            print("    ❌ Failed to restore training_dataset")

        print("  📥 Restoring livenifty50...")
        data = pd.read_csv("datasets/livenifty50.csv", index_col=0, parse_dates=True)
        if db.save_ohlc_data(data, "livenifty50"):
            print("    ✅ Restored livenifty50")
        else:
            print("    ❌ Failed to restore livenifty50")

        # Restore trained models
        print("🤖 Restoring trained models...")
        try:
            with open("models/trained_models.pkl", "rb") as f:
                trained_models = pickle.load(f)
            
            if db.save_trained_models(trained_models):
                print(f"    ✅ Restored {len(trained_models)} models")
            else:
                print("    ❌ Failed to restore trained models")
        except Exception as e:
            print(f"    ❌ Error restoring models: {str(e)}")

        print("✅ Database restoration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Database restoration failed: {str(e)}")
        return False

if __name__ == "__main__":
    restore_database()
