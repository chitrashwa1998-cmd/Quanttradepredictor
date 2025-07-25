# Applying the provided changes to the original code, focusing on adding the missing upload endpoint, fixing API routes, adding missing imports, and adding all missing API endpoints.
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
from typing import Dict, Any, List, Optional
import json
from utils.row_based_database import RowBasedPostgresDatabase
from utils.data_processing import DataProcessor

app = FastAPI(title="TribexAlpha Simple Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize row-based database
try:
    db = RowBasedPostgresDatabase()
    print("✅ Connected to Row-Based PostgreSQL database")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    db = None

@app.get("/")
async def root():
    return {"message": "TribexAlpha Simple Backend with Row-Based PostgreSQL"}

@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...), 
                     dataset_name: str = "main_dataset",
                     dataset_purpose: str = "training",
                     data_only_mode: bool = False):
    """Upload data file and save to row-based database"""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")

        # Read file content
        content = await file.read()

        # Try to read as CSV
        try:
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        except Exception as e:
            print(f"CSV read error: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {str(e)}")

        # Process timestamp column
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Find timestamp column
        timestamp_columns = ['timestamp', 'Timestamp', 'date', 'Date', 'datetime', 'DateTime']
        timestamp_col = None

        for col in timestamp_columns:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col:
            # Convert timestamp column to datetime and set as index
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
        else:
            # Try to use first column as timestamp if it looks like dates
            if len(df.columns) > 0:
                first_col = df.columns[0]
                try:
                    df[first_col] = pd.to_datetime(df[first_col])
                    df = df.set_index(first_col)
                except:
                    # If no timestamp column found, create a simple index
                    pass

        # Use DataProcessor to clean the data
        try:
            processor = DataProcessor()
            df = processor.clean_data(df)
        except Exception as e:
            print(f"Data processing warning: {e}")
            # Continue without processing if it fails

        # Save to row-based database
        success = db.save_ohlc_data(
            df, 
            dataset_name=dataset_name, 
            data_only_mode=data_only_mode,
            dataset_purpose=dataset_purpose
        )

        if success:
            return {
                "message": f"Successfully uploaded {len(df)} rows to dataset '{dataset_name}'",
                "dataset_name": dataset_name,
                "rows": len(df),
                "columns": list(df.columns),
                "purpose": dataset_purpose,
                "start_date": df.index.min().strftime('%Y-%m-%d') if len(df) > 0 else None,
                "end_date": df.index.max().strftime('%Y-%m-%d') if len(df) > 0 else None
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save data to database")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/data/database/info")
async def get_database_info():
    """Get database information"""
    try:
        if not db:
            return {"error": "Database not connected"}

        info = db.get_database_info()
        return info
    except Exception as e:
        print(f"Error getting database info: {e}")
        return {
            'database_type': 'postgresql_row_based',
            'total_datasets': 0,
            'total_records': 0,
            'total_models': 0,
            'total_trained_models': 0,
            'total_predictions': 0,
            'datasets': [],
            'backend': 'PostgreSQL (Row-Based)',
            'storage_type': 'Row-Based',
            'supports_append': True,
            'supports_range_queries': True,
            'error': str(e)
        }

@app.get("/api/data/datasets")
async def list_datasets():
    """List all datasets"""
    try:
        if not db:
            return {"datasets": [], "error": "Database not connected"}

        datasets = db.get_dataset_list()
        return {"datasets": datasets}
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return {"datasets": [], "error": str(e)}

@app.get("/api/data/datasets/{dataset_name}")
async def get_dataset(dataset_name: str, limit: Optional[int] = None, 
                     start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get specific dataset with optional filters"""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")

        data = db.load_ohlc_data(dataset_name, limit=limit, start_date=start_date, end_date=end_date)

        if data is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

        # Convert to JSON-serializable format
        result = {
            "dataset_name": dataset_name,
            "total_rows": len(data),
            "columns": list(data.columns),
            "data": data.reset_index().to_dict('records')
        }

        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/datasets/{dataset_name}/stats")
async def get_dataset_stats(dataset_name: str):
    """Get dataset statistics"""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")

        data = db.load_ohlc_data(dataset_name)

        if data is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

        # Calculate basic statistics
        stats = {
            "total_rows": len(data),
            "columns": list(data.columns),
            "start_date": data.index.min().strftime('%Y-%m-%d') if len(data) > 0 else None,
            "end_date": data.index.max().strftime('%Y-%m-%d') if len(data) > 0 else None,
            "summary": data.describe().to_dict()
        }

        return stats
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting dataset stats for {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/data/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a specific dataset"""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")

        success = db.delete_dataset(dataset_name)

        if success:
            return {"message": f"Dataset '{dataset_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete dataset")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/data/datasets")
async def clear_all_data():
    """Clear all data from database"""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")

        success = db.clear_all_data()

        if success:
            return {"message": "All data cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear database")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/clean-mode")
async def clean_data_mode():
    """Enable clean data mode (data-only mode)"""
    return {"message": "Clean data mode enabled", "mode": "data_only"}

@app.get("/api/data/datasets/{dataset_name}/export")
async def export_dataset(dataset_name: str):
    """Export dataset as CSV"""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")

        data = db.load_ohlc_data(dataset_name)

        if data is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

        # Convert to CSV
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer)
        csv_content = csv_buffer.getvalue()

        return JSONResponse(
            content={"csv_data": csv_content},
            headers={"Content-Disposition": f"attachment; filename={dataset_name}.csv"}
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error exporting dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/models/status")
async def get_models_status():
    """Get status of trained models"""
    try:
        if not db:
            return {"models": [], "error": "Database not connected"}

        # Try to load trained models from database
        models = db.load_trained_models()

        if models:
            model_status = []
            for model_name in models.keys():
                model_status.append({
                    "name": model_name,
                    "status": "trained",
                    "last_updated": "Available in database"
                })
            return {"models": model_status}
        else:
            return {"models": []}

    except Exception as e:
        print(f"Error getting model status: {e}")
        return {"models": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
`