"""
Data management API endpoints
Handles dataset operations and database interactions
"""

from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import logging
import io

from core.database import get_database_dependency

logger = logging.getLogger(__name__)
router = APIRouter()

class DatasetInfo(BaseModel):
    name: str
    rows: int
    columns: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets(
    db = Depends(get_database_dependency)
):
    """List all available datasets"""
    try:
        datasets = db.list_datasets()
        
        result = []
        for dataset in datasets:
            result.append(DatasetInfo(
                name=dataset['name'],
                rows=dataset.get('rows', 0),
                columns=dataset.get('columns', []),
                start_date=dataset.get('start_date'),
                end_date=dataset.get('end_date'),
                created_at=dataset.get('created_at'),
                updated_at=dataset.get('updated_at')
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = None,
    db = Depends(get_database_dependency)
):
    """Upload CSV data file"""
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Basic validation
        if df.empty:
            raise ValueError("Uploaded file is empty")
        
        # Use filename as dataset name if not provided
        if not dataset_name:
            dataset_name = (file.filename or "uploaded_data").replace('.csv', '')
        
        # Store in database
        success = db.store_dataset(dataset_name, df)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store dataset")
        
        return {
            "success": True,
            "dataset_name": dataset_name,
            "rows": len(df),
            "columns": list(df.columns),
            "message": f"Successfully uploaded {len(df)} rows"
        }
        
    except Exception as e:
        logger.error(f"Data upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/clear-all")
async def clear_all_data(
    db = Depends(get_database_dependency)
):
    """Clear all data from database"""
    try:
        # This would need implementation in the database adapter
        return {
            "success": True,
            "message": "Clear all data functionality needs implementation"
        }
    except Exception as e:
        logger.error(f"Failed to clear all data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clean-mode")
async def activate_clean_mode(
    db = Depends(get_database_dependency)
):
    """Activate clean data mode"""
    try:
        # This would need implementation in the database adapter
        return {
            "success": True,
            "message": "Clean data mode functionality needs implementation"
        }
    except Exception as e:
        logger.error(f"Failed to activate clean mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_name}/export")
async def export_dataset(
    dataset_name: str,
    db = Depends(get_database_dependency)
):
    """Export dataset as CSV"""
    try:
        dataset = db.get_dataset(dataset_name)
        
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
        
        # Convert to CSV
        csv_content = dataset.to_csv(index=False)
        
        return {
            "success": True,
            "data": csv_content,
            "filename": f"{dataset_name}.csv"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_name}")
async def get_dataset(
    dataset_name: str,
    limit: Optional[int] = Query(None, description="Limit number of rows returned"),
    offset: Optional[int] = Query(0, description="Offset for pagination"),
    db = Depends(get_database_dependency)
):
    """Get specific dataset with optional pagination"""
    try:
        dataset = db.get_dataset(dataset_name)
        
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
        
        # Apply pagination if requested
        if limit is not None:
            end_index = offset + limit
            dataset = dataset.iloc[offset:end_index]
        
        return {
            "success": True,
            "dataset_name": dataset_name,
            "data": dataset.to_dict('records'),
            "total_rows": len(dataset),
            "columns": list(dataset.columns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/datasets/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    db = Depends(get_database_dependency)
):
    """Delete a specific dataset"""
    try:
        success = db.delete_dataset(dataset_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
        
        return {
            "success": True,
            "message": f"Dataset {dataset_name} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database/info")
async def get_database_info(
    db = Depends(get_database_dependency)
):
    """Get database information and statistics"""
    try:
        info = db.get_database_info()
        
        return {
            "success": True,
            "info": info
        }
        
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/live-data/latest")
async def get_latest_live_data(
    limit: Optional[int] = Query(10, description="Number of latest records to return"),
    db = Depends(get_database_dependency)
):
    """Get latest live market data"""
    try:
        # Try to get live data from livenifty50 dataset
        try:
            live_data = db.get_dataset('livenifty50')
            if live_data is not None and not live_data.empty:
                # Get latest records
                latest_data = live_data.tail(limit)
                
                return {
                    "success": True,
                    "data": latest_data.to_dict('records'),
                    "count": len(latest_data),
                    "last_update": latest_data['Date'].max() if 'Date' in latest_data.columns else None
                }
        except Exception:
            pass
        
        # If no live data available, return empty response
        return {
            "success": True,
            "data": [],
            "count": 0,
            "last_update": None,
            "message": "No live data available"
        }
        
    except Exception as e:
        logger.error(f"Failed to get latest live data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_name}/stats")
async def get_dataset_statistics(
    dataset_name: str,
    db = Depends(get_database_dependency)
):
    """Get statistical information for a dataset"""
    try:
        dataset = db.get_dataset(dataset_name)
        
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
        
        # Calculate basic statistics
        stats = {}
        
        # Numeric columns statistics
        numeric_columns = dataset.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            stats['numeric'] = dataset[numeric_columns].describe().to_dict()
        
        # Basic info
        stats['info'] = {
            'total_rows': len(dataset),
            'total_columns': len(dataset.columns),
            'columns': list(dataset.columns),
            'dtypes': dataset.dtypes.astype(str).to_dict()
        }
        
        # Date range if Date column exists
        if 'Date' in dataset.columns:
            stats['date_range'] = {
                'start': dataset['Date'].min(),
                'end': dataset['Date'].max(),
                'total_days': (pd.to_datetime(dataset['Date'].max()) - pd.to_datetime(dataset['Date'].min())).days
            }
        
        return {
            "success": True,
            "dataset_name": dataset_name,
            "statistics": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))