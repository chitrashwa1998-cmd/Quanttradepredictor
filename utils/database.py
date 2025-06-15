
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from replit import db
import json
from datetime import datetime

class TradingDatabase:
    """Database utility for storing and retrieving trading data using Replit Key-Value Store."""
    
    def __init__(self):
        self.db = db
        
    def save_ohlc_data(self, data: pd.DataFrame, dataset_name: str = "main_dataset") -> bool:
        """Save OHLC dataframe to database."""
        try:
            # Convert DataFrame to dictionary format for storage
            data_dict = {
                'data': data.to_dict('records'),
                'index': data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'columns': data.columns.tolist(),
                'metadata': {
                    'rows': len(data),
                    'start_date': data.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': data.index.max().strftime('%Y-%m-%d %H:%M:%S'),
                    'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Store in database
            self.db[f"ohlc_data_{dataset_name}"] = data_dict
            
            # Update dataset list
            existing_datasets = self.get_dataset_list()
            if dataset_name not in existing_datasets:
                existing_datasets.append(dataset_name)
                self.db["dataset_list"] = existing_datasets
            
            return True
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False
    
    def load_ohlc_data(self, dataset_name: str = "main_dataset") -> Optional[pd.DataFrame]:
        """Load OHLC dataframe from database."""
        try:
            key = f"ohlc_data_{dataset_name}"
            if key not in self.db:
                return None
            
            data_dict = self.db[key]
            
            # Reconstruct DataFrame
            df = pd.DataFrame(data_dict['data'])
            df.index = pd.to_datetime(data_dict['index'])
            df.index.name = 'Date'
            
            # Ensure proper column order
            if 'columns' in data_dict:
                df = df[data_dict['columns']]
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def get_dataset_list(self) -> list:
        """Get list of saved datasets."""
        try:
            return self.db.get("dataset_list", [])
        except:
            return []
    
    def get_dataset_metadata(self, dataset_name: str = "main_dataset") -> Optional[Dict[str, Any]]:
        """Get metadata for a dataset."""
        try:
            key = f"ohlc_data_{dataset_name}"
            if key not in self.db:
                return None
            
            data_dict = self.db[key]
            return data_dict.get('metadata', {})
            
        except Exception as e:
            print(f"Error getting metadata: {str(e)}")
            return None
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a dataset from database."""
        try:
            key = f"ohlc_data_{dataset_name}"
            if key in self.db:
                del self.db[key]
            
            # Update dataset list
            existing_datasets = self.get_dataset_list()
            if dataset_name in existing_datasets:
                existing_datasets.remove(dataset_name)
                self.db["dataset_list"] = existing_datasets
            
            return True
            
        except Exception as e:
            print(f"Error deleting dataset: {str(e)}")
            return False
    
    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        """Save model training results."""
        try:
            key = f"model_results_{model_name}"
            self.db[key] = {
                'results': results,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return True
        except Exception as e:
            print(f"Error saving model results: {str(e)}")
            return False
    
    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model training results."""
        try:
            key = f"model_results_{model_name}"
            if key not in self.db:
                return None
            return self.db[key]['results']
        except Exception as e:
            print(f"Error loading model results: {str(e)}")
            return None
    
    def save_predictions(self, predictions: pd.DataFrame, model_name: str) -> bool:
        """Save model predictions."""
        try:
            predictions_dict = {
                'data': predictions.to_dict('records'),
                'index': predictions.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'columns': predictions.columns.tolist(),
                'model_name': model_name,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            key = f"predictions_{model_name}"
            self.db[key] = predictions_dict
            return True
            
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return False
    
    def load_predictions(self, model_name: str) -> Optional[pd.DataFrame]:
        """Load model predictions."""
        try:
            key = f"predictions_{model_name}"
            if key not in self.db:
                return None
            
            pred_dict = self.db[key]
            df = pd.DataFrame(pred_dict['data'])
            df.index = pd.to_datetime(pred_dict['index'])
            
            if 'columns' in pred_dict:
                df = df[pred_dict['columns']]
            
            return df
            
        except Exception as e:
            print(f"Error loading predictions: {str(e)}")
            return None
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about stored data."""
        try:
            datasets = self.get_dataset_list()
            
            info = {
                'total_datasets': len(datasets),
                'datasets': [],
                'total_keys': len(list(self.db.keys())),
                'available_keys': list(self.db.keys())
            }
            
            for dataset in datasets:
                metadata = self.get_dataset_metadata(dataset)
                if metadata:
                    info['datasets'].append({
                        'name': dataset,
                        'rows': metadata.get('rows', 0),
                        'date_range': f"{metadata.get('start_date', 'N/A')} to {metadata.get('end_date', 'N/A')}",
                        'saved_at': metadata.get('saved_at', 'N/A')
                    })
            
            return info
            
        except Exception as e:
            print(f"Error getting database info: {str(e)}")
            return {'error': str(e)}
    
    def recover_data(self) -> Optional[pd.DataFrame]:
        """Try to recover any available OHLC data from database."""
        try:
            # Look for any OHLC data keys
            ohlc_keys = [key for key in self.db.keys() if key.startswith('ohlc_data_')]
            
            if not ohlc_keys:
                return None
            
            # Try to load the most recent dataset
            latest_key = None
            latest_time = None
            
            for key in ohlc_keys:
                try:
                    data_dict = self.db[key]
                    if 'metadata' in data_dict and 'saved_at' in data_dict['metadata']:
                        saved_time = datetime.strptime(data_dict['metadata']['saved_at'], '%Y-%m-%d %H:%M:%S')
                        if latest_time is None or saved_time > latest_time:
                            latest_time = saved_time
                            latest_key = key
                except:
                    continue
            
            if latest_key:
                dataset_name = latest_key.replace('ohlc_data_', '')
                return self.load_ohlc_data(dataset_name)
            
            return None
            
        except Exception as e:
            print(f"Error recovering data: {str(e)}")
            return None

    def clear_all_data(self) -> bool:
        """Clear all data from database."""
        try:
            # Get all keys and delete them
            keys_to_delete = list(self.db.keys())
            for key in keys_to_delete:
                del self.db[key]
            return True
        except Exception as e:
            print(f"Error clearing database: {str(e)}")
            return False
