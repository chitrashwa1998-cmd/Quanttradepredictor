
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
        
    def save_ohlc_data(self, data: pd.DataFrame, dataset_name: str = "main_dataset", preserve_full_data: bool = False) -> bool:
        """Save OHLC dataframe to database with chunking for very large datasets."""
        try:
            original_rows = len(data)
            
            if preserve_full_data and len(data) > 100000:
                # For datasets over 100k rows, use chunked storage
                print(f"Large dataset detected ({len(data)} rows), using chunked storage...")
                return self._save_large_dataset_chunked(data, dataset_name)
            elif not preserve_full_data:
                # Standard sampling for manageable sizes
                max_rows = 50000
                if len(data) > max_rows:
                    recent_data = data.tail(30000)
                    if len(data) > 30000:
                        older_data = data.head(len(data) - 30000)
                        step = max(1, len(older_data) // 20000)
                        sampled_older = older_data.iloc[::step]
                        data_sampled = pd.concat([sampled_older, recent_data])
                    else:
                        data_sampled = recent_data
                    print(f"Data optimized ({len(data)} rows), preserved {len(data_sampled)} rows with recent data priority")
                    data = data_sampled
            
            # Convert to simpler format to reduce size
            data_records = []
            for idx, row in data.iterrows():
                record = {
                    'date': idx.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close'])
                }
                if 'Volume' in row and pd.notna(row['Volume']):
                    record['volume'] = float(row['Volume'])
                data_records.append(record)
            
            # Create metadata
            metadata = {
                'rows': len(data),
                'original_rows': len(data),
                'start_date': data.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': data.index.max().strftime('%Y-%m-%d %H:%M:%S'),
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'columns': data.columns.tolist()
            }
            
            # Try to save in chunks if still too large
            chunk_size = 1000
            if len(data_records) > chunk_size:
                # Save metadata first
                self.db[f"ohlc_metadata_{dataset_name}"] = metadata
                
                # Save data in chunks
                for i in range(0, len(data_records), chunk_size):
                    chunk = data_records[i:i+chunk_size]
                    chunk_key = f"ohlc_chunk_{dataset_name}_{i//chunk_size}"
                    self.db[chunk_key] = chunk
                
                # Save chunk info
                chunk_info = {
                    'total_chunks': (len(data_records) + chunk_size - 1) // chunk_size,
                    'chunk_size': chunk_size,
                    'total_records': len(data_records)
                }
                self.db[f"ohlc_info_{dataset_name}"] = chunk_info
                
            else:
                # Save as single object if small enough
                data_dict = {
                    'data': data_records,
                    'metadata': metadata
                }
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
    
    def _save_large_dataset_chunked(self, data: pd.DataFrame, dataset_name: str) -> bool:
        """Save large datasets using chunked storage strategy."""
        try:
            print(f"Saving {len(data)} rows using chunked storage...")
            
            # Clear any existing chunks for this dataset
            self._clear_dataset_chunks(dataset_name)
            
            # Convert to efficient format
            data_records = []
            for idx, row in data.iterrows():
                record = {
                    'date': idx.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close'])
                }
                if 'Volume' in row and pd.notna(row['Volume']):
                    record['volume'] = float(row['Volume'])
                data_records.append(record)
            
            # Save in chunks of 10,000 records each
            chunk_size = 10000
            total_chunks = (len(data_records) + chunk_size - 1) // chunk_size
            
            print(f"Splitting into {total_chunks} chunks of {chunk_size} records each...")
            
            for i in range(0, len(data_records), chunk_size):
                chunk_num = i // chunk_size
                chunk = data_records[i:i+chunk_size]
                chunk_key = f"ohlc_chunk_{dataset_name}_{chunk_num}"
                
                # Save each chunk
                self.db[chunk_key] = {
                    'chunk_data': chunk,
                    'chunk_number': chunk_num,
                    'chunk_size': len(chunk)
                }
                print(f"Saved chunk {chunk_num + 1}/{total_chunks}")
            
            # Save metadata and chunk information
            metadata = {
                'total_rows': len(data),
                'total_chunks': total_chunks,
                'chunk_size': chunk_size,
                'start_date': data.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': data.index.max().strftime('%Y-%m-%d %H:%M:%S'),
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'columns': data.columns.tolist(),
                'is_chunked': True
            }
            
            self.db[f"ohlc_metadata_{dataset_name}"] = metadata
            
            # Update dataset list
            existing_datasets = self.get_dataset_list()
            if dataset_name not in existing_datasets:
                existing_datasets.append(dataset_name)
                self.db["dataset_list"] = existing_datasets
            
            print(f"Successfully saved {len(data)} rows in {total_chunks} chunks")
            return True
            
        except Exception as e:
            print(f"Error in chunked storage: {str(e)}")
            return False
    
    def _clear_dataset_chunks(self, dataset_name: str):
        """Clear existing chunks for a dataset."""
        try:
            # Find and delete existing chunks
            keys_to_delete = []
            for key in self.db.keys():
                if key.startswith(f"ohlc_chunk_{dataset_name}_") or key == f"ohlc_metadata_{dataset_name}":
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.db[key]
                
        except Exception as e:
            print(f"Error clearing chunks: {str(e)}")
    
    def _load_chunked_dataset(self, dataset_name: str, metadata: dict) -> Optional[pd.DataFrame]:
        """Load a chunked dataset and reconstruct the complete DataFrame efficiently."""
        try:
            total_chunks = metadata['total_chunks']
            print(f"Loading chunked dataset: {total_chunks} chunks, {metadata['total_rows']} total rows")
            
            # Pre-allocate lists for better performance
            opens, highs, lows, closes, volumes, dates = [], [], [], [], [], []
            
            # Load chunks in batches for better memory efficiency
            for chunk_num in range(total_chunks):
                chunk_key = f"ohlc_chunk_{dataset_name}_{chunk_num}"
                if chunk_key in self.db:
                    chunk_data = self.db[chunk_key]
                    chunk_records = chunk_data['chunk_data']
                    
                    # Extract data directly into lists
                    for record in chunk_records:
                        opens.append(record['open'])
                        highs.append(record['high'])
                        lows.append(record['low'])
                        closes.append(record['close'])
                        volumes.append(record.get('volume', 0))
                        dates.append(record['date'])
                    
                    print(f"Loaded chunk {chunk_num + 1}/{total_chunks}")
                else:
                    print(f"Warning: Missing chunk {chunk_num}")
            
            if not dates:
                print("No chunk data found")
                return None
            
            # Create DataFrame directly from lists (much faster)
            df = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': volumes
            })
            
            df.index = pd.to_datetime(dates)
            df.index.name = 'Date'
            
            print(f"Successfully reconstructed dataset: {len(df)} rows")
            return df
            
        except Exception as e:
            print(f"Error loading chunked dataset: {str(e)}")
            return None
    
    def load_ohlc_data(self, dataset_name: str = "main_dataset") -> Optional[pd.DataFrame]:
        """Load OHLC dataframe from database, handling both chunked and regular data."""
        try:
            # Check if this is a chunked dataset
            metadata_key = f"ohlc_metadata_{dataset_name}"
            if metadata_key in self.db:
                metadata = self.db[metadata_key]
                if metadata.get('is_chunked', False):
                    return self._load_chunked_dataset(dataset_name, metadata)
            
            # Try loading regular format
            key = f"ohlc_data_{dataset_name}"
            if key in self.db:
                data_dict = self.db[key]
                
                # New format with simplified records
                if 'data' in data_dict and isinstance(data_dict['data'], list):
                    records = data_dict['data']
                    df_data = []
                    for record in records:
                        row_data = {
                            'Open': record['open'],
                            'High': record['high'],
                            'Low': record['low'],
                            'Close': record['close']
                        }
                        if 'volume' in record:
                            row_data['Volume'] = record['volume']
                        df_data.append(row_data)
                    
                    df = pd.DataFrame(df_data)
                    dates = [record['date'] for record in records]
                    df.index = pd.to_datetime(dates)
                    df.index.name = 'Date'
                    return df
                
                # Old format compatibility
                elif 'data' in data_dict and 'index' in data_dict:
                    df = pd.DataFrame(data_dict['data'])
                    df.index = pd.to_datetime(data_dict['index'])
                    df.index.name = 'Date'
                    
                    if 'columns' in data_dict:
                        df = df[data_dict['columns']]
                    return df
            
            # Try loading chunked format
            info_key = f"ohlc_info_{dataset_name}"
            if info_key in self.db:
                chunk_info = self.db[info_key]
                metadata = self.db.get(f"ohlc_metadata_{dataset_name}", {})
                
                all_records = []
                for chunk_idx in range(chunk_info['total_chunks']):
                    chunk_key = f"ohlc_chunk_{dataset_name}_{chunk_idx}"
                    if chunk_key in self.db:
                        chunk_data = self.db[chunk_key]
                        all_records.extend(chunk_data)
                
                if all_records:
                    df_data = []
                    for record in all_records:
                        row_data = {
                            'Open': record['open'],
                            'High': record['high'],
                            'Low': record['low'],
                            'Close': record['close']
                        }
                        if 'volume' in record:
                            row_data['Volume'] = record['volume']
                        df_data.append(row_data)
                    
                    df = pd.DataFrame(df_data)
                    dates = [record['date'] for record in all_records]
                    df.index = pd.to_datetime(dates)
                    df.index.name = 'Date'
                    return df
            
            # Try loading fallback format
            fallback_key = f"ohlc_fallback_{dataset_name}"
            if fallback_key in self.db:
                fallback_dict = self.db[fallback_key]
                records = fallback_dict['data']
                
                df_data = []
                for record in records:
                    df_data.append({'Close': record['close']})
                
                df = pd.DataFrame(df_data)
                dates = [record['date'] for record in records]
                df.index = pd.to_datetime(dates)
                df.index.name = 'Date'
                return df
            
            return None
            
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
    
    def save_trained_models(self, models_dict: Dict[str, Any]) -> bool:
        """Save trained model objects for persistence."""
        try:
            import pickle
            import base64
            
            serialized_models = {}
            for model_name, model_data in models_dict.items():
                try:
                    # Serialize the model ensemble
                    model_bytes = pickle.dumps(model_data['ensemble'])
                    model_b64 = base64.b64encode(model_bytes).decode('utf-8')
                    
                    serialized_models[model_name] = {
                        'ensemble': model_b64,
                        'feature_names': model_data.get('feature_names', []),
                        'task_type': model_data.get('task_type', 'classification'),
                        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'model_type': str(type(model_data['ensemble']).__name__)
                    }
                    print(f"Serialized model: {model_name}")
                except Exception as e:
                    print(f"Error serializing model {model_name}: {str(e)}")
                    continue
            
            if serialized_models:
                self.db['trained_models'] = serialized_models
                print(f"Saved {len(serialized_models)} trained models to database")
                return True
            else:
                print("No models were successfully serialized")
                return False
                
        except Exception as e:
            print(f"Error saving trained models: {str(e)}")
            return False
    
    def load_trained_models(self) -> Optional[Dict[str, Any]]:
        """Load trained model objects from database."""
        try:
            import pickle
            import base64
            
            if 'trained_models' not in self.db:
                return None
            
            serialized_models = self.db['trained_models']
            loaded_models = {}
            
            for model_name, model_data in serialized_models.items():
                try:
                    # Deserialize the model ensemble
                    model_b64 = model_data['ensemble']
                    model_bytes = base64.b64decode(model_b64.encode('utf-8'))
                    ensemble = pickle.loads(model_bytes)
                    
                    loaded_models[model_name] = {
                        'ensemble': ensemble,
                        'feature_names': model_data.get('feature_names', []),
                        'task_type': model_data.get('task_type', 'classification'),
                        'trained_at': model_data.get('trained_at', ''),
                        'model_type': model_data.get('model_type', 'Unknown')
                    }
                    print(f"Loaded model: {model_name}")
                    
                except Exception as e:
                    print(f"Error loading model {model_name}: {str(e)}")
                    continue
            
            if loaded_models:
                print(f"Successfully loaded {len(loaded_models)} trained models")
                return loaded_models
            else:
                print("No models could be loaded")
                return None
                
        except Exception as e:
            print(f"Error loading trained models: {str(e)}")
            return None
    
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
