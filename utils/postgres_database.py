

"""
PostgreSQL database implementation for TribexAlpha trading app
Uses SQLAlchemy ORM for database operations
"""
import os
import json
import pickle
import base64
import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

Base = declarative_base()

class OHLCDataset(Base):
    __tablename__ = 'ohlc_datasets'
    
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String(255), unique=True, nullable=False)
    data_json = Column(Text, nullable=False)
    metadata_json = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class ModelResult(Base):
    __tablename__ = 'model_results'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), unique=True, nullable=False)
    results_json = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class TrainedModel(Base):
    __tablename__ = 'trained_models'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), unique=True, nullable=False)
    model_data = Column(Text, nullable=False)
    task_type = Column(String(100))
    trained_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False)
    predictions_json = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class PostgresTradingDatabase:
    """PostgreSQL database using SQLAlchemy for storing trading data, models, and predictions."""

    def __init__(self):
        """Initialize PostgreSQL connection with SQLAlchemy."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Create SQLAlchemy engine
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        self._create_tables()

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            
            # Create indexes
            with self.engine.connect() as connection:
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_ohlc_dataset_name ON ohlc_datasets(dataset_name);"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_model_name ON model_results(model_name);"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_trained_model_name ON trained_models(model_name);"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);"))
                connection.commit()
            
            print("✅ PostgreSQL tables created successfully")
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
            raise e

    def test_connection(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1;"))
                return result.fetchone()[0] == 1
        except Exception as e:
            print(f"PostgreSQL connection test failed: {str(e)}")
            return False

    def save_ohlc_data(self, data: pd.DataFrame, dataset_name: str = "main_dataset", preserve_full_data: bool = False) -> bool:
        """Save OHLC dataframe to PostgreSQL database."""
        try:
            # Convert DataFrame to JSON for storage
            data_json = data.to_json(orient='records', date_format='iso')

            # Create metadata
            metadata = {
                'rows': len(data),
                'columns': list(data.columns),
                'start_date': str(data.index[0]) if not data.empty else None,
                'end_date': str(data.index[-1]) if not data.empty else None,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'preserve_full_data': preserve_full_data
            }
            metadata_json = json.dumps(metadata)

            with self.SessionLocal() as session:
                # Check if dataset exists
                existing = session.query(OHLCDataset).filter_by(dataset_name=dataset_name).first()
                
                if existing:
                    existing.data_json = data_json
                    existing.metadata_json = metadata_json
                    existing.updated_at = datetime.now()
                else:
                    new_dataset = OHLCDataset(
                        dataset_name=dataset_name,
                        data_json=data_json,
                        metadata_json=metadata_json
                    )
                    session.add(new_dataset)
                
                session.commit()

            print(f"✅ Successfully saved {len(data)} rows to PostgreSQL")
            return True

        except Exception as e:
            print(f"Error saving OHLC data: {str(e)}")
            return False

    def load_ohlc_data(self, dataset_name: str = "main_dataset") -> Optional[pd.DataFrame]:
        """Load OHLC dataframe from PostgreSQL database."""
        try:
            with self.SessionLocal() as session:
                dataset = session.query(OHLCDataset).filter_by(dataset_name=dataset_name).first()
                
                if dataset:
                    df = pd.read_json(dataset.data_json, orient='records')

                    # Set datetime index if present
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    elif 'Datetime' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Datetime'])
                        df.set_index('Datetime', inplace=True)

                    print(f"✅ Loaded {len(df)} rows from PostgreSQL")
                    return df

            return None

        except Exception as e:
            print(f"Error loading OHLC data: {str(e)}")
            return None

    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """Get list of saved datasets."""
        try:
            with self.SessionLocal() as session:
                datasets_query = session.query(OHLCDataset).order_by(OHLCDataset.updated_at.desc()).all()

            datasets = []
            for dataset in datasets_query:
                metadata = json.loads(dataset.metadata_json) if dataset.metadata_json else {}
                datasets.append({
                    'name': dataset.dataset_name,
                    'rows': metadata.get('rows', 0),
                    'columns': metadata.get('columns', []),
                    'start_date': metadata.get('start_date'),
                    'end_date': metadata.get('end_date'),
                    'created_at': dataset.created_at,
                    'updated_at': dataset.updated_at
                })

            return datasets

        except Exception as e:
            print(f"Error getting dataset list: {str(e)}")
            return []

    def get_dataset_metadata(self, dataset_name: str = "main_dataset") -> Optional[Dict[str, Any]]:
        """Get metadata for a dataset."""
        try:
            with self.SessionLocal() as session:
                dataset = session.query(OHLCDataset).filter_by(dataset_name=dataset_name).first()
                
                if dataset and dataset.metadata_json:
                    return json.loads(dataset.metadata_json)

            return None

        except Exception as e:
            print(f"Error getting dataset metadata: {str(e)}")
            return None

    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a dataset from database."""
        try:
            with self.SessionLocal() as session:
                dataset = session.query(OHLCDataset).filter_by(dataset_name=dataset_name).first()
                if dataset:
                    session.delete(dataset)
                    session.commit()
                    return True
                return False

        except Exception as e:
            print(f"Error deleting dataset: {str(e)}")
            return False

    def save_model_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        """Save model training results."""
        try:
            results_json = json.dumps(results, default=str)

            with self.SessionLocal() as session:
                existing = session.query(ModelResult).filter_by(model_name=model_name).first()
                
                if existing:
                    existing.results_json = results_json
                    existing.updated_at = datetime.now()
                else:
                    new_result = ModelResult(
                        model_name=model_name,
                        results_json=results_json
                    )
                    session.add(new_result)
                
                session.commit()

            return True

        except Exception as e:
            print(f"Error saving model results: {str(e)}")
            return False

    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model training results."""
        try:
            with self.SessionLocal() as session:
                result = session.query(ModelResult).filter_by(model_name=model_name).first()
                
                if result:
                    return json.loads(result.results_json)

            return None

        except Exception as e:
            print(f"Error loading model results: {str(e)}")
            return None

    def save_trained_models(self, models_dict: Dict[str, Any]) -> bool:
        """Save trained model objects for persistence."""
        try:
            success_count = 0

            for model_name, model_data in models_dict.items():
                try:
                    # Serialize model data using pickle and base64 encoding
                    model_pickle = pickle.dumps(model_data)
                    model_b64 = base64.b64encode(model_pickle).decode('utf-8')

                    task_type = model_data.get('task_type', 'unknown') if isinstance(model_data, dict) else 'unknown'

                    with self.SessionLocal() as session:
                        existing = session.query(TrainedModel).filter_by(model_name=model_name).first()
                        
                        if existing:
                            existing.model_data = model_b64
                            existing.task_type = task_type
                            existing.updated_at = datetime.now()
                        else:
                            new_model = TrainedModel(
                                model_name=model_name,
                                model_data=model_b64,
                                task_type=task_type
                            )
                            session.add(new_model)
                        
                        session.commit()

                    success_count += 1
                    print(f"Serialized model: {model_name}")

                except Exception as e:
                    print(f"Error saving model {model_name}: {str(e)}")
                    continue

            return success_count > 0

        except Exception as e:
            print(f"Error saving trained models: {str(e)}")
            return False

    def load_trained_models(self) -> Optional[Dict[str, Any]]:
        """Load trained model objects from database."""
        try:
            with self.SessionLocal() as session:
                models_query = session.query(TrainedModel).all()

            if not models_query:
                return None

            models_dict = {}
            for model in models_query:
                try:
                    # Deserialize model data
                    model_pickle = base64.b64decode(model.model_data.encode('utf-8'))
                    model_data = pickle.loads(model_pickle)
                    models_dict[model.model_name] = model_data
                except Exception as e:
                    print(f"Error deserializing model {model.model_name}: {str(e)}")
                    continue

            return models_dict if models_dict else None

        except Exception as e:
            print(f"Error loading trained models: {str(e)}")
            return None

    def save_predictions(self, predictions: pd.DataFrame, model_name: str) -> bool:
        """Save model predictions."""
        try:
            predictions_json = predictions.to_json(orient='records', date_format='iso')

            with self.SessionLocal() as session:
                new_prediction = Prediction(
                    model_name=model_name,
                    predictions_json=predictions_json
                )
                session.add(new_prediction)
                session.commit()

            return True

        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return False

    def load_predictions(self, model_name: str) -> Optional[pd.DataFrame]:
        """Load model predictions."""
        try:
            with self.SessionLocal() as session:
                prediction = session.query(Prediction).filter_by(model_name=model_name).order_by(Prediction.created_at.desc()).first()
                
                if prediction:
                    return pd.read_json(prediction.predictions_json, orient='records')

            return None

        except Exception as e:
            print(f"Error loading predictions: {str(e)}")
            return None

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about stored data."""
        try:
            with self.SessionLocal() as session:
                # Count datasets
                dataset_count = session.query(OHLCDataset).count()
                
                # Count model results
                model_count = session.query(ModelResult).count()
                
                # Count trained models
                trained_model_count = session.query(TrainedModel).count()
                
                # Count predictions
                prediction_count = session.query(Prediction).count()
                
                # Get dataset info
                datasets_query = session.query(OHLCDataset).all()

            datasets = []
            for dataset in datasets_query:
                metadata = json.loads(dataset.metadata_json) if dataset.metadata_json else {}
                datasets.append({
                    'name': dataset.dataset_name,
                    'rows': metadata.get('rows', 0),
                    'columns': metadata.get('columns', [])
                })

            return {
                'total_datasets': dataset_count,
                'total_models': model_count,
                'total_trained_models': trained_model_count,
                'total_predictions': prediction_count,
                'datasets': datasets,
                'database_type': 'PostgreSQL'
            }

        except Exception as e:
            print(f"Error getting database info: {str(e)}")
            return {'error': str(e), 'database_type': 'PostgreSQL'}

    def recover_data(self) -> Optional[pd.DataFrame]:
        """Try to recover any available OHLC data from database."""
        try:
            datasets = self.get_dataset_list()
            if not datasets:
                return None

            # Get the most recent dataset
            latest_dataset = datasets[0]  # Already sorted by updated_at DESC
            return self.load_ohlc_data(latest_dataset['name'])

        except Exception as e:
            print(f"Error recovering data: {str(e)}")
            return None

    def clear_all_data(self) -> bool:
        """Clear all data from database."""
        try:
            with self.SessionLocal() as session:
                session.query(Prediction).delete()
                session.query(TrainedModel).delete()
                session.query(ModelResult).delete()
                session.query(OHLCDataset).delete()
                session.commit()

            return True

        except Exception as e:
            print(f"Error clearing database: {str(e)}")
            return False
