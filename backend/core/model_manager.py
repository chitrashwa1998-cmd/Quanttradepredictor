"""
Model management for FastAPI backend
Handles loading, training, and prediction with existing ML models
"""

import sys
import os
import logging
from typing import Dict, Any, Optional, List

# Add parent directory to path to import existing models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.model_manager import ModelManager as ExistingModelManager
from models.volatility_model import VolatilityModel
from models.direction_model import DirectionModel  
from models.profit_probability_model import ProfitProbabilityModel
from models.reversal_model import ReversalModel
from core.database import get_database

logger = logging.getLogger(__name__)

class ModelManager:
    """FastAPI model manager wrapping existing model functionality"""
    
    def __init__(self):
        self.db = get_database()
        self.existing_manager = ExistingModelManager()
        self.models = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize all models from database"""
        if self.initialized:
            return
            
        try:
            # Load existing models from database
            model_info = self.db.get_database_info()
            logger.info(f"Database info: {model_info}")
            
            # Initialize individual models
            self.models = {
                'volatility': VolatilityModel(),
                'direction': DirectionModel(),
                'profit_probability': ProfitProbabilityModel(), 
                'reversal': ReversalModel()
            }
            
            # Load trained models from database
            loaded_models = self.db.load_trained_models()
            if loaded_models:
                for model_name, model_data in loaded_models.items():
                    if model_name in self.models:
                        # For now, just log the availability
                        logger.info(f"✅ {model_name} model available in database")
            else:
                logger.info("No pre-trained models found in database")
            
            self.initialized = True
            logger.info("✅ Model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Model manager initialization failed: {e}")
            raise
    
    def get_model(self, model_name: str):
        """Get specific model by name"""
        if not self.initialized:
            raise RuntimeError("Model manager not initialized")
        return self.models.get(model_name)
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all loaded models"""
        if not self.initialized:
            raise RuntimeError("Model manager not initialized")
        return self.models
    
    async def train_model(self, model_name: str, data: Any) -> Dict[str, Any]:
        """Train specific model with provided data"""
        if not self.initialized:
            raise RuntimeError("Model manager not initialized")
            
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            # Train model using existing logic
            result = model.train(data)
            
            # Save to database
            model.save_to_database()
            
            logger.info(f"✅ Model {model_name} trained successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Model {model_name} training failed: {e}")
            raise
    
    async def predict(self, model_name: str, data: Any) -> Dict[str, Any]:
        """Generate predictions using specific model"""
        if not self.initialized:
            raise RuntimeError("Model manager not initialized")
            
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            prediction = model.predict(data)
            logger.info(f"✅ Generated prediction with {model_name} model")
            return prediction
        except Exception as e:
            logger.error(f"❌ Prediction with {model_name} failed: {e}")
            raise
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        if not self.initialized:
            return {"initialized": False, "models": {}}
        
        status = {
            "initialized": True,
            "models": {}
        }
        
        for name, model in self.models.items():
            status["models"][name] = {
                "loaded": hasattr(model, 'model') and model.model is not None,
                "features": getattr(model, 'feature_names', []),
                "last_trained": getattr(model, 'last_trained', None)
            }
        
        return status