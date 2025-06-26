import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import streamlit as st

from .direction_model import DirectionModel
from .magnitude_model import MagnitudeModel
from .volatility_model import VolatilityModel
from .profit_probability_model import ProfitProbabilityModel
from .trend_sideways_model import TrendSidewaysModel
from .reversal_model import ReversalModel

class ModelManager:
    """Centralized manager for prediction models."""

    def __init__(self):
        self.models = {
            'direction': DirectionModel(),
            'magnitude': MagnitudeModel(),
            'volatility': VolatilityModel(),
            'profit_prob': ProfitProbabilityModel(),
            'trend_sideways': TrendSidewaysModel(),
            'reversal': ReversalModel(),
        }
        self.trained_models = {}
        self._load_existing_models()

    def _load_existing_models(self):
        """Load previously trained models from database if available."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()
            loaded_models = db.load_trained_models()

            if loaded_models:
                for model_name, model_data in loaded_models.items():
                    # Ensure task_type is present
                    if 'task_type' not in model_data:
                        if model_name in ['direction', 'profit_prob', 'trend_sideways', 'reversal']:
                            model_data['task_type'] = 'classification'
                        elif model_name in ['magnitude', 'volatility']:
                            model_data['task_type'] = 'regression'
                        else:
                            model_data['task_type'] = 'classification'

                self.trained_models = loaded_models
                print(f"Loaded {len(loaded_models)} existing trained models from database")
            else:
                print("No existing models found in database")

        except Exception as e:
            print(f"Could not load existing models: {str(e)}")

    def _save_models_to_database(self):
        """Save trained models to database for persistence."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()

            models_to_save = {}
            for model_name, model_data in self.trained_models.items():
                if 'model' in model_data:
                    models_to_save[model_name] = {
                        'ensemble': model_data['model'],  # Use 'ensemble' key for database compatibility
                        'feature_names': model_data.get('feature_names', []),
                        'task_type': model_data.get('task_type', 'classification')
                    }

            if models_to_save:
                success = db.save_trained_models(models_to_save)
                if success:
                    print(f"Saved {len(models_to_save)} trained models to database")
                else:
                    print("Failed to save models to database")

        except Exception as e:
            print(f"Error saving models to database: {str(e)}")

    def prepare_all_features_and_targets(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """Prepare features and targets for all models."""
        features = {}
        targets = {}

        for model_name, model in self.models.items():
            try:
                # Prepare features
                features[model_name] = model.prepare_features(df)

                # Create targets
                targets[model_name] = model.create_target(df)

                print(f"✅ Prepared {model_name}: {features[model_name].shape[1]} features, {len(targets[model_name])} targets")

            except Exception as e:
                print(f"❌ Error preparing {model_name}: {str(e)}")
                features[model_name] = None
                targets[model_name] = None

        return features, targets

    def train_all_models(self, df: pd.DataFrame, train_split: float = 0.8) -> Dict[str, Any]:
        """Train all models."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare features and targets for all models
        status_text.text("Preparing features and targets for all models...")
        features, targets = self.prepare_all_features_and_targets(df)

        results = {}
        total_models = len(self.models)

        for i, (model_name, model) in enumerate(self.models.items()):
            status_text.text(f"Training {model_name} model...")

            try:
                if features[model_name] is not None and targets[model_name] is not None:
                    # Train the model
                    result = model.train(features[model_name], targets[model_name], train_split)

                    # Store the result
                    self.trained_models[model_name] = result
                    results[model_name] = result

                    st.success(f"✅ {model_name} model trained successfully")
                else:
                    st.warning(f"⚠️ Could not prepare data for {model_name}")
                    results[model_name] = None

            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                results[model_name] = None

            progress_bar.progress((i + 1) / total_models)

        # Save all trained models
        status_text.text("Saving trained models to database...")
        self._save_models_to_database()

        status_text.text("All models trained and saved!")
        progress_bar.empty()
        status_text.empty()

        return results

    def train_selected_models(self, df: pd.DataFrame, selected_models: List[str], train_split: float = 0.8) -> Dict[str, Any]:
        """Train only selected models."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare features and targets for all models (needed for consistency)
        status_text.text("Preparing features and targets...")
        features, targets = self.prepare_all_features_and_targets(df)

        results = {}
        total_models = len(selected_models)

        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name} model...")

            try:
                if model_name in self.models:
                    model = self.models[model_name]

                    if features[model_name] is not None and targets[model_name] is not None:
                        # Show target distribution
                        target_series = targets[model_name]
                        if model.task_type == 'classification':
                            target_counts = target_series.value_counts()
                            st.info(f"📊 {model_name} target distribution: {target_counts.to_dict()}")
                        else:
                            target_stats = target_series.describe()
                            st.info(f"📊 {model_name} target range: {target_stats['min']:.4f} to {target_stats['max']:.4f}")

                        # Train the model
                        result = model.train(features[model_name], targets[model_name], train_split)

                        # Store the result
                        self.trained_models[model_name] = result
                        results[model_name] = result

                        # Show immediate results
                        if result and 'metrics' in result:
                            metrics = result['metrics']
                            if model.task_type == 'classification':
                                accuracy = metrics.get('accuracy', 0)
                                st.success(f"✅ {model_name} trained - Accuracy: {accuracy:.3f}")
                            else:
                                rmse = metrics.get('rmse', 0)
                                st.success(f"✅ {model_name} trained - RMSE: {rmse:.4f}")
                        else:
                            st.success(f"✅ {model_name} model trained successfully")
                    else:
                        st.warning(f"⚠️ Could not prepare data for {model_name}")
                        results[model_name] = None
                else:
                    st.error(f"❌ Model {model_name} not found")
                    results[model_name] = None

            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                import traceback
                error_details = traceback.format_exc()
                st.error(f"Detailed error: {error_details}")
                results[model_name] = None

            progress_bar.progress((i + 1) / total_models)

        # Save trained models
        status_text.text("Saving trained models to database...")
        try:
            self._save_models_to_database()
            status_text.text("✅ Selected models trained and saved!")
        except Exception as e:
            st.warning(f"⚠️ Models trained but database save failed: {str(e)}")
            status_text.text("✅ Selected models trained!")

        progress_bar.empty()
        status_text.empty()

        return results

    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using a specific trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained. Available models: {list(self.trained_models.keys())}")

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in model definitions")

        # Get the model instance and trained model data
        model_instance = self.models[model_name]
        trained_model_data = self.trained_models[model_name]

        # Set the trained model in the instance
        model_instance.model = trained_model_data['model']
        model_instance.scaler = trained_model_data.get('scaler')
        model_instance.selector = trained_model_data.get('selector')  # For direction model
        model_instance.feature_names = trained_model_data.get('feature_names', [])

        # Prepare features for this specific model
        X_features = model_instance.prepare_features(X)

        # Make predictions
        return model_instance.predict(X_features)

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for a specific model."""
        if model_name not in self.trained_models:
            print(f"Model {model_name} not found in trained models: {list(self.trained_models.keys())}")
            return {}

        model_data = self.trained_models[model_name]
        feature_importance = model_data.get('feature_importance', {})

        print(f"Getting feature importance for {model_name}: {len(feature_importance)} features")
        return feature_importance

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())

    def get_trained_models(self) -> List[str]:
        """Get list of trained model names."""
        return list(self.trained_models.keys())

    def is_model_trained(self, model_name: str) -> bool:
        """Check if a specific model is trained."""
        return model_name in self.trained_models

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_name in self.trained_models:
            return self.trained_models[model_name]
        else:
            return {}