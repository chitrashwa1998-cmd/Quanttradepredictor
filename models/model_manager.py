import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import streamlit as st

from .volatility_model import VolatilityModel
from .direction_model import DirectionModel
from .profit_probability_model import ProfitProbabilityModel
from .reversal_model import ReversalModel

class ModelManager:
    """Centralized manager for all 4 prediction models."""

    def __init__(self):
        self.models = {
            'volatility': VolatilityModel(),
            'direction': DirectionModel(),
            'profit_probability': ProfitProbabilityModel(),
            'reversal': ReversalModel()
        }
        self.trained_models = {}
        self._load_existing_models()

    def _load_existing_models(self):
        """Load previously trained models from database and session state."""
        try:
            # Load from database
            from utils.database_adapter import get_trading_database
            db = get_trading_database()
            loaded_models = db.load_trained_models()

            if loaded_models:
                for model_name in ['volatility', 'direction', 'profit_probability', 'reversal']:
                    if model_name in loaded_models:
                        model_data = loaded_models[model_name]
                        if 'task_type' not in model_data:
                            model_data['task_type'] = 'regression' if model_name == 'volatility' else 'classification'
                        self.trained_models[model_name] = model_data
                        print(f"Loaded {model_name} model from database")

            # Also check session state for direction model specifically
            if hasattr(st, 'session_state'):
                if hasattr(st.session_state, 'direction_trained_models') and st.session_state.direction_trained_models:
                    if 'direction' in st.session_state.direction_trained_models:
                        direction_model_instance = st.session_state.direction_trained_models['direction']
                        if hasattr(direction_model_instance, 'model') and direction_model_instance.model is not None:
                            # Convert model instance to the format expected by ModelManager
                            self.trained_models['direction'] = {
                                'model': direction_model_instance.model,
                                'scaler': direction_model_instance.scaler,
                                'feature_names': getattr(direction_model_instance, 'selected_features', []),
                                'task_type': 'classification'
                            }
                            print("Loaded direction model from session state")

                # Check for other models in session state
                for model_name in ['profit_probability', 'reversal']:
                    session_key = f'{model_name.replace("_", "_")}_trained_models'
                    if hasattr(st.session_state, session_key):
                        session_models = getattr(st.session_state, session_key)
                        if model_name in session_models:
                            model_instance = session_models[model_name]
                            if hasattr(model_instance, 'model') and model_instance.model is not None:
                                self.trained_models[model_name] = {
                                    'model': model_instance.model,
                                    'scaler': model_instance.scaler,
                                    'feature_names': getattr(model_instance, 'selected_features', []),
                                    'task_type': 'classification'
                                }
                                print(f"Loaded {model_name} model from session state")

        except Exception as e:
            print(f"Could not load existing models: {str(e)}")

    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using any trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")

        if model_name not in self.trained_models:
            raise ValueError(f"{model_name.title()} model not trained")

        # Get the model instance and trained model data
        model_instance = self.models[model_name]
        trained_model_data = self.trained_models[model_name]

        # Handle both 'model' and 'ensemble' keys for backward compatibility
        ensemble_model = trained_model_data.get('model') or trained_model_data.get('ensemble')
        if ensemble_model is None:
            raise ValueError("No trained model found in model data")

        # Set the trained model in the instance
        model_instance.model = ensemble_model
        model_instance.scaler = trained_model_data.get('scaler')

        # Handle feature names for different model types
        if hasattr(model_instance, 'feature_names'):
            model_instance.feature_names = trained_model_data.get('feature_names', [])
        if hasattr(model_instance, 'selected_features'):
            model_instance.selected_features = trained_model_data.get('feature_names', [])

        # Verify scaler is available
        if model_instance.scaler is None:
            raise ValueError("Scaler not found in trained model data. Please retrain the model.")

        # Prepare features for the specific model
        X_features = model_instance.prepare_features(X)

        # Make predictions
        return model_instance.predict(X_features)

    def is_model_trained(self, model_name: str) -> bool:
        """Check if a specific model is trained."""
        return model_name in self.trained_models and self.trained_models[model_name] is not None

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())

    def get_trained_models(self) -> List[str]:
        """Get list of trained model names."""
        return list(self.trained_models.keys())

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_name in self.trained_models:
            return self.trained_models[model_name]
        else:
            return {}

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for a specific model."""
        if model_name not in self.trained_models:
            print(f"{model_name.title()} model not found in trained models")
            return {}

        model_data = self.trained_models[model_name]
        feature_importance = model_data.get('feature_importance', {})

        print(f"Getting feature importance for {model_name}: {len(feature_importance)} features")
        return feature_importance
    
    def prepare_all_features_and_targets(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """Prepare features and targets for all models."""
        features = {}
        targets = {}
        
        # Volatility Model
        try:
            from features.technical_indicators import TechnicalIndicators
            from features.custom_engineered import compute_custom_volatility_features
            from features.lagged_features import add_volatility_lagged_features
            from features.time_context_features import add_time_context_features

            # Calculate all features
            df_with_features = df.copy()
            df_with_features = TechnicalIndicators.calculate_volatility_indicators(df_with_features)
            df_with_features = compute_custom_volatility_features(df_with_features)
            df_with_features = add_volatility_lagged_features(df_with_features)
            df_with_features = add_time_context_features(df_with_features)

            # Prepare features for volatility model
            X_volatility = self.models['volatility'].prepare_features(df_with_features)

            # Create targets for volatility model
            y_volatility = self.models['volatility'].create_target(df)

            features['volatility'] = X_volatility
            targets['volatility'] = y_volatility

            print(f"✅ Prepared volatility features: {X_volatility.shape} with all feature types")
        except Exception as e:
            print(f"❌ Error preparing volatility model: {str(e)}")
            features['volatility'] = None
            targets['volatility'] = None

        # Direction Model
        try:
            # Assuming DirectionModel requires similar feature engineering as VolatilityModel
            # Modify feature engineering as needed for DirectionModel
            df_with_features = df.copy()
            df_with_features = TechnicalIndicators.calculate_all_indicators(df_with_features)  # Example
            X_direction = self.models['direction'].prepare_features(df_with_features)
            y_direction = self.models['direction'].create_target(df)
            features['direction'] = X_direction
            targets['direction'] = y_direction
            print(f"✅ Prepared direction features: {X_direction.shape}")
        except Exception as e:
            print(f"❌ Error preparing direction model: {str(e)}")
            features['direction'] = None
            targets['direction'] = None

        # Profit Probability Model
        try:
            df_with_features = df.copy()
            df_with_features = TechnicalIndicators.calculate_all_indicators(df_with_features)  # Example
            X_profit_probability = self.models['profit_probability'].prepare_features(df_with_features)
            y_profit_probability = self.models['profit_probability'].create_target(df)
            features['profit_probability'] = X_profit_probability
            targets['profit_probability'] = y_profit_probability
            print(f"✅ Prepared profit_probability features: {X_profit_probability.shape}")
        except Exception as e:
            print(f"❌ Error preparing profit_probability model: {str(e)}")
            features['profit_probability'] = None
            targets['profit_probability'] = None

        # Reversal Model
        try:
            df_with_features = df.copy()
            df_with_features = TechnicalIndicators.calculate_all_indicators(df_with_features)  # Example
            X_reversal = self.models['reversal'].prepare_features(df_with_features)
            y_reversal = self.models['reversal'].create_target(df)
            features['reversal'] = X_reversal
            targets['reversal'] = y_reversal
            print(f"✅ Prepared reversal features: {X_reversal.shape}")
        except Exception as e:
            print(f"❌ Error preparing reversal model: {str(e)}")
            features['reversal'] = None
            targets['reversal'] = None
            
        return features, targets

    def train_all_models(self, df: pd.DataFrame, train_split: float = 0.8) -> Dict[str, Any]:
        """Train all available models."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare features and targets for all models
        status_text.text("Preparing features and targets for all models...")
        features, targets = self.prepare_all_features_and_targets(df)

        results = {}

        for model_name in self.models:
            status_text.text(f"Training {model_name} model...")

            try:
                if features[model_name] is not None and targets[model_name] is not None:
                    # Train the model
                    result = self.models[model_name].train(features[model_name], targets[model_name], train_split)

                    # Store the result
                    self.trained_models[model_name] = result
                    results[model_name] = result

                    st.success(f"✅ {model_name} model trained successfully")
                else:
                    st.warning(f"⚠️ Could not prepare data for {model_name} model")
                    results[model_name] = None

            except Exception as e:
                st.error(f"❌ Error training {model_name} model: {str(e)}")
                results[model_name] = None

            progress_bar.progress(1.0 / len(self.models) * (list(self.models.keys()).index(model_name) + 1))

        # Save trained models
        status_text.text("Saving trained models to database...")
        self._save_models_to_database()

        status_text.text("All models trained and saved!")
        progress_bar.empty()
        status_text.empty()

        return results
    
    def train_selected_models(self, df: pd.DataFrame, selected_models: List[str], train_split: float = 0.8) -> Dict[str, Any]:
        """Train selected models."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare features and targets
        status_text.text("Preparing features and targets...")
        features, targets = self.prepare_all_features_and_targets(df)

        results = {}

        for model_name in selected_models:
            if model_name not in self.models:
                st.warning(f"Model '{model_name}' is not available.")
                results[model_name] = None
                continue

            status_text.text(f"Training {model_name} model...")

            try:
                if features[model_name] is not None and targets[model_name] is not None:
                    # Train the model
                    result = self.models[model_name].train(features[model_name], targets[model_name], train_split)

                    # Store the result
                    self.trained_models[model_name] = result
                    results[model_name] = result

                    st.success(f"✅ {model_name} trained successfully")
                else:
                    st.warning(f"⚠️ Could not prepare data for {model_name}")
                    results[model_name] = None

            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                import traceback
                error_details = traceback.format_exc()
                st.error(f"Detailed error: {error_details}")
                results[model_name] = None

            progress_bar.progress(1.0 / len(selected_models) * (selected_models.index(model_name) + 1))

        # Save trained models
        status_text.text("Saving trained models to database...")
        try:
            self._save_models_to_database()
            status_text.text("✅ Models trained and saved!")
        except Exception as e:
            st.warning(f"⚠️ Model trained but database save failed: {str(e)}")
            status_text.text("✅ Models trained!")

        progress_bar.empty()
        status_text.empty()

        return results

    def _save_models_to_database(self):
        """Save trained models to database for persistence."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()

            models_to_save = {}
            for model_name in self.trained_models:
                model_data = self.trained_models[model_name]
                if 'model' in model_data:
                    models_to_save[model_name] = {
                        'ensemble': model_data['model'],
                        'scaler': model_data.get('scaler'),  # Include the scaler!
                        'feature_names': model_data.get('feature_names', []),
                        'task_type': model_data.get('task_type', 'regression')
                    }

            if models_to_save:
                success = db.save_trained_models(models_to_save)
                if success:
                    print(f"Saved models to database")
                else:
                    print("Failed to save models to database")

        except Exception as e:
            print(f"Error saving models to database: {str(e)}")