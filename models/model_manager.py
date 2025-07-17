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
            
            print(f"🔍 Database loaded models: {list(loaded_models.keys()) if loaded_models else 'None'}")

            if loaded_models:
                for model_name in ['volatility', 'direction', 'profit_probability', 'reversal']:
                    if model_name in loaded_models:
                        model_data = loaded_models[model_name]
                        
                        # Ensure task type is set
                        if 'task_type' not in model_data:
                            model_data['task_type'] = 'regression' if model_name == 'volatility' else 'classification'
                        
                        # Ensure model has all required keys - check for both 'model' and 'ensemble'
                        model_obj = model_data.get('model') or model_data.get('ensemble')
                        if model_obj is not None:
                            # Normalize to use 'model' key
                            model_data['model'] = model_obj
                            if 'ensemble' in model_data and 'model' not in model_data:
                                model_data['model'] = model_data['ensemble']
                        
                        # Restore training results if available
                        if 'training_results' in model_data:
                            training_results = model_data['training_results']
                            # Merge training results back into model_data
                            for key, value in training_results.items():
                                if key not in model_data:
                                    model_data[key] = value
                        
                        # Validate that we have the minimum required components
                        has_model = model_data.get('model') is not None or model_data.get('ensemble') is not None
                        has_scaler = model_data.get('scaler') is not None
                        has_features = model_data.get('feature_names') is not None
                        
                        if has_model and has_scaler and has_features:
                            self.trained_models[model_name] = model_data
                            feature_count = len(model_data.get('feature_names', []))
                            print(f"✅ Loaded {model_name} model from database with {feature_count} features")
                        else:
                            print(f"⚠️ {model_name} model incomplete - Model: {has_model}, Scaler: {has_scaler}, Features: {has_features}")

            # Also check session state for any models not loaded from database
            if hasattr(st, 'session_state'):
                # Check for trained models in session state
                if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
                    session_models = st.session_state.trained_models
                    print(f"🔍 Session state models: {list(session_models.keys())}")
                    
                    for model_name in ['volatility', 'direction', 'profit_probability', 'reversal']:
                        if model_name not in self.trained_models and model_name in session_models:
                            model_data = session_models[model_name]
                            if model_data and isinstance(model_data, dict):
                                # Validate session state model
                                has_model = model_data.get('model') is not None or model_data.get('ensemble') is not None
                                has_scaler = model_data.get('scaler') is not None
                                
                                if has_model and has_scaler:
                                    self.trained_models[model_name] = model_data
                                    print(f"✅ Loaded {model_name} model from session state")
                
                # Check individual model session states
                model_session_keys = {
                    'direction': 'direction_trained_models',
                    'profit_probability': 'profit_prob_trained_models', 
                    'reversal': 'reversal_trained_models'
                }
                
                for model_name, session_key in model_session_keys.items():
                    if model_name not in self.trained_models and hasattr(st.session_state, session_key):
                        session_models = getattr(st.session_state, session_key, {})
                        if model_name in session_models:
                            model_instance = session_models[model_name]
                            if hasattr(model_instance, 'model') and model_instance.model is not None:
                                self.trained_models[model_name] = {
                                    'model': model_instance.model,
                                    'scaler': model_instance.scaler,
                                    'feature_names': getattr(model_instance, 'feature_names', getattr(model_instance, 'selected_features', [])),
                                    'task_type': 'classification'
                                }
                                print(f"✅ Loaded {model_name} model from individual session state")
                
                # Update session state with all loaded models
                if self.trained_models:
                    st.session_state.trained_models = self.trained_models
                    print(f"✅ ModelManager initialized with {len(self.trained_models)} trained models: {list(self.trained_models.keys())}")

        except Exception as e:
            print(f"❌ Error loading existing models: {str(e)}")
            import traceback
            traceback.print_exc()

    def predict(self, model_name: str, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model_info = self.trained_models[model_name]
        # Handle both 'model' and 'ensemble' keys for compatibility
        model = model_info.get('model') or model_info.get('ensemble')
        if model is None:
            raise ValueError(f"No trained model found for {model_name}")
        scaler = model_info['scaler']
        expected_features = model_info.get('feature_names', [])

        # Ensure features match what model expects
        if expected_features:
            # Check for feature alignment
            available_features = [col for col in expected_features if col in features.columns]
            missing_features = [col for col in expected_features if col not in features.columns]

            if missing_features:
                print(f"Warning: Missing features for {model_name}: {missing_features[:5]}...")

            if len(available_features) < len(expected_features) * 0.8:
                raise ValueError(f"Too many missing features for {model_name}. Available: {len(available_features)}, Expected: {len(expected_features)}")

            # Use only available features that match training
            features = features[available_features]

            # Update scaler features if needed
            if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                scaler_features = list(scaler.feature_names_in_)
                if scaler_features != available_features:
                    print(f"Feature mismatch detected for {model_name}")
                    # Reorder features to match scaler expectations
                    common_features = [f for f in scaler_features if f in features.columns]
                    if len(common_features) >= len(scaler_features) * 0.8:
                        features = features[common_features]
                    else:
                        raise ValueError(f"Cannot align features for {model_name}")

        # Scale features
        features_scaled = scaler.transform(features)

        # Make predictions
        predictions = model.predict(features_scaled)
        probabilities = None

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)

        return predictions, probabilities

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
                if 'model' in model_data or 'ensemble' in model_data:
                    # Handle both 'model' and 'ensemble' keys
                    model_obj = model_data.get('model') or model_data.get('ensemble')
                    if model_obj is not None:
                        # Ensure metrics are properly extracted and saved
                        metrics = model_data.get('metrics', {})
                        if not metrics:
                            # Try to find metrics in alternative locations
                            for key in ['training_metrics', 'performance', 'results']:
                                if key in model_data and isinstance(model_data[key], dict):
                                    metrics = model_data[key]
                                    break
                        
                        models_to_save[model_name] = {
                            'ensemble': model_obj,
                            'scaler': model_data.get('scaler'),
                            'feature_names': model_data.get('feature_names', []),
                            'task_type': model_data.get('task_type', 'regression'),
                            'metrics': metrics,
                            'feature_importance': model_data.get('feature_importance', {}),
                            # Preserve all original data for debugging
                            'training_results': model_data
                        }
                        print(f"✅ Prepared {model_name} model for database save with metrics: {list(metrics.keys())}")

            if models_to_save:
                success = db.save_trained_models(models_to_save)
                if success:
                    print(f"✅ Saved {len(models_to_save)} models to database: {list(models_to_save.keys())}")
                else:
                    print("❌ Failed to save models to database")
            else:
                print("⚠️ No models to save to database")

        except Exception as e:
            print(f"❌ Error saving models to database: {str(e)}")
            import traceback
            traceback.print_exc()