import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import streamlit as st

from .volatility_model import VolatilityModel

class ModelManager:
    """Centralized manager for prediction models - Volatility only."""

    def __init__(self):
        self.models = {
            'volatility': VolatilityModel(),
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
                # Only load volatility model
                if 'volatility' in loaded_models:
                    model_data = loaded_models['volatility']
                    if 'task_type' not in model_data:
                        model_data['task_type'] = 'regression'
                    self.trained_models['volatility'] = model_data
                    print(f"Loaded volatility model from database")
                else:
                    print("No volatility model found in database")
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
            if 'volatility' in self.trained_models:
                model_data = self.trained_models['volatility']
                if 'model' in model_data:
                    models_to_save['volatility'] = {
                        'ensemble': model_data['model'],
                        'feature_names': model_data.get('feature_names', []),
                        'task_type': model_data.get('task_type', 'regression')
                    }

            if models_to_save:
                success = db.save_trained_models(models_to_save)
                if success:
                    print(f"Saved volatility model to database")
                else:
                    print("Failed to save volatility model to database")

        except Exception as e:
            print(f"Error saving models to database: {str(e)}")

    def prepare_all_features_and_targets(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """Prepare features and targets for volatility model."""
        features = {}
        targets = {}

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

        return features, targets

    def train_all_models(self, df: pd.DataFrame, train_split: float = 0.8) -> Dict[str, Any]:
        """Train volatility model."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare features and targets for volatility model
        status_text.text("Preparing features and targets for volatility model...")
        features, targets = self.prepare_all_features_and_targets(df)

        results = {}

        status_text.text("Training volatility model...")

        try:
            if features['volatility'] is not None and targets['volatility'] is not None:
                # Train the volatility model
                result = self.models['volatility'].train(features['volatility'], targets['volatility'], train_split)

                # Store the result
                self.trained_models['volatility'] = result
                results['volatility'] = result

                st.success(f"✅ Volatility model trained successfully")
            else:
                st.warning(f"⚠️ Could not prepare data for volatility model")
                results['volatility'] = None

        except Exception as e:
            st.error(f"❌ Error training volatility model: {str(e)}")
            results['volatility'] = None

        progress_bar.progress(1.0)

        # Save trained model
        status_text.text("Saving trained model to database...")
        self._save_models_to_database()

        status_text.text("Volatility model trained and saved!")
        progress_bar.empty()
        status_text.empty()

        return results

    def train_selected_models(self, df: pd.DataFrame, selected_models: List[str], train_split: float = 0.8) -> Dict[str, Any]:
        """Train volatility model if selected."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Only process if volatility is selected
        if 'volatility' not in selected_models:
            st.warning("Only volatility model is available")
            return {}

        # Prepare features and targets
        status_text.text("Preparing features and targets...")
        features, targets = self.prepare_all_features_and_targets(df)

        results = {}

        status_text.text("Training volatility model...")

        try:
            if features['volatility'] is not None and targets['volatility'] is not None:
                # Show target distribution
                target_series = targets['volatility']
                target_stats = target_series.describe()
                st.info(f"📊 Volatility target range: {target_stats['min']:.4f} to {target_stats['max']:.4f}")

                # Train the model
                result = self.models['volatility'].train(features['volatility'], targets['volatility'], train_split)

                # Store the result
                self.trained_models['volatility'] = result
                results['volatility'] = result

                # Show immediate results
                if result and 'metrics' in result:
                    metrics = result['metrics']
                    rmse = metrics.get('rmse', 0)
                    st.success(f"✅ Volatility trained - RMSE: {rmse:.4f}")
                else:
                    st.success(f"✅ Volatility model trained successfully")
            else:
                st.warning(f"⚠️ Could not prepare data for volatility model")
                results['volatility'] = None

        except Exception as e:
            st.error(f"❌ Error training volatility model: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            st.error(f"Detailed error: {error_details}")
            results['volatility'] = None

        progress_bar.progress(1.0)

        # Save trained models
        status_text.text("Saving trained model to database...")
        try:
            self._save_models_to_database()
            status_text.text("✅ Volatility model trained and saved!")
        except Exception as e:
            st.warning(f"⚠️ Model trained but database save failed: {str(e)}")
            status_text.text("✅ Volatility model trained!")

        progress_bar.empty()
        status_text.empty()

        return results

    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using volatility model."""
        if model_name != 'volatility':
            raise ValueError(f"Only volatility model is available")

        if 'volatility' not in self.trained_models:
            raise ValueError(f"Volatility model not trained")

        # Get the model instance and trained model data
        model_instance = self.models['volatility']
        trained_model_data = self.trained_models['volatility']

        # Set the trained model in the instance
        model_instance.model = trained_model_data['model']
        model_instance.scaler = trained_model_data.get('scaler')
        model_instance.feature_names = trained_model_data.get('feature_names', [])

        # Prepare features for volatility model
        X_features = model_instance.prepare_features(X)

        # Make predictions
        return model_instance.predict(X_features)

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for volatility model."""
        if model_name != 'volatility':
            print(f"Only volatility model is available")
            return {}

        if 'volatility' not in self.trained_models:
            print(f"Volatility model not found in trained models")
            return {}

        model_data = self.trained_models['volatility']
        feature_importance = model_data.get('feature_importance', {})

        print(f"Getting feature importance for volatility: {len(feature_importance)} features")
        return feature_importance

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return ['volatility']

    def get_trained_models(self) -> List[str]:
        """Get list of trained model names."""
        return list(self.trained_models.keys())

    def is_model_trained(self, model_name: str) -> bool:
        """Check if volatility model is trained."""
        return model_name == 'volatility' and 'volatility' in self.trained_models

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about volatility model."""
        if model_name == 'volatility' and 'volatility' in self.trained_models:
            return self.trained_models['volatility']
        else:
            return {}