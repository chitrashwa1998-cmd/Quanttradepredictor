
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import streamlit as st

from .volatility_model import VolatilityModel
from .direction_model import DirectionModel
from .profit_probability_model import ProfitProbabilityModel
from .reversal_model import ReversalModel

class ModelManager:
    """Centralized manager for all prediction models."""

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
        """Load previously trained models from database if available."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()
            loaded_models = db.load_trained_models()

            if loaded_models:
                for model_name in ['volatility', 'direction', 'profit_probability', 'reversal']:
                    if model_name in loaded_models:
                        model_data = loaded_models[model_name]
                        if 'task_type' not in model_data:
                            # Set default task types
                            if model_name == 'volatility':
                                model_data['task_type'] = 'regression'
                            else:
                                model_data['task_type'] = 'classification'
                        self.trained_models[model_name] = model_data
                        print(f"Loaded {model_name} model from database")

        except Exception as e:
            print(f"Could not load existing models: {str(e)}")

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
                        'scaler': model_data.get('scaler'),
                        'feature_names': model_data.get('feature_names', []),
                        'task_type': model_data.get('task_type', 'classification')
                    }

            if models_to_save:
                success = db.save_trained_models(models_to_save)
                if success:
                    print(f"Saved {len(models_to_save)} models to database")
                else:
                    print("Failed to save models to database")

        except Exception as e:
            print(f"Error saving models to database: {str(e)}")

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

            df_with_features = df.copy()
            df_with_features = TechnicalIndicators.calculate_volatility_indicators(df_with_features)
            df_with_features = compute_custom_volatility_features(df_with_features)
            df_with_features = add_volatility_lagged_features(df_with_features)
            df_with_features = add_time_context_features(df_with_features)

            X_volatility = self.models['volatility'].prepare_features(df_with_features)
            y_volatility = self.models['volatility'].create_target(df)

            features['volatility'] = X_volatility
            targets['volatility'] = y_volatility
            print(f"✅ Prepared volatility features: {X_volatility.shape}")
        except Exception as e:
            print(f"❌ Error preparing volatility model: {str(e)}")
            features['volatility'] = None
            targets['volatility'] = None

        # Direction Model
        try:
            X_direction = self.models['direction'].prepare_features(df)
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
            X_profit = self.models['profit_probability'].prepare_features(df)
            y_profit = self.models['profit_probability'].create_target(df)

            features['profit_probability'] = X_profit
            targets['profit_probability'] = y_profit
            print(f"✅ Prepared profit probability features: {X_profit.shape}")
        except Exception as e:
            print(f"❌ Error preparing profit probability model: {str(e)}")
            features['profit_probability'] = None
            targets['profit_probability'] = None

        # Reversal Model
        try:
            X_reversal = self.models['reversal'].prepare_features(df)
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
        """Train all models."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare features and targets for all models
        status_text.text("Preparing features and targets for all models...")
        features, targets = self.prepare_all_features_and_targets(df)

        results = {}
        model_names = ['volatility', 'direction', 'profit_probability', 'reversal']

        for i, model_name in enumerate(model_names):
            status_text.text(f"Training {model_name} model...")
            progress_bar.progress((i + 0.5) / len(model_names))

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

            progress_bar.progress((i + 1) / len(model_names))

        # Save trained models
        status_text.text("Saving trained models to database...")
        self._save_models_to_database()

        status_text.text("All models trained and saved!")
        progress_bar.empty()
        status_text.empty()

        return results

    def train_selected_models(self, df: pd.DataFrame, selected_models: List[str], train_split: float = 0.8) -> Dict[str, Any]:
        """Train selected models."""
        if not selected_models:
            st.warning("No models selected for training")
            return {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare features and targets
        status_text.text("Preparing features and targets...")
        features, targets = self.prepare_all_features_and_targets(df)

        results = {}

        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name} model...")
            progress_bar.progress((i + 0.5) / len(selected_models))

            try:
                if features[model_name] is not None and targets[model_name] is not None:
                    # Show target distribution for classification models
                    if model_name != 'volatility':
                        target_counts = targets[model_name].value_counts()
                        st.info(f"📊 {model_name} target distribution: {dict(target_counts)}")
                    else:
                        target_stats = targets[model_name].describe()
                        st.info(f"📊 Volatility target range: {target_stats['min']:.4f} to {target_stats['max']:.4f}")

                    # Train the model
                    result = self.models[model_name].train(features[model_name], targets[model_name], train_split)

                    # Store the result
                    self.trained_models[model_name] = result
                    results[model_name] = result

                    # Show immediate results
                    if result and 'metrics' in result:
                        metrics = result['metrics']
                        if model_name == 'volatility':
                            rmse = metrics.get('rmse', 0)
                            st.success(f"✅ {model_name} trained - RMSE: {rmse:.4f}")
                        else:
                            accuracy = metrics.get('accuracy', 0)
                            st.success(f"✅ {model_name} trained - Accuracy: {accuracy:.2%}")
                    else:
                        st.success(f"✅ {model_name} model trained successfully")
                else:
                    st.warning(f"⚠️ Could not prepare data for {model_name} model")
                    results[model_name] = None

            except Exception as e:
                st.error(f"❌ Error training {model_name} model: {str(e)}")
                results[model_name] = None

            progress_bar.progress((i + 1) / len(selected_models))

        # Save trained models
        status_text.text("Saving trained models to database...")
        try:
            self._save_models_to_database()
            status_text.text("✅ All selected models trained and saved!")
        except Exception as e:
            st.warning(f"⚠️ Models trained but database save failed: {str(e)}")
            status_text.text("✅ All selected models trained!")

        progress_bar.empty()
        status_text.empty()

        return results

    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

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
        model_instance.feature_names = trained_model_data.get('feature_names', [])

        # Verify scaler is available
        if model_instance.scaler is None:
            raise ValueError("Scaler not found in trained model data. Please retrain the model.")

        # Prepare features for the specific model
        X_features = model_instance.prepare_features(X)

        # Make predictions
        return model_instance.predict(X_features)

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for specified model."""
        if model_name not in self.models:
            print(f"Model {model_name} not available")
            return {}

        if model_name not in self.trained_models:
            print(f"Model {model_name} not found in trained models")
            return {}

        model_data = self.trained_models[model_name]
        feature_importance = model_data.get('feature_importance', {})

        print(f"Getting feature importance for {model_name}: {len(feature_importance)} features")
        return feature_importance

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return ['volatility', 'direction', 'profit_probability', 'reversal']

    def get_trained_models(self) -> List[str]:
        """Get list of trained model names."""
        return list(self.trained_models.keys())

    def is_model_trained(self, model_name: str) -> bool:
        """Check if specified model is trained."""
        return model_name in self.trained_models

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about specified model."""
        if model_name in self.trained_models:
            return self.trained_models[model_name]
        else:
            return {}
