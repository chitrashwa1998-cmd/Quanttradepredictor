import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Model Training")
st.markdown("Train prediction models using your processed data.")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("❌ No data available. Please upload data first in the Data Upload page.")
    st.stop()

# Feature Engineering Section
st.header("🔧 Feature Engineering")

if 'features' not in st.session_state or st.session_state.features is None:
    st.warning("⚠️ No features calculated yet. Please calculate technical indicators first.")
    
    if st.button("🔧 Calculate Technical Indicators Now", type="primary"):
        with st.spinner("Calculating technical indicators..."):
            try:
                from features.technical_indicators import TechnicalIndicators
                from utils.data_processing import DataProcessor
                
                # Calculate technical indicators
                features_data = TechnicalIndicators.calculate_all_indicators(st.session_state.data)
                
                # Clean the data
                features_clean = DataProcessor.clean_data(features_data)
                
                # Store in session state
                st.session_state.features = features_clean
                
                st.success("✅ Technical indicators calculated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error calculating indicators: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
else:
    st.success("✅ Technical indicators ready")
    
    # Show feature summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", len(st.session_state.features.columns))
    with col2:
        st.metric("Data Points", len(st.session_state.features))
    with col3:
        # Count non-OHLC features
        ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in st.session_state.features.columns if col not in ohlc_cols]
        st.metric("Engineered Features", len(feature_cols))
    
    # Show sample of features
    with st.expander("View Feature Sample"):
        st.dataframe(st.session_state.features.head(10), use_container_width=True)

# Training Configuration
st.header("⚙️ Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    train_split = st.slider("Training Split", 0.6, 0.9, 0.8, 0.05,
                           help="Percentage of data used for training")

with col2:
    max_depth = st.selectbox("Max Depth", [4, 6, 8, 10, 12], index=1,
                            help="Maximum depth of decision trees")

with col3:
    n_estimators = st.selectbox("Number of Estimators", [50, 100, 150, 200, 250, 300], index=1,
                               help="Number of trees in the ensemble")

st.info(f"Training: {int(train_split*100)}% | Testing: {int((1-train_split)*100)}%")

# Model Selection and Training
st.header("🎯 Model Selection")

tab1, tab2 = st.tabs(["Volatility Model", "Direction Model"])

# Volatility Model Tab
with tab1:
    st.subheader("📈 Volatility Prediction Model")
    st.markdown("Predicts future market volatility using technical indicators.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("This model forecasts the magnitude of price movements without predicting direction.")
    
    with col2:
        if st.button("🚀 Train Volatility Model", type="primary", key="train_vol"):
            with st.spinner("Training volatility model..."):
                try:
                    # Initialize volatility model
                    from models.xgboost_models import QuantTradingModels
                    
                    model_trainer = QuantTradingModels()
                    
                    # Prepare data
                    if 'features' not in st.session_state or st.session_state.features is None:
                        st.info("Calculating technical indicators...")
                        from features.technical_indicators import TechnicalIndicators
                        from utils.data_processing import DataProcessor
                        
                        features_data = TechnicalIndicators.calculate_all_indicators(st.session_state.data)
                        combined_data = DataProcessor.clean_data(features_data)
                        st.session_state.features = combined_data
                    else:
                        combined_data = st.session_state.features.copy()
                    
                    # Ensure OHLC columns are present
                    for col in ['Open', 'High', 'Low', 'Close']:
                        if col in st.session_state.data.columns and col not in combined_data.columns:
                            combined_data[col] = st.session_state.data[col]
                    
                    # Validate data
                    if len(combined_data) < 100:
                        st.error("❌ Insufficient data for training. Need at least 100 rows.")
                        st.stop()
                    
                    st.info(f"📊 Training data: {len(combined_data)} rows with {len(combined_data.columns)} features")
                    
                    # Train the model with configuration parameters
                    selected_models = ['volatility']
                    training_results = model_trainer.train_selected_models(
                        combined_data, 
                        selected_models,
                        train_split,
                        max_depth=max_depth,
                        n_estimators=n_estimators
                    )
                    
                    # Store results
                    if 'trained_models' not in st.session_state:
                        st.session_state.trained_models = {}
                    st.session_state.trained_models['volatility'] = training_results.get('volatility')
                    st.session_state.volatility_trainer = model_trainer
                    
                    # Display results
                    if training_results.get('volatility') is not None:
                        result = training_results['volatility']
                        metrics = result.get('metrics', {})
                        
                        st.success("✅ Volatility model trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            rmse = metrics.get('rmse', 0)
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col2:
                            mae = metrics.get('mae', 0)
                            st.metric("MAE", f"{mae:.4f}")
                        with col3:
                            mse = metrics.get('mse', 0)
                            st.metric("MSE", f"{mse:.4f}")
                        
                        # Feature importance
                        if 'feature_importance' in result and result['feature_importance']:
                            with st.expander("📊 Feature Importance"):
                                features = list(result['feature_importance'].keys())
                                importances = list(result['feature_importance'].values())
                                importance_df = pd.DataFrame({
                                    'Feature': features,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                st.dataframe(importance_df.head(10))
                                
                                fig = px.bar(
                                    importance_df.head(10),
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title="Top 10 Most Important Features"
                                )
                                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Failed to train volatility model")
                        
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())

# Direction Model Tab
with tab2:
    st.subheader("🎯 Direction Prediction Model")
    st.markdown("Predicts whether price will move up or down.")
    
    # Direction model features status
    st.subheader("Direction Model Features")
    
    if 'direction_features' not in st.session_state or st.session_state.direction_features is None:
        st.warning("⚠️ Direction model features not calculated yet.")
        
        if st.button("🔧 Calculate Direction Features", type="primary", key="calc_direction_features"):
            with st.spinner("Calculating direction-specific features..."):
                try:
                    from models.direction_model import DirectionModel
                    
                    direction_model = DirectionModel()
                    direction_features = direction_model.prepare_features(st.session_state.data)
                    
                    st.session_state.direction_features = direction_features
                    st.success("✅ Direction features calculated successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error calculating direction features: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
    else:
        st.success("✅ Direction features ready")
        
        # Show direction feature summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Direction Features", len(st.session_state.direction_features.columns))
        with col2:
            st.metric("Data Points", len(st.session_state.direction_features))
        
        # Show sample of direction features
        with st.expander("View Direction Feature Sample"):
            st.dataframe(st.session_state.direction_features.head(10), use_container_width=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("This model predicts the direction of price movement (bullish/bearish).")
    
    with col2:
        if st.button("🎯 Train Direction Model", type="primary", key="train_dir"):
            with st.spinner("Training direction model..."):
                try:
                    # Initialize direction model
                    from models.direction_model import DirectionModel
                    
                    direction_model = DirectionModel()
                    
                    # Use pre-calculated direction features if available, otherwise calculate them
                    if 'direction_features' in st.session_state and st.session_state.direction_features is not None:
                        direction_features = st.session_state.direction_features
                    else:
                        st.info("Calculating direction-specific features...")
                        direction_features = direction_model.prepare_features(st.session_state.data)
                        st.session_state.direction_features = direction_features
                    
                    # Create direction target
                    direction_target = direction_model.create_target(st.session_state.data)
                    
                    # Validate data
                    if len(direction_features) < 100:
                        st.error("❌ Insufficient data for training. Need at least 100 rows.")
                        st.stop()
                    
                    st.info(f"📊 Direction data: {len(direction_features)} samples with {len(direction_features.columns)} features")
                    
                    # Train direction model with configuration parameters
                    training_result = direction_model.train(
                        direction_features, 
                        direction_target, 
                        train_split,
                        max_depth=max_depth,
                        n_estimators=n_estimators
                    )
                    
                    # Store results
                    if 'trained_models' not in st.session_state:
                        st.session_state.trained_models = {}
                    st.session_state.trained_models['direction'] = training_result
                    st.session_state.direction_model = direction_model
                    
                    # Display results
                    if training_result is not None:
                        metrics = training_result.get('metrics', {})
                        
                        st.success("✅ Direction model trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            accuracy = metrics.get('accuracy', 0)
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        with col2:
                            precision = metrics.get('precision', 0)
                            st.metric("Precision", f"{precision:.2%}")
                        with col3:
                            recall = metrics.get('recall', 0)
                            st.metric("Recall", f"{recall:.2%}")
                        
                        # Feature importance for direction model
                        if 'feature_importance' in training_result and training_result['feature_importance']:
                            with st.expander("📊 Direction Feature Importance"):
                                features = list(training_result['feature_importance'].keys())
                                importances = list(training_result['feature_importance'].values())
                                importance_df = pd.DataFrame({
                                    'Feature': features,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                st.dataframe(importance_df.head(10))
                                
                                fig = px.bar(
                                    importance_df.head(10),
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title="Top 10 Direction Features"
                                )
                                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Failed to train direction model")
                        
                except Exception as e:
                    st.error(f"❌ Direction training failed: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())

# Model Status Section
st.header("📊 Model Status")

if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volatility Model")
        if 'volatility' in st.session_state.trained_models and st.session_state.trained_models['volatility']:
            result = st.session_state.trained_models['volatility']
            metrics = result.get('metrics', {})
            rmse = metrics.get('rmse', 0)
            st.success(f"✅ Trained - RMSE: {rmse:.4f}")
        else:
            st.warning("⚠️ Not trained")
    
    with col2:
        st.subheader("Direction Model")
        if 'direction' in st.session_state.trained_models and st.session_state.trained_models['direction']:
            result = st.session_state.trained_models['direction']
            metrics = result.get('metrics', {})
            accuracy = metrics.get('accuracy', 0)
            st.success(f"✅ Trained - Accuracy: {accuracy:.2%}")
        else:
            st.warning("⚠️ Not trained")
else:
    st.info("ℹ️ No models trained yet")

# Export Models Section
if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
    st.header("💾 Export Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Save Volatility Model", disabled='volatility' not in st.session_state.trained_models):
            try:
                if hasattr(st.session_state, 'volatility_trainer'):
                    st.session_state.volatility_trainer._save_models_to_database()
                    st.success("✅ Volatility model saved to database!")
                else:
                    st.error("❌ Volatility trainer not available")
            except Exception as e:
                st.error(f"❌ Failed to save volatility model: {str(e)}")
    
    with col2:
        if st.button("📥 Save Direction Model", disabled='direction' not in st.session_state.trained_models):
            try:
                from utils.database_adapter import get_trading_database
                db = get_trading_database()
                
                # Save direction model results
                if 'direction' in st.session_state.trained_models:
                    db.save_model_results('direction', st.session_state.trained_models['direction'])
                    st.success("✅ Direction model saved to database!")
                else:
                    st.error("❌ Direction model not available")
            except Exception as e:
                st.error(f"❌ Failed to save direction model: {str(e)}")