import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Model Training")
st.markdown("Train the volatility forecasting model using your processed data.")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("❌ No data available. Please upload data first in the Data Upload page.")
    st.stop()

# Check if features are available - if not, offer to calculate them
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
    st.stop()

# Training configuration
st.header("Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    train_split = st.slider("Training Split", 0.6, 0.9, 0.8, 0.05,
                           help="Percentage of data used for training")

with col2:
    max_depth = st.selectbox("Max Depth", [4, 6, 8, 10, 12], index=1)

with col3:
    n_estimators = st.selectbox("Number of Estimators", [50, 100, 150, 200, 250, 300], index=1)

# Model selection
st.header("Volatility Model Training")
st.markdown("Train the volatility forecasting model to predict future market volatility.")

train_volatility = st.checkbox("Train Volatility Model", value=True,
                              help="Forecast future volatility")

# Feature engineering status
st.header("Feature Engineering")

if 'features' in st.session_state and st.session_state.features is not None:
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
        
else:
    st.info("ℹ️ Features will be automatically calculated when training starts")

# Training section
st.header("Model Training")

if st.button("🚀 Train Volatility Model", type="primary", disabled=not train_volatility):
    if not train_volatility:
        st.warning("Please select the volatility model to train.")
    else:
        try:
            # Initialize model
            from models.xgboost_models import QuantTradingModels

            with st.spinner("Initializing model trainer..."):
                model_trainer = QuantTradingModels()

            # Prepare data
            with st.spinner("Preparing training data..."):
                # If features aren't calculated, calculate them now
                if 'features' not in st.session_state or st.session_state.features is None:
                    st.info("Calculating technical indicators...")
                    from features.technical_indicators import TechnicalIndicators
                    from utils.data_processing import DataProcessor
                    
                    features_data = TechnicalIndicators.calculate_all_indicators(st.session_state.data)
                    combined_data = DataProcessor.clean_data(features_data)
                    st.session_state.features = combined_data
                else:
                    combined_data = st.session_state.features.copy()

                # Ensure we have the required OHLC columns for target creation
                if not all(col in combined_data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    # Add OHLC data if missing
                    for col in ['Open', 'High', 'Low', 'Close']:
                        if col in st.session_state.data.columns and col not in combined_data.columns:
                            combined_data[col] = st.session_state.data[col]

                # Validate data
                if len(combined_data) < 100:
                    st.error("❌ Insufficient data for training. Need at least 100 rows.")
                    st.stop()

                st.info(f"📊 Training data prepared: {len(combined_data)} rows with {len(combined_data.columns)} features")

            # Train the model
            with st.spinner("Training volatility model..."):
                selected_models = ['volatility']
                training_results = model_trainer.train_selected_models(
                    combined_data, 
                    selected_models,
                    train_split
                )

            # Store results in session state
            st.session_state.trained_models = training_results
            st.session_state.model_trainer = model_trainer

            # Display results
            st.success("🎉 Model training completed!")

            if 'volatility' in training_results and training_results['volatility'] is not None:
                result = training_results['volatility']
                metrics = result.get('metrics', {})

                st.subheader("📊 Training Results")

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
                if 'feature_importance' in result:
                    st.subheader("🎯 Feature Importance")

                    importance_data = result['feature_importance']
                    if importance_data:
                        importance_df = pd.DataFrame(
                            list(importance_data.items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)

                        # Show top 10 features
                        st.dataframe(importance_df.head(10), use_container_width=True)

                        # Feature importance chart
                        import plotly.express as px

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
                        st.info("Feature importance data not available")

                # Individual model performance
                st.subheader("🔧 Individual Model Performance")

                model_metrics = {}
                for key, value in metrics.items():
                    if '_mse' in key or '_mae' in key:
                        model_name = key.split('_')[0]
                        metric_type = key.split('_')[1]

                        if model_name not in model_metrics:
                            model_metrics[model_name] = {}
                        model_metrics[model_name][metric_type] = value

                if model_metrics:
                    cols = st.columns(len(model_metrics))
                    for i, (model_name, model_metric) in enumerate(model_metrics.items()):
                        with cols[i]:
                            st.write(f"**{model_name.upper()}**")
                            if 'mse' in model_metric:
                                st.metric("MSE", f"{model_metric['mse']:.4f}")
                            if 'mae' in model_metric:
                                st.metric("MAE", f"{model_metric['mae']:.4f}")

                st.success("✅ Volatility model is ready for predictions!")
            else:
                st.error("❌ Failed to train volatility model")

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

# Model status
st.header("📈 Model Status")

if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
    if 'volatility' in st.session_state.trained_models and st.session_state.trained_models['volatility'] is not None:
        st.success("✅ Volatility Model: Trained and Ready")

        # Show model info
        result = st.session_state.trained_models['volatility']
        if 'metrics' in result:
            metrics = result['metrics']
            rmse = metrics.get('rmse', 0)
            st.info(f"📊 Model RMSE: {rmse:.4f}")
    else:
        st.warning("⚠️ Volatility Model: Not trained")
else:
    st.info("ℹ️ No models trained yet")

# Export trained models
if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
    st.header("💾 Export Models")

    if st.button("📥 Save Models to Database"):
        try:
            model_trainer = st.session_state.model_trainer
            model_trainer._save_models_to_database()
            st.success("✅ Models saved to database successfully!")
        except Exception as e:
            st.error(f"❌ Failed to save models: {str(e)}")