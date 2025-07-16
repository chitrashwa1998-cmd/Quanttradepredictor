import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Model Training")
st.markdown("Train prediction models using your processed data.")

# Dataset Selection
st.header("📊 Dataset Selection")

try:
    from utils.database_adapter import get_trading_database
    db = get_trading_database()
    datasets = db.get_dataset_list()
    
    if datasets:
        dataset_names = [d['name'] for d in datasets]
        dataset_info = {d['name']: f"{d['name']} ({d['rows']} rows)" for d in datasets}
        
        # Default to training_dataset if available, otherwise first dataset
        default_index = 0
        if "training_dataset" in dataset_names:
            default_index = dataset_names.index("training_dataset")
        
        selected_dataset = st.selectbox(
            "Select Dataset for Training:",
            options=dataset_names,
            format_func=lambda x: dataset_info[x],
            index=default_index,
            help="Choose which dataset to use for model training"
        )
        
        # Load selected dataset
        if st.button("🔄 Load Selected Dataset", type="primary"):
            selected_data = db.load_ohlc_data(selected_dataset)
            if selected_data is not None:
                st.session_state.data = selected_data
                st.success(f"✅ Loaded {selected_dataset}: {len(selected_data)} rows")
                st.rerun()
            else:
                st.error(f"❌ Failed to load {selected_dataset}")
        
        # Show current dataset info
        if hasattr(st.session_state, 'data') and st.session_state.data is not None:
            st.info(f"📈 Current dataset: {len(st.session_state.data)} rows loaded")
        
    else:
        st.warning("⚠️ No datasets found in database.")
        
except Exception as e:
    st.error(f"❌ Error loading datasets: {str(e)}")

# Check if data is available and prioritize training dataset
if 'data' not in st.session_state or st.session_state.data is None:
    # Try to load the training dataset automatically
    try:
        from utils.database_adapter import get_trading_database
        db = get_trading_database()
        training_data = db.load_ohlc_data("training_dataset")
        
        if training_data is not None and len(training_data) > 0:
            st.session_state.data = training_data
            st.info(f"✅ Automatically loaded training dataset: {len(training_data)} rows")
        else:
            st.error("❌ No training data available. Please upload data first in the Data Upload page.")
            st.stop()
    except Exception as e:
        st.error("❌ Could not load training data. Please upload data first in the Data Upload page.")
        st.stop()

# Feature Engineering Section will be handled within each model tab

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
    n_estimators = st.selectbox("Number of Estimators", [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], index=1,
                               help="Number of trees in the ensemble")

col1, col2 = st.columns(2)
with col1:
    st.info(f"Training: {int(train_split*100)}% | Testing: {int((1-train_split)*100)}%")
with col2:
    st.info(f"Max Depth: {max_depth} | Estimators: {n_estimators}")

# Model Selection and Training
st.header("🎯 Model Selection")

tab1, tab2, tab3, tab4 = st.tabs(["Volatility Model", "Direction Model", "Profit Probability Model", "Reversal Model"])

# Volatility Model Tab
with tab1:
    st.subheader("📈 Volatility Prediction Model")
    st.markdown("Predicts future market volatility using technical indicators.")
    
    # Volatility model features section
    st.subheader("Volatility Model Features")
    
    if 'features' not in st.session_state or st.session_state.features is None:
        st.warning("⚠️ Volatility features not calculated yet.")
        
        if st.button("🔧 Calculate Technical Indicators", type="primary", key="calc_volatility_features"):
            with st.spinner("Calculating volatility-specific technical indicators..."):
                try:
                    from features.technical_indicators import TechnicalIndicators
                    from utils.data_processing import DataProcessor
                    
                    # Calculate technical indicators for volatility
                    features_data = TechnicalIndicators.calculate_all_indicators(st.session_state.data)
                    
                    # Clean the data
                    features_clean = DataProcessor.clean_data(features_data)
                    
                    # Store in session state
                    st.session_state.features = features_clean
                    
                    st.success("✅ Volatility technical indicators calculated successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error calculating volatility indicators: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
    else:
        st.success("✅ Volatility features ready")
        
        # Show volatility feature summary
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
        
        # Show sample of volatility features
        with st.expander("View Volatility Feature Sample"):
            st.dataframe(st.session_state.features.head(10), use_container_width=True)
    
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
                        train_split
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
    
    # Direction model features section
    st.subheader("Direction Model Features")
    
    if 'direction_features' not in st.session_state or st.session_state.direction_features is None:
        st.warning("⚠️ Direction features not calculated yet.")
        
        if st.button("🔧 Calculate Technical Indicators", type="primary", key="calc_direction_features"):
            with st.spinner("Calculating direction-specific technical indicators..."):
                try:
                    from features.direction_technical_indicators import DirectionTechnicalIndicators
                    
                    # Calculate direction indicators directly
                    st.info("Starting direction indicator calculation...")
                    direction_features = DirectionTechnicalIndicators.calculate_all_direction_indicators(st.session_state.data)
                    
                    st.session_state.direction_features = direction_features
                    st.success("✅ Direction technical indicators calculated successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error calculating direction indicators: {str(e)}")
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"Full error: {error_details}")
                    
                    # Try fallback calculation
                    try:
                        st.warning("Attempting fallback calculation...")
                        from features.direction_technical_indicators import DirectionTechnicalIndicators
                        
                        # Calculate only basic direction indicators
                        direction_features = DirectionTechnicalIndicators.calculate_direction_indicators(st.session_state.data)
                        st.session_state.direction_features = direction_features
                        st.success("✅ Basic direction indicators calculated successfully!")
                        st.rerun()
                        
                    except Exception as e2:
                        st.error(f"❌ Fallback also failed: {str(e2)}")
                        with st.expander("Show fallback error details"):
                            st.code(traceback.format_exc())
    else:
        st.success("✅ Direction features ready")
        
        # Show direction feature summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", len(st.session_state.direction_features.columns))
        with col2:
            st.metric("Data Points", len(st.session_state.direction_features))
        with col3:
            # Count direction-specific features (engineered features)
            ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            direction_feature_cols = [col for col in st.session_state.direction_features.columns if col not in ohlc_cols]
            st.metric("Engineered Features", len(direction_feature_cols))
        
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
                    
                    # Store direction models separately for predictions
                    if 'direction_trained_models' not in st.session_state:
                        st.session_state.direction_trained_models = {}
                    st.session_state.direction_trained_models['direction'] = direction_model
                    
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

# Profit Probability Model Tab
with tab3:
    st.subheader("💰 Profit Probability Prediction Model")
    st.markdown("Predicts the likelihood of profitable trades within the next 5 periods.")
    
    # Profit probability model features section
    st.subheader("Profit Probability Model Features")
    
    if 'profit_prob_features' not in st.session_state or st.session_state.profit_prob_features is None:
        st.warning("⚠️ Profit probability features not calculated yet.")
        
        if st.button("🔧 Calculate Technical Indicators", type="primary", key="calc_profit_prob_features"):
            with st.spinner("Calculating profit probability-specific technical indicators..."):
                try:
                    from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
                    from features.profit_probability_custom_engineered import add_custom_profit_features
                    from features.profit_probability_lagged_features import add_lagged_features_profit_prob
                    from features.profit_probability_time_context import add_time_context_features_profit_prob
                    
                    # Calculate all profit probability features
                    st.info("Starting profit probability feature calculation...")
                    profit_prob_features = ProfitProbabilityTechnicalIndicators.calculate_all_profit_probability_indicators(st.session_state.data)
                    
                    # Add custom engineered features
                    profit_prob_features = add_custom_profit_features(profit_prob_features)
                    
                    # Add lagged features
                    profit_prob_features = add_lagged_features_profit_prob(profit_prob_features)
                    
                    # Add time/context features
                    profit_prob_features = add_time_context_features_profit_prob(profit_prob_features)
                    
                    st.session_state.profit_prob_features = profit_prob_features
                    st.success("✅ Profit probability features calculated successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error calculating profit probability features: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
    else:
        st.success("✅ Profit probability features ready")
        
        # Show profit probability feature summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", len(st.session_state.profit_prob_features.columns))
        with col2:
            st.metric("Data Points", len(st.session_state.profit_prob_features))
        with col3:
            # Count profit probability-specific features
            ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']
            profit_prob_feature_cols = [col for col in st.session_state.profit_prob_features.columns if col not in ohlc_cols]
            st.metric("Engineered Features", len(profit_prob_feature_cols))
        
        # Show sample of profit probability features
        with st.expander("View Profit Probability Feature Sample"):
            st.dataframe(st.session_state.profit_prob_features.head(10), use_container_width=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("This model predicts the probability of making a profit within the next 5 trading periods.")
    
    with col2:
        if st.button("💰 Train Profit Probability Model", type="primary", key="train_profit_prob"):
            with st.spinner("Training profit probability model..."):
                try:
                    # Initialize profit probability model
                    from models.profit_probability_model import ProfitProbabilityModel
                    
                    profit_prob_model = ProfitProbabilityModel()
                    
                    # Use pre-calculated profit probability features if available, otherwise calculate them
                    if 'profit_prob_features' in st.session_state and st.session_state.profit_prob_features is not None:
                        profit_prob_features = st.session_state.profit_prob_features
                    else:
                        st.info("Calculating profit probability-specific features...")
                        profit_prob_features = profit_prob_model.prepare_features(st.session_state.data)
                        st.session_state.profit_prob_features = profit_prob_features
                    
                    # Create profit probability target
                    profit_prob_target = profit_prob_model.create_target(st.session_state.data)
                    
                    # Validate data
                    if len(profit_prob_features) < 100:
                        st.error("❌ Insufficient data for training. Need at least 100 rows.")
                        st.stop()
                    
                    st.info(f"📊 Profit probability data: {len(profit_prob_features)} samples with {len(profit_prob_features.columns)} features")
                    
                    # Train profit probability model with configuration parameters
                    training_result = profit_prob_model.train(
                        profit_prob_features, 
                        profit_prob_target, 
                        train_split
                    )
                    
                    # Store results
                    if 'trained_models' not in st.session_state:
                        st.session_state.trained_models = {}
                    st.session_state.trained_models['profit_probability'] = training_result
                    
                    # Store profit probability models separately for predictions
                    if 'profit_prob_trained_models' not in st.session_state:
                        st.session_state.profit_prob_trained_models = {}
                    st.session_state.profit_prob_trained_models['profit_probability'] = profit_prob_model
                    
                    # Display results
                    if training_result is not None:
                        metrics = training_result.get('metrics', {})
                        
                        st.success("✅ Profit probability model trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            accuracy = metrics.get('accuracy', 0)
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        with col2:
                            classification_metrics = metrics.get('classification_report', {})
                            precision = classification_metrics.get('weighted avg', {}).get('precision', 0)
                            st.metric("Precision", f"{precision:.2%}")
                        with col3:
                            recall = classification_metrics.get('weighted avg', {}).get('recall', 0)
                            st.metric("Recall", f"{recall:.2%}")
                        
                        # Feature importance for profit probability model
                        if 'feature_importance' in training_result and training_result['feature_importance']:
                            with st.expander("📊 Profit Probability Feature Importance"):
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
                                    title="Top 10 Profit Probability Features"
                                )
                                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Failed to train profit probability model")
                        
                except Exception as e:
                    st.error(f"❌ Profit probability training failed: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())

# Reversal Model Tab
with tab4:
    st.subheader("🔄 Reversal Prediction Model")
    st.markdown("Predicts market reversal points using specialized technical indicators.")
    
    # Reversal model features section
    st.subheader("Reversal Model Features")
    
    if 'reversal_features' not in st.session_state or st.session_state.reversal_features is None:
        st.warning("⚠️ Reversal features not calculated yet.")
        
        if st.button("🔧 Calculate Comprehensive Reversal Features", type="primary", key="calc_reversal_features"):
            with st.spinner("Calculating comprehensive reversal features..."):
                try:
                    from models.reversal_model import ReversalModel
                    
                    # Use comprehensive feature preparation
                    st.info("Starting comprehensive reversal feature calculation...")
                    reversal_model = ReversalModel()
                    reversal_features = reversal_model.prepare_features(st.session_state.data)
                    
                    st.session_state.reversal_features = reversal_features
                    st.success(f"✅ Comprehensive reversal features calculated successfully! Generated {len(reversal_features.columns)} features.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error calculating reversal features: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
    else:
        st.success("✅ Reversal features ready")
        
        # Show reversal feature summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", len(st.session_state.reversal_features.columns))
        with col2:
            st.metric("Data Points", len(st.session_state.reversal_features))
        with col3:
            # Count reversal-specific features
            ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']
            reversal_feature_cols = [col for col in st.session_state.reversal_features.columns if col not in ohlc_cols]
            st.metric("Engineered Features", len(reversal_feature_cols))
        
        # Show sample of reversal features
        with st.expander("View Reversal Feature Sample"):
            st.dataframe(st.session_state.reversal_features.head(10), use_container_width=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("This model predicts potential market reversal points using specialized indicators like RSI, MACD, and momentum signals.")
    
    with col2:
        if st.button("🔄 Train Reversal Model", type="primary", key="train_reversal"):
            with st.spinner("Training reversal model..."):
                try:
                    # Initialize reversal model
                    from models.reversal_model import ReversalModel
                    
                    reversal_model = ReversalModel()
                    
                    # Use pre-calculated reversal features if available, otherwise calculate them
                    if 'reversal_features' in st.session_state and st.session_state.reversal_features is not None:
                        reversal_features = st.session_state.reversal_features
                    else:
                        st.info("Calculating reversal-specific features...")
                        reversal_features = reversal_model.prepare_features(st.session_state.data)
                        st.session_state.reversal_features = reversal_features
                    
                    # Create reversal target
                    reversal_target = reversal_model.create_target(st.session_state.data)
                    
                    # Validate data
                    if len(reversal_features) < 100:
                        st.error("❌ Insufficient data for training. Need at least 100 rows.")
                        st.stop()
                    
                    st.info(f"📊 Reversal data: {len(reversal_features)} samples with {len(reversal_features.columns)} features")
                    
                    # Train reversal model with configuration parameters
                    training_result = reversal_model.train(
                        reversal_features, 
                        reversal_target, 
                        train_split,
                        max_depth=max_depth,
                        n_estimators=n_estimators
                    )
                    
                    # Store results
                    if 'trained_models' not in st.session_state:
                        st.session_state.trained_models = {}
                    st.session_state.trained_models['reversal'] = training_result
                    
                    # Store reversal models separately for predictions
                    if 'reversal_trained_models' not in st.session_state:
                        st.session_state.reversal_trained_models = {}
                    st.session_state.reversal_trained_models['reversal'] = reversal_model
                    
                    # Display results
                    if training_result is not None:
                        metrics = training_result.get('metrics', {})
                        
                        st.success("✅ Reversal model trained successfully!")
                        
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
                        
                        # Feature importance for reversal model
                        if 'feature_importance' in training_result and training_result['feature_importance']:
                            with st.expander("📊 Reversal Feature Importance"):
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
                                    title="Top 10 Reversal Features"
                                )
                                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Failed to train reversal model")
                        
                except Exception as e:
                    st.error(f"❌ Reversal training failed: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())

# Model Status Section
st.header("📊 Model Status")

if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col3:
        st.subheader("Profit Probability Model")
        if 'profit_probability' in st.session_state.trained_models and st.session_state.trained_models['profit_probability']:
            result = st.session_state.trained_models['profit_probability']
            metrics = result.get('metrics', {})
            accuracy = metrics.get('accuracy', 0)
            st.success(f"✅ Trained - Accuracy: {accuracy:.2%}")
        else:
            st.warning("⚠️ Not trained")
    
    with col4:
        st.subheader("Reversal Model")
        if 'reversal' in st.session_state.trained_models and st.session_state.trained_models['reversal']:
            result = st.session_state.trained_models['reversal']
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
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            if st.button("📥 Save Volatility Model", disabled='volatility' not in st.session_state.trained_models):
                try:
                    if hasattr(st.session_state, 'volatility_trainer'):
                        st.session_state.volatility_trainer._save_models_to_database()
                        st.success("✅ Volatility model saved to database!")
                    else:
                        st.error("❌ Volatility trainer not available")
                except Exception as e:
                    st.error(f"❌ Failed to save volatility model: {str(e)}")
        
        with subcol2:
            if st.button("📥 Save Direction Model", disabled='direction' not in st.session_state.trained_models):
                try:
                    from utils.database_adapter import get_trading_database
                    db = get_trading_database()
                    
                    # Save direction model object for persistence
                    if ('direction_trained_models' in st.session_state and 
                        'direction' in st.session_state.direction_trained_models and
                        st.session_state.direction_trained_models['direction'] is not None):
                        
                        direction_model = st.session_state.direction_trained_models['direction']
                        
                        # Prepare model for database save
                        models_to_save = {
                            'direction': {
                                'ensemble': direction_model.model,
                                'scaler': direction_model.scaler,
                                'feature_names': getattr(direction_model, 'selected_features', []),
                                'task_type': 'classification',
                                'metrics': st.session_state.trained_models.get('direction', {}).get('metrics', {}),
                                'feature_importance': st.session_state.trained_models.get('direction', {}).get('feature_importance', {})
                            }
                        }
                        
                        success = db.save_trained_models(models_to_save)
                        if success:
                            st.success("✅ Direction model saved to database!")
                        else:
                            st.error("❌ Failed to save direction model to database")
                    else:
                        st.error("❌ Direction model not available")
                except Exception as e:
                    st.error(f"❌ Failed to save direction model: {str(e)}")
    
    with col2:
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            if st.button("📥 Save Profit Probability Model", disabled='profit_probability' not in st.session_state.trained_models):
                try:
                    from utils.database_adapter import get_trading_database
                    db = get_trading_database()
                    
                    # Save profit probability model object for persistence
                    if ('profit_prob_trained_models' in st.session_state and 
                        'profit_probability' in st.session_state.profit_prob_trained_models and
                        st.session_state.profit_prob_trained_models['profit_probability'] is not None):
                        
                        profit_model = st.session_state.profit_prob_trained_models['profit_probability']
                        
                        # Prepare model for database save
                        models_to_save = {
                            'profit_probability': {
                                'ensemble': profit_model.model,
                                'scaler': profit_model.scaler,
                                'feature_names': getattr(profit_model, 'feature_names', []),
                                'task_type': 'classification',
                                'metrics': st.session_state.trained_models.get('profit_probability', {}).get('metrics', {}),
                                'feature_importance': st.session_state.trained_models.get('profit_probability', {}).get('feature_importance', {})
                            }
                        }
                        
                        success = db.save_trained_models(models_to_save)
                        if success:
                            st.success("✅ Profit probability model saved to database!")
                        else:
                            st.error("❌ Failed to save profit probability model to database")
                    else:
                        st.error("❌ Profit probability model not available")
                except Exception as e:
                    st.error(f"❌ Failed to save profit probability model: {str(e)}")
        
        with subcol2:
            if st.button("📥 Save Reversal Model", disabled='reversal' not in st.session_state.trained_models):
                try:
                    from utils.database_adapter import get_trading_database
                    db = get_trading_database()
                    
                    # Save reversal model object for persistence
                    if ('reversal_trained_models' in st.session_state and 
                        'reversal' in st.session_state.reversal_trained_models and
                        st.session_state.reversal_trained_models['reversal'] is not None):
                        
                        reversal_model = st.session_state.reversal_trained_models['reversal']
                        
                        # Prepare model for database save
                        models_to_save = {
                            'reversal': {
                                'ensemble': reversal_model.model,
                                'scaler': reversal_model.scaler,
                                'feature_names': getattr(reversal_model, 'feature_names', []),
                                'task_type': 'classification',
                                'metrics': st.session_state.trained_models.get('reversal', {}).get('metrics', {}),
                                'feature_importance': st.session_state.trained_models.get('reversal', {}).get('feature_importance', {})
                            }
                        }
                        
                        success = db.save_trained_models(models_to_save)
                        if success:
                            st.success("✅ Reversal model saved to database!")
                        else:
                            st.error("❌ Failed to save reversal model to database")
                    else:
                        st.error("❌ Reversal model not available")
                except Exception as e:
                    st.error(f"❌ Failed to save reversal model: {str(e)}")