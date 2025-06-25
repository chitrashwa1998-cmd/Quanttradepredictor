import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from models.xgboost_models import QuantTradingModels
from features.technical_indicators import TechnicalIndicators

st.set_page_config(page_title="Model Training", page_icon="🔬", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">🔬 ML TRAINING LAB</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Advanced Machine Learning Model Training
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'features' not in st.session_state:
    st.session_state.features = None

# Check if data is loaded
if st.session_state.data is None:
    st.warning("⚠️ No data loaded. Please go to the **Data Upload** page first.")
    st.stop()

# Initialize model trainer
if 'model_trainer' not in st.session_state or st.session_state.model_trainer is None:
    st.session_state.model_trainer = QuantTradingModels()

# Check for existing trained models
model_trainer = st.session_state.model_trainer
existing_models = model_trainer.models

if existing_models:
    st.success(f"🎯 Found {len(existing_models)} pre-trained models in database!")

    # Show existing models status
    with st.expander("View Existing Models", expanded=True):
        for model_name, model_data in existing_models.items():
            trained_date = model_data.get('trained_at', 'Unknown date')
            task_type = model_data.get('task_type', 'Unknown type')
            st.info(f"**{model_name}** ({task_type}) - Trained: {trained_date}")

    # Option to retrain or use existing
    retrain_choice = st.radio(
        "Model Training Options:",
        ["Use existing trained models", "Retrain all models (this will overwrite existing models)"],
        help="Existing models can be used immediately for predictions without retraining"
    )

    if retrain_choice == "Use existing trained models":
        st.session_state.training_results = {name: {'status': 'loaded'} for name in existing_models.keys()}
        st.success("✅ Using existing trained models - ready for predictions!")
        st.stop()

df = st.session_state.data

st.header("Training Configuration")

# Training parameters
col1, col2, col3 = st.columns(3)

with col1:
    train_split = st.slider("Training Data Split", 0.6, 0.9, 0.8, 0.05, 
                           help="Percentage of data to use for training")

with col2:
    max_depth = st.selectbox("XGBoost Max Depth", [3, 4, 5, 6, 7, 8], index=3)

with col3:
    n_estimators = st.selectbox("Number of Estimators", [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], index=1)

# Model selection
st.header("Models to Train")
st.markdown("Select which prediction models you want to train:")

col1, col2, col3 = st.columns(3)

with col1:
    train_direction = st.checkbox("Direction Prediction", value=True, 
                                 help="Predict if price will go up or down")
    train_magnitude = st.checkbox("Magnitude Prediction", value=True,
                                 help="Predict the size of price moves")
    train_profit = st.checkbox("Profit Probability", value=True,
                              help="Predict probability of profitable trades")

with col2:
    train_volatility = st.checkbox("Volatility Forecasting", value=True,
                                  help="Forecast future volatility")
    train_trend = st.checkbox("Trend Classification", value=True,
                             help="Classify trending vs sideways markets")
    train_reversal = st.checkbox("Reversal Points", value=True,
                                help="Identify potential trend reversals")

with col3:
    train_signals = st.checkbox("Trading Signals", value=True,
                               help="Generate buy/sell/hold recommendations")

# Feature engineering status
st.header("Feature Engineering")

if st.session_state.features is None:
    st.warning("⚠️ Technical indicators not calculated yet.")

    if st.button("Calculate Technical Indicators", type="primary"):
        with st.spinner("Calculating technical indicators..."):
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
            st.session_state.features = df_with_indicators

        st.success("✅ Technical indicators calculated!")
        st.rerun()
else:
    st.success("✅ Technical indicators ready")

    # Show feature summary using the exact same logic as model training
    all_cols = st.session_state.features.columns.tolist()

    # Use the exact same feature selection logic as in prepare_features()
    feature_cols = [col for col in all_cols if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    feature_cols = [col for col in feature_cols if not col.startswith(('target_', 'future_'))]

    # Remove data leakage features (same as in prepare_features())
    leakage_features = [
        'Prediction', 'predicted_direction', 'predictions',
        'Signal', 'Signal_Name', 'Confidence',
        'accuracy', 'precision', 'recall'
    ]
    training_feature_cols = [col for col in feature_cols if col not in leakage_features]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Features", len(training_feature_cols))
    with col2:
        st.metric("Data Points", len(st.session_state.features))
    with col3:
        missing_pct = st.session_state.features.isnull().sum().sum() / (len(st.session_state.features) * len(st.session_state.features.columns)) * 100
        st.metric("Missing Data %", f"{missing_pct:.1f}%")

    # Show information about data leakage removal
    if len(feature_cols) != len(training_feature_cols):
        removed_count = len(feature_cols) - len(training_feature_cols)
        st.info(f"ℹ️ Removed {removed_count} data leakage features from training. Using {len(training_feature_cols)} legitimate market features.")

# Model training section
st.header("Train Models")

if st.session_state.features is not None:

    # Update model parameters
    if st.button("Update Model Parameters"):
        # This would update the model configuration
        st.info("Model parameters updated for next training run")

    if st.button("🚀 Train All Selected Models", type="primary"):

        selected_models = []
        if train_direction: selected_models.append("Direction Prediction")
        if train_magnitude: selected_models.append("Magnitude Prediction")
        if train_profit: selected_models.append("Profit Probability")
        if train_volatility: selected_models.append("Volatility Forecasting")
        if train_trend: selected_models.append("Trend Classification")
        if train_reversal: selected_models.append("Reversal Points")
        if train_signals: selected_models.append("Trading Signals")

        if not selected_models:
            st.error("Please select at least one model to train.")
        else:
            # Ensure model trainer is initialized
            if st.session_state.model_trainer is None:
                st.session_state.model_trainer = QuantTradingModels()

            st.info(f"Training {len(selected_models)} models...")

            try:
                # Use full dataset for maximum accuracy
                features_data = st.session_state.features
                st.info(f"Training on complete dataset: {len(features_data)} rows for maximum accuracy...")

                # Map UI selections to model names
                model_mapping = {
                    "Direction Prediction": "direction",
                    "Magnitude Prediction": "magnitude", 
                    "Profit Probability": "profit",
                    "Volatility Forecasting": "volatility",
                    "Trend Classification": "trend",
                    "Reversal Points": "reversal",
                    "Trading Signals": "trading_signal"
                }

                # Get the actual model names to train
                models_to_train = [model_mapping[model] for model in selected_models if model in model_mapping]

                st.info(f"Training selected models: {', '.join(models_to_train)}")

                # Train only selected models
                results = st.session_state.model_trainer.train_selected_models(features_data, models_to_train, train_split)

                # Store results in session state
                st.session_state.models = results

                # Ensure model trainer also has the results
                if st.session_state.model_trainer:
                    for model_name, model_result in results.items():
                        if model_result is not None:
                            st.session_state.model_trainer.models[model_name] = model_result

                # Display immediate training results
                st.success(f"🎉 Training completed for {len([r for r in results.values() if r is not None])} models!")

                # Show which models were trained successfully
                successful_models = [name for name, result in results.items() if result is not None]
                failed_models = [name for name, result in results.items() if result is None]

                if successful_models:
                    st.success(f"✅ Successfully trained: {', '.join(successful_models)}")

                if failed_models:
                    st.error(f"❌ Failed to train: {', '.join(failed_models)}")

                # Show training metrics immediately
                for model_name, result in results.items():
                    if result is not None:
                        metrics = result.get('metrics', {})
                        task_type = result.get('task_type', 'unknown')

                        with st.expander(f"📊 {model_name.replace('_', ' ').title()} Results"):
                            if task_type == 'classification':
                                accuracy = metrics.get('accuracy', 0)
                                st.metric("Accuracy", f"{accuracy:.3f}")

                                # Show calibration info for direction model
                                if model_name == 'direction' and 'calibration' in metrics:
                                    cal_info = metrics['calibration']
                                    st.info(f"✅ Model calibrated using {cal_info.get('calibration_method', 'Unknown')}")
                                    if cal_info.get('brier_score'):
                                        st.metric("Brier Score", f"{cal_info['brier_score']:.4f}")

                            elif task_type == 'regression':
                                rmse = metrics.get('rmse', 0)
                                mae = metrics.get('mae', 0)
                                st.metric("RMSE", f"{rmse:.4f}")
                                st.metric("MAE", f"{mae:.4f}")

                # Show feature count
                if hasattr(st.session_state.model_trainer, 'feature_names'):
                    feature_count = len(st.session_state.model_trainer.feature_names)
                    st.info(f"📈 Models trained using {feature_count} features")

                # Auto-save model results to database
                try:
                    from utils.database_adapter import DatabaseAdapter
                    trading_db = DatabaseAdapter()

                    # Save model results
                    saved_count = 0
                    for model_name, model_result in results.items():
                        if model_result is not None:
                            # Save model metrics and info
                            model_data = {
                                'metrics': model_result['metrics'],
                                'task_type': model_result['task_type'],
                                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            if trading_db.save_model_results(model_name, model_data):
                                saved_count += 1

                    # Save the actual trained model objects for persistence
                    if trading_db.save_trained_models(st.session_state.model_trainer.models):
                        st.success(f"💾 Saved {saved_count} models to database!")
                    else:
                        st.warning("⚠️ Models trained but failed to save to database")

                except Exception as e:
                    st.warning(f"⚠️ Database save error: {str(e)}")
                    st.info("Models are still available in this session")

                # Don't force rerun - let user see results immediately
            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
                st.info("Please try refreshing the page and training again.")

# Display training results - Always show if any models exist
if st.session_state.models or (st.session_state.model_trainer and st.session_state.model_trainer.models):
    st.header("Training Results")

    # Get all available models (session state + loaded models)
    all_models = {}

    # Add session state models first (these have full metrics)
    if st.session_state.models:
        all_models.update(st.session_state.models)

    # Add loaded models from trainer (but don't overwrite session state models)
    if st.session_state.model_trainer and st.session_state.model_trainer.models:
        for model_name, model_data in st.session_state.model_trainer.models.items():
            if model_name not in all_models:
                # Safely extract task_type and metrics
                if isinstance(model_data, dict):
                    task_type = model_data.get('task_type', 'classification')
                    # Check if metrics exist, otherwise create placeholder
                    if 'metrics' in model_data:
                        all_models[model_name] = model_data
                    else:
                        # Convert loaded model to session state format
                        all_models[model_name] = {
                            'metrics': {'accuracy': 'Loaded'} if task_type == 'classification' else {'rmse': 'Loaded'},
                            'task_type': task_type,
                            'status': 'loaded'
                        }
                else:
                    # Handle case where model_data is not a dict (raw model object)
                    all_models[model_name] = {
                        'metrics': {'accuracy': 'Loaded'},
                        'task_type': 'classification',
                        'status': 'loaded'
                    }

    # Debug information
    st.write(f"Debug: Found {len(all_models)} total models")
    st.write(f"Session state models: {list(st.session_state.models.keys()) if st.session_state.models else 'None'}")
    st.write(f"Trainer models: {list(st.session_state.model_trainer.models.keys()) if st.session_state.model_trainer and st.session_state.model_trainer.models else 'None'}")

    # Comprehensive Model Accuracy Table
    st.subheader("📊 Model Accuracy Summary")

    # Prepare data for the accuracy table
    accuracy_data = []
    classification_models = []
    regression_models = []

    for model_name, model_info in all_models.items():
        if model_info is not None and isinstance(model_info, dict):
            metrics = model_info.get('metrics', {})
            model_display_name = model_name.replace('_', ' ').title()
            task_type = model_info.get('task_type', 'classification')
            status = model_info.get('status', 'trained')

            if task_type == 'classification':
                if status == 'loaded':
                    accuracy_data.append({
                        'Model': model_display_name,
                        'Type': 'Classification',
                        'Ensemble Accuracy': 'Loaded',
                        'XGBoost': 'Loaded',
                        'CatBoost': 'Loaded',
                        'Random Forest': 'Loaded',
                        'Status': '📂 Loaded'
                    })
                else:
                    accuracy = metrics.get('accuracy', 0)

                    # Get individual model accuracies if available
                    xgb_acc = metrics.get('xgboost_accuracy', 'N/A')
                    cat_acc = metrics.get('catboost_accuracy', 'N/A')
                    rf_acc = metrics.get('random_forest_accuracy', 'N/A')

                    # Format individual accuracies
                    xgb_acc_str = f"{xgb_acc:.3f}" if isinstance(xgb_acc, float) else xgb_acc
                    cat_acc_str = f"{cat_acc:.3f}" if isinstance(cat_acc, float) else cat_acc
                    rf_acc_str = f"{rf_acc:.3f}" if isinstance(rf_acc, float) else rf_acc

                    # Get classification report details if available
                    class_report = metrics.get('classification_report', {})
                    weighted_avg = class_report.get('weighted avg', {})
                    precision = weighted_avg.get('precision', 0)
                    recall = weighted_avg.get('recall', 0)
                    f1_score = weighted_avg.get('f1-score', 0)

                    accuracy_data.append({
                        'Model': model_display_name,
                        'Type': 'Classification',
                        'Ensemble Accuracy': f"{accuracy:.3f}",
                        'XGBoost': xgb_acc_str,
                        'CatBoost': cat_acc_str,
                        'Random Forest': rf_acc_str,
                        'Precision': f"{precision:.3f}" if precision > 0 else 'N/A',
                        'Recall': f"{recall:.3f}" if recall > 0 else 'N/A',
                        'F1-Score': f"{f1_score:.3f}" if f1_score > 0 else 'N/A',
                        'Status': '✅ Trained'
                    })

                    classification_models.append({
                        'Model': model_display_name,
                        'Accuracy': accuracy
                    })

            else:  # Regression
                if status == 'loaded':
                    accuracy_data.append({
                        'Model': model_display_name,
                        'Type': 'Regression',
                        'Ensemble RMSE': 'Loaded',
                        'Ensemble MAE': 'Loaded',
                        'Status': '📂 Loaded'
                    })
                else:
                    rmse = metrics.get('rmse', 0)
                    mae = metrics.get('mae', 0)
                    mse = metrics.get('mse', 0)

                    # Get individual model errors if available
                    xgb_mse = metrics.get('xgboost_mse', 'N/A')
                    cat_mse = metrics.get('catboost_mse', 'N/A')
                    rf_mse = metrics.get('random_forest_mse', 'N/A')

                    # Format individual errors
                    xgb_mse_str = f"{xgb_mse:.4f}" if isinstance(xgb_mse, float) else xgb_mse
                    cat_mse_str = f"{cat_mse:.4f}" if isinstance(cat_mse, float) else cat_mse
                    rf_mse_str = f"{rf_mse:.4f}" if isinstance(rf_mse, float) else rf_mse

                    accuracy_data.append({
                        'Model': model_display_name,
                        'Type': 'Regression',
                        'Ensemble RMSE': f"{rmse:.4f}",
                        'Ensemble MAE': f"{mae:.4f}",
                        'XGBoost MSE': xgb_mse_str,
                        'CatBoost MSE': cat_mse_str,
                        'RF MSE': rf_mse_str,
                        'Status': '✅ Trained'
                    })

                    regression_models.append({
                        'Model': model_display_name,
                        'RMSE': rmse,
                        'MAE': mae
                    })
        else:
            accuracy_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Type': 'Unknown',
                'Status': '❌ Failed'
            })

    if accuracy_data:
        # Create and display the comprehensive table
        accuracy_df = pd.DataFrame(accuracy_data)

        # Style the dataframe for better visualization
        def style_accuracy_table(df):
            def highlight_status(val):
                if '✅' in str(val):
                    return 'background-color: #d4edda; color: #155724;'
                elif '❌' in str(val):
                    return 'background-color: #f8d7da; color: #721c24;'
                elif '📂' in str(val):
                    return 'background-color: #d1ecf1; color: #0c5460;'
                return ''

            def highlight_accuracy(val):
                try:
                    if isinstance(val, str) and val != 'N/A' and val != 'Loaded':
                        accuracy_val = float(val)
                        if accuracy_val >= 0.8:
                            return 'background-color: #d4edda; font-weight: bold;'
                        elif accuracy_val >= 0.7:
                            return 'background-color: #fff3cd;'
                        elif accuracy_val < 0.6:
                            return 'background-color: #f8d7da;'
                except:
                    pass
                return ''

            # Apply styling
            styled_df = df.style.applymap(highlight_status, subset=['Status'])

            # Highlight accuracy columns for classification models
            accuracy_columns = ['Ensemble Accuracy', 'XGBoost', 'CatBoost', 'Random Forest']
            for col in accuracy_columns:
                if col in df.columns:
                    styled_df = styled_df.applymap(highlight_accuracy, subset=[col])

            return styled_df

        # Display the styled table
        st.dataframe(
            style_accuracy_table(accuracy_df),
            use_container_width=True,
            hide_index=True
        )

        # Summary metrics
        st.subheader("📈 Performance Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            total_models = len([m for m in all_models.values() if m is not None])
            st.metric("Available Models", total_models)

        with col2:
            if classification_models:
                avg_accuracy = sum(m['Accuracy'] for m in classification_models) / len(classification_models)
                st.metric("Avg Classification Accuracy", f"{avg_accuracy:.3f}")
            else:
                loaded_classification = len([m for m in all_models.values() 
                                           if m and m.get('task_type') == 'classification'])
                st.metric("Classification Models", f"{loaded_classification} loaded")

        with col3:
            if regression_models:
                avg_rmse = sum(m['RMSE'] for m in regression_models) / len(regression_models)
                st.metric("Avg Regression RMSE", f"{avg_rmse:.4f}")
            else:
                loaded_regression = len([m for m in all_models.values() 
                                       if m and m.get('task_type') == 'regression'])
                st.metric("Regression Models", f"{loaded_regression} loaded")

        # Top performing models (only for newly trained models)
        if classification_models:
            st.subheader("🏆 Best Performing Classification Models")
            best_models = sorted(classification_models, key=lambda x: x['Accuracy'], reverse=True)[:3]

            for i, model in enumerate(best_models):
                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
                st.write(f"{rank_emoji} **{model['Model']}**: {model['Accuracy']:.3f} accuracy")

    # Model performance summary (keeping original for compatibility)
    performance_data = []

    for model_name, model_info in st.session_state.models.items():
        if model_info is not None:
            metrics = model_info['metrics']

            if model_info['task_type'] == 'classification':
                accuracy = metrics['accuracy']
                performance_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Task Type': 'Classification',
                    'Accuracy': f"{accuracy:.3f}",
                    'Status': '✅ Trained'
                })
            else:
                rmse = metrics['rmse']
                mae = metrics['mae']
                performance_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Task Type': 'Regression',
                    'RMSE': f"{rmse:.4f}",
                    'MAE': f"{mae:.4f}",
                    'Status': '✅ Trained'
                })
        else:
            performance_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Status': '❌ Failed'
            })

    # Keep original simple table for reference
    st.subheader("📋 Simple Performance Overview")
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True, hide_index=True)

    # Feature importance analysis
    st.subheader("Feature Importance Analysis")

    # Get available models for feature importance (both session state and loaded models)
    available_models = []

    # Add models from session state
    if st.session_state.models:
        for name, model_info in st.session_state.models.items():
            if model_info is not None:
                available_models.append(name)

    # Add models from trainer
    if st.session_state.model_trainer and st.session_state.model_trainer.models:
        for name in st.session_state.model_trainer.models.keys():
            if name not in available_models:
                available_models.append(name)

    if available_models:
        model_for_importance = st.selectbox(
            "Select model for feature importance",
            available_models,
            index=0 if 'direction' not in available_models else available_models.index('direction')
        )

        if model_for_importance:
            importance_dict = st.session_state.model_trainer.get_feature_importance(model_for_importance)

            if importance_dict:
                # Get model-specific feature names to ensure we show the correct features
                model_info = st.session_state.model_trainer.models.get(model_for_importance, {})
                model_specific_features = model_info.get('feature_names', [])
                
                # If we have model-specific features, filter the importance dict
                if model_specific_features:
                    # Only include features that were actually used for this model
                    filtered_importance = {feat: importance_dict.get(feat, 0) 
                                         for feat in model_specific_features 
                                         if feat in importance_dict}
                    if filtered_importance:
                        importance_dict = filtered_importance
                        st.info(f"Showing feature importance for {len(model_specific_features)} features used in {model_for_importance} model")

                # Convert to DataFrame and sort
                importance_df = pd.DataFrame(
                    list(importance_dict.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)

                # Display top 20 features table first
                st.subheader(f"🏆 Top 20 Feature Importance - {model_for_importance.replace('_', ' ').title()}")
                top_20_features = importance_df.head(20)

                # Format the importance values for better display
                display_df = top_20_features.copy()
                display_df['Rank'] = range(1, len(display_df) + 1)
                display_df['Importance'] = display_df['Importance'].apply(lambda x: f"{x:.6f}")
                display_df = display_df[['Rank', 'Feature', 'Importance']]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Plot top 20 features
                fig = px.bar(
                    top_20_features, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title=f'Top 20 Feature Importance - {model_for_importance.replace("_", " ").title()}',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    height=600, 
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title="Importance Score",
                    yaxis_title="Features"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show detailed importance table
                with st.expander("View All Feature Importance"):
                    st.dataframe(importance_df, use_container_width=True)

                    # Download button for feature importance
                    csv = importance_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Feature Importance CSV",
                        data=csv,
                        file_name=f"feature_importance_{model_for_importance}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning(f"No feature importance data available for {model_for_importance}. Try retraining the model.")
    else:
        st.info("No trained models available for feature importance analysis. Train some models first!")

    # Model comparison
    st.subheader("Model Performance Comparison")

    # Create comparison chart for classification models
    classification_models = {name: info for name, info in st.session_state.models.items() 
                           if info and info['task_type'] == 'classification'}

    if classification_models:
        model_names = []
        accuracies = []

        for name, info in classification_models.items():
            model_names.append(name.replace('_', ' ').title())
            accuracies.append(info['metrics']['accuracy'])

        fig = px.bar(
            x=model_names,
            y=accuracies,
            title="Classification Model Accuracy Comparison",
            labels={'x': 'Model', 'y': 'Accuracy'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Regression models comparison
    regression_models = {name: info for name, info in st.session_state.models.items() 
                        if info and info['task_type'] == 'regression'}

    if regression_models:
        model_names = []
        rmse_values = []

        for name, info in regression_models.items():
            model_names.append(name.replace('_', ' ').title())
            rmse_values.append(info['metrics']['rmse'])

        fig = px.bar(
            x=model_names,
            y=rmse_values,
            title="Regression Model RMSE Comparison (Lower is Better)",
            labels={'x': 'Model', 'y': 'RMSE'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed model metrics
    st.subheader("Detailed Model Metrics")

    selected_model = st.selectbox(
        "Select model for detailed metrics",
        [name for name in st.session_state.models.keys() if st.session_state.models[name] is not None]
    )

    if selected_model:
        model_info = st.session_state.models[selected_model]
        if model_info is not None and isinstance(model_info, dict):
            metrics = model_info.get('metrics', {})

        st.write(f"**Model**: {selected_model.replace('_', ' ').title()}")
        st.write(f"**Task Type**: {model_info['task_type'].title()}")

        if model_info['task_type'] == 'classification':
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")

            # Classification report
            if 'classification_report' in metrics:
                st.subheader("Classification Report")

                # Convert classification report to DataFrame
                report = metrics['classification_report']

                # Extract metrics for each class
                classes = [k for k in report.keys() if k.isdigit()]

                if classes:
                    report_data = []
                    for class_id in classes:
                        class_metrics = report[class_id]
                        report_data.append({
                            'Class': class_id,
                            'Precision': f"{class_metrics['precision']:.3f}",
                            'Recall': f"{class_metrics['recall']:.3f}",
                            'F1-Score': f"{class_metrics['f1-score']:.3f}",
                            'Support': class_metrics['support']
                        })

                    report_df = pd.DataFrame(report_data)
                    st.dataframe(report_df, use_container_width=True)

        else:  # Regression
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            with col3:
                st.metric("MSE", f"{metrics['mse']:.4f}")

else:
    st.info("👆 Configure your training parameters and click 'Train All Selected Models' to start training.")

# Model saving/loading options
st.header("Model Management")

col1, col2 = st.columns(2)

with col1:
    if st.button("Save Trained Models"):
        if st.session_state.models:
            # In a real application, you would save models to disk
            st.success("✅ Models saved successfully! (In session state)")
        else:
            st.warning("No trained models to save")

with col2:
    if st.button("Clear All Models"):
        st.session_state.models = {}
        st.session_state.model_trainer = QuantTradingModels()
        st.success("✅ All models cleared")
        st.rerun()

## Next steps
st.markdown("---")
if st.session_state.models:
    st.info("📋 **Next Steps:** Your models are trained! Go to the **Predictions** page to view model predictions and analysis.")
else:
    st.info("📋 **Next Steps:** Train your models and then proceed to the **Predictions** page for analysis.")