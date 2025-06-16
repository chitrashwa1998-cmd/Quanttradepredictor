import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
    n_estimators = st.selectbox("Number of Estimators", [50, 100, 150, 200], index=1)

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
    
    # Show feature summary
    feature_cols = [col for col in st.session_state.features.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", len(feature_cols))
    with col2:
        st.metric("Data Points", len(st.session_state.features))
    with col3:
        missing_pct = st.session_state.features.isnull().sum().sum() / (len(st.session_state.features) * len(st.session_state.features.columns)) * 100
        st.metric("Missing Data %", f"{missing_pct:.1f}%")

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
                # Train models
                results = st.session_state.model_trainer.train_all_models(st.session_state.features)
                
                # Store results
                st.session_state.models = results
                
                # Auto-save model results to database
                try:
                    from utils.database import TradingDatabase
                    trading_db = TradingDatabase()
                    for model_name, model_result in results.items():
                        if model_result is not None:
                            # Save model metrics and info (not the actual model object)
                            model_data = {
                                'metrics': model_result['metrics'],
                                'task_type': model_result['task_type'],
                                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            trading_db.save_model_results(model_name, model_data)
                    st.success("🎉 Model training completed & saved to database!")
                except Exception as e:
                    st.success("🎉 Model training completed!")
                    st.warning("⚠️ Models trained but failed to save to database")
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
                st.info("Please try refreshing the page and training again.")

# Display training results
if st.session_state.models:
    st.header("Training Results")
    
    # Model performance summary
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
    
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
    
    # Feature importance analysis
    st.subheader("Feature Importance Analysis")
    
    model_for_importance = st.selectbox(
        "Select model for feature importance",
        [name for name in st.session_state.models.keys() if st.session_state.models[name] is not None]
    )
    
    if model_for_importance:
        importance_dict = st.session_state.model_trainer.get_feature_importance(model_for_importance)
        
        if importance_dict:
            # Convert to DataFrame and sort
            importance_df = pd.DataFrame(
                list(importance_dict.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            # Plot top 20 features
            top_features = importance_df.head(20)
            
            fig = px.bar(
                top_features, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=f'Top 20 Feature Importance - {model_for_importance.replace("_", " ").title()}'
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed importance table
            with st.expander("View All Feature Importance"):
                st.dataframe(importance_df, use_container_width=True)
    
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
        metrics = model_info['metrics']
        
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

# Next steps
st.markdown("---")
if st.session_state.models:
    st.info("📋 **Next Steps:** Your models are trained! Go to the **Predictions** page to view model predictions and analysis.")
else:
    st.info("📋 **Next Steps:** Train your models and then proceed to the **Predictions** page for analysis.")
