import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None

# Check if data and models are available
if st.session_state.data is None:
    st.warning("⚠️ No data loaded. Please go to the **Data Upload** page first.")
    st.stop()

# Check for models in both session state and model trainer
has_models = False
if st.session_state.models:
    has_models = True
elif st.session_state.model_trainer and st.session_state.model_trainer.models:
    has_models = True
    # Load models from trainer into session state for compatibility
    st.session_state.models = {}
    for model_name, model_data in st.session_state.model_trainer.models.items():
        if isinstance(model_data, dict):
            st.session_state.models[model_name] = model_data
        else:
            # Create a basic model info structure
            st.session_state.models[model_name] = {
                'metrics': {'accuracy': 'Loaded'},
                'task_type': 'classification',
                'status': 'loaded'
            }

if not has_models:
    st.warning("⚠️ No trained models found. Please go to the **Model Training** page first.")

    # Add quick training button for convenience
    st.markdown("---")
    st.subheader("Quick Model Training")
    st.info("Train essential models directly from this page for immediate predictions.")

    if st.button("🚀 Train Essential Models Now", type="primary"):
        try:
            from models.xgboost_models import QuantTradingModels
            from features.technical_indicators import TechnicalIndicators
            from utils.database_adapter import get_trading_database
            from datetime import datetime

            with st.spinner("Training models on full dataset... This may take 5-10 minutes for maximum accuracy."):
                # Use full dataset for maximum accuracy
                data = st.session_state.data
                st.info(f"Training on complete dataset: {len(data)} rows for maximum accuracy")

                # Calculate features
                features_data = TechnicalIndicators.calculate_all_indicators(data)
                features_data = features_data.dropna()

                if len(features_data) < 100:
                    st.error("Not enough clean data for training")
                    st.stop()

                # Initialize trainer
                model_trainer = QuantTradingModels()
                st.session_state.model_trainer = model_trainer

                # Prepare data
                X = model_trainer.prepare_features(features_data)
                targets = model_trainer.create_targets(features_data)

                # Train essential models
                essential_models = ['direction', 'magnitude', 'trading_signal']
                results = {}

                for model_name in essential_models:
                    if model_name in targets:
                        y = targets[model_name]
                        task_type = 'classification' if model_name in ['direction', 'trading_signal'] else 'regression'

                        result = model_trainer.train_model(model_name, X, y, task_type)
                        if result:
                            results[model_name] = result

                # Save results
                st.session_state.models = results

                # Save to database
                try:
                    db = get_trading_database()
                    for model_name, model_result in results.items():
                        if model_result:
                            model_data = {
                                'metrics': model_result,
                                'task_type': 'classification' if model_name in ['direction', 'trading_signal'] else 'regression',
                                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            db.save_model_results(model_name, model_data)
                    db.save_trained_models(model_trainer.models)
                except:
                    pass

                st.success("Models trained successfully! Refreshing page...")
                st.rerun()

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.info("Please try using the Model Training page instead.")

    st.stop()

if st.session_state.model_trainer is None:
    st.warning("⚠️ Model trainer not initialized. Please go to the **Model Training** page first.")
    st.stop()

df = st.session_state.data
models = st.session_state.models
model_trainer = st.session_state.model_trainer

# Available models - check both session state and trainer
available_models = []
if st.session_state.models:
    available_models.extend([name for name, info in st.session_state.models.items() if info is not None])

# Also check trainer models
if st.session_state.model_trainer and st.session_state.model_trainer.models:
    trainer_models = [name for name in st.session_state.model_trainer.models.keys() 
                     if name not in available_models]
    available_models.extend(trainer_models)

if not available_models:
    st.error("❌ No successfully trained models found.")
    st.info("Please go to the **Model Training** page to train models first.")
    st.stop()

# Header
st.title("🎯 Prediction Engine")
st.markdown("Real-time Market Analysis & Forecasting")
st.markdown("---")

# Model and time range selection
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_model = st.selectbox(
        "🤖 Select AI Model",
        available_models,
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Choose a trained model for predictions"
    )

with col2:
    date_range = st.selectbox(
        "📅 Time Period",
        ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All data"],
        index=1
    )

with col3:
    # Handle missing task_type with fallback
    task_type = models[selected_model].get('task_type', 'unknown')
    if not task_type or task_type == 'unknown':
        # Try to infer task type from model name
        if selected_model in ['direction', 'profit_prob', 'trend_sideways', 'reversal', 'trading_signal']:
            task_type = 'classification'
        elif selected_model in ['magnitude', 'volatility']:
            task_type = 'regression'
        else:
            task_type = 'unknown'
    st.metric("Model Type", task_type.title())

# Ensure Date is not both index and column
if 'Date' in df.columns:
    df = df.drop(columns=['Date'], errors='ignore')

if st.session_state.features is not None:
    if hasattr(st.session_state.features, 'columns') and 'Date' in st.session_state.features.columns:
        st.session_state.features = st.session_state.features.drop(columns=['Date'], errors='ignore')

# Ensure DataFrame has proper datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    # Try to convert index to datetime
    try:
        df.index = pd.to_datetime(df.index)
    except:
        # If conversion fails, create a simple date range
        if len(df) > 0:
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        else:
            st.error("Cannot create proper date index for the data")
            st.stop()

# Filter data based on selection
try:
    if date_range == "Last 30 days":
        start_date = df.index.max() - timedelta(days=30)
    elif date_range == "Last 90 days":
        start_date = df.index.max() - timedelta(days=90)
    elif date_range == "Last 6 months":
        start_date = df.index.max() - timedelta(days=180)
    elif date_range == "Last year":
        start_date = df.index.max() - timedelta(days=365)
    else:
        start_date = df.index.min()

    df_filtered = df[df.index >= start_date].copy()
except Exception as e:
    st.warning(f"Error filtering data by date range: {str(e)}. Using all available data.")
    df_filtered = df.copy()
    start_date = df.index.min() if len(df) > 0 else pd.Timestamp.now()

# Check if features exist, if not prepare them
if st.session_state.features is None:
    st.warning("Features not found. Preparing features from current data...")
    try:
        # First calculate technical indicators if not present
        from features.technical_indicators import TechnicalIndicators

        # Check if technical indicators are already calculated
        required_indicators = ['sma_5', 'ema_5', 'rsi', 'macd_histogram']
        missing_indicators = [ind for ind in required_indicators if ind not in df_filtered.columns]

        if missing_indicators:
            st.info("Calculating missing technical indicators...")
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(df_filtered)
            df_full_indicators = TechnicalIndicators.calculate_all_indicators(df)
        else:
            df_with_indicators = df_filtered
            df_full_indicators = df

        features_filtered = model_trainer.prepare_features(df_with_indicators)
        st.session_state.features = model_trainer.prepare_features(df_full_indicators)

        st.success(f"✅ Prepared {len(features_filtered)} feature rows with {features_filtered.shape[1]} features")

    except Exception as e:
        st.error(f"❌ Error preparing features: {str(e)}")
        st.info("Please ensure your data has technical indicators calculated. Go to Model Training page first.")
        st.stop()
else:
    # Ensure features have proper datetime index
    features = st.session_state.features.copy()
    if not isinstance(features.index, pd.DatetimeIndex):
        try:
            features.index = pd.to_datetime(features.index)
        except:
            features.index = pd.date_range(start='2020-01-01', periods=len(features), freq='D')

    try:
        features_filtered = features[features.index >= start_date].copy()
        if features_filtered.empty:
            st.warning("No features found for selected date range. Using recent data...")
            features_filtered = features.tail(100).copy()
    except Exception as e:
        st.warning(f"Error filtering features: {str(e)}. Using all features.")
        features_filtered = features.copy()

def safe_format_date(idx):
    """Safely format date index to string"""
    try:
        if hasattr(idx, 'strftime'):
            return idx.strftime('%Y-%m-%d %H:%M')
        else:
            return pd.to_datetime(idx).strftime('%Y-%m-%d %H:%M')
    except:
        return str(idx)

def create_display_dataframe(pred_df):
    """Create a safely formatted display dataframe"""
    display_df = pred_df.copy()
    display_df = display_df.reset_index()

    # Safely handle date column
    if len(display_df.columns) > 0:
        index_col = display_df.columns[0]
        display_df['Date'] = display_df[index_col].apply(safe_format_date)

        # Remove original index column if different from Date
        if index_col != 'Date':
            display_df = display_df.drop(columns=[index_col], errors='ignore')

    # Format numeric columns
    if 'Price' in display_df.columns:
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")

    if 'Confidence' in display_df.columns:
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.3f}")

    # Reorder columns
    date_cols = [col for col in display_df.columns if 'Date' in col]
    other_cols = [col for col in display_df.columns if 'Date' not in col]
    ordered_cols = date_cols + other_cols

    return display_df[ordered_cols]

# Generate predictions
try:
    predictions, probabilities = model_trainer.predict(selected_model, features_filtered)

    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'Price': df_filtered['Close'],
        'Prediction': predictions,
        'Direction': ['Up' if p == 1 else 'Down' for p in predictions]
    }, index=features_filtered.index)

    if probabilities is not None:
        pred_df['Confidence'] = np.max(probabilities, axis=1)

    # Save predictions to database
    try:
        from utils.database_adapter import get_trading_database
        db = get_trading_database()
        
        # Save the prediction dataframe
        save_success = db.save_predictions(pred_df, selected_model)
        if save_success:
            st.success(f"✅ Predictions saved to database for model: {selected_model}")
        else:
            st.warning("⚠️ Failed to save predictions to database")
    except Exception as save_error:
        st.warning(f"⚠️ Could not save predictions: {str(save_error)}")

    # Create tabs for different views
    if selected_model == 'direction':
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Statistics", "📋 Data Table", "🔍 Analysis"])

        with tab1:
            st.subheader("📈 Direction Predictions")

            # Create price chart with predictions
            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2),
                opacity=0.7
            ))

            # Prediction markers
            up_predictions = pred_df[pred_df['Prediction'] == 1]
            down_predictions = pred_df[pred_df['Prediction'] == 0]

            if len(up_predictions) > 0:
                # Build hover template with confidence if available
                if 'Confidence' in up_predictions.columns:
                    hover_template_up = '<b>Prediction:</b> Up<br><b>Price:</b> $%{y:.2f}<br><b>Confidence:</b> %{customdata:.3f}<extra></extra>'
                    customdata_up = up_predictions['Confidence']
                else:
                    hover_template_up = '<b>Prediction:</b> Up<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                    customdata_up = None

                fig.add_trace(go.Scatter(
                    x=up_predictions.index,
                    y=up_predictions['Price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=8),
                    name='Predicted Up',
                    hovertemplate=hover_template_up,
                    customdata=customdata_up
                ))

            if len(down_predictions) > 0:
                # Build hover template with confidence if available
                if 'Confidence' in down_predictions.columns:
                    hover_template_down = '<b>Prediction:</b> Down<br><b>Price:</b> $%{y:.2f}<br><b>Confidence:</b> %{customdata:.3f}<extra></extra>'
                    customdata_down = down_predictions['Confidence']
                else:
                    hover_template_down = '<b>Prediction:</b> Down<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                    customdata_down = None

                fig.add_trace(go.Scatter(
                    x=down_predictions.index,
                    y=down_predictions['Price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=8),
                    name='Predicted Down',
                    hovertemplate=hover_template_down,
                    customdata=customdata_down
                ))

            fig.update_layout(
                height=500,
                title="Direction Predictions Overview",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📊 Prediction Statistics")

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                up_pct = (predictions == 1).mean() * 100
                st.metric("Predicted Up", f"{up_pct:.1f}%")
            with col2:
                down_pct = (predictions == 0).mean() * 100
                st.metric("Predicted Down", f"{down_pct:.1f}%")
            with col3:
                if 'Confidence' in pred_df.columns:
                    avg_conf = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                else:
                    st.metric("Data Points", len(pred_df))
            with col4:
                current_pred = "Up" if predictions[-1] == 1 else "Down"
                st.metric("Latest Signal", current_pred)

        with tab3:
            st.subheader("📋 Detailed Predictions Data")

            # Create display dataframe
            display_df = create_display_dataframe(pred_df)

            # Show recent predictions
            st.dataframe(
                display_df.tail(50),
                use_container_width=True,
                hide_index=True
            )

            # Download button for complete dataset
            st.subheader("📥 Export Complete Predictions")
            
            # Create a clean export dataframe from the original pred_df
            complete_export_df = pred_df.copy()
            complete_export_df = complete_export_df.reset_index()

            # Format the complete export dataframe properly
            if len(complete_export_df.columns) > 0:
                index_col = complete_export_df.columns[0]
                complete_export_df['Date'] = complete_export_df[index_col].apply(safe_format_date)
                if index_col != 'Date':
                    complete_export_df = complete_export_df.drop(columns=[index_col], errors='ignore')

            # Ensure all data is included (not just tail(50))
            csv_data = complete_export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Complete Direction Predictions CSV",
                data=csv_data,
                file_name=f"direction_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download all {len(complete_export_df)} prediction records ({len(complete_export_df)} rows)"
            )
            
            # Show download confirmation
            st.info(f"📊 Ready to download: {len(complete_export_df)} rows of direction prediction data")

        with tab4:
            st.subheader("🔍 Recent Predictions")

            # Create display dataframe for recent predictions
            display_df = create_display_dataframe(pred_df)

            # Show the most recent 20 predictions with better formatting
            recent_df = display_df.tail(20).copy()

            # Add signal interpretation
            if 'Direction' in recent_df.columns:
                recent_df['Signal'] = recent_df['Direction'].apply(
                    lambda x: "🟢 BUY" if x == "Up" else "🔴 SELL"
                )

            st.dataframe(
                recent_df[['Date', 'Price', 'Direction', 'Signal'] + 
                         ([col for col in recent_df.columns if 'Confidence' in col] if 'Confidence' in recent_df.columns else [])
                        ],
                use_container_width=True,
                hide_index=True
            )

    elif selected_model == 'magnitude':
        tab1, tab2, tab3 = st.tabs(["📊 Magnitude Analysis", "📈 Price Movement", "📋 Data Table"])

        # Add magnitude-specific fields - predictions are already in percentage format
        pred_df['Magnitude'] = np.abs(predictions)  # Keep original scale
        
        # Calculate actual magnitude for comparison
        actual_returns = df_filtered['Close'].pct_change().shift(-1)  # Next period return
        pred_df['Actual_Magnitude'] = np.abs(actual_returns) * 100  # Convert to percentage
        
        # Create dynamic magnitude categories based on actual data distribution
        magnitude_values = pred_df['Magnitude'].dropna()
        
        if len(magnitude_values) > 0:
            # Use quantile-based binning for balanced distribution
            q25 = np.percentile(magnitude_values, 25)
            q50 = np.percentile(magnitude_values, 50)
            q75 = np.percentile(magnitude_values, 75)
            
            pred_df['Magnitude_Category'] = pd.cut(pred_df['Magnitude'], 
                                                 bins=[0, q25, q50, q75, float('inf')],
                                                 labels=[f'Low (0-{q25:.3f})', f'Medium ({q25:.3f}-{q50:.3f})', 
                                                        f'High ({q50:.3f}-{q75:.3f})', f'Extreme ({q75:.3f}+)'],
                                                 include_lowest=True)
        else:
            # Fallback categories
            pred_df['Magnitude_Category'] = pd.cut(pred_df['Magnitude'], 
                                                 bins=[0, 0.5, 1.0, 2.0, float('inf')],
                                                 labels=['Low (0-0.5)', 'Medium (0.5-1.0)', 'High (1.0-2.0)', 'Extreme (2.0+)'],
                                                 include_lowest=True)

        with tab1:
            st.subheader("📊 Price Movement Magnitude Analysis")

            # Magnitude comparison chart
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=('Predicted vs Actual Magnitude', 'Magnitude Over Time', 
                                            'Predicted Magnitude Distribution', 'Actual Magnitude Distribution'),
                              vertical_spacing=0.15, horizontal_spacing=0.1)

            # Predicted vs Actual scatter plot
            clean_data = pred_df.dropna(subset=['Magnitude', 'Actual_Magnitude'])
            if len(clean_data) > 0:
                fig.add_trace(go.Scatter(
                    x=clean_data['Actual_Magnitude'],
                    y=clean_data['Magnitude'],
                    mode='markers',
                    name='Predicted vs Actual',
                    marker=dict(color='blue', size=4, opacity=0.6),
                    hovertemplate='Actual: %{x:.4f}%<br>Predicted: %{y:.4f}<extra></extra>'
                ), row=1, col=1)
                
                # Add perfect prediction line
                max_val = max(clean_data['Actual_Magnitude'].max(), clean_data['Magnitude'].max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ), row=1, col=1)

            # Magnitude timeline - both predicted and actual
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Magnitude'],
                mode='lines',
                name='Predicted Magnitude',
                line=dict(color='purple', width=2)
            ), row=1, col=2)
            
            if 'Actual_Magnitude' in pred_df.columns:
                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Actual_Magnitude'],
                    mode='lines',
                    name='Actual Magnitude',
                    line=dict(color='orange', width=1, dash='dot'),
                    opacity=0.7
                ), row=1, col=2)

            # Predicted magnitude histogram
            fig.add_trace(go.Histogram(
                x=pred_df['Magnitude'],
                nbinsx=25,
                name='Predicted Distribution',
                marker_color='lightblue',
                showlegend=False
            ), row=2, col=1)
            
            # Actual magnitude histogram
            if 'Actual_Magnitude' in pred_df.columns:
                fig.add_trace(go.Histogram(
                    x=pred_df['Actual_Magnitude'].dropna(),
                    nbinsx=25,
                    name='Actual Distribution',
                    marker_color='lightcoral',
                    showlegend=False
                ), row=2, col=2)

            fig.update_layout(height=700, showlegend=True)
            fig.update_xaxes(title_text="Actual Magnitude (%)", row=1, col=1)
            fig.update_yaxes(title_text="Predicted Magnitude", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=1, col=2)
            fig.update_yaxes(title_text="Magnitude", row=1, col=2)
            fig.update_xaxes(title_text="Predicted Magnitude", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_xaxes(title_text="Actual Magnitude (%)", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)

            st.plotly_chart(fig, use_container_width=True)

            # Enhanced statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Predicted", f"{pred_df['Magnitude'].mean():.4f}%")
            with col2:
                if 'Actual_Magnitude' in pred_df.columns:
                    st.metric("Avg Actual", f"{pred_df['Actual_Magnitude'].mean():.4f}%")
                else:
                    st.metric("Max Predicted", f"{pred_df['Magnitude'].max():.4f}%")
            with col3:
                # Calculate correlation if both exist
                if 'Actual_Magnitude' in pred_df.columns:
                    correlation = pred_df[['Magnitude', 'Actual_Magnitude']].corr().iloc[0, 1]
                    st.metric("Correlation", f"{correlation:.3f}")
                else:
                    high_magnitude = (pred_df['Magnitude_Category'].str.contains('High|Extreme', na=False)).mean() * 100
                    st.metric("High Magnitude %", f"{high_magnitude:.1f}%")
            with col4:
                st.metric("Latest Predicted", f"{pred_df['Magnitude'].iloc[-1]:.4f}%")
            
            # Show magnitude category distribution
            st.subheader("📊 Magnitude Category Distribution")
            if 'Magnitude_Category' in pred_df.columns:
                category_counts = pred_df['Magnitude_Category'].value_counts()
                if len(category_counts) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        for category, count in category_counts.items():
                            percentage = count / len(pred_df) * 100
                            st.metric(f"{category}", f"{count} ({percentage:.1f}%)")
                    
                    with col2:
                        # Show actual ranges
                        st.markdown("**Magnitude Ranges:**")
                        magnitude_stats = pred_df['Magnitude'].describe()
                        st.text(f"Min: {magnitude_stats['min']:.4f}%")
                        st.text(f"25th percentile: {magnitude_stats['25%']:.4f}%")
                        st.text(f"Median: {magnitude_stats['50%']:.4f}%")
                        st.text(f"75th percentile: {magnitude_stats['75%']:.4f}%")
                        st.text(f"Max: {magnitude_stats['max']:.4f}%")

        with tab2:
            st.subheader("📈 Price Movement with Magnitude Analysis")

            # Create subplot with price and magnitude
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Price with Magnitude Indicators', 'Predicted Magnitude Values'),
                              vertical_spacing=0.15,
                              row_heights=[0.7, 0.3])

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Price:</b> $%{y:.2f}<br><b>Date:</b> %{x}<extra></extra>'
            ), row=1, col=1)

            # Enhanced color-coded magnitude markers with better visibility
            colors = {'Low': '#27AE60', 'Medium': '#F39C12', 'High': '#E67E22', 'Extreme': '#E74C3C'}
            symbols = {'Low': 'circle', 'Medium': 'square', 'High': 'diamond', 'Extreme': 'triangle-up'}
            sizes = {'Low': 8, 'Medium': 10, 'High': 12, 'Extreme': 14}
            
            for category in colors:
                if category in pred_df['Magnitude_Category'].astype(str).values:
                    category_data = pred_df[pred_df['Magnitude_Category'].astype(str).str.contains(category.split(' ')[0], na=False)]
                    if len(category_data) > 0:
                        fig.add_trace(go.Scatter(
                            x=category_data.index,
                            y=category_data['Price'],
                            mode='markers',
                            marker=dict(
                                color=colors[category], 
                                size=sizes[category],
                                symbol=symbols[category],
                                line=dict(width=2, color='white')
                            ),
                            name=f'{category} Magnitude',
                            hovertemplate=f'<b>Magnitude:</b> {category}<br><b>Price:</b> $%{{y:.2f}}<br><b>Predicted:</b> %{{customdata:.4f}}%<extra></extra>',
                            customdata=category_data['Magnitude']
                        ), row=1, col=1)

            # Add magnitude values as a separate chart
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Magnitude'],
                mode='lines+markers',
                name='Predicted Magnitude',
                line=dict(color='purple', width=2),
                marker=dict(size=4, color='purple'),
                hovertemplate='<b>Predicted Magnitude:</b> %{y:.4f}%<br><b>Date:</b> %{x}<extra></extra>'
            ), row=2, col=1)

            # Add actual volatility if available
            if 'Actual_Volatility' in pred_df.columns:
                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Actual_Volatility'],
                    mode='lines+markers',
                    name='Actual Volatility',
                    line=dict(color='orange', width=1, dash='dot'),
                    marker=dict(size=3, color='orange'),
                    opacity=0.7,
                    hovertemplate='<b>Actual Volatility:</b> %{y:.4f}<br><b>Date:</b> %{x}<extra></extra>'
                ), row=2, col=1)

            # Add magnitude threshold lines
            if len(pred_df['Magnitude'].dropna()) > 0:
                q25 = np.percentile(pred_df['Magnitude'].dropna(), 25)
                q50 = np.percentile(pred_df['Magnitude'].dropna(), 50)
                q75 = np.percentile(pred_df['Magnitude'].dropna(), 75)

                fig.add_hline(y=q25, line_dash="dash", line_color="green", 
                             annotation_text=f"Low/Medium ({q25:.4f}%)", row=2, col=1)
                fig.add_hline(y=q50, line_dash="dash", line_color="orange", 
                             annotation_text=f"Medium/High ({q50:.4f}%)", row=2, col=1)
                fig.add_hline(y=q75, line_dash="dash", line_color="red", 
                             annotation_text=f"High/Extreme ({q75:.4f}%)", row=2, col=1)

            fig.update_layout(
                height=700, 
                showlegend=True,
                title_text="Price Movement Analysis with Magnitude Predictions"
            )
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude (%)", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Add magnitude statistics summary
            st.subheader("📊 Current Magnitude Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                recent_avg = pred_df['Magnitude'].tail(10).mean()
                st.metric("Recent Avg (10 periods)", f"{recent_avg:.4f}%")
            with col2:
                current_magnitude = pred_df['Magnitude'].iloc[-1]
                st.metric("Latest Prediction", f"{current_magnitude:.4f}%")
            with col3:
                high_magnitude_count = (pred_df['Magnitude'] > pred_df['Magnitude'].quantile(0.75)).sum()
                st.metric("High Magnitude Periods", high_magnitude_count)
            with col4:
                current_category = pred_df['Magnitude_Category'].iloc[-1] if 'Magnitude_Category' in pred_df.columns else 'N/A'
                st.metric("Current Category", str(current_category))

        with tab3:
            st.subheader("📋 Magnitude Predictions Data")

            display_df = create_display_dataframe(pred_df)
            
            # Format magnitude values properly - both should have % symbol for consistency
            if 'Magnitude' in display_df.columns:
                display_df['Predicted_Magnitude'] = display_df['Magnitude'].apply(
                    lambda x: f"{float(x):.4f}%" if pd.notna(x) and isinstance(x, (int, float, str)) else "N/A"
                )
            
            if 'Actual_Magnitude' in display_df.columns:
                display_df['Actual_Magnitude'] = display_df['Actual_Magnitude'].apply(
                    lambda x: f"{float(x):.4f}%" if pd.notna(x) and isinstance(x, (int, float, str)) else "N/A"
                )
            
            # Calculate prediction error - handle NaN values properly
            if 'Magnitude' in pred_df.columns and 'Actual_Magnitude' in pred_df.columns:
                # Clean the data before calculating error
                pred_clean = pd.to_numeric(pred_df['Magnitude'], errors='coerce')
                actual_clean = pd.to_numeric(pred_df['Actual_Magnitude'], errors='coerce')
                
                # Calculate absolute error only where both values are valid
                valid_mask = pd.notna(pred_clean) & pd.notna(actual_clean)
                pred_df['Prediction_Error'] = np.nan
                pred_df.loc[valid_mask, 'Prediction_Error'] = np.abs(pred_clean[valid_mask] - actual_clean[valid_mask])
                
                # Format prediction error for display
                display_df['Prediction_Error'] = pred_df['Prediction_Error'].apply(
                    lambda x: f"{float(x):.4f}%" if pd.notna(x) else "N/A"
                )
            
            # Select columns to display
            columns_to_show = ['Date', 'Price']
            if 'Predicted_Magnitude' in display_df.columns:
                columns_to_show.append('Predicted_Magnitude')
            if 'Actual_Magnitude' in display_df.columns:
                columns_to_show.append('Actual_Magnitude')
            if 'Prediction_Error' in display_df.columns:
                columns_to_show.append('Prediction_Error')
            if 'Magnitude_Category' in display_df.columns:
                columns_to_show.append('Magnitude_Category')
            
            # Filter to available columns
            available_columns = [col for col in columns_to_show if col in display_df.columns]
            
            # Show summary statistics
            st.subheader("📊 Prediction Accuracy Summary")
            if 'Prediction_Error' in pred_df.columns:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    mean_error = pred_df['Prediction_Error'].mean()
                    st.metric("Mean Abs Error", f"{mean_error:.4f}")
                with col2:
                    median_error = pred_df['Prediction_Error'].median()
                    st.metric("Median Abs Error", f"{median_error:.4f}")
                with col3:
                    rmse = np.sqrt(np.mean(pred_df['Prediction_Error']**2))
                    st.metric("RMSE", f"{rmse:.4f}")
                with col4:
                    if 'Actual_Magnitude' in pred_df.columns:
                        mape = np.mean(np.abs(pred_df['Prediction_Error'] / pred_df['Actual_Magnitude'])) * 100
                        st.metric("MAPE", f"{mape:.1f}%")

            st.dataframe(
                display_df[available_columns].tail(50), 
                use_container_width=True, 
                hide_index=True
            )

            # Download button for complete dataset
            st.subheader("📥 Export Complete Predictions")
            
            # Create a clean export dataframe from the original pred_df
            complete_export_df = pred_df.copy()
            complete_export_df = complete_export_df.reset_index()

            # Format the complete export dataframe properly
            if len(complete_export_df.columns) > 0:
                index_col = complete_export_df.columns[0]
                complete_export_df['Date'] = complete_export_df[index_col].apply(safe_format_date)
                if index_col != 'Date':
                    complete_export_df = complete_export_df.drop(columns=[index_col], errors='ignore')

            # Ensure all data is included
            csv_data = complete_export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Complete Magnitude Predictions CSV",
                data=csv_data,
                file_name=f"magnitude_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download all {len(complete_export_df)} prediction records ({len(complete_export_df)} rows)"
            )
            
            # Show download confirmation
            st.info(f"📊 Ready to download: {len(complete_export_df)} rows of magnitude prediction data")

    elif selected_model in ['profit_probability', 'profit_prob', 'profit_prob_regression']:
        tab1, tab2, tab3 = st.tabs(["🎯 Profit Probability", "📊 Risk Analysis", "📋 Data Table"])

        # Add profit probability specific fields
        if selected_model == 'profit_prob_regression':
            # For regression model, predictions are actual probabilities (0-1)
            pred_df['Profit_Prob'] = np.clip(predictions, 0, 1)
            pred_df['Prediction_Type'] = 'Regression (Probability)'
        else:
            # For classification model
            if probabilities is not None and probabilities.shape[1] > 1:
                # Use the probability of the positive class (class 1)
                pred_df['Profit_Prob'] = probabilities[:, 1]
                pred_df['Prediction_Type'] = 'Classification (Confidence)'
            elif probabilities is not None:
                # Single probability array
                pred_df['Profit_Prob'] = probabilities[:, 0]
                pred_df['Prediction_Type'] = 'Classification (Confidence)'
            else:
                # Binary predictions only - convert to meaningful probabilities
                # High confidence for 1, low confidence for 0
                pred_df['Profit_Prob'] = np.where(predictions == 1, 0.75, 0.25)
                pred_df['Prediction_Type'] = 'Classification (Binary)'

        # Create more balanced risk levels based on data distribution
        # Use quantile-based binning for better distribution
        prob_values = pred_df['Profit_Prob'].dropna()

        if len(prob_values) > 0:
            # Create quartile-based risk levels for better balance
            q25 = np.percentile(prob_values, 25)
            q50 = np.percentile(prob_values, 50)
            q75 = np.percentile(prob_values, 75)

            pred_df['Risk_Level'] = pd.cut(pred_df['Profit_Prob'], 
                                         bins=[0, q25, q50, q75, 1.0],
                                         labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'],
                                         include_lowest=True)
        else:
            # Fallback to original binning if no data
            pred_df['Risk_Level'] = pd.cut(pred_df['Profit_Prob'], 
                                         bins=[0, 0.25, 0.5, 0.75, 1.0],
                                         labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'],
                                         include_lowest=True)

        # Dynamic signal interpretation based on actual data distribution
        if len(pred_df) > 0:
            prob_median = pred_df['Profit_Prob'].median()
            prob_75th = pred_df['Profit_Prob'].quantile(0.75)

            pred_df['Signal'] = np.where(pred_df['Profit_Prob'] >= prob_75th, '🟢 HIGH PROFIT', 
                                       np.where(pred_df['Profit_Prob'] >= prob_median, '🟡 MEDIUM PROFIT', '🔴 LOW PROFIT'))
        else:
            pred_df['Signal'] = '🔴 LOW PROFIT'

        with tab1:
            st.subheader("🎯 Profit Probability Analysis")

            # Probability over time with risk zones
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Profit_Prob'],
                mode='lines+markers',
                name='Profit Probability',
                line=dict(color='green', width=2),
                fill='tonexty'
            ))

            # Add risk zone backgrounds
            fig.add_hline(y=0.75, line_dash="dash", line_color="green", 
                         annotation_text="Low Risk Zone")
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Risk Zone")
            fig.add_hline(y=0.25, line_dash="dash", line_color="red", 
                         annotation_text="High Risk Zone")

            fig.update_layout(
                height=500,
                title="Profit Probability Over Time",
                xaxis_title="Date",
                yaxis_title="Profit Probability",
                yaxis=dict(range=[0, 1])
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Probability", f"{pred_df['Profit_Prob'].mean():.3f}")
            with col2:
                high_prob = (pred_df['Profit_Prob'] >= 0.6).mean() * 100
                st.metric("High Confidence %", f"{high_prob:.1f}%")
            with col3:
                low_risk = (pred_df['Risk_Level'] == 'Low Risk').mean() * 100
                st.metric("Low Risk %", f"{low_risk:.1f}%")
            with col4:
                st.metric("Latest Probability", f"{pred_df['Profit_Prob'].iloc[-1]:.3f}")

        with tab2:
            st.subheader("📊 Risk Distribution Analysis")

            # Risk level pie chart
            risk_counts = pred_df['Risk_Level'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.3,
                marker_colors=['red', 'orange', 'lightgreen', 'darkgreen']
            )])

            fig.update_layout(
                title="Risk Level Distribution",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Price chart with risk indicators
            st.subheader("Price Movement with Risk Levels")

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))

            # Risk level markers
            risk_colors = {'High Risk': 'red', 'Medium Risk': 'orange', 
                          'Low Risk': 'lightgreen', 'Very Low Risk': 'darkgreen'}

            for risk_level in risk_colors:
                risk_data = pred_df[pred_df['Risk_Level'] == risk_level]
                if len(risk_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=risk_data.index,
                        y=risk_data['Price'],
                        mode='markers',
                        marker=dict(color=risk_colors[risk_level], size=8),
                        name=risk_level,
                        hovertemplate=f'<b>Risk:</b> {risk_level}<br><b>Price:</b> $%{{y:.2f}}<extra></extra>'
                    ))

            fig.update_layout(
                height=500,
                title="Price Movement with Risk Level Indicators",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("📋 Profit Probability Data")

            display_df = create_display_dataframe(pred_df)
            if 'Profit_Prob' in display_df.columns:
                display_df['Profit_Prob'] = display_df['Profit_Prob'].apply(lambda x: f"{float(x):.3f}" if isinstance(x, (int, float, str)) else x)
            if 'Signal' in display_df.columns:
                display_df['Signal'] = display_df['Signal'].astype(str)

            st.dataframe(display_df.tail(50), use_container_width=True, hide_index=True)

            # Download button for complete dataset
            st.subheader("📥 Export Complete Predictions")
            
            # Create a clean export dataframe from the original pred_df
            complete_export_df = pred_df.copy()
            complete_export_df = complete_export_df.reset_index()

            # Format the complete export dataframe properly
            if len(complete_export_df.columns) > 0:
                index_col = complete_export_df.columns[0]
                complete_export_df['Date'] = complete_export_df[index_col].apply(safe_format_date)
                if index_col != 'Date':
                    complete_export_df = complete_export_df.drop(columns=[index_col], errors='ignore')

            # Ensure all data is included
            csv_data = complete_export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Complete Profit Probability Predictions CSV",
                data=csv_data,
                file_name=f"profit_prob_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download all {len(complete_export_df)} prediction records ({len(complete_export_df)} rows)"
            )
            
            # Show download confirmation
            st.info(f"📊 Ready to download: {len(complete_export_df)} rows of profit probability prediction data")

    elif selected_model == 'volatility':
        tab1, tab2, tab3 = st.tabs(["📊 Volatility Forecast", "📈 Volatility Trends", "📋 Data Table"])

        # Add volatility-specific fields
        pred_df['Volatility_Forecast'] = predictions
        
        # Calculate actual volatility for comparison
        # Use rolling standard deviation of returns as actual volatility
        returns = df_filtered['Close'].pct_change()
        pred_df['Actual_Volatility'] = returns.rolling(window=5).std() * np.sqrt(252)  # Annualized volatility

        # Create balanced volatility categories using data distribution
        vol_values = pred_df['Volatility_Forecast'].dropna()

        if len(vol_values) > 0:
            # Use quartile-based binning for balanced distribution
            q25 = np.percentile(vol_values, 25)
            q50 = np.percentile(vol_values, 50) 
            q75 = np.percentile(vol_values, 75)

            pred_df['Volatility_Category'] = pd.cut(pred_df['Volatility_Forecast'], 
                                                   bins=[0, q25, q50, q75, float('inf')],
                                                   labels=['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol'],
                                                   include_lowest=True)
        else:
            # Fallback to original binning if no data
            pred_df['Volatility_Category'] = pd.cut(pred_df['Volatility_Forecast'], 
                                                   bins=[0, 0.01, 0.02, 0.05, float('inf')],
                                                   labels=['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol'])

        with tab1:
            st.subheader("📊 Volatility Forecasting Analysis")

            # Volatility forecast over time
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Predicted Volatility Over Time', 'Volatility Distribution'),
                              vertical_spacing=0.1)

            # Volatility timeline
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Volatility_Forecast'],
                mode='lines+markers',
                name='Volatility Forecast',
                line=dict(color='orange', width=2),
                marker=dict(size=4)
            ), row=1, col=1)

            # Volatility histogram
            fig.add_trace(go.Histogram(
                x=pred_df['Volatility_Forecast'],
                nbinsx=30,
                name='Distribution',
                marker_color='lightcoral'
            ), row=2, col=1)

            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", row=1, col=1)
            fig.update_xaxes(title_text="Volatility Value", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Volatility distribution pie chart
            vol_counts = pred_df['Volatility_Category'].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=vol_counts.index,
                    values=vol_counts.values,
                    hole=0.3,
                    marker_colors=['green', 'yellow', 'orange', 'red']
                )])

                fig_pie.update_layout(
                    title="Volatility Distribution",
                    height=400
                )

                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # Statistics
                col1a, col2a = st.columns(2)
                with col1a:
                    st.metric("Avg Volatility", f"{pred_df['Volatility_Forecast'].mean():.4f}")
                    st.metric("Max Volatility", f"{pred_df['Volatility_Forecast'].max():.4f}")
                with col2a:
                    high_vol = (pred_df['Volatility_Category'].isin(['High Vol', 'Extreme Vol'])).mean() * 100
                    st.metric("High Volatility %", f"{high_vol:.1f}%")
                    st.metric("Latest Volatility", f"{pred_df['Volatility_Forecast'].iloc[-1]:.4f}")

                # Volatility range info
                st.subheader("📊 Volatility Ranges")
                if len(vol_counts) > 0:
                    vol_ranges = []
                    vol_values = pred_df['Volatility_Forecast'].dropna()
                    if len(vol_values) > 0:
                        q25 = np.percentile(vol_values, 25)
                        q50 = np.percentile(vol_values, 50)
                        q75 = np.percentile(vol_values, 75)
                        vol_max = vol_values.max()

                        vol_ranges = [
                            f"Low: 0 - {q25:.4f}",
                            f"Medium: {q25:.4f} - {q50:.4f}",
                            f"High: {q50:.4f} - {q75:.4f}",
                            f"Extreme: {q75:.4f} - {vol_max:.4f}"
                        ]

                        for range_info in vol_ranges:
                            st.text(range_info)

        with tab2:
            st.subheader("📈 Price with Volatility Indicators")

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))

            # Volatility-based markers
            vol_colors = {'Low Vol': 'green', 'Medium Vol': 'yellow', 'High Vol': 'orange', 'Extreme Vol': 'red'}
            for vol_level in vol_colors:
                vol_data = pred_df[pred_df['Volatility_Category'] == vol_level]
                if len(vol_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=vol_data.index,
                        y=vol_data['Price'],
                        mode='markers',
                        marker=dict(color=vol_colors[vol_level], size=8),
                        name=vol_level,
                        hovertemplate=f'<b>Volatility:</b> {vol_level}<br><b>Price:</b> $%{{y:.2f}}<extra></extra>'
                    ))

            fig.update_layout(
                height=500,
                title="Price Movement with Volatility Level Indicators",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("📋 Volatility Predictions Data")

            # Create enhanced display dataframe for volatility
            display_df = create_display_dataframe(pred_df)

            # Format volatility values
            if 'Volatility_Forecast' in display_df.columns:
                display_df['Predicted_Volatility'] = pred_df['Volatility_Forecast'].apply(
                    lambda x: f"{float(x):.4f}" if pd.notna(x) and isinstance(x, (int, float, str)) else "N/A"
                )
            
            if 'Actual_Volatility' in display_df.columns:
                display_df['Actual_Volatility'] = pred_df['Actual_Volatility'].apply(
                    lambda x: f"{float(x):.4f}" if pd.notna(x) and isinstance(x, (int, float, str)) else "N/A"
                )
            
            # Calculate prediction error for volatility
            if 'Volatility_Forecast' in pred_df.columns and 'Actual_Volatility' in pred_df.columns:
                # Clean the data before calculating error
                pred_clean = pd.to_numeric(pred_df['Volatility_Forecast'], errors='coerce')
                actual_clean = pd.to_numeric(pred_df['Actual_Volatility'], errors='coerce')
                
                # Calculate absolute error only where both values are valid
                valid_mask = pd.notna(pred_clean) & pd.notna(actual_clean)
                pred_df['Volatility_Error'] = np.nan
                pred_df.loc[valid_mask, 'Volatility_Error'] = np.abs(pred_clean[valid_mask] - actual_clean[valid_mask])
                
                # Format prediction error for display
                display_df['Prediction_Error'] = pred_df['Volatility_Error'].apply(
                    lambda x: f"{float(x):.4f}" if pd.notna(x) else "N/A"
                )

            # Add volatility interpretation
            if 'Volatility_Category' in display_df.columns:
                display_df['Vol_Level'] = display_df['Volatility_Category'].apply(
                    lambda x: f"📊 {x}" if pd.notna(x) else "N/A"
                )

            # Show validation stats
            st.subheader("📊 Volatility Analysis Summary")
            if 'Volatility_Error' in pred_df.columns:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    mean_error = pred_df['Volatility_Error'].mean()
                    st.metric("Mean Abs Error", f"{mean_error:.4f}" if pd.notna(mean_error) else "N/A")
                with col2:
                    median_error = pred_df['Volatility_Error'].median()
                    st.metric("Median Abs Error", f"{median_error:.4f}" if pd.notna(median_error) else "N/A")
                with col3:
                    correlation = pred_df[['Volatility_Forecast', 'Actual_Volatility']].corr().iloc[0, 1]
                    st.metric("Correlation", f"{correlation:.3f}" if pd.notna(correlation) else "N/A")
                with col4:
                    if pd.notna(mean_error) and pred_df['Actual_Volatility'].mean() > 0:
                        mape = (mean_error / pred_df['Actual_Volatility'].mean()) * 100
                        st.metric("MAPE", f"{mape:.1f}%" if pd.notna(mape) else "N/A")
                    else:
                        st.metric("MAPE", "N/A")

            # Detailed data table
            columns_to_show = ['Date', 'Price']
            if 'Predicted_Volatility' in display_df.columns:
                columns_to_show.append('Predicted_Volatility')
            if 'Actual_Volatility' in display_df.columns:
                columns_to_show.append('Actual_Volatility')
            if 'Prediction_Error' in display_df.columns:
                columns_to_show.append('Prediction_Error')
            if 'Vol_Level' in display_df.columns:
                columns_to_show.append('Vol_Level')
            if 'Volatility_Category' in display_df.columns:
                columns_to_show.append('Volatility_Category')

            # Filter columns that exist
            available_columns = [col for col in columns_to_show if col in display_df.columns]

            # Add volatility statistics for recent data
            st.subheader("📈 Recent Volatility Data")
            recent_vol_data = pred_df.tail(20)

            # Show volatility statistics for recent data
            if len(recent_vol_data) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    recent_avg = recent_vol_data['Volatility_Forecast'].mean()
                    st.metric("Recent Avg Vol", f"{recent_avg:.4f}")
                with col2:
                    recent_max = recent_vol_data['Volatility_Forecast'].max()
                    st.metric("Recent Max Vol", f"{recent_max:.4f}")
                with col3:
                    latest_vol = recent_vol_data['Volatility_Forecast'].iloc[-1]
                    st.metric("Latest Vol", f"{latest_vol:.4f}")

            st.dataframe(
                display_df[available_columns].tail(50), 
                use_container_width=True, 
                hide_index=True
            )

            # Download button for complete dataset
            st.subheader("📥 Export Complete Predictions")
            
            # Create a clean export dataframe from the original pred_df
            complete_export_df = pred_df.copy()
            complete_export_df = complete_export_df.reset_index()

            # Format the complete export dataframe properly
            if len(complete_export_df.columns) > 0:
                index_col = complete_export_df.columns[0]
                complete_export_df['Date'] = complete_export_df[index_col].apply(safe_format_date)
                if index_col != 'Date':
                    complete_export_df = complete_export_df.drop(columns=[index_col], errors='ignore')

            # Ensure all data is included
            csv_data = complete_export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Complete Volatility Predictions CSV",
                data=csv_data,
                file_name=f"volatility_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download all {len(complete_export_df)} prediction records ({len(complete_export_df)} rows)"
            )
            
            # Show download confirmation
            st.info(f"📊 Ready to download: {len(complete_export_df)} rows of volatility prediction data")

    elif selected_model == 'trend_sideways':
        tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "📊 Market State", "📋 Data Table"])

        # Add trend analysis fields
        pred_df['Market_State'] = np.where(predictions == 1, 'Trending', 'Sideways')
        pred_df['Trend_Signal'] = predictions

        with tab1:
            st.subheader("📈 Trend vs Sideways Market Analysis")

            # Trend analysis chart
            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))

            # Trend/Sideways markers
            trending_data = pred_df[pred_df['Market_State'] == 'Trending']
            sideways_data = pred_df[pred_df['Market_State'] == 'Sideways']

            if len(trending_data) > 0:
                fig.add_trace(go.Scatter(
                    x=trending_data.index,
                    y=trending_data['Price'],
                    mode='markers',
                    marker=dict(color='green', size=8, symbol='triangle-up'),
                    name='Trending Market',
                    hovertemplate='<b>State:</b> Trending<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ))

            if len(sideways_data) > 0:
                fig.add_trace(go.Scatter(
                    x=sideways_data.index,
                    y=sideways_data['Price'],
                    mode='markers',
                    marker=dict(color='orange', size=8, symbol='circle'),
                    name='Sideways Market',
                    hovertemplate='<b>State:</b> Sideways<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ))

            fig.update_layout(
                height=500,
                title="Market State Classification",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📊 Market State Distribution")

            # Market state pie chart
            state_counts = pred_df['Market_State'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=state_counts.index,
                values=state_counts.values,
                hole=0.3,
                marker_colors=['green', 'orange']
            )])

            fig.update_layout(
                title="Trending vs Sideways Market Distribution",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                trending_pct = (predictions == 1).mean() * 100
                st.metric("Trending %", f"{trending_pct:.1f}%")
            with col2:
                sideways_pct = (predictions == 0).mean() * 100
                st.metric("Sideways %", f"{sideways_pct:.1f}%")
            with col3:
                if 'Confidence' in pred_df.columns:
                    avg_conf = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                else:
                    st.metric("Data Points", len(pred_df))
            with col4:
                current_state = "Trending" if predictions[-1] == 1 else "Sideways"
                st.metric("Current State", current_state)

        with tab3:
            st.subheader("📋 Trend Classification Data")

            display_df = create_display_dataframe(pred_df)
            st.dataframe(display_df.tail(50), use_container_width=True, hide_index=True)

            # Download button for complete dataset
            st.subheader("📥 Export Complete Predictions")
            
            # Create a clean export dataframe from the original pred_df
            complete_export_df = pred_df.copy()
            complete_export_df = complete_export_df.reset_index()

            # Format the complete export dataframe properly
            if len(complete_export_df.columns) > 0:
                index_col = complete_export_df.columns[0]
                complete_export_df['Date'] = complete_export_df[index_col].apply(safe_format_date)
                if index_col != 'Date':
                    complete_export_df = complete_export_df.drop(columns=[index_col], errors='ignore')

            # Ensure all data is included
            csv_data = complete_export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Complete Trend Classification Predictions CSV",
                data=csv_data,
                file_name=f"trend_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download all {len(complete_export_df)} prediction records ({len(complete_export_df)} rows)"
            )
            
            # Show download confirmation
            st.info(f"📊 Ready to download: {len(complete_export_df)} rows of trend classification prediction data")

    elif selected_model == 'reversal':
        tab1, tab2, tab3, tab4 = st.tabs(["🔄 Reversal Signals", "📊 Analysis & Validation", "📈 Technical Context", "📋 Data Table"])

        # Add reversal fields with enhanced analysis
        pred_df['Reversal_Signal'] = np.where(predictions == 1, 'Reversal Expected', 'No Reversal')
        pred_df['Signal_Type'] = predictions

        # Calculate additional technical indicators for context
        pred_df['Price_Change_1'] = pred_df['Price'].pct_change(1) * 100
        pred_df['Price_Change_3'] = pred_df['Price'].pct_change(3) * 100
        pred_df['SMA_5'] = pred_df['Price'].rolling(5).mean()
        pred_df['SMA_20'] = pred_df['Price'].rolling(20).mean()

        # Calculate price position in recent range
        pred_df['High_10'] = pred_df['Price'].rolling(10).max()
        pred_df['Low_10'] = pred_df['Price'].rolling(10).min()
        pred_df['Price_Position'] = (pred_df['Price'] - pred_df['Low_10']) / (pred_df['High_10'] - pred_df['Low_10'])

        # Identify reversal types based on price position
        reversal_data = pred_df[pred_df['Reversal_Signal'] == 'Reversal Expected'].copy()
        if len(reversal_data) > 0:
            reversal_data['Reversal_Type'] = np.where(
                reversal_data['Price_Position'] <= 0.4, 
                'Bullish (Bounce)', 
                'Bearish (Pullback)'
            )

        with tab1:
            st.subheader("🔄 Price Reversal Detection Points")

            fig = go.Figure()

            # Price line with moving averages
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='#2E86C1', width=2)
            ))

            # Add SMA lines for context
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['SMA_5'],
                mode='lines',
                name='SMA 5',
                line=dict(color='orange', width=1, dash='dot'),
                opacity=0.7
            ))

            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='purple', width=1, dash='dash'),
                opacity=0.7
            ))

            # Reversal markers with different colors for bullish/bearish
            if len(reversal_data) > 0:
                bullish_reversals = reversal_data[reversal_data['Reversal_Type'] == 'Bullish (Bounce)']
                bearish_reversals = reversal_data[reversal_data['Reversal_Type'] == 'Bearish (Pullback)']

                if len(bullish_reversals) > 0:
                    fig.add_trace(go.Scatter(
                        x=bullish_reversals.index,
                        y=bullish_reversals['Price'],
                        mode='markers',
                        marker=dict(color='green', size=12, symbol='triangle-up', 
                                   line=dict(width=2, color='darkgreen')),
                        name='Bullish Reversal',
                        hovertemplate='<b>Bullish Reversal</b><br>Price: $%{y:.2f}<br>Position: %{customdata:.1%}<extra></extra>',
                        customdata=bullish_reversals['Price_Position']
                    ))

                if len(bearish_reversals) > 0:
                    fig.add_trace(go.Scatter(
                        x=bearish_reversals.index,
                        y=bearish_reversals['Price'],
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='triangle-down',
                                   line=dict(width=2, color='darkred')),
                        name='Bearish Reversal',
                        hovertemplate='<b>Bearish Reversal</b><br>Price: $%{y:.2f}<br>Position: %{customdata:.1%}<extra></extra>',
                        customdata=bearish_reversals['Price_Position']
                    ))

            fig.update_layout(
                height=600,
                title="Price Reversal Detection with Technical Context",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Current market status
            if len(pred_df) > 0:
                latest_signal = pred_df['Reversal_Signal'].iloc[-1]
                latest_price_pos = pred_df['Price_Position'].iloc[-1]
                latest_momentum = pred_df['Price_Change_3'].iloc[-1]

                if latest_signal == 'Reversal Expected':
                    reversal_type = 'Bullish Reversal' if latest_price_pos <= 0.4 else 'Bearish Reversal'
                    signal_color = '#27AE60' if latest_price_pos <= 0.4 else '#E74C3C'

                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, {signal_color}20, {signal_color}10); 
                         border-left: 4px solid {signal_color}; padding: 1rem; margin: 1rem 0; border-radius: 8px;">
                        <h3 style="color: {signal_color}; margin: 0;">🔄 {reversal_type} Signal Detected</h3>
                        <p>Price Position in Range: <strong>{latest_price_pos:.1%}</strong></p>
                        <p>Recent 3-Period Change: <strong>{latest_momentum:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            st.subheader("📊 Reversal Signal Analysis & Validation")

            # Reversal distribution pie chart
            reversal_counts = pred_df['Reversal_Signal'].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure(data=[go.Pie(
                    labels=reversal_counts.index,
                    values=reversal_counts.values,
                    hole=0.3,
                    marker_colors=['#85C1E9', '#E74C3C']
                )])

                fig.update_layout(
                    title="Overall Reversal Signal Distribution",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if len(reversal_data) > 0:
                    reversal_type_counts = reversal_data['Reversal_Type'].value_counts()

                    fig = go.Figure(data=[go.Pie(
                        labels=reversal_type_counts.index,
                        values=reversal_type_counts.values,
                        hole=0.3,
                        marker_colors=['#27AE60', '#E74C3C']
                    )])

                    fig.update_layout(
                        title="Reversal Type Distribution",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                reversal_pct = (predictions == 1).mean() * 100
                st.metric("Reversal Signals %", f"{reversal_pct:.1f}%")
            with col2:
                if len(reversal_data) > 0:
                    bullish_count = len(reversal_data[reversal_data['Reversal_Type'] == 'Bullish (Bounce)'])
                    st.metric("Bullish Reversals", bullish_count)
                else:
                    st.metric("Bullish Reversals", 0)
            with col3:
                if len(reversal_data) > 0:
                    bearish_count = len(reversal_data[reversal_data['Reversal_Type'] == 'Bearish (Pullback)'])
                    st.metric("Bearish Reversals", bearish_count)
                else:
                    st.metric("Bearish Reversals", 0)
            with col4:
                if 'Confidence' in pred_df.columns:
                    avg_conf = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                else:
                    st.metric("Total Signals", len(reversal_data))

        with tab3:
            st.subheader("📈 Technical Analysis Context")

            # Price position distribution
            if len(pred_df) > 0:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Price_Position'],
                    mode='lines',
                    name='Price Position in Range',
                    line=dict(color='blue', width=2)
                ))

                # Add horizontal lines for key levels
                fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                             annotation_text="Overbought Zone (70%)")
                fig.add_hline(y=0.3, line_dash="dash", line_color="green", 
                             annotation_text="Oversold Zone (30%)")
                fig.add_hline(y=0.5, line_dash="dot", line_color="gray", 
                             annotation_text="Midpoint")

                # Mark reversal points
                if len(reversal_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=reversal_data.index,
                        y=reversal_data['Price_Position'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='diamond'),
                        name='Reversal Signals'
                    ))

                fig.update_layout(
                    height=400,
                    title="Price Position in 10-Period Range (Reversal Context)",
                    xaxis_title="Date",
                    yaxis_title="Position in Range (0-1)",
                    yaxis=dict(range=[0, 1])
                )

                st.plotly_chart(fig, use_container_width=True)

            # Reversal analysis and debugging
            st.subheader("🔍 Reversal Detection Analysis")

            # Show reversal statistics
            total_data_points = len(pred_df)
            reversal_count = len(reversal_data)
            reversal_percentage = (reversal_count / total_data_points * 100) if total_data_points > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Data Points", total_data_points)
            with col2:
                st.metric("Reversal Signals", reversal_count)
            with col3:
                st.metric("Reversal Rate", f"{reversal_percentage:.1f}%")

            # Show recent reversal signals if any
            if len(reversal_data) > 0:
                st.subheader("📋 Recent Reversal Signals")

                recent_reversals = reversal_data.tail(10)

                # Create detailed table
                analysis_data = []
                for idx, row in recent_reversals.iterrows():
                    analysis_data.append({
                        'Date': idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx),
                        'Price': f"${row['Price']:.2f}",
                        'Type': row['Reversal_Type'],
                        'Position': f"{row['Price_Position']:.1%}",
                        '1-Period Change': f"{row['Price_Change_1']:.2f}%",
                        '3-Period Change': f"{row['Price_Change_3']:.2f}%",
                        'Confidence': f"{row.get('Confidence', 0):.3f}" if 'Confidence' in pred_df.columns else 'N/A'
                    })

                if analysis_data:
                    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ No reversal signals detected in the current time period.")

                # Show debugging information
                st.subheader("🔧 Reversal Detection Debug Info")

                # Check if we have the necessary data for analysis
                debug_info = []
                if 'Price_Position' in pred_df.columns:
                    price_pos_stats = pred_df['Price_Position'].describe()
                    debug_info.append(f"Price Position Range: {price_pos_stats['min']:.2f} to {price_pos_stats['max']:.2f}")
                    debug_info.append(f"Low Range (<25%): {(pred_df['Price_Position'] <= 0.25).sum()} points")
                    debug_info.append(f"High Range (>75%): {(pred_df['Price_Position'] >= 0.75).sum()} points")

                if 'Price_Change_3' in pred_df.columns:
                    momentum_stats = pred_df['Price_Change_3'].describe()
                    debug_info.append(f"3-Period Change Range: {momentum_stats['min']:.3f} to {momentum_stats['max']:.3f}")
                    debug_info.append(f"Significant Declines (<-0.3%): {(pred_df['Price_Change_3'] < -0.003).sum()} points")
                    debug_info.append(f"Significant Rallies (>0.3%): {(pred_df['Price_Change_3'] > 0.003).sum()} points")

                for info in debug_info:
                    st.text(info)

                st.info("💡 Tip: Reversal signals depend on price reaching extreme positions (top/bottom 25% of recent range) combined with momentum conditions. Consider using a longer time period or different assets if no signals are detected.")

        with tab4:
            st.subheader("📋 Complete Reversal Detection Data")

            # Enhanced display dataframe
            display_df = create_display_dataframe(pred_df)

            # Add formatted columns
            if 'Price_Position' in display_df.columns:
                display_df['Price_Position'] = display_df['Price_Position'].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                )

            if 'Price_Change_1' in display_df.columns:
                display_df['1-Period Change'] = display_df['Price_Change_1'].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                )

            if 'Price_Change_3' in display_df.columns:
                display_df['3-Period Change'] = display_df['Price_Change_3'].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                )

            # Select and reorder columns for display
            columns_to_show = ['Date', 'Price', 'Reversal_Signal']
            if 'Price_Position' in display_df.columns:
                columns_to_show.append('Price_Position')
            if '1-Period Change' in display_df.columns:
                columns_to_show.append('1-Period Change')
            if '3-Period Change' in display_df.columns:
                columns_to_show.append('3-Period Change')
            if 'Confidence' in display_df.columns:
                columns_to_show.append('Confidence')

            # Filter to existing columns
            available_columns = [col for col in columns_to_show if col in display_df.columns]

            st.dataframe(
                display_df[available_columns].tail(50), 
                use_container_width=True, 
                hide_index=True
            )

            # Download button for complete dataset
            st.subheader("📥 Export Complete Predictions")
            
            # Create a clean export dataframe from the original pred_df
            complete_export_df = pred_df.copy()
            complete_export_df = complete_export_df.reset_index()

            # Format the complete export dataframe properly
            if len(complete_export_df.columns) > 0:
                index_col = complete_export_df.columns[0]
                complete_export_df['Date'] = complete_export_df[index_col].apply(safe_format_date)
                if index_col != 'Date':
                    complete_export_df = complete_export_df.drop(columns=[index_col], errors='ignore')

            # Ensure all data is included
            csv_data = complete_export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Complete Reversal Predictions CSV",
                data=csv_data,
                file_name=f"reversal_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download all {len(complete_export_df)} prediction records ({len(complete_export_df)} rows)"
            )
            
            # Show download confirmation
            st.info(f"📊 Ready to download: {len(complete_export_df)} rows of reversal prediction data")

    elif selected_model == 'trading_signal':
        tab1, tab2, tab3 = st.tabs(["📊 Trading Signals", "📈 Signal Analysis", "📋 Signal History"])

        signal_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
        signal_colors = {0: '#E74C3C', 1: '#F39C12', 2: '#27AE60'}

        pred_df['Signal'] = predictions
        pred_df['Signal_Name'] = [signal_map[p] for p in predictions]

        with tab1:
            st.subheader("📊 Trading Signal Overview")

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index, 
                y=pred_df['Price'],
                name='Price',
                line=dict(color='#2E86C1', width=2)
            ))

            # Signal markers
            for signal_value, signal_name in signal_map.items():
                signal_data = pred_df[pred_df['Signal'] == signal_value]
                if len(signal_data) > 0:
                    marker_symbol = 'triangle-up' if signal_value == 2 else ('triangle-down' if signal_value == 0 else 'circle')

                    fig.add_trace(go.Scatter(
                        x=signal_data.index,
                        y=signal_data['Price'],
                        mode='markers',
                        marker=dict(
                            symbol=marker_symbol,
                            color=signal_colors[signal_value],
                            size=10
                        ),
                        name=f'{signal_name} Signal',
                        hovertemplate=f'<b>{signal_name} Signal</b><br>Price: $%{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
                    ))

            fig.update_layout(
                height=500,
                title="Trading Signals on Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Current signal
            current_signal = signal_map[predictions[-1]]
            signal_color = signal_colors[predictions[-1]]

            st.markdown(f"""
            <div style="background: rgba{signal_color[3:-1]}, 0.1); border: 2px solid {signal_color}; 
                 border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
                <h3 style="color: {signal_color}; margin: 0;">Current Signal: {current_signal}</h3>
                <p style="margin: 0.5rem 0 0 0;">Latest trading recommendation</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.subheader("📈 Signal Distribution Analysis")

            signal_counts = pd.Series(predictions).value_counts()

            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Sell', 'Hold', 'Buy'],
                values=[signal_counts.get(0, 0), signal_counts.get(1, 0), signal_counts.get(2, 0)],
                hole=0.3,
                marker_colors=['#E74C3C', '#F39C12', '#27AE60']
            )])

            fig.update_layout(
                title="Trading Signal Distribution",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Signal metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                buy_pct = (signal_counts.get(2, 0) / len(predictions)) * 100
                st.metric("Buy Signals", f"{buy_pct:.1f}%")
            with col2:
                hold_pct = (signal_counts.get(1, 0) / len(predictions)) * 100
                st.metric("Hold Signals", f"{hold_pct:.1f}%")
            with col3:
                sell_pct = (signal_counts.get(0, 0) / len(predictions)) * 100
                st.metric("Sell Signals", f"{sell_pct:.1f}%")

        with tab3:
            st.subheader("📋 Signal History")

            display_df = create_display_dataframe(pred_df)

            st.dataframe(
                display_df.tail(50),
                use_container_width=True,
                hide_index=True
            )

            # Download button for complete dataset
            st.subheader("📥 Export Complete Predictions")
            
            # Create a clean export dataframe from the original pred_df
            complete_export_df = pred_df.copy()
            complete_export_df = complete_export_df.reset_index()

            # Format the complete export dataframe properly
            if len(complete_export_df.columns) > 0:
                index_col = complete_export_df.columns[0]
                complete_export_df['Date'] = complete_export_df[index_col].apply(safe_format_date)
                if index_col != 'Date':
                    complete_export_df = complete_export_df.drop(columns=[index_col], errors='ignore')

            # Ensure all data is included
            csv_data = complete_export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Complete Trading Signal Predictions CSV",
                data=csv_data,
                file_name=f"trading_signal_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download all {len(complete_export_df)} prediction records ({len(complete_export_df)} rows)"
            )
            
            # Show download confirmation
            st.info(f"📊 Ready to download: {len(complete_export_df)} rows of trading signal prediction data")

    else:
        # Handle any other model types
        st.subheader(f"📊 {selected_model.replace('_', ' ').title()} Predictions")

        # Create display dataframe
        display_df = create_display_dataframe(pred_df)

        # Show chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Price'],
            mode='lines+markers',
            name='Price with Predictions'
        ))

        fig.update_layout(
            height=400,
            title=f"{selected_model.replace('_', ' ').title()} Predictions",
            xaxis_title="Date",
            yaxis_title="Price ($)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        st.subheader("Recent Predictions")
        st.dataframe(display_df.tail(20), use_container_width=True, hide_index=True)

        # Download button for complete dataset
        st.subheader("📥 Export Complete Predictions")
        
        # Create a clean export dataframe from the original pred_df
        complete_export_df = pred_df.copy()
        complete_export_df = complete_export_df.reset_index()

        # Format the complete export dataframe properly
        if len(complete_export_df.columns) > 0:
            index_col = complete_export_df.columns[0]
            complete_export_df['Date'] = complete_export_df[index_col].apply(safe_format_date)
            if index_col != 'Date':
                complete_export_df = complete_export_df.drop(columns=[index_col], errors='ignore')

        # Ensure all data is included
        csv_data = complete_export_df.to_csv(index=False)
        
        st.download_button(
            label=f"📥 Download Complete {selected_model.replace('_', ' ').title()} Predictions CSV",
            data=csv_data,
            file_name=f"{selected_model}_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help=f"Download all {len(complete_export_df)} prediction records ({len(complete_export_df)} rows)"
        )
        
        # Show download confirmation
        st.info(f"📊 Ready to download: {len(complete_export_df)} rows of {selected_model.replace('_', ' ')} prediction data")

except Exception as e:
    st.error(f"Error generating predictions: {str(e)}")
    st.info("Please try refreshing the page or check your model training.")

    # Show debug information
    st.subheader("Debug Information")
    st.write("Available models:", available_models)
    st.write("Selected model:", selected_model)
    st.write("Features shape:", features_filtered.shape if features_filtered is not None else "None")
    st.write("Data shape:", df_filtered.shape)

# Model comparison section
st.markdown("---")
st.header("🔍 Model Comparison")

if len(available_models) > 1:
    compare_models = st.multiselect(
        "Select models to compare performance",
        available_models,
        default=available_models[:3] if len(available_models) >= 3 else available_models,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    if compare_models:
        st.info(f"Comparing {len(compare_models)} models. Feature comparison would be displayed here.")
else:
    st.info("Train more models to enable comparison features.")