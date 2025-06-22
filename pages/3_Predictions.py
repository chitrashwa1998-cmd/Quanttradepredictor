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

# Ensure models is a proper dictionary
if not st.session_state.models or not isinstance(st.session_state.models, dict):
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

# Available models - ensure models is properly structured
if isinstance(models, dict):
    available_models = []
    for name, info in models.items():
        # Check if info is a valid model info dictionary
        if isinstance(info, dict) and info is not None:
            available_models.append(name)
        elif info is not None and not isinstance(info, bool):
            available_models.append(name)
else:
    available_models = []

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
    # Handle missing task_type with proper validation
    task_type = 'unknown'

    if isinstance(models.get(selected_model), dict):
        task_type = models[selected_model].get('task_type', 'unknown')

    if not task_type or task_type == 'unknown':
        # Try to infer task type from model name
        if selected_model in ['direction', 'profit_prob', 'trend_sideways', 'reversal', 'trading_signal', 'magnitude', 'volatility']:
            task_type = 'classification'  # All models are now classification
        else:
            task_type = 'classification'  # Default

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
    # Ensure features_filtered is properly aligned and has the correct features
    if features_filtered.empty:
        st.error("No features available for prediction. Please check your data preparation.")
        st.stop()

    # Validate that the model trainer has the required feature names
    if not hasattr(model_trainer, 'feature_names') or not model_trainer.feature_names:
        st.error("Model trainer feature names not found. Please retrain models.")
        st.stop()

    # Check for missing features
    missing_features = [col for col in model_trainer.feature_names if col not in features_filtered.columns]
    if missing_features:
        st.error(f"Missing required features for prediction: {missing_features}")
        st.info("Please ensure your data has all required technical indicators calculated.")
        st.stop()

    predictions, probabilities = model_trainer.predict(selected_model, features_filtered)

    # Ensure predictions and features have compatible indices
    common_index = features_filtered.index[:len(predictions)]
    df_filtered_aligned = df_filtered.loc[df_filtered.index.isin(common_index)]

    # Create prediction dataframe with proper alignment
    pred_df = pd.DataFrame({
        'Price': df_filtered_aligned['Close'].iloc[:len(predictions)],
        'Prediction': predictions,
        'Direction': ['Up' if p == 1 else 'Down' for p in predictions]
    }, index=common_index)

    if probabilities is not None:
        pred_df['Confidence'] = np.max(probabilities, axis=1)

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
        tab1, tab2, tab3 = st.tabs(["📊 Magnitude Analysis", "📈 Distribution", "📋 Data Table"])

        # Add magnitude-specific fields
        pred_df['Magnitude'] = np.abs(predictions)
        pred_df['Magnitude_Category'] = pd.cut(pred_df['Magnitude'], 
                                             bins=[0, 0.01, 0.02, 0.05, float('inf')],
                                             labels=['Low', 'Medium', 'High', 'Extreme'])

        with tab1:
            st.subheader("📊 Price Magnitude Analysis")

            # Magnitude over time
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Magnitude'],
                mode='lines+markers',
                name='Magnitude',
                line=dict(color='purple', width=2)
            ))

            fig.update_layout(
                height=500,
                title="Price Movement Magnitude Over Time",
                xaxis_title="Date",
                yaxis_title="Magnitude"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📈 Magnitude Distribution")

            # Distribution chart
            fig = go.Figure(data=[go.Histogram(x=pred_df['Magnitude'], nbinsx=30)])
            fig.update_layout(title="Magnitude Distribution", xaxis_title="Magnitude", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("📋 Magnitude Data")
            display_df = create_display_dataframe(pred_df)
            st.dataframe(display_df.tail(50), use_container_width=True, hide_index=True)

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

    elif selected_model == 'volatility':
        tab1, tab2, tab3 = st.tabs(["📊 Volatility Forecast", "📈 Volatility Trends", "📋 Data Table"])

        # Add volatility-specific fields
        pred_df['Volatility_Forecast'] = predictions

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
                        
                        for i, range_info in enumerate(vol_ranges):
                            st.text(range_info)