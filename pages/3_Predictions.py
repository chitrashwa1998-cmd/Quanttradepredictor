import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

def safe_format_date_range(index):
    """Safely format date range from index, handling both datetime and non-datetime indexes."""
    try:
        if pd.api.types.is_datetime64_any_dtype(index):
            return f"{index[0].strftime('%Y-%m-%d')} to {index[-1].strftime('%Y-%m-%d')}"
        elif pd.api.types.is_numeric_dtype(index):
            # Try to convert numeric timestamps to datetime
            start_date = pd.to_datetime(index[0], unit='s', errors='coerce')
            end_date = pd.to_datetime(index[-1], unit='s', errors='coerce')
            if pd.notna(start_date) and pd.notna(end_date):
                return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            else:
                return f"Row {index[0]} to Row {index[-1]}"
        else:
            return f"Row {index[0]} to Row {index[-1]}"
    except Exception:
        return f"Row {index[0]} to Row {index[-1]}"

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None
if 'direction_features' not in st.session_state:
    st.session_state.direction_features = None
if 'direction_trained_models' not in st.session_state:
    st.session_state.direction_trained_models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'volatility_predictions' not in st.session_state:
    st.session_state.volatility_predictions = None
if 'direction_predictions' not in st.session_state:
    st.session_state.direction_predictions = None
if 'direction_probabilities' not in st.session_state:
    st.session_state.direction_probabilities = None

st.title("🔮 Model Predictions")
st.markdown("Generate and analyze predictions using the trained models.")

# Auto-restore data if missing
if st.session_state.data is None:
    try:
        from utils.database_adapter import get_trading_database
        db = get_trading_database()

        # Try to load data from database
        recovered_data = db.recover_data()
        if recovered_data is not None and not recovered_data.empty:
            st.session_state.data = recovered_data
            st.success("✅ Data restored from database")
        else:
            st.warning("⚠️ No data available. Please go to Data Upload page to upload data first.")
            st.stop()
    except Exception as e:
        st.warning("⚠️ No data available. Please go to Data Upload page to upload data first.")
        st.stop()

# Create tabs for different prediction types
volatility_tab, direction_tab, profit_prob_tab = st.tabs(["📊 Volatility Predictions", "🎯 Direction Predictions", "💰 Profit Probability"])

# Volatility Predictions Tab
with volatility_tab:
    st.header("📊 Volatility Predictions")

    # Check if volatility features are available
    if st.session_state.features is None:
        st.error("❌ No volatility features calculated. Please calculate technical indicators first.")
    else:
        # Check if volatility model is available in database or session state
        model_available = False

        # Check session state first
        if (hasattr(st.session_state, 'trained_models') and 
            st.session_state.trained_models and 
            'volatility' in st.session_state.trained_models and 
            st.session_state.trained_models['volatility'] is not None):
            model_available = True
        else:
            # Check database for trained models
            try:
                from utils.database_adapter import get_trading_database
                db = get_trading_database()

                # Check for trained model objects first (primary storage)
                trained_models = db.load_trained_models()
                if trained_models and 'volatility' in trained_models:
                    model_available = True
                    st.info("✅ Volatility model found in database")
                else:
                    # Fallback: check for model results
                    db_model = db.load_model_results('volatility')
                    if db_model and 'metrics' in db_model:
                        model_available = True
                        st.info("✅ Volatility model results found in database")
            except Exception as e:
                st.warning(f"Database check failed: {str(e)}")
                pass

        if not model_available:
            st.error("❌ Volatility model not trained. Please train the model first.")
        else:
            # Volatility prediction controls
            st.subheader("🎯 Prediction Controls")

        col1, col2 = st.columns(2)

        with col1:
            vol_filter = st.selectbox(
                "📅 Time Period Filter",
                ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All data"],
                index=1,
                help="Select the time period for volatility predictions",
                key="vol_filter"
            )

        with col2:
            st.metric("Volatility Model Status", "✅ Ready", help="Volatility model is trained and ready")

        # Generate volatility predictions button
        if st.button("🚀 Generate Volatility Predictions", type="primary", key="vol_predict"):
            try:
                with st.spinner("Generating volatility predictions..."):
                    # Load model from database if not in session state
                    if not hasattr(st.session_state, 'model_trainer') or st.session_state.model_trainer is None:
                        try:
                            from utils.database_adapter import get_trading_database
                            from models.model_manager import ModelManager

                            db = get_trading_database()

                            # Try to load trained model objects first
                            trained_models = db.load_trained_models()
                            if trained_models and 'volatility' in trained_models:
                                # Initialize model manager and set the loaded model
                                model_trainer = ModelManager()

                                # Properly structure the model data
                                volatility_model_data = trained_models['volatility']
                                if 'ensemble' in volatility_model_data:
                                    # Restructure the data to match expected format
                                    model_trainer.trained_models['volatility'] = {
                                        'model': volatility_model_data['ensemble'],
                                        'feature_names': volatility_model_data.get('feature_names', []),
                                        'task_type': volatility_model_data.get('task_type', 'regression'),
                                        'scaler': volatility_model_data.get('scaler')  # Include the scaler from database
                                    }
                                else:
                                    model_trainer.trained_models['volatility'] = volatility_model_data

                                st.session_state.model_trainer = model_trainer
                                st.info("✅ Loaded volatility model from database")
                            else:
                                st.error("❌ No trained volatility model found in database. Please train the model first.")
                                st.stop()
                        except Exception as load_error:
                            st.error(f"❌ Error loading model from database: {str(load_error)}")
                            st.error(f"Debug: {str(load_error)}")
                            st.stop()

                    model_trainer = st.session_state.model_trainer

                    # Verify the model is properly loaded
                    if 'volatility' not in model_trainer.trained_models:
                        st.error("❌ Volatility model not found in trainer")
                        st.stop()

                    volatility_model_data = model_trainer.trained_models['volatility']
                    if 'model' not in volatility_model_data and 'ensemble' not in volatility_model_data:
                        st.error("❌ Model object not found in volatility model data")
                        st.stop()

                    # Apply time filter to features for prediction
                    features_for_prediction = st.session_state.features.copy()

                    # Apply volatility filter using realistic row counts for trading data
                    total_rows = len(features_for_prediction)
                    
                    if vol_filter == "Last 30 days":
                        # 30 days × ~78 bars/day (6.5 trading hours × 12 bars/hour) = ~2,340 bars
                        target_rows = min(2500, total_rows // 8)  # Cap at 12.5% of total data
                    elif vol_filter == "Last 90 days":
                        # 90 days × ~78 bars/day = ~7,020 bars
                        target_rows = min(7500, total_rows // 4)  # Cap at 25% of total data
                    elif vol_filter == "Last 6 months":
                        # ~180 days × ~78 bars/day = ~14,040 bars
                        target_rows = min(15000, total_rows // 2)  # Cap at 50% of total data
                    elif vol_filter == "Last year":
                        # ~252 trading days × ~78 bars/day = ~19,656 bars
                        target_rows = min(20000, int(total_rows * 0.75))  # Cap at 75% of total data
                    else:  # All data
                        target_rows = total_rows
                    
                    start_idx = max(0, total_rows - target_rows)
                    features_filtered = features_for_prediction.iloc[start_idx:]

                    st.info(f"Processing {len(features_filtered)} data points for volatility predictions ({vol_filter})")

                    # Generate predictions using the model manager
                    predictions, _ = model_trainer.predict('volatility', features_filtered)

                    # Store predictions
                    st.session_state.volatility_predictions = predictions

                    st.success("✅ Volatility predictions generated successfully!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating volatility predictions: {str(e)}")

        # Display volatility predictions if available
        if hasattr(st.session_state, 'volatility_predictions') and st.session_state.volatility_predictions is not None:
            st.subheader("📈 Volatility Prediction Results")

            # Show prediction statistics
            predictions = st.session_state.volatility_predictions

            # Calculate key statistics
            mean_vol = float(np.mean(predictions))
            max_vol = float(np.max(predictions))
            min_vol = float(np.min(predictions))
            std_vol = float(np.std(predictions))
            median_vol = float(np.median(predictions))
            percentile_95 = float(np.percentile(predictions, 95))
            percentile_75 = float(np.percentile(predictions, 75))
            percentile_25 = float(np.percentile(predictions, 25))

            # Enhanced statistics with more details
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Avg Volatility", f"{mean_vol:.6f}")
            with col2:
                st.metric("Max Volatility", f"{max_vol:.6f}")
            with col3:
                st.metric("Min Volatility", f"{min_vol:.6f}")
            with col4:
                st.metric("Volatility Range", f"{max_vol - min_vol:.6f}")

            # Additional volatility statistics
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.metric("Std Dev", f"{std_vol:.6f}")
            with col6:
                st.metric("Median Volatility", f"{median_vol:.6f}")
            with col7:
                st.metric("95th Percentile", f"{percentile_95:.6f}")
            with col8:
                high_vol_count = int(np.sum(predictions > median_vol * 1.5))
                st.metric("High Vol Periods", f"{high_vol_count}")

            # Create comprehensive tabbed analysis for volatility predictions
            vol_tab1, vol_tab2, vol_tab3, vol_tab4, vol_tab5 = st.tabs([
                "📊 Interactive Chart", 
                "📋 Detailed Data Table", 
                "📈 Distribution Analysis", 
                "🔍 Statistical Analysis",
                "📈 Performance Metrics"
            ])

            with vol_tab1:
                st.markdown("**Price vs Predicted Volatility Analysis**")

                # Enhanced volatility prediction chart with multiple visualizations
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=('Price Chart with Volume', 'Predicted Volatility', 'Volatility Regime'),
                    row_heights=[0.5, 0.3, 0.2]
                )

                # Filter data for display consistency
                data_len = min(len(st.session_state.data), len(predictions))
                recent_data = st.session_state.data.tail(data_len)
                recent_predictions = predictions[-data_len:]

                # Add candlestick chart for price
                fig.add_trace(go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ), row=1, col=1)

                # Add volume bars
                if 'Volume' in recent_data.columns:
                    fig.add_trace(go.Bar(
                        x=recent_data.index,
                        y=recent_data['Volume'],
                        name='Volume',
                        marker_color='lightblue',
                        opacity=0.3,
                        yaxis='y2'
                    ), row=1, col=1)

                # Add volatility predictions with enhanced color coding
                volatility_colors = []
                volatility_sizes = []
                for vol in recent_predictions:
                    if vol > percentile_95:
                        volatility_colors.append('red')  # High volatility
                        volatility_sizes.append(6)
                    elif vol > percentile_75:
                        volatility_colors.append('orange')  # Medium-high volatility
                        volatility_sizes.append(5)
                    elif vol > median_vol:
                        volatility_colors.append('yellow')  # Medium volatility
                        volatility_sizes.append(4)
                    else:
                        volatility_colors.append('green')  # Low volatility
                        volatility_sizes.append(3)

                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_predictions,
                    mode='lines+markers',
                    name='Predicted Volatility',
                    line=dict(color='purple', width=2),
                    marker=dict(color=volatility_colors, size=volatility_sizes),
                    hovertemplate='Date: %{x}<br>Volatility: %{y:.6f}<extra></extra>'
                ), row=2, col=1)

                # Add volatility threshold lines
                fig.add_hline(y=median_vol, line_dash="dash", line_color="blue", 
                             annotation_text="Median", row=2, col=1)
                fig.add_hline(y=percentile_95, line_dash="dash", line_color="red", 
                             annotation_text="95th %ile", row=2, col=1)
                fig.add_hline(y=percentile_75, line_dash="dot", line_color="orange", 
                             annotation_text="75th %ile", row=2, col=1)

                # Add volatility regime visualization
                regime_colors = ['green' if vol <= percentile_25 else 
                               'yellow' if vol <= percentile_75 else 
                               'orange' if vol <= percentile_95 else 'red' 
                               for vol in recent_predictions]

                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=[1] * len(recent_predictions),
                    mode='markers',
                    name='Volatility Regime',
                    marker=dict(color=regime_colors, size=8, symbol='square'),
                    showlegend=False,
                    hovertemplate='Regime: %{marker.color}<extra></extra>'
                ), row=3, col=1)

                # Update layout
                fig.update_layout(
                    title="Comprehensive Volatility Analysis Dashboard",
                    height=900,
                    showlegend=True,
                    hovermode='x unified'
                )

                fig.update_xaxes(title_text="Time", row=3, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volatility", row=2, col=1)
                fig.update_yaxes(title_text="Regime", row=3, col=1, range=[0.5, 1.5])

                st.plotly_chart(fig, use_container_width=True)

            with vol_tab2:
                st.markdown("**Comprehensive Volatility Predictions Data Table**")

                # Create comprehensive predictions dataframe with proper data types
                num_recent = min(150, len(predictions))  # Show more predictions
                recent_predictions = predictions[-num_recent:]
                recent_prices = st.session_state.data.tail(num_recent).copy()

                try:
                    # Calculate actual volatility for comparison
                    actual_returns = recent_prices['Close'].pct_change()
                    actual_volatility = actual_returns.rolling(10).std().shift(-1)

                    # Calculate prediction accuracy metrics
                    valid_indices = ~pd.isna(actual_volatility)
                    prediction_error = np.abs(recent_predictions - actual_volatility.values)
                    relative_error = (prediction_error / actual_volatility.values) * 100

                    # Create regime classification
                    def classify_volatility(vol):
                        if vol > percentile_95:
                            return "🔴 Very High"
                        elif vol > percentile_75:
                            return "🟠 High" 
                        elif vol > median_vol:
                            return "🟡 Medium"
                        elif vol > percentile_25:
                            return "🟢 Low"
                        else:
                            return "🔵 Very Low"

                    # Create the main predictions dataframe with improved datetime handling
                    try:
                        # Debug print to understand the index format
                        print(f"DEBUG Volatility: Index type: {type(recent_prices.index)}")
                        print(f"DEBUG Volatility: Index dtype: {recent_prices.index.dtype}")
                        print(f"DEBUG Volatility: First few index values: {recent_prices.index[:5].tolist()}")
                        
                        if pd.api.types.is_datetime64_any_dtype(recent_prices.index):
                            # Already datetime index
                            date_col = recent_prices.index.strftime('%Y-%m-%d')
                            time_col = recent_prices.index.strftime('%H:%M:%S')
                        elif pd.api.types.is_numeric_dtype(recent_prices.index):
                            # Handle different timestamp formats
                            sample_val = recent_prices.index[0]
                            print(f"DEBUG Volatility: Sample timestamp value: {sample_val}")
                            
                            # Try different timestamp conversion approaches
                            datetime_index = None
                            if sample_val > 1e12:  # Millisecond timestamps
                                datetime_index = pd.to_datetime(recent_prices.index, unit='ms', errors='coerce')
                                print("DEBUG Volatility: Trying millisecond conversion")
                            elif sample_val > 1e9:  # Second timestamps
                                datetime_index = pd.to_datetime(recent_prices.index, unit='s', errors='coerce')
                                print("DEBUG Volatility: Trying second conversion")
                            else:  # Might be days since epoch or other format
                                # Try interpreting as days since epoch
                                datetime_index = pd.to_datetime(recent_prices.index, unit='D', errors='coerce', origin='1970-01-01')
                                print("DEBUG Volatility: Trying days conversion")
                            
                            if datetime_index is not None and not datetime_index.isna().all():
                                print(f"DEBUG Volatility: Converted datetime sample: {datetime_index[:3].tolist()}")
                                date_col = datetime_index.strftime('%Y-%m-%d')
                                time_col = datetime_index.strftime('%H:%M:%S')
                            else:
                                print("DEBUG Volatility: Using sequential numbering")
                                # Use sequential numbering based on actual data range
                                date_col = [f"Data_{i+1}" for i in range(len(recent_prices))]
                                time_col = [f"{(9 + i // 12) % 24:02d}:{((i*5) % 60):02d}:00" for i in range(len(recent_prices))]
                        else:
                            print("DEBUG Volatility: Non-numeric index, using sequential")
                            # Non-numeric index - use sequential numbering
                            date_col = [f"Data_{i+1}" for i in range(len(recent_prices))]
                            time_col = [f"{(9 + i // 12) % 24:02d}:{((i*5) % 60):02d}:00" for i in range(len(recent_prices))]
                            
                    except Exception as e:
                        print(f"DEBUG Volatility: Error in datetime conversion: {e}")
                        # Fallback with more realistic time simulation
                        date_col = [f"Data_{i+1}" for i in range(len(recent_prices))]
                        time_col = [f"{(9 + i // 12) % 24:02d}:{((i*5) % 60):02d}:00" for i in range(len(recent_prices))]

                    predictions_df = pd.DataFrame({
                        'Date': date_col,
                        'Time': time_col,
                        'Open': recent_prices['Open'].round(4),
                        'High': recent_prices['High'].round(4),
                        'Low': recent_prices['Low'].round(4),
                        'Close': recent_prices['Close'].round(4),
                        'Predicted_Vol': recent_predictions.round(6),
                        'Actual_Vol': actual_volatility.round(6),
                        'Error': prediction_error.round(6),
                        'Error_Pct': relative_error.round(2),
                        'Vol_Regime': [classify_volatility(pred) for pred in recent_predictions],
                        'Price_Change': recent_prices['Close'].pct_change().round(4),
                        'Vol_Rank': pd.qcut(recent_predictions, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
                    })

                    # Display the dataframe with enhanced formatting
                    st.dataframe(
                        predictions_df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Date": st.column_config.DateColumn("Date"),
                            "Time": st.column_config.TimeColumn("Time"),
                            "Open": st.column_config.NumberColumn("Open", format="%.4f"),
                            "High": st.column_config.NumberColumn("High", format="%.4f"),
                            "Low": st.column_config.NumberColumn("Low", format="%.4f"),
                            "Close": st.column_config.NumberColumn("Close", format="%.4f"),
                            "Predicted_Vol": st.column_config.NumberColumn("Pred Vol", format="%.6f"),
                            "Actual_Vol": st.column_config.NumberColumn("Actual Vol", format="%.6f"),
                            "Error": st.column_config.NumberColumn("Error", format="%.6f"),
                            "Error_Pct": st.column_config.NumberColumn("Error %", format="%.2f%%"),
                            "Price_Change": st.column_config.NumberColumn("Price Δ", format="%.4f"),
                        }
                    )

                    # Show summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_error = np.nanmean(prediction_error)
                        st.metric("Avg Prediction Error", f"{avg_error:.6f}")
                    with col2:
                        avg_rel_error = np.nanmean(relative_error)
                        st.metric("Avg Relative Error", f"{avg_rel_error:.2f}%")
                    with col3:
                        correlation = np.corrcoef(recent_predictions[valid_indices], 
                                                actual_volatility[valid_indices])[0,1]
                        st.metric("Prediction Correlation", f"{correlation:.3f}")
                    with col4:
                        rmse = np.sqrt(np.nanmean(prediction_error**2))
                        st.metric("RMSE", f"{rmse:.6f}")

                except Exception as df_error:
                    st.warning("Creating simplified data table due to data processing issue")
                    # Fallback simplified table
                    # Handle different index types safely
                    if hasattr(recent_prices.index, 'strftime'):
                        date_col = recent_prices.index.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        date_col = [f"Point_{i+1}" for i in range(len(recent_prices))]

                    simple_df = pd.DataFrame({
                        'Date': date_col,
                        'Close_Price': recent_prices['Close'].round(4),
                        'Predicted_Volatility': recent_predictions.round(6),
                        'Volatility_Level': [classify_volatility(pred) for pred in recent_predictions],
                        'Vol_Percentile': pd.qcut(recent_predictions, 10, labels=False) + 1
                    })
                    st.dataframe(simple_df, use_container_width=True, hide_index=True)

                # Enhanced download options
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        csv_data = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Data CSV",
                            data=csv_data,
                            file_name=f"volatility_predictions_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except:
                        st.info("Full download not available")

                with col2:
                    # Simple summary download
                    summary_df = pd.DataFrame({
                        'Timestamp': recent_prices.index,
                        'Predicted_Volatility': recent_predictions,
                        'Volatility_Regime': [classify_volatility(pred) for pred in recent_predictions]
                    })
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=summary_csv,
                        file_name=f"volatility_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            with vol_tab3:
                st.markdown("**Volatility Distribution and Pattern Analysis**")

                # Enhanced distribution analysis with multiple charts
                col1, col2 = st.columns(2)

                with col1:
                    # Volatility histogram with statistical overlays
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=predictions,
                        nbinsx=50,
                        name='Volatility Distribution',
                        marker_color='lightblue',
                        opacity=0.7,
                        histnorm='probability'
                    ))

                    # Add statistical lines
                    fig_hist.add_vline(x=mean_vol, line_dash="solid", line_color="red", 
                                      annotation_text=f"Mean: {mean_vol:.6f}")
                    fig_hist.add_vline(x=median_vol, line_dash="dash", line_color="blue", 
                                      annotation_text=f"Median: {median_vol:.6f}")
                    fig_hist.add_vline(x=percentile_95, line_dash="dot", line_color="orange", 
                                      annotation_text=f"95th: {percentile_95:.6f}")

                    fig_hist.update_layout(
                        title="Volatility Distribution with Statistics",
                        xaxis_title="Volatility",
                        yaxis_title="Probability Density",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    # Enhanced box plot with outlier analysis
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=predictions,
                        name='Volatility',
                        boxpoints='suspectedoutliers',
                        marker_color='lightgreen',
                        fillcolor='rgba(144,238,144,0.5)',
                        line_color='green'
                    ))
                    fig_box.update_layout(
                        title="Volatility Box Plot with Outliers",
                        yaxis_title="Volatility",
                        height=400
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

                # Time series decomposition of volatility
                st.markdown("**Volatility Time Series Analysis**")

                # Volatility over time with trend
                fig_ts = go.Figure()

                # Add the main volatility line
                data_len = min(len(st.session_state.data), len(predictions))
                recent_data = st.session_state.data.tail(data_len)
                recent_predictions = predictions[-data_len:]

                fig_ts.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_predictions,
                    mode='lines',
                    name='Predicted Volatility',
                    line=dict(color='blue', width=1)
                ))

                # Add moving average trend
                if len(recent_predictions) >= 20:
                    vol_ma = pd.Series(recent_predictions).rolling(20).mean()
                    fig_ts.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=vol_ma,
                        mode='lines',
                        name='20-Period MA',
                        line=dict(color='red', width=2)
                    ))

                # Add regime bands
                fig_ts.add_hline(y=percentile_95, line_dash="dash", line_color="red", 
                                annotation_text="High Vol Threshold")
                fig_ts.add_hline(y=median_vol, line_dash="dash", line_color="blue", 
                                annotation_text="Median")
                fig_ts.add_hline(y=percentile_25, line_dash="dash", line_color="green", 
                                annotation_text="Low Vol Threshold")

                fig_ts.update_layout(
                    title="Volatility Time Series with Trend and Regimes",
                    xaxis_title="Time",
                    yaxis_title="Volatility",
                    height=400
                )
                st.plotly_chart(fig_ts, use_container_width=True)

            with vol_tab4:
                st.markdown("**Advanced Statistical Analysis**")

                # Comprehensive statistics table
                stats_data = {
                    'Statistic': [
                        'Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Range',
                        '1st Percentile', '5th Percentile', '25th Percentile', 'Median (50th)', 
                        '75th Percentile', '95th Percentile', '99th Percentile',
                        'Skewness', 'Kurtosis', 'Coefficient of Variation'
                    ],
                    'Value': [
                        f"{len(predictions):,}",
                        f"{mean_vol:.6f}",
                        f"{std_vol:.6f}",
                        f"{min_vol:.6f}",
                        f"{max_vol:.6f}",
                        f"{max_vol - min_vol:.6f}",
                        f"{np.percentile(predictions, 1):.6f}",
                        f"{np.percentile(predictions, 5):.6f}",
                        f"{percentile_25:.6f}",
                        f"{median_vol:.6f}",
                        f"{percentile_75:.6f}",
                        f"{percentile_95:.6f}",
                        f"{np.percentile(predictions, 99):.6f}",
                        f"{float(pd.Series(predictions).skew()):.4f}",
                        f"{float(pd.Series(predictions).kurtosis()):.4f}",
                        f"{(std_vol/mean_vol):.4f}"
                    ]
                }

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

                # Volatility regime analysis
                st.markdown("**Volatility Regime Classification**")

                low_vol_threshold = percentile_25
                med_low_threshold = median_vol
                med_high_threshold = percentile_75
                high_vol_threshold = percentile_95

                very_low_count = int(np.sum(predictions <= low_vol_threshold))
                low_count = int(np.sum((predictions > low_vol_threshold) & (predictions <= med_low_threshold)))
                medium_count = int(np.sum((predictions > med_low_threshold) & (predictions <= med_high_threshold)))
                high_count = int(np.sum((predictions > med_high_threshold) & (predictions <= high_vol_threshold)))
                very_high_count = int(np.sum(predictions > high_vol_threshold))

                total_count = len(predictions)

                regime_df = pd.DataFrame({
                    'Volatility_Regime': ['Very Low (≤25th)', 'Low (25th-50th)', 'Medium (50th-75th)', 
                                         'High (75th-95th)', 'Very High (>95th)'],
                    'Count': [very_low_count, low_count, medium_count, high_count, very_high_count],
                    'Percentage': [
                        f"{(very_low_count/total_count*100):.1f}%",
                        f"{(low_count/total_count*100):.1f}%",
                        f"{(medium_count/total_count*100):.1f}%",
                        f"{(high_count/total_count*100):.1f}%",
                        f"{(very_high_count/total_count*100):.1f}%"
                    ],
                    'Threshold_Range': [
                        f"≤ {low_vol_threshold:.6f}",
                        f"{low_vol_threshold:.6f} - {med_low_threshold:.6f}",
                        f"{med_low_threshold:.6f} - {med_high_threshold:.6f}",
                        f"{med_high_threshold:.6f} - {high_vol_threshold:.6f}",
                        f"> {high_vol_threshold:.6f}"
                    ]
                })

                st.dataframe(regime_df, use_container_width=True, hide_index=True)

                # Volatility trend analysis
                st.markdown("**Volatility Trend Analysis**")
                if len(predictions) >= 40:
                    recent_40 = predictions[-40:]
                    recent_20 = recent_40[-20:]
                    previous_20 = recent_40[:20]

                    recent_avg = float(np.mean(recent_20))
                    previous_avg = float(np.mean(previous_20))
                    volatility_trend = "📈 Increasing" if recent_avg > previous_avg else "📉 Decreasing"
                    trend_change = ((recent_avg - previous_avg) / previous_avg) * 100

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Recent Avg (Last 20)", f"{recent_avg:.6f}")
                    with col2:
                        st.metric("Previous Avg (20 before)", f"{previous_avg:.6f}")
                    with col3:
                        st.metric("Trend Direction", volatility_trend)
                    with col4:
                        st.metric("Trend Change", f"{trend_change:+.2f}%")

            with vol_tab5:
                st.markdown("**Model Performance and Validation Metrics**")

                # Model performance analysis
                if len(predictions) >= 20:
                    try:
                        # Calculate rolling performance metrics
                        recent_data = st.session_state.data.tail(len(predictions))
                        actual_returns = recent_data['Close'].pct_change()
                        actual_volatility = actual_returns.rolling(10).std()

                        # Performance metrics where we have actual data
                        valid_mask = ~pd.isna(actual_volatility)
                        if valid_mask.sum() > 10:
                            pred_valid = predictions[valid_mask.values]
                            actual_valid = actual_volatility[valid_mask].values

                            # Calculate comprehensive metrics
                            mae = float(np.mean(np.abs(pred_valid - actual_valid)))
                            rmse = float(np.sqrt(np.mean((pred_valid - actual_valid)**2)))
                            mape = float(np.mean(np.abs((pred_valid - actual_valid) / actual_valid)) * 100)
                            correlation = float(np.corrcoef(pred_valid, actual_valid)[0,1])
                            r_squared = float(correlation ** 2)

                            # Display performance metrics
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("MAE", f"{mae:.6f}")
                            with col2:
                                st.metric("RMSE", f"{rmse:.6f}")
                            with col3:
                                st.metric("MAPE", f"{mape:.2f}%")
                            with col4:
                                st.metric("Correlation", f"{correlation:.4f}")
                            with col5:
                                st.metric("R²", f"{r_squared:.4f}")

                            # Prediction vs Actual scatter plot
                            fig_scatter = go.Figure()
                            fig_scatter.add_trace(go.Scatter(
                                x=actual_valid,
                                y=pred_valid,
                                mode='markers',
                                name='Predictions vs Actual',
                                marker=dict(color='blue', size=6, opacity=0.6)
                            ))

                            # Add perfect prediction line
                            min_val = min(np.min(actual_valid), np.min(pred_valid))
                            max_val = max(np.max(actual_valid), np.max(pred_valid))
                            fig_scatter.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            ))

                            fig_scatter.update_layout(
                                title="Predicted vs Actual Volatility",
                                xaxis_title="Actual Volatility",
                                yaxis_title="Predicted Volatility",
                                height=400
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)

                        else:
                            st.info("Insufficient actual volatility data for validation")

                    except Exception as perf_error:
                        st.warning("Performance metrics calculation not available")

                # Feature importance (if available from model)
                st.markdown("**Model Feature Analysis**")
                try:
                    model_trainer = st.session_state.model_trainer
                    feature_importance = model_trainer.get_feature_importance('volatility')

                    if feature_importance:
                        # Convert to sorted dataframe
                        importance_df = pd.DataFrame(
                            list(feature_importance.items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)

                        # Show top features
                        st.markdown("**Top 15 Most Important Features**")
                        top_features = importance_df.head(15)
                        st.dataframe(top_features, use_container_width=True, hide_index=True)

                        # Feature importance chart
                        fig_importance = go.Figure()
                        fig_importance.add_trace(go.Bar(
                            x=top_features['Importance'],
                            y=top_features['Feature'],
                            orientation='h',
                            name='Feature Importance',
                            marker_color='lightcoral'
                        ))
                        fig_importance.update_layout(
                            title="Top Feature Importance",
                            xaxis_title="Importance Score",
                            yaxis_title="Features",
                            height=500
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("Feature importance data not available")

                except Exception as feature_error:
                    st.info("Feature importance analysis not available")

                # Model configuration summary
                st.markdown("**Model Configuration Summary**")
                config_info = {
                    'Model Type': 'Ensemble (Volatility Prediction)',
                    'Task Type': 'Regression',
                    'Prediction Window': '1 Period Ahead',
                    'Total Predictions': f"{len(predictions):,}",
                    'Feature Count': f"{len(st.session_state.features.columns) if st.session_state.features is not None else 'N/A'}",
                    'Data Points Used': f"{len(st.session_state.data):,}",
                    'Prediction Date Range': safe_format_date_range(st.session_state.data.index)
                }

                config_df = pd.DataFrame(
                    list(config_info.items()),
                    columns=['Configuration', 'Value']
                )
                st.dataframe(config_df, use_container_width=True, hide_index=True)

# Direction Predictions Tab
with direction_tab:
    st.header("🎯 Direction Predictions")

    # Check if direction features and model are available
    direction_features_available = (hasattr(st.session_state, 'direction_features') and 
                                   st.session_state.direction_features is not None)
    direction_model_available = (hasattr(st.session_state, 'direction_trained_models') and 
                                st.session_state.direction_trained_models and
                                'direction' in st.session_state.direction_trained_models and
                                st.session_state.direction_trained_models['direction'] is not None)

    # Show status information
    col1, col2 = st.columns(2)
    with col1:
        if direction_features_available:
            st.success("✅ Direction features calculated")
        else:
            st.warning("⚠️ Direction features not calculated")

    with col2:
        if direction_model_available:
            st.success("✅ Direction model trained")
        else:
            st.warning("⚠️ Direction model not trained")

    # Show instructions if prerequisites are missing
    if not direction_features_available or not direction_model_available:
        st.info("""
        📋 **To use Direction Predictions:**
        1. Go to **Model Training** page
        2. Click on **Direction Predictions** tab
        3. Train the direction model
        4. Return here to generate predictions
        """)



        # Show preview of what will be available
        st.subheader("🔮 Preview: Direction Prediction Features")
        st.markdown("""
        **Once the direction model is trained, you'll see:**
        - 📈 **Interactive Candlestick Chart** with bullish/bearish signals
        - 🎯 **Confidence-Based Visualization** with signal strength indicators
        - 📊 **Comprehensive Analysis Tabs:**
          - Recent Predictions with OHLC data and accuracy validation
          - Performance Metrics with confidence distribution
          - Signal Quality Assessment with strength categories
        - 📋 **Real-time Statistics** including prediction accuracy and confidence levels
        """)

        # Show sample chart placeholder
        st.info("💡 **Sample visualization will appear here after model training**")

    # Show prediction interface if everything is available
    if direction_features_available and direction_model_available:
        # Direction prediction controls
        st.subheader("🎯 Prediction Controls")

        col1, col2 = st.columns(2)

        with col1:
            dir_filter = st.selectbox(
                "📅 Time Period Filter",
                ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All data"],
                index=1,
                help="Select the time period for direction predictions",
                key="dir_filter"
            )

        with col2:
            st.metric("Direction Model Status", "✅ Ready", help="Direction model is trained and ready")

        # Generate direction predictions button
        if st.button("🚀 Generate Direction Predictions", type="primary", key="dir_predict"):
            try:
                with st.spinner("Generating direction predictions..."):
                    # Get direction model and features
                    direction_model = st.session_state.direction_trained_models['direction']
                    direction_features = st.session_state.direction_features.copy()

                    # Apply direction filter using realistic row counts for trading data
                    total_rows = len(direction_features)
                    
                    if dir_filter == "Last 30 days":
                        # 30 days × ~78 bars/day (6.5 trading hours × 12 bars/hour) = ~2,340 bars
                        target_rows = min(2500, total_rows // 8)  # Cap at 12.5% of total data
                    elif dir_filter == "Last 90 days":
                        # 90 days × ~78 bars/day = ~7,020 bars
                        target_rows = min(7500, total_rows // 4)  # Cap at 25% of total data
                    elif dir_filter == "Last 6 months":
                        # ~180 days × ~78 bars/day = ~14,040 bars
                        target_rows = min(15000, total_rows // 2)  # Cap at 50% of total data
                    elif dir_filter == "Last year":
                        # ~252 trading days × ~78 bars/day = ~19,656 bars
                        target_rows = min(20000, int(total_rows * 0.75))  # Cap at 75% of total data
                    else:  # All data
                        target_rows = total_rows
                    
                    start_idx = max(0, total_rows - target_rows)
                    direction_features_filtered = direction_features.iloc[start_idx:]

                    st.info(f"Processing {len(direction_features_filtered)} data points for {dir_filter}")

                    # Generate predictions
                    predictions, probabilities = direction_model.predict(direction_features_filtered)

                    # Store predictions
                    st.session_state.direction_predictions = predictions
                    st.session_state.direction_probabilities = probabilities

                    st.success("✅ Direction predictions generated successfully!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating direction predictions: {str(e)}")
                # Show only the main error message, not full traceback
                if "Missing required features" in str(e):
                    st.error("The model was trained with different features. Please retrain the model.")
                elif "No selected features" in str(e):
                    st.error("Model training data is incomplete. Please retrain the model.")
                else:
                    st.error(f"Technical error: {str(e)[:100]}...")



        # Display direction predictions if available
        if hasattr(st.session_state, 'direction_predictions') and st.session_state.direction_predictions is not None:
            st.subheader("📈 Direction Prediction Results")

            # Show prediction statistics
            predictions = st.session_state.direction_predictions
            probabilities = st.session_state.direction_probabilities

            # Apply same time filter to data for display consistency
            if dir_filter == "Last 30 days":
                cutoff_date = st.session_state.data.index.max() - pd.Timedelta(days=30)
            elif dir_filter == "Last 90 days":
                cutoff_date = st.session_state.data.index.max() - pd.Timedelta(days=90)
            elif dir_filter == "Last 6 months":
                cutoff_date = st.session_state.data.index.max() - pd.Timedelta(days=180)
            elif dir_filter == "Last year":
                cutoff_date = st.session_state.data.index.max() - pd.Timedelta(days=365)
            else:  # All data
                cutoff_date = st.session_state.data.index.min()

            # Filter data for display
            filtered_data = st.session_state.data[st.session_state.data.index >= cutoff_date]

            # Enhanced statistics
            bullish_count = np.sum(predictions == 1)
            bearish_count = np.sum(predictions == 0)
            bullish_pct = (bullish_count / len(predictions)) * 100

            if probabilities is not None:
                avg_confidence = np.mean(np.max(probabilities, axis=1)) * 100
                high_confidence = np.sum(np.max(probabilities, axis=1) > 0.7)
                high_conf_pct = (high_confidence / len(predictions)) * 100
            else:
                avg_confidence = 0
                high_confidence = 0
                high_conf_pct = 0

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Predictions", f"{len(predictions)}")
            with col2:
                st.metric("Bullish Signals", f"{bullish_count} ({bullish_pct:.1f}%)")
            with col3:
                st.metric("Bearish Signals", f"{bearish_count} ({100-bullish_pct:.1f}%)")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

            # Additional statistics
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.metric("High Confidence", f"{high_confidence} ({high_conf_pct:.1f}%)")
            with col6:
                recent_bullish = np.sum(predictions[-20:] == 1) if len(predictions) >= 20 else np.sum(predictions == 1)
                recent_pct = (recent_bullish / min(20, len(predictions))) * 100
                st.metric("Recent Bullish (Last 20)", f"{recent_bullish} ({recent_pct:.1f}%)")
            with col7:
                if probabilities is not None:
                    recent_conf = np.mean(np.max(probabilities[-20:], axis=1)) * 100 if len(probabilities) >= 20 else avg_confidence
                    st.metric("Recent Confidence", f"{recent_conf:.1f}%")
                else:
                    st.metric("Recent Confidence", "N/A")
            with col8:
                price_change = ((filtered_data['Close'].iloc[-1] - filtered_data['Close'].iloc[0]) / filtered_data['Close'].iloc[0]) * 100
                st.metric("Price Change", f"{price_change:.2f}%")

            # Create direction prediction chart
            try:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Price Chart', 'Direction Predictions'),
                    row_heights=[0.7, 0.3]
                )

                # Use filtered data for chart (matching the prediction timeframe)
                data_len = min(len(filtered_data), len(predictions))
                recent_data = filtered_data.tail(data_len)

                # Add candlestick chart for price
                fig.add_trace(go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ), row=1, col=1)

                # Add direction predictions with confidence coloring
                pred_data = predictions[-data_len:]
                prob_data = probabilities[-data_len:] if probabilities is not None else None

                # Create confidence-based colors
                if prob_data is not None:
                    confidences = np.max(prob_data, axis=1)
                    colors = ['rgba(0, 255, 0, ' + str(conf) + ')' for conf in confidences]
                    sizes = [6 + 6 * conf for conf in confidences]  # Size based on confidence
                else:
                    confidences = np.ones(len(pred_data)) * 0.5  # Default confidence for fallback
                    colors = ['green'] * len(pred_data)
                    sizes = [8] * len(pred_data)

                # Bullish signals
                bullish_mask = pred_data == 1
                if np.any(bullish_mask):
                    bullish_colors = [colors[i] for i in range(len(colors)) if bullish_mask[i]]
                    bullish_sizes = [sizes[i] for i in range(len(sizes)) if bullish_mask[i]]

                    fig.add_trace(go.Scatter(
                        x=recent_data.index[bullish_mask],
                        y=[1] * np.sum(bullish_mask),
                        mode='markers',
                        name='Bullish',
                        marker=dict(color=bullish_colors if prob_data is not None else 'green', 
                                   size=bullish_sizes if prob_data is not None else 10, 
                                   symbol='triangle-up'),
                        text=[f'Confidence: {confidences[i]:.1%}' for i in range(len(confidences)) if bullish_mask[i]] if prob_data is not None else None,
                        hovertemplate='Bullish Signal<br>%{text}<extra></extra>' if prob_data is not None else 'Bullish Signal<extra></extra>'
                    ), row=2, col=1)

                # Bearish signals
                bearish_mask = pred_data == 0
                if np.any(bearish_mask):
                    bearish_colors = [colors[i] for i in range(len(colors)) if bearish_mask[i]]
                    bearish_sizes = [sizes[i] for i in range(len(sizes)) if bearish_mask[i]]

                    fig.add_trace(go.Scatter(
                        x=recent_data.index[bearish_mask],
                        y=[0] * np.sum(bearish_mask),
                        mode='markers',
                        name='Bearish',
                        marker=dict(color=bearish_colors if prob_data is not None else 'red', 
                                   size=bearish_sizes if prob_data is not None else 10, 
                                   symbol='triangle-down'),
                        text=[f'Confidence: {confidences[i]:.1%}' for i in range(len(confidences)) if bearish_mask[i]] if prob_data is not None else None,
                        hovertemplate='Bearish Signal<br>%{text}<extra></extra>' if prob_data is not None else 'Bearish Signal<extra></extra>'
                    ), row=2, col=1)

                # Update layout
                fig.update_layout(
                    title="Price vs Direction Predictions",
                    height=600,
                    showlegend=True
                )

                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Direction", row=2, col=1, range=[-0.1, 1.1])

                st.plotly_chart(fig, use_container_width=True)

            except Exception as chart_error:
                st.error(f"Chart generation error: {str(chart_error)[:100]}...")
                # Show basic prediction data without chart
                st.subheader("📊 Basic Prediction Results")
                bullish_signals = np.sum(predictions == 1)
                total_signals = len(predictions)
                st.metric("Bullish Signals", f"{bullish_signals}/{total_signals}")
                st.metric("Bearish Signals", f"{total_signals - bullish_signals}/{total_signals}")

            # Show detailed analysis section
            st.subheader("📊 Detailed Direction Analysis")

            # Create comprehensive tabbed analysis for direction predictions (5 tabs like volatility)
            dir_tab1, dir_tab2, dir_tab3, dir_tab4, dir_tab5 = st.tabs([
                "📊 Interactive Chart", 
                "📋 Detailed Data Table", 
                "📈 Distribution Analysis", 
                "🔍 Statistical Analysis",
                "📈 Performance Metrics"
            ])

            with dir_tab1:
                st.markdown("**Enhanced Price vs Direction Predictions Analysis**")

                # Enhanced direction prediction chart with additional analysis
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=('Price Chart with Direction Signals', 'Signal Confidence', 'Direction Pattern'),
                    row_heights=[0.5, 0.3, 0.2]
                )

                # Use filtered data for chart
                data_len = min(len(filtered_data), len(predictions))
                recent_data = filtered_data.tail(data_len)
                pred_data = predictions[-data_len:]
                prob_data = probabilities[-data_len:] if probabilities is not None else None

                # Add candlestick chart for price
                fig.add_trace(go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ), row=1, col=1)

                # Add direction arrows on price chart
                for i, (idx, pred) in enumerate(zip(recent_data.index, pred_data)):
                    if i % 5 == 0:  # Show every 5th prediction to avoid clutter
                        if pred == 1:  # Bullish
                            fig.add_annotation(
                                x=idx, y=recent_data.iloc[i]['High'],
                                text="▲", showarrow=False,
                                font=dict(color="green", size=12),
                                row=1, col=1
                            )
                        else:  # Bearish
                            fig.add_annotation(
                                x=idx, y=recent_data.iloc[i]['Low'],
                                text="▼", showarrow=False,
                                font=dict(color="red", size=12),
                                row=1, col=1
                            )

                # Add confidence line chart
                if prob_data is not None:
                    confidences = np.max(prob_data, axis=1)
                    fig.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=confidences,
                        mode='lines+markers',
                        name='Prediction Confidence',
                        line=dict(color='purple', width=2),
                        marker=dict(size=4),
                        hovertemplate='Confidence: %{y:.1%}<extra></extra>'
                    ), row=2, col=1)

                    # Add confidence threshold lines
                    fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                                 annotation_text="High Confidence", row=2, col=1)
                    fig.add_hline(y=0.6, line_dash="dot", line_color="orange", 
                                 annotation_text="Medium Confidence", row=2, col=1)

                # Add direction pattern visualization
                direction_y = [1 if p == 1 else 0 for p in pred_data]
                colors = ['green' if p == 1 else 'red' for p in pred_data]

                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=direction_y,
                    mode='markers',
                    name='Direction Pattern',
                    marker=dict(color=colors, size=6),
                    showlegend=False,
                    hovertemplate='Direction: %{text}<extra></extra>',
                    text=['Bullish' if p == 1 else 'Bearish' for p in pred_data]
                ), row=3, col=1)

                # Update layout
                fig.update_layout(
                    title="Comprehensive Direction Analysis Dashboard",
                    height=800,
                    showlegend=True,
                    hovermode='x unified'
                )

                fig.update_xaxes(title_text="Time", row=3, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Confidence", row=2, col=1, range=[0, 1])
                fig.update_yaxes(title_text="Direction", row=3, col=1, range=[-0.1, 1.1])

                st.plotly_chart(fig, use_container_width=True)

            with dir_tab2:
                st.markdown("**Comprehensive Direction Predictions Data Table**")

                # Create comprehensive predictions dataframe
                # Use all available data from the selected time filter instead of limiting to 150 rows
                num_recent = len(predictions)  # Use all predictions from the selected time filter
                recent_predictions = predictions
                recent_probs = probabilities if probabilities is not None else None
                recent_prices = filtered_data.tail(num_recent)

                # Ensure data alignment - match lengths
                data_len = min(len(recent_prices), len(recent_predictions))
                if recent_probs is not None and len(recent_probs) > 0:
                    data_len = min(data_len, len(recent_probs))

                # Trim all arrays to the same length
                recent_prices_aligned = recent_prices.tail(data_len)
                recent_predictions_aligned = recent_predictions[-data_len:]
                recent_probs_aligned = recent_probs[-data_len:] if recent_probs is not None else None

                # Calculate price changes for validation
                price_changes = recent_prices_aligned['Close'].pct_change().shift(-1) * 100
                actual_direction = (price_changes > 0).astype(int)

                # Calculate prediction accuracy metrics
                valid_indices = ~pd.isna(actual_direction)
                prediction_correct = (recent_predictions_aligned == actual_direction.values) & valid_indices

                # Create signal classification
                def classify_signal_strength(pred, conf):
                    if conf is None:
                        return "🟡 Medium"
                    if conf > 0.8:
                        return "🟢 Very Strong" if pred == 1 else "🔴 Very Strong"
                    elif conf > 0.7:
                        return "🟢 Strong" if pred == 1 else "🔴 Strong"
                    elif conf > 0.6:
                        return "🟢 Medium" if pred == 1 else "🔴 Medium"
                    else:
                        return "🟡 Weak"

                # Create the main predictions dataframe with improved datetime handling
                try:
                    # Debug print to understand the index format
                    print(f"DEBUG Direction: Index type: {type(recent_prices_aligned.index)}")
                    print(f"DEBUG Direction: Index dtype: {recent_prices_aligned.index.dtype}")
                    print(f"DEBUG Direction: First few index values: {recent_prices_aligned.index[:5].tolist()}")
                    
                    if pd.api.types.is_datetime64_any_dtype(recent_prices_aligned.index):
                        # Already datetime index
                        date_col = recent_prices_aligned.index.strftime('%Y-%m-%d')
                        time_col = recent_prices_aligned.index.strftime('%H:%M:%S')
                    elif pd.api.types.is_numeric_dtype(recent_prices_aligned.index):
                        # Handle different timestamp formats
                        sample_val = recent_prices_aligned.index[0]
                        print(f"DEBUG Direction: Sample timestamp value: {sample_val}")
                        
                        # Try different timestamp conversion approaches
                        datetime_index = None
                        if sample_val > 1e12:  # Millisecond timestamps
                            datetime_index = pd.to_datetime(recent_prices_aligned.index, unit='ms', errors='coerce')
                            print("DEBUG Direction: Trying millisecond conversion")
                        elif sample_val > 1e9:  # Second timestamps
                            datetime_index = pd.to_datetime(recent_prices_aligned.index, unit='s', errors='coerce')
                            print("DEBUG Direction: Trying second conversion")
                        else:  # Might be days since epoch or other format
                            # Try interpreting as days since epoch
                            datetime_index = pd.to_datetime(recent_prices_aligned.index, unit='D', errors='coerce', origin='1970-01-01')
                            print("DEBUG Direction: Trying days conversion")
                        
                        if datetime_index is not None and not datetime_index.isna().all():
                            print(f"DEBUG Direction: Converted datetime sample: {datetime_index[:3].tolist()}")
                            date_col = datetime_index.strftime('%Y-%m-%d')
                            time_col = datetime_index.strftime('%H:%M:%S')
                        else:
                            print("DEBUG Direction: Using sequential numbering")
                            # Use sequential numbering based on actual data range
                            date_col = [f"Data_{i+1}" for i in range(len(recent_prices_aligned))]
                            time_col = [f"{(9 + i // 12) % 24:02d}:{((i*5) % 60):02d}:00" for i in range(len(recent_prices_aligned))]
                    else:
                        print("DEBUG Direction: Non-numeric index, using sequential")
                        # Non-numeric index - use sequential numbering
                        date_col = [f"Data_{i+1}" for i in range(len(recent_prices_aligned))]
                        time_col = [f"{(9 + i // 12) % 24:02d}:{((i*5) % 60):02d}:00" for i in range(len(recent_prices_aligned))]
                        
                except Exception as e:
                    print(f"DEBUG Direction: Error in datetime conversion: {e}")
                    # Fallback with more realistic time simulation
                    date_col = [f"Data_{i+1}" for i in range(len(recent_prices_aligned))]
                    time_col = [f"{(9 + i // 12) % 24:02d}:{((i*5) % 60):02d}:00" for i in range(len(recent_prices_aligned))]

                # Debug: ensure all arrays have the same length
                print(f"DEBUG: data_len={data_len}, recent_prices_aligned={len(recent_prices_aligned)}, "
                      f"recent_predictions_aligned={len(recent_predictions_aligned)}, "
                      f"price_changes={len(price_changes)}, actual_direction={len(actual_direction)}")

                # Ensure all arrays match the exact same length
                actual_len = len(recent_prices_aligned)
                recent_predictions_aligned = recent_predictions_aligned[:actual_len]
                if recent_probs_aligned is not None:
                    recent_probs_aligned = recent_probs_aligned[:actual_len]
                price_changes = price_changes[:actual_len]
                actual_direction = actual_direction[:actual_len]
                prediction_correct = prediction_correct[:actual_len]
                valid_indices = valid_indices[:actual_len]

                # Ensure date/time columns match the same length
                date_col = date_col[:actual_len]
                time_col = time_col[:actual_len]

                # Debug: Print lengths after alignment
                print(f"DEBUG FINAL: actual_len={actual_len}, date_col={len(date_col)}, time_col={len(time_col)}, "
                      f"recent_predictions_aligned={len(recent_predictions_aligned)}, "
                      f"recent_probs_aligned={len(recent_probs_aligned) if recent_probs_aligned is not None else 'None'}, "
                      f"price_changes={len(price_changes)}, actual_direction={len(actual_direction)}")

                # Build DataFrame step by step to avoid index mismatch
                # Convert pandas series to plain lists to avoid index conflicts
                ohlc_dict = {
                    'Date': date_col,
                    'Time': time_col,
                    'Open': recent_prices_aligned['Open'].values.round(4).tolist(),
                    'High': recent_prices_aligned['High'].values.round(4).tolist(),
                    'Low': recent_prices_aligned['Low'].values.round(4).tolist(),
                    'Close': recent_prices_aligned['Close'].values.round(4).tolist(),
                }

                # Convert all other arrays to simple lists
                pred_dict = {
                    'Predicted_Dir': ['🟢 Bullish' if p == 1 else '🔴 Bearish' for p in recent_predictions_aligned],
                    'Confidence': ([f"{np.max(prob):.3f}" for prob in recent_probs_aligned] 
                                  if recent_probs_aligned is not None else ['N/A'] * actual_len),
                    'Next_Change_%': [f"{change:.2f}%" if not pd.isna(change) else 'N/A' for change in price_changes],
                    'Actual_Dir': ['🟢 Up' if actual == 1 and not pd.isna(actual) 
                                  else '🔴 Down' if actual == 0 and not pd.isna(actual) 
                                  else '⏳ Pending' for actual in actual_direction],
                    'Correct': ['✅' if correct else '❌' if valid else '⏳' 
                               for correct, valid in zip(prediction_correct, valid_indices)],
                }

                # Add signal strength and derived columns
                signal_strength = [classify_signal_strength(pred, np.max(prob) if prob is not None else None) 
                                  for pred, prob in zip(recent_predictions_aligned, 
                                                       recent_probs_aligned if recent_probs_aligned is not None 
                                                       else [None]*len(recent_predictions_aligned))]

                price_change_list = recent_prices_aligned['Close'].pct_change().round(4).fillna(0).tolist()
                direction_streak_list = (pd.Series(recent_predictions_aligned)
                                       .rolling(3).apply(lambda x: (x == x.iloc[-1]).sum())
                                       .fillna(1).astype(int).tolist())

                # Combine all dictionaries
                all_data = {**ohlc_dict, **pred_dict, 
                           'Signal_Strength': signal_strength,
                           'Price_Change': price_change_list,
                           'Direction_Streak': direction_streak_list}

                predictions_df = pd.DataFrame(all_data)

                # Display the dataframe with enhanced formatting
                st.dataframe(
                    predictions_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Date": st.column_config.DateColumn("Date"),
                        "Time": st.column_config.TimeColumn("Time"),
                        "Open": st.column_config.NumberColumn("Open", format="%.4f"),
                        "High": st.column_config.NumberColumn("High", format="%.4f"),
                        "Low": st.column_config.NumberColumn("Low", format="%.4f"),
                        "Close": st.column_config.NumberColumn("Close", format="%.4f"),
                        "Confidence": st.column_config.TextColumn("Confidence"),
                        "Next_Change_%": st.column_config.TextColumn("Next Δ%"),
                        "Price_Change": st.column_config.NumberColumn("Price Δ", format="%.4f"),
                        "Direction_Streak": st.column_config.NumberColumn("Streak", format="%d"),
                    }
                )

                # Show summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_correct = prediction_correct.sum()
                    total_valid = valid_indices.sum()
                    accuracy = total_correct / total_valid if total_valid > 0 else 0
                    st.metric("Prediction Accuracy", f"{accuracy:.1%}")
                with col2:
                    if recent_probs_aligned is not None:
                        avg_confidence = np.mean([np.max(prob) for prob in recent_probs_aligned])
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                with col3:
                    bullish_correct = prediction_correct[recent_predictions_aligned == 1].sum()
                    bullish_total = (recent_predictions_aligned == 1).sum()
                    bullish_acc = bullish_correct / bullish_total if bullish_total > 0 else 0
                    st.metric("Bullish Accuracy", f"{bullish_acc:.1%}")
                with col4:
                    bearish_correct = prediction_correct[recent_predictions_aligned == 0].sum()
                    bearish_total = (recent_predictions_aligned == 0).sum()
                    bearish_acc = bearish_correct / bearish_total if bearish_total > 0 else 0
                    st.metric("Bearish Accuracy", f"{bearish_acc:.1%}")

                # Enhanced download options
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        csv_data = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Data CSV",
                            data=csv_data,
                            file_name=f"direction_predictions_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except:
                        st.info("Full download not available")

                with col2:
                    # Simple summary download
                    summary_df = pd.DataFrame({
                        'Timestamp': recent_prices.index,
                        'Predicted_Direction': recent_predictions,
                        'Direction_Label': ['Bullish' if p == 1 else 'Bearish' for p in recent_predictions],
                        'Confidence': [np.max(prob) if prob is not None else 0.5 for prob in recent_probs] if recent_probs is not None else [0.5] * len(recent_predictions)
                    })
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=summary_csv,
                        file_name=f"direction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            with dir_tab3:
                st.markdown("**Direction Distribution and Pattern Analysis**")

                # Enhanced distribution analysis with multiple charts
                col1, col2 = st.columns(2)

                with col1:
                    # Direction distribution pie chart
                    bullish_count = (predictions == 1).sum()
                    bearish_count = (predictions == 0).sum()

                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Bullish', 'Bearish'],
                        values=[bullish_count, bearish_count],
                        marker_colors=['green', 'red'],
                        textinfo='label+percent+value'
                    )])
                    fig_pie.update_layout(
                        title="Direction Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Confidence distribution if available
                    if probabilities is not None:
                        conf_scores = np.max(probabilities, axis=1)
                        fig_conf = go.Figure()
                        fig_conf.add_trace(go.Histogram(
                            x=conf_scores,
                            nbinsx=30,
                            name='Confidence Distribution',
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        fig_conf.update_layout(
                            title="Confidence Score Distribution",
                            xaxis_title="Confidence Level",
                            yaxis_title="Frequency",
                            height=400
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    else:
                        st.info("Confidence distribution not available (no probabilities)")

                # Direction patterns over time
                st.markdown("**Direction Patterns Over Time**")

                # Create rolling direction analysis
                direction_series = pd.Series(predictions, index=filtered_data.index[-len(predictions):])
                rolling_bullish = direction_series.rolling(20).mean()

                fig_pattern = go.Figure()
                fig_pattern.add_trace(go.Scatter(
                    x=direction_series.index,
                    y=rolling_bullish,
                    mode='lines',
                    name='20-Period Bullish Ratio',
                    line=dict(color='blue', width=2)
                ))

                # Add horizontal reference lines
                fig_pattern.add_hline(y=0.7, line_dash="dash", line_color="green", 
                                     annotation_text="Strong Bullish")
                fig_pattern.add_hline(y=0.5, line_dash="solid", line_color="gray", 
                                     annotation_text="Neutral")
                fig_pattern.add_hline(y=0.3, line_dash="dash", line_color="red", 
                                     annotation_text="Strong Bearish")

                fig_pattern.update_layout(
                    title="Direction Bias Over Time (20-Period Rolling Average)",
                    xaxis_title="Time",
                    yaxis_title="Bullish Ratio",
                    height=400,
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig_pattern, use_container_width=True)

                # Direction streaks analysis
                st.markdown("**Direction Streaks Analysis**")

                # Calculate consecutive direction streaks
                direction_changes = np.diff(predictions, prepend=predictions[0])
                streak_lengths = []
                current_streak = 1

                for change in direction_changes[1:]:
                    if change == 0:
                        current_streak += 1
                    else:
                        streak_lengths.append(current_streak)
                        current_streak = 1
                streak_lengths.append(current_streak)

                if len(streak_lengths) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Streak Length", f"{max(streak_lengths)}")
                    with col2:
                        st.metric("Avg Streak Length", f"{np.mean(streak_lengths):.1f}")
                    with col3:
                        st.metric("Total Direction Changes", f"{len(streak_lengths)}")

            with dir_tab4:
                st.markdown("**Advanced Statistical Analysis**")

                # Comprehensive statistics table
                bullish_pct = (predictions == 1).mean() * 100
                bearish_pct = (predictions == 0).mean() * 100

                if probabilities is not None:
                    conf_scores = np.max(probabilities, axis=1)
                    avg_conf = np.mean(conf_scores)
                    std_conf = np.std(conf_scores)
                    min_conf = np.min(conf_scores)
                    max_conf = np.max(conf_scores)
                    median_conf = np.median(conf_scores)
                else:
                    avg_conf = std_conf = min_conf = max_conf = median_conf = 0

                stats_data = {
                    'Statistic': [
                        'Total Predictions', 'Bullish Signals', 'Bearish Signals', 'Bullish %', 'Bearish %',
                        'Avg Confidence', 'Std Confidence', 'Min Confidence', 'Max Confidence', 'Median Confidence',
                        'High Confidence (>80%)', 'Medium Confidence (60-80%)', 'Low Confidence (<60%)'
                    ],
                    'Value': [
                        f"{len(predictions):,}",
                        f"{(predictions == 1).sum():,}",
                        f"{(predictions == 0).sum():,}",
                        f"{bullish_pct:.1f}%",
                        f"{bearish_pct:.1f}%",
                        f"{avg_conf:.3f}" if probabilities is not None else "N/A",
                        f"{std_conf:.3f}" if probabilities is not None else "N/A",
                        f"{min_conf:.3f}" if probabilities is not None else "N/A",
                        f"{max_conf:.3f}" if probabilities is not None else "N/A",
                        f"{median_conf:.3f}" if probabilities is not None else "N/A",
                        f"{(conf_scores > 0.8).sum():,}" if probabilities is not None else "N/A",
                        f"{((conf_scores >= 0.6) & (conf_scores <= 0.8)).sum():,}" if probabilities is not None else "N/A",
                        f"{(conf_scores < 0.6).sum():,}" if probabilities is not None else "N/A"
                    ]
                }

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

                # Direction regime analysis
                st.markdown("**Direction Regime Classification**")

                if probabilities is not None:
                    # Classify signals by confidence and direction
                    high_conf_bullish = ((conf_scores > 0.7) & (predictions == 1)).sum()
                    high_conf_bearish = ((conf_scores > 0.7) & (predictions == 0)).sum()
                    med_conf_bullish = ((conf_scores >= 0.5) & (conf_scores <= 0.7) & (predictions == 1)).sum()
                    med_conf_bearish = ((conf_scores >= 0.5) & (conf_scores <= 0.7) & (predictions == 0)).sum()
                    low_conf_bullish = ((conf_scores < 0.5) & (predictions == 1)).sum()
                    low_conf_bearish = ((conf_scores < 0.5) & (predictions == 0)).sum()

                    total_signals = len(predictions)

                    regime_df = pd.DataFrame({
                        'Signal_Category': [
                            'High Conf Bullish', 'High Conf Bearish', 
                            'Med Conf Bullish', 'Med Conf Bearish',
                            'Low Conf Bullish', 'Low Conf Bearish'
                        ],
                        'Count': [
                            high_conf_bullish, high_conf_bearish,
                            med_conf_bullish, med_conf_bearish,
                            low_conf_bullish, low_conf_bearish
                        ],
                        'Percentage': [
                            f"{(high_conf_bullish/total_signals*100):.1f}%",
                            f"{(high_conf_bearish/total_signals*100):.1f}%",
                            f"{(med_conf_bullish/total_signals*100):.1f}%",
                            f"{(med_conf_bearish/total_signals*100):.1f}%",
                            f"{(low_conf_bullish/total_signals*100):.1f}%",
                            f"{(low_conf_bearish/total_signals*100):.1f}%"
                        ],
                        'Confidence_Range': [
                            ">70%", ">70%", "50-70%", "50-70%", "<50%", "<50%"
                        ]
                    })

                    st.dataframe(regime_df, use_container_width=True, hide_index=True)

                # Direction trend analysis
                st.markdown("**Recent Direction Trend Analysis**")
                if len(predictions) >= 40:
                    recent_40 = predictions[-40:]
                    recent_20 = recent_40[-20:]
                    previous_20 = recent_40[:20]

                    recent_bullish = (recent_20 == 1).mean()
                    previous_bullish = (previous_20 == 1).mean()
                    trend_direction = "📈 More Bullish" if recent_bullish > previous_bullish else "📉 More Bearish"
                    trend_change = (recent_bullish - previous_bullish) * 100

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Recent Bullish % (Last 20)", f"{recent_bullish:.1%}")
                    with col2:
                        st.metric("Previous Bullish % (20 before)", f"{previous_bullish:.1%}")
                    with col3:
                        st.metric("Trend Direction", trend_direction)
                    with col4:
                        st.metric("Trend Change", f"{trend_change:+.1f}%")

            with dir_tab5:
                st.markdown("**Model Performance and Validation Metrics**")

                # Model performance analysis
                if len(predictions) >= 20:
                    try:
                        # Calculate performance metrics where we have actual data
                        recent_data = filtered_data.tail(len(predictions))
                        price_changes = recent_data['Close'].pct_change().shift(-1)
                        actual_directions = (price_changes > 0).astype(int)

                        # Performance metrics where we have actual data
                        valid_mask = ~pd.isna(actual_directions)
                        if valid_mask.sum() > 10:
                            pred_valid = predictions[valid_mask.values]
                            actual_valid = actual_directions[valid_mask].values

                            # Calculate comprehensive metrics
                            accuracy = (pred_valid == actual_valid).mean()
                            precision_bull = ((pred_valid == 1) & (actual_valid == 1)).sum() / ((pred_valid == 1).sum() + 1e-8)
                            recall_bull = ((pred_valid == 1) & (actual_valid == 1)).sum() / ((actual_valid == 1).sum() + 1e-8)
                            precision_bear = ((pred_valid == 0) & (actual_valid == 0)).sum() / ((pred_valid == 0).sum() + 1e-8)
                            recall_bear = ((pred_valid == 0) & (actual_valid == 0)).sum() / ((actual_valid == 0).sum() + 1e-8)

                            # Display performance metrics
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Overall Accuracy", f"{accuracy:.1%}")
                            with col2:
                                st.metric("Bullish Precision", f"{precision_bull:.1%}")
                            with col3:
                                st.metric("Bullish Recall", f"{recall_bull:.1%}")
                            with col4:
                                st.metric("Bearish Precision", f"{precision_bear:.1%}")
                            with col5:
                                st.metric("Bearish Recall", f"{recall_bear:.1%}")

                            # Confusion matrix visualization
                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(actual_valid, pred_valid)

                            fig_cm = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=['Predicted Bearish', 'Predicted Bullish'],
                                y=['Actual Bearish', 'Actual Bullish'],
                                colorscale='Blues',
                                text=cm,
                                texttemplate="%{text}",
                                textfont={"size": 20}
                            ))
                            fig_cm.update_layout(
                                title="Prediction Confusion Matrix",
                                height=400
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)

                        else:
                            st.info("Insufficient actual direction data for validation")

                    except Exception as perf_error:
                        st.warning("Performance metrics calculation not available")

                # Feature importance (if available from model)
                st.markdown("**Model Feature Analysis**")
                try:
                    if hasattr(st.session_state, 'direction_trained_models') and st.session_state.direction_trained_models:
                        direction_model_data = st.session_state.direction_trained_models.get('direction', {})
                        feature_importance = direction_model_data.get('feature_importance', {})

                        if feature_importance:
                            # Convert to sorted dataframe
                            importance_df = pd.DataFrame(
                                list(feature_importance.items()),
                                columns=['Feature', 'Importance']
                            ).sort_values('Importance', ascending=False)

                            # Show top features
                            st.markdown("**Top 15 Most Important Features**")
                            top_features = importance_df.head(15)
                            st.dataframe(top_features, use_container_width=True, hide_index=True)

                            # Feature importance chart
                            fig_importance = go.Figure()
                            fig_importance.add_trace(go.Bar(
                                x=top_features['Importance'],
                                y=top_features['Feature'],
                                orientation='h',
                                name='Feature Importance',
                                marker_color='lightcoral'
                            ))
                            fig_importance.update_layout(
                                title="Top Feature Importance",
                                xaxis_title="Importance Score",
                                yaxis_title="Features",
                                height=500
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                        else:
                            st.info("Feature importance data not available")

                except Exception as feature_error:
                    st.info("Feature importance analysis not available")

                # Model configuration summary
                st.markdown("**Model Configuration Summary**")
                config_info = {
                    'Model Type': 'Ensemble (Direction Prediction)',
                    'Task Type': 'Classification',
                    'Classes': 'Bullish (1), Bearish (0)',
                    'Total Predictions': f"{len(predictions):,}",
                    'Feature Count': f"{len(st.session_state.direction_features.columns) if st.session_state.direction_features is not None else 'N/A'}",
                    'Data Points Used': f"{len(st.session_state.data):,}",
                    'Prediction Date Range': safe_format_date_range(st.session_state.data.index)
                }

                config_df = pd.DataFrame(
                    list(config_info.items()),
                    columns=['Configuration', 'Value']
                )
                st.dataframe(config_df, use_container_width=True, hide_index=True)

# Profit Probability Predictions Tab
with profit_prob_tab:
    st.header("💰 Profit Probability Predictions")

    # Check if profit probability features and model are available
    profit_prob_features_available = (hasattr(st.session_state, 'profit_prob_features') and 
                                     st.session_state.profit_prob_features is not None)
    profit_prob_model_available = (hasattr(st.session_state, 'profit_prob_trained_models') and 
                                  st.session_state.profit_prob_trained_models and
                                  'profit_probability' in st.session_state.profit_prob_trained_models and
                                  st.session_state.profit_prob_trained_models['profit_probability'] is not None)

    # Show status information
    col1, col2 = st.columns(2)
    with col1:
        if profit_prob_features_available:
            st.success("✅ Profit probability features calculated")
        else:
            st.warning("⚠️ Profit probability features not calculated")

    with col2:
        if profit_prob_model_available:
            st.success("✅ Profit probability model trained")
        else:
            st.warning("⚠️ Profit probability model not trained")

    # Show instructions if prerequisites are missing
    if not profit_prob_features_available or not profit_prob_model_available:
        st.info("""
        📋 **To use Profit Probability Predictions:**
        1. Go to **Model Training** page
        2. Click on **Profit Probability Predictions** tab
        3. Train the profit probability model
        4. Return here to generate predictions
        """)

        # Show preview of what will be available
        st.subheader("🔮 Preview: Profit Probability Features")
        st.markdown("""
        **Once the profit probability model is trained, you'll see:**
        - 📈 **Interactive Chart** with profit opportunity signals
        - 🎯 **Confidence-Based Visualization** with probability scores
        - 📊 **Comprehensive Analysis Tabs:**
          - Recent Predictions with OHLC data and accuracy validation
          - Performance Metrics with confidence distribution
          - Signal Quality Assessment with profit probability categories
        - 📋 **Real-time Statistics** including prediction accuracy and probability levels
        """)

        # Show sample chart placeholder
        st.info("💡 **Sample visualization will appear here after model training**")

    # Show prediction interface if everything is available
    if profit_prob_features_available and profit_prob_model_available:
        # Profit probability prediction controls
        st.subheader("🎯 Prediction Controls")

        col1, col2 = st.columns(2)

        with col1:
            profit_filter = st.selectbox(
                "📅 Time Period Filter",
                ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All data"],
                index=1,
                help="Select the time period for profit probability predictions",
                key="profit_filter"
            )

        with col2:
            st.metric("Profit Probability Model Status", "✅ Ready", help="Profit probability model is trained and ready")

        # Generate profit probability predictions button
        if st.button("🚀 Generate Profit Probability Predictions", type="primary", key="profit_predict"):
            try:
                with st.spinner("Generating profit probability predictions..."):
                    # Get profit probability model and features
                    profit_prob_model = st.session_state.profit_prob_trained_models['profit_probability']
                    profit_prob_features = st.session_state.profit_prob_features.copy()

                    # Apply time filter to prevent system hang - use realistic row-based limits for trading data
                    try:
                        total_rows = len(profit_prob_features)
                        
                        # Use realistic row counts for trading data (assuming 5-min bars, ~6.5 hours trading per day)
                        if profit_filter == "Last 30 days":
                            # 30 days × ~78 bars/day (6.5 trading hours × 12 bars/hour) = ~2,340 bars
                            target_rows = min(2500, total_rows // 8)  # Cap at 12.5% of total data
                        elif profit_filter == "Last 90 days":
                            # 90 days × ~78 bars/day = ~7,020 bars
                            target_rows = min(7500, total_rows // 4)  # Cap at 25% of total data
                        elif profit_filter == "Last 6 months":
                            # ~180 days × ~78 bars/day = ~14,040 bars
                            target_rows = min(15000, total_rows // 2)  # Cap at 50% of total data
                        elif profit_filter == "Last year":
                            # ~252 trading days × ~78 bars/day = ~19,656 bars
                            target_rows = min(20000, int(total_rows * 0.75))  # Cap at 75% of total data
                        else:  # All data
                            target_rows = total_rows
                        
                        start_idx = max(0, total_rows - target_rows)
                        profit_prob_features_filtered = profit_prob_features.iloc[start_idx:]
                        
                        # Log the actual filtering result
                        original_count = len(profit_prob_features)
                        filtered_count = len(profit_prob_features_filtered)
                        filter_percent = (filtered_count / original_count) * 100 if original_count > 0 else 0
                        print(f"Profit probability filter '{profit_filter}': {original_count} → {filtered_count} rows ({filter_percent:.1f}%)")
                        
                    except Exception as filter_error:
                        st.warning(f"Time filtering failed, using all data: {str(filter_error)}")
                        profit_prob_features_filtered = profit_prob_features

                    # Show filtering information to user
                    original_size = len(profit_prob_features)
                    filtered_size = len(profit_prob_features_filtered)
                    reduction_pct = ((original_size - filtered_size) / original_size * 100) if original_size > 0 else 0
                    
                    if profit_filter != "All data":
                        st.info(f"Time filter '{profit_filter}': Processing {filtered_size:,} data points ({100-reduction_pct:.1f}% of total {original_size:,} points)")
                    else:
                        st.info(f"Processing all {filtered_size:,} data points")

                    # Generate predictions
                    predictions, probabilities = profit_prob_model.predict(profit_prob_features_filtered)

                    # Store predictions
                    st.session_state.profit_prob_predictions = predictions
                    st.session_state.profit_prob_probabilities = probabilities

                    st.success("✅ Profit probability predictions generated successfully!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating profit probability predictions: {str(e)}")
                # Show only the main error message, not full traceback
                if "Missing required features" in str(e):
                    st.error("The model was trained with different features. Please retrain the model.")
                elif "No selected features" in str(e):
                    st.error("Model training data is incomplete. Please retrain the model.")
                else:
                    st.error(f"Technical error: {str(e)[:100]}...")

        # Display profit probability predictions if available
        if hasattr(st.session_state, 'profit_prob_predictions') and st.session_state.profit_prob_predictions is not None:
            st.subheader("📈 Profit Probability Prediction Results")

            # Show prediction statistics
            predictions = st.session_state.profit_prob_predictions
            probabilities = st.session_state.profit_prob_probabilities

            # Apply same time filter to data for display consistency
            # Use the length of predictions to determine the data slice
            predictions_length = len(predictions)
            
            # Filter data to match the prediction length and respect the time filter
            try:
                if hasattr(st.session_state.data.index, 'max') and pd.api.types.is_datetime64_any_dtype(st.session_state.data.index):
                    # Datetime index - use Timedelta
                    if profit_filter == "Last 30 days":
                        cutoff_date = st.session_state.data.index.max() - pd.Timedelta(days=30)
                    elif profit_filter == "Last 90 days":
                        cutoff_date = st.session_state.data.index.max() - pd.Timedelta(days=90)
                    elif profit_filter == "Last 6 months":
                        cutoff_date = st.session_state.data.index.max() - pd.Timedelta(days=180)
                    elif profit_filter == "Last year":
                        cutoff_date = st.session_state.data.index.max() - pd.Timedelta(days=365)
                    else:  # All data
                        cutoff_date = st.session_state.data.index.min()
                    filtered_data = st.session_state.data[st.session_state.data.index >= cutoff_date]
                else:
                    # Non-datetime index - filter based on predictions length
                    total_rows = len(st.session_state.data)
                    start_idx = max(0, total_rows - predictions_length)
                    filtered_data = st.session_state.data.iloc[start_idx:]
                    
                # Ensure filtered_data matches predictions length exactly
                if len(filtered_data) > predictions_length:
                    filtered_data = filtered_data.tail(predictions_length)
                    
                # Log the actual filtering result for display data
                display_count = len(filtered_data)
                print(f"Profit probability display data: {len(st.session_state.data)} → {display_count} rows (matching {predictions_length} predictions)")
                
            except Exception as filter_error:
                st.warning(f"Display filtering failed, using all data: {str(filter_error)}")
                filtered_data = st.session_state.data.tail(predictions_length)

            # Enhanced statistics
            profitable_count = np.sum(predictions == 1)
            unprofitable_count = np.sum(predictions == 0)
            profitable_pct = (profitable_count / len(predictions)) * 100

            if probabilities is not None:
                avg_confidence = np.mean(np.max(probabilities, axis=1)) * 100
                high_confidence = np.sum(np.max(probabilities, axis=1) > 0.7)
                high_conf_pct = (high_confidence / len(predictions)) * 100
            else:
                avg_confidence = 0
                high_confidence = 0
                high_conf_pct = 0

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Predictions", f"{len(predictions)}")
            with col2:
                st.metric("Profit Opportunities", f"{profitable_count} ({profitable_pct:.1f}%)")
            with col3:
                st.metric("Low Profit Signals", f"{unprofitable_count} ({100-profitable_pct:.1f}%)")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

            # Additional statistics
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.metric("High Confidence", f"{high_confidence} ({high_conf_pct:.1f}%)")
            with col6:
                recent_profitable = np.sum(predictions[-20:] == 1) if len(predictions) >= 20 else np.sum(predictions == 1)
                recent_pct = (recent_profitable / min(20, len(predictions))) * 100
                st.metric("Recent Profitable (Last 20)", f"{recent_profitable} ({recent_pct:.1f}%)")
            with col7:
                if probabilities is not None:
                    recent_conf = np.mean(np.max(probabilities[-20:], axis=1)) * 100 if len(probabilities) >= 20 else avg_confidence
                    st.metric("Recent Confidence", f"{recent_conf:.1f}%")
                else:
                    st.metric("Recent Confidence", "N/A")
            with col8:
                price_change = ((filtered_data['Close'].iloc[-1] - filtered_data['Close'].iloc[0]) / filtered_data['Close'].iloc[0]) * 100
                st.metric("Price Change", f"{price_change:.2f}%")

            # Create profit probability prediction chart
            try:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Price Chart', 'Profit Probability Predictions'),
                    row_heights=[0.7, 0.3]
                )

                # Use filtered data for chart (matching the prediction timeframe)
                data_len = min(len(filtered_data), len(predictions))
                recent_data = filtered_data.tail(data_len)

                # Add candlestick chart for price
                fig.add_trace(go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ), row=1, col=1)

                # Add profit probability predictions with confidence coloring
                pred_data = predictions[-data_len:]
                prob_data = probabilities[-data_len:] if probabilities is not None else None

                # Create confidence-based colors
                if prob_data is not None:
                    confidences = np.max(prob_data, axis=1)
                    colors = ['rgba(0, 255, 0, ' + str(conf) + ')' for conf in confidences]
                    sizes = [6 + 6 * conf for conf in confidences]  # Size based on confidence
                else:
                    confidences = np.ones(len(pred_data)) * 0.5  # Default confidence for fallback
                    colors = ['green'] * len(pred_data)
                    sizes = [8] * len(pred_data)

                # Profitable signals
                profitable_mask = pred_data == 1
                if np.any(profitable_mask):
                    profitable_colors = [colors[i] for i in range(len(colors)) if profitable_mask[i]]
                    profitable_sizes = [sizes[i] for i in range(len(sizes)) if profitable_mask[i]]

                    fig.add_trace(go.Scatter(
                        x=recent_data.index[profitable_mask],
                        y=[1] * np.sum(profitable_mask),
                        mode='markers',
                        name='Profit Opportunity',
                        marker=dict(color=profitable_colors if prob_data is not None else 'green', 
                                   size=profitable_sizes if prob_data is not None else 10, 
                                   symbol='triangle-up'),
                        text=[f'Confidence: {confidences[i]:.1%}' for i in range(len(confidences)) if profitable_mask[i]] if prob_data is not None else None,
                        hovertemplate='Profit Opportunity<br>%{text}<extra></extra>' if prob_data is not None else 'Profit Opportunity<extra></extra>'
                    ), row=2, col=1)

                # Low profit signals
                low_profit_mask = pred_data == 0
                if np.any(low_profit_mask):
                    low_profit_colors = [colors[i] for i in range(len(colors)) if low_profit_mask[i]]
                    low_profit_sizes = [sizes[i] for i in range(len(sizes)) if low_profit_mask[i]]

                    fig.add_trace(go.Scatter(
                        x=recent_data.index[low_profit_mask],
                        y=[0] * np.sum(low_profit_mask),
                        mode='markers',
                        name='Low Profit',
                        marker=dict(color=low_profit_colors if prob_data is not None else 'red', 
                                   size=low_profit_sizes if prob_data is not None else 10, 
                                   symbol='triangle-down'),
                        text=[f'Confidence: {confidences[i]:.1%}' for i in range(len(confidences)) if low_profit_mask[i]] if prob_data is not None else None,
                        hovertemplate='Low Profit<br>%{text}<extra></extra>' if prob_data is not None else 'Low Profit<extra></extra>'
                    ), row=2, col=1)

                # Update layout
                fig.update_layout(
                    title="Price vs Profit Probability Predictions",
                    height=600,
                    showlegend=True
                )

                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Profit Probability", row=2, col=1, range=[-0.1, 1.1])

                st.plotly_chart(fig, use_container_width=True)

            except Exception as chart_error:
                st.error(f"Chart generation error: {str(chart_error)[:100]}...")
                # Show basic prediction data without chart
                st.subheader("📊 Basic Prediction Results")
                profitable_signals = np.sum(predictions == 1)
                total_signals = len(predictions)
                st.metric("Profitable Signals", f"{profitable_signals}/{total_signals}")
                st.metric("Low Profit Signals", f"{total_signals - profitable_signals}/{total_signals}")

            # Show detailed analysis section
            st.subheader("📊 Detailed Profit Probability Analysis")

            # Create comprehensive tabbed analysis for profit probability predictions
            prob_tab1, prob_tab2, prob_tab3, prob_tab4, prob_tab5 = st.tabs([
                "📊 Interactive Chart", 
                "📋 Detailed Data Table", 
                "📈 Distribution Analysis", 
                "🔍 Statistical Analysis",
                "📈 Performance Metrics"
            ])

            with prob_tab1:
                st.markdown("**Enhanced Price vs Profit Probability Analysis**")

                # Enhanced profit probability chart with additional analysis
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=('Price Chart with Profit Signals', 'Profit Confidence', 'Profit Pattern'),
                    row_heights=[0.5, 0.3, 0.2]
                )

                # Use filtered data for chart
                data_len = min(len(filtered_data), len(predictions))
                recent_data = filtered_data.tail(data_len)
                pred_data = predictions[-data_len:]
                prob_data = probabilities[-data_len:] if probabilities is not None else None

                # Add candlestick chart for price
                fig.add_trace(go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ), row=1, col=1)

                # Add profit arrows on price chart
                for i, (idx, pred) in enumerate(zip(recent_data.index, pred_data)):
                    if i % 5 == 0:  # Show every 5th prediction to avoid clutter
                        if pred == 1:  # Profitable
                            fig.add_annotation(
                                x=idx, y=recent_data.iloc[i]['High'],
                                text="💰", showarrow=False,
                                font=dict(color="green", size=12),
                                row=1, col=1
                            )
                        else:  # Low profit
                            fig.add_annotation(
                                x=idx, y=recent_data.iloc[i]['Low'],
                                text="⚠️", showarrow=False,
                                font=dict(color="red", size=12),
                                row=1, col=1
                            )

                # Add confidence line chart
                if prob_data is not None:
                    confidences = np.max(prob_data, axis=1)
                    fig.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=confidences,
                        mode='lines+markers',
                        name='Prediction Confidence',
                        line=dict(color='purple', width=2),
                        marker=dict(size=4),
                        hovertemplate='Confidence: %{y:.1%}<extra></extra>'
                    ), row=2, col=1)

                    # Add confidence threshold lines
                    fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                                 annotation_text="High Confidence", row=2, col=1)
                    fig.add_hline(y=0.6, line_dash="dot", line_color="orange", 
                                 annotation_text="Medium Confidence", row=2, col=1)

                # Add profit pattern visualization
                profit_y = [1 if p == 1 else 0 for p in pred_data]
                colors = ['green' if p == 1 else 'red' for p in pred_data]

                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=profit_y,
                    mode='markers',
                    name='Profit Pattern',
                    marker=dict(color=colors, size=6),
                    showlegend=False,
                    hovertemplate='Profit: %{text}<extra></extra>',
                    text=['High Profit' if p == 1 else 'Low Profit' for p in pred_data]
                ), row=3, col=1)

                # Update layout
                fig.update_layout(
                    title="Comprehensive Profit Probability Analysis Dashboard",
                    height=800,
                    showlegend=True,
                    hovermode='x unified'
                )

                fig.update_xaxes(title_text="Time", row=3, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Confidence", row=2, col=1, range=[0, 1])
                fig.update_yaxes(title_text="Profit Probability", row=3, col=1, range=[-0.1, 1.1])

                st.plotly_chart(fig, use_container_width=True)

            with prob_tab2:
                st.markdown("**Comprehensive Profit Probability Data Table**")

                # Create comprehensive predictions dataframe
                num_recent = len(predictions)
                recent_predictions = predictions
                recent_probs = probabilities if probabilities is not None else None
                recent_prices = filtered_data.tail(num_recent)

                # Ensure data alignment
                data_len = min(len(recent_prices), len(recent_predictions))
                if recent_probs is not None and len(recent_probs) > 0:
                    data_len = min(data_len, len(recent_probs))

                # Trim all arrays to the same length
                recent_prices_aligned = recent_prices.tail(data_len)
                recent_predictions_aligned = recent_predictions[-data_len:]
                recent_probs_aligned = recent_probs[-data_len:] if recent_probs is not None else None

                # Calculate future price changes for validation (5 periods ahead)
                future_returns = []
                for i in range(len(recent_prices_aligned)):
                    if i < len(recent_prices_aligned) - 5:
                        # Look ahead 5 periods for profit validation
                        max_return = 0
                        for j in range(1, 6):
                            if i + j < len(recent_prices_aligned):
                                return_val = (recent_prices_aligned.iloc[i + j]['Close'] - recent_prices_aligned.iloc[i]['Close']) / recent_prices_aligned.iloc[i]['Close']
                                max_return = max(max_return, return_val)
                        future_returns.append(max_return * 100)
                    else:
                        future_returns.append(np.nan)

                # Calculate actual profit opportunities
                profit_threshold = 0.1  # 0.1% profit threshold
                actual_profit = [(ret > profit_threshold) if not pd.isna(ret) else None for ret in future_returns]

                # Create signal classification
                def classify_profit_strength(pred, conf):
                    if conf is None:
                        return "🟡 Medium"
                    if conf > 0.8:
                        return "🟢 Very Strong" if pred == 1 else "🔴 Very Weak"
                    elif conf > 0.7:
                        return "🟢 Strong" if pred == 1 else "🔴 Weak"
                    elif conf > 0.6:
                        return "🟢 Medium" if pred == 1 else "🔴 Medium"
                    else:
                        return "🟡 Uncertain"

                # Create the main predictions dataframe with improved datetime handling
                try:
                    # Debug print to understand the index format
                    print(f"DEBUG Profit Prob: Index type: {type(recent_prices_aligned.index)}")
                    print(f"DEBUG Profit Prob: Index dtype: {recent_prices_aligned.index.dtype}")
                    print(f"DEBUG Profit Prob: First few index values: {recent_prices_aligned.index[:5].tolist()}")
                    
                    if pd.api.types.is_datetime64_any_dtype(recent_prices_aligned.index):
                        # Already datetime index
                        date_col = recent_prices_aligned.index.strftime('%Y-%m-%d')
                        time_col = recent_prices_aligned.index.strftime('%H:%M:%S')
                    elif pd.api.types.is_numeric_dtype(recent_prices_aligned.index):
                        # Handle different timestamp formats
                        sample_val = recent_prices_aligned.index[0]
                        print(f"DEBUG Profit Prob: Sample timestamp value: {sample_val}")
                        
                        # Try different timestamp conversion approaches
                        datetime_index = None
                        if sample_val > 1e12:  # Millisecond timestamps
                            datetime_index = pd.to_datetime(recent_prices_aligned.index, unit='ms', errors='coerce')
                            print("DEBUG Profit Prob: Trying millisecond conversion")
                        elif sample_val > 1e9:  # Second timestamps
                            datetime_index = pd.to_datetime(recent_prices_aligned.index, unit='s', errors='coerce')
                            print("DEBUG Profit Prob: Trying second conversion")
                        else:  # Might be days since epoch or other format
                            # Try interpreting as days since epoch
                            datetime_index = pd.to_datetime(recent_prices_aligned.index, unit='D', errors='coerce', origin='1970-01-01')
                            print("DEBUG Profit Prob: Trying days conversion")
                        
                        if datetime_index is not None and not datetime_index.isna().all():
                            print(f"DEBUG Profit Prob: Converted datetime sample: {datetime_index[:3].tolist()}")
                            date_col = datetime_index.strftime('%Y-%m-%d')
                            time_col = datetime_index.strftime('%H:%M:%S')
                        else:
                            print("DEBUG Profit Prob: Using sequential numbering")
                            # Use sequential numbering based on actual data range
                            date_col = [f"Data_{i+1}" for i in range(len(recent_prices_aligned))]
                            time_col = [f"{(9 + i // 12) % 24:02d}:{((i*5) % 60):02d}:00" for i in range(len(recent_prices_aligned))]
                    else:
                        print("DEBUG Profit Prob: Non-numeric index, using sequential")
                        # Non-numeric index - use sequential numbering
                        date_col = [f"Data_{i+1}" for i in range(len(recent_prices_aligned))]
                        time_col = [f"{(9 + i // 12) % 24:02d}:{((i*5) % 60):02d}:00" for i in range(len(recent_prices_aligned))]
                        
                except Exception as e:
                    print(f"DEBUG Profit Prob: Error in datetime conversion: {e}")
                    # Fallback with more realistic time simulation
                    date_col = [f"Data_{i+1}" for i in range(len(recent_prices_aligned))]
                    time_col = [f"{(9 + i // 12) % 24:02d}:{((i*5) % 60):02d}:00" for i in range(len(recent_prices_aligned))]

                # Ensure all arrays match the exact same length
                actual_len = len(recent_prices_aligned)
                recent_predictions_aligned = recent_predictions_aligned[:actual_len]
                if recent_probs_aligned is not None:
                    recent_probs_aligned = recent_probs_aligned[:actual_len]
                future_returns = future_returns[:actual_len]
                actual_profit = actual_profit[:actual_len]
                date_col = date_col[:actual_len]
                time_col = time_col[:actual_len]

                # Build DataFrame
                ohlc_dict = {
                    'Date': date_col,
                    'Time': time_col,
                    'Open': recent_prices_aligned['Open'].values.round(4).tolist(),
                    'High': recent_prices_aligned['High'].values.round(4).tolist(),
                    'Low': recent_prices_aligned['Low'].values.round(4).tolist(),
                    'Close': recent_prices_aligned['Close'].values.round(4).tolist(),
                }

                pred_dict = {
                    'Predicted_Profit': ['🟢 High Profit' if p == 1 else '🔴 Low Profit' for p in recent_predictions_aligned],
                    'Confidence': ([f"{np.max(prob):.3f}" for prob in recent_probs_aligned] 
                                  if recent_probs_aligned is not None else ['N/A'] * actual_len),
                    'Max_Return_5P_%': [f"{ret:.2f}%" if not pd.isna(ret) else '⏳ Pending' for ret in future_returns],
                    'Actual_Profit': ['✅ Profitable' if profit is True else '❌ Not Profitable' if profit is False else '⏳ Pending' 
                                     for profit in actual_profit],
                    'Correct': ['✅' if (pred == 1 and profit is True) or (pred == 0 and profit is False) 
                               else '❌' if profit is not None else '⏳' 
                               for pred, profit in zip(recent_predictions_aligned, actual_profit)],
                }

                # Add signal strength and derived columns
                signal_strength = [classify_profit_strength(pred, np.max(prob) if prob is not None else None) 
                                  for pred, prob in zip(recent_predictions_aligned, 
                                                       recent_probs_aligned if recent_probs_aligned is not None 
                                                       else [None]*len(recent_predictions_aligned))]

                price_change_list = recent_prices_aligned['Close'].pct_change().round(4).fillna(0).tolist()
                profit_streak_list = (pd.Series(recent_predictions_aligned)
                                     .rolling(3).apply(lambda x: (x == x.iloc[-1]).sum())
                                     .fillna(1).astype(int).tolist())

                # Combine all dictionaries
                all_data = {**ohlc_dict, **pred_dict, 
                           'Signal_Strength': signal_strength,
                           'Price_Change': price_change_list,
                           'Profit_Streak': profit_streak_list}

                predictions_df = pd.DataFrame(all_data)

                # Display the dataframe with enhanced formatting
                st.dataframe(
                    predictions_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Date": st.column_config.DateColumn("Date"),
                        "Time": st.column_config.TimeColumn("Time"),
                        "Open": st.column_config.NumberColumn("Open", format="%.4f"),
                        "High": st.column_config.NumberColumn("High", format="%.4f"),
                        "Low": st.column_config.NumberColumn("Low", format="%.4f"),
                        "Close": st.column_config.NumberColumn("Close", format="%.4f"),
                        "Confidence": st.column_config.TextColumn("Confidence"),
                        "Max_Return_5P_%": st.column_config.TextColumn("Max Return 5P"),
                        "Price_Change": st.column_config.NumberColumn("Price Δ", format="%.4f"),
                        "Profit_Streak": st.column_config.NumberColumn("Streak", format="%d"),
                    }
                )

                # Show summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    correct_predictions = [pred for pred in predictions_df['Correct'] if pred in ['✅', '❌']]
                    accuracy = correct_predictions.count('✅') / len(correct_predictions) if correct_predictions else 0
                    st.metric("Prediction Accuracy", f"{accuracy:.1%}")
                with col2:
                    if recent_probs_aligned is not None:
                        avg_confidence = np.mean([np.max(prob) for prob in recent_probs_aligned])
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                with col3:
                    high_profit_correct = sum(1 for i, (pred, actual) in enumerate(zip(recent_predictions_aligned, actual_profit)) 
                                             if pred == 1 and actual is True)
                    high_profit_total = sum(1 for pred in recent_predictions_aligned if pred == 1)
                    high_profit_acc = high_profit_correct / high_profit_total if high_profit_total > 0 else 0
                    st.metric("High Profit Accuracy", f"{high_profit_acc:.1%}")
                with col4:
                    low_profit_correct = sum(1 for i, (pred, actual) in enumerate(zip(recent_predictions_aligned, actual_profit)) 
                                            if pred == 0 and actual is False)
                    low_profit_total = sum(1 for pred in recent_predictions_aligned if pred == 0)
                    low_profit_acc = low_profit_correct / low_profit_total if low_profit_total > 0 else 0
                    st.metric("Low Profit Accuracy", f"{low_profit_acc:.1%}")

                # Enhanced download options
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        csv_data = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Data CSV",
                            data=csv_data,
                            file_name=f"profit_probability_predictions_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except:
                        st.info("Full download not available")

                with col2:
                    # Simple summary download
                    summary_df = pd.DataFrame({
                        'Timestamp': recent_prices.index,
                        'Predicted_Profit_Probability': recent_predictions,
                        'Profit_Label': ['High Profit' if p == 1 else 'Low Profit' for p in recent_predictions],
                        'Confidence': [np.max(prob) if prob is not None else 0.5 for prob in recent_probs] if recent_probs is not None else [0.5] * len(recent_predictions)
                    })
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=summary_csv,
                        file_name=f"profit_probability_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            with prob_tab3:
                st.markdown("**Profit Probability Distribution and Pattern Analysis**")

                # Enhanced distribution analysis with multiple charts
                col1, col2 = st.columns(2)

                with col1:
                    # Profit probability distribution pie chart
                    profitable_count = (predictions == 1).sum()
                    unprofitable_count = (predictions == 0).sum()

                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['High Profit', 'Low Profit'],
                        values=[profitable_count, unprofitable_count],
                        marker_colors=['green', 'red'],
                        textinfo='label+percent+value'
                    )])
                    fig_pie.update_layout(
                        title="Profit Probability Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Confidence distribution if available
                    if probabilities is not None:
                        conf_scores = np.max(probabilities, axis=1)
                        fig_conf = go.Figure()
                        fig_conf.add_trace(go.Histogram(
                            x=conf_scores,
                            nbinsx=30,
                            name='Confidence Distribution',
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        fig_conf.update_layout(
                            title="Confidence Score Distribution",
                            xaxis_title="Confidence Level",
                            yaxis_title="Frequency",
                            height=400
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    else:
                        st.info("Confidence distribution not available (no probabilities)")

                # Profit patterns over time
                st.markdown("**Profit Patterns Over Time**")

                # Create rolling profit analysis
                profit_series = pd.Series(predictions, index=filtered_data.index[-len(predictions):])
                rolling_profitable = profit_series.rolling(20).mean()

                fig_pattern = go.Figure()
                fig_pattern.add_trace(go.Scatter(
                    x=profit_series.index,
                    y=rolling_profitable,
                    mode='lines',
                    name='20-Period Profit Ratio',
                    line=dict(color='blue', width=2)
                ))

                # Add horizontal reference lines
                fig_pattern.add_hline(y=0.7, line_dash="dash", line_color="green", 
                                     annotation_text="High Profit Period")
                fig_pattern.add_hline(y=0.5, line_dash="solid", line_color="gray", 
                                     annotation_text="Neutral")
                fig_pattern.add_hline(y=0.3, line_dash="dash", line_color="red", 
                                     annotation_text="Low Profit Period")

                fig_pattern.update_layout(
                    title="Profit Opportunity Over Time (20-Period Rolling Average)",
                    xaxis_title="Time",
                    yaxis_title="Profit Ratio",
                    height=400,
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig_pattern, use_container_width=True)

            with prob_tab4:
                st.markdown("**Advanced Statistical Analysis**")

                # Comprehensive statistics table
                profitable_pct = (predictions == 1).mean() * 100
                unprofitable_pct = (predictions == 0).mean() * 100

                if probabilities is not None:
                    conf_scores = np.max(probabilities, axis=1)
                    avg_conf = np.mean(conf_scores)
                    std_conf = np.std(conf_scores)
                    min_conf = np.min(conf_scores)
                    max_conf = np.max(conf_scores)
                    median_conf = np.median(conf_scores)
                else:
                    avg_conf = std_conf = min_conf = max_conf = median_conf = 0

                stats_data = {
                    'Statistic': [
                        'Total Predictions', 'High Profit Signals', 'Low Profit Signals', 'High Profit %', 'Low Profit %',
                        'Avg Confidence', 'Std Confidence', 'Min Confidence', 'Max Confidence', 'Median Confidence',
                        'High Confidence (>80%)', 'Medium Confidence (60-80%)', 'Low Confidence (<60%)'
                    ],
                    'Value': [
                        f"{len(predictions):,}",
                        f"{(predictions == 1).sum():,}",
                        f"{(predictions == 0).sum():,}",
                        f"{profitable_pct:.1f}%",
                        f"{unprofitable_pct:.1f}%",
                        f"{avg_conf:.3f}" if probabilities is not None else "N/A",
                        f"{std_conf:.3f}" if probabilities is not None else "N/A",
                        f"{min_conf:.3f}" if probabilities is not None else "N/A",
                        f"{max_conf:.3f}" if probabilities is not None else "N/A",
                        f"{median_conf:.3f}" if probabilities is not None else "N/A",
                        f"{(conf_scores > 0.8).sum():,}" if probabilities is not None else "N/A",
                        f"{((conf_scores >= 0.6) & (conf_scores <= 0.8)).sum():,}" if probabilities is not None else "N/A",
                        f"{(conf_scores < 0.6).sum():,}" if probabilities is not None else "N/A"
                    ]
                }

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            with prob_tab5:
                st.markdown("**Model Performance and Validation Metrics**")

                # Model configuration summary
                st.markdown("**Model Configuration Summary**")
                config_info = {
                    'Model Type': 'Ensemble (Profit Probability Prediction)',
                    'Task Type': 'Classification',
                    'Classes': 'High Profit (1), Low Profit (0)',
                    'Prediction Window': 'Next 5 Periods',
                    'Total Predictions': f"{len(predictions):,}",
                    'Feature Count': f"{len(st.session_state.profit_prob_features.columns) if st.session_state.profit_prob_features is not None else 'N/A'}",
                    'Data Points Used': f"{len(st.session_state.data):,}",
                    'Prediction Date Range': safe_format_date_range(st.session_state.data.index)
                }

                config_df = pd.DataFrame(
                    list(config_info.items()),
                    columns=['Configuration', 'Value']
                )
                st.dataframe(config_df, use_container_width=True, hide_index=True)