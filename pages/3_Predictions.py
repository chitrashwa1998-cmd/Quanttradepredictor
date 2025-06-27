import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

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
volatility_tab, direction_tab = st.tabs(["📊 Volatility Predictions", "🎯 Direction Predictions"])

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
                    
                    # Use volatility features for prediction
                    data_for_prediction = st.session_state.features.copy()
                    
                    # Generate predictions using the model manager
                    predictions, _ = model_trainer.predict('volatility', data_for_prediction)
                    
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
                    
                    # Create the main predictions dataframe with proper data types
                    predictions_df = pd.DataFrame({
                        'Date': recent_prices.index.strftime('%Y-%m-%d'),
                        'Time': recent_prices.index.strftime('%H:%M:%S'),
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
                    simple_df = pd.DataFrame({
                        'Date': recent_prices.index.strftime('%Y-%m-%d %H:%M:%S'),
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
                    'Prediction Date Range': f"{st.session_state.data.index[0].strftime('%Y-%m-%d')} to {st.session_state.data.index[-1].strftime('%Y-%m-%d')}"
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
                    
                    # Apply time filter to prevent system hang
                    if dir_filter == "Last 30 days":
                        cutoff_date = direction_features.index.max() - pd.Timedelta(days=30)
                    elif dir_filter == "Last 90 days":
                        cutoff_date = direction_features.index.max() - pd.Timedelta(days=90)
                    elif dir_filter == "Last 6 months":
                        cutoff_date = direction_features.index.max() - pd.Timedelta(days=180)
                    elif dir_filter == "Last year":
                        cutoff_date = direction_features.index.max() - pd.Timedelta(days=365)
                    else:  # All data
                        cutoff_date = direction_features.index.min()
                    
                    # Filter direction features based on selected time period
                    direction_features_filtered = direction_features[direction_features.index >= cutoff_date]
                    
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
            
            # Create tabbed analysis
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📋 Recent Predictions", "📈 Performance Metrics", "🔍 Signal Quality"])
            
            with analysis_tab1:
                st.markdown("**Last 30 Direction Predictions**")
                
                # Create comprehensive predictions dataframe
                num_recent = min(30, len(predictions))
                recent_predictions = predictions[-num_recent:]
                recent_probs = probabilities[-num_recent:] if probabilities is not None else None
                recent_prices = filtered_data.tail(num_recent)
                
                # Calculate price changes for validation
                price_changes = recent_prices['Close'].pct_change().shift(-1) * 100  # Next period change
                actual_direction = (price_changes > 0).astype(int)
                
                predictions_df = pd.DataFrame({
                    'Timestamp': recent_prices.index,
                    'Open': recent_prices['Open'].round(2),
                    'High': recent_prices['High'].round(2),
                    'Low': recent_prices['Low'].round(2),
                    'Close': recent_prices['Close'].round(2),
                    'Predicted Direction': ['🟢 Bullish' if p == 1 else '🔴 Bearish' for p in recent_predictions],
                    'Confidence': [f"{np.max(prob):.1f}%" for prob in recent_probs] if recent_probs is not None else ['N/A'] * num_recent,
                    'Next Change %': [f"{change:.2f}%" if not pd.isna(change) else 'N/A' for change in price_changes],
                    'Correct': ['✅' if pred == actual and not pd.isna(actual) else '❌' if not pd.isna(actual) else '⏳' 
                               for pred, actual in zip(recent_predictions, actual_direction)]
                })
                
                st.dataframe(predictions_df, use_container_width=True)
            
            with analysis_tab2:
                st.markdown("**Model Performance Analysis**")
                
                # Calculate accuracy where we have actual data
                valid_comparisons = ~pd.isna(actual_direction[:-1])  # Exclude last as it has no future data
                if valid_comparisons.sum() > 0:
                    accuracy = (recent_predictions[:-1][valid_comparisons] == actual_direction[:-1][valid_comparisons]).mean()
                    st.metric("Prediction Accuracy", f"{accuracy:.1%}")
                
                # Confidence distribution
                if probabilities is not None:
                    conf_dist = np.max(probabilities, axis=1)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Confidence histogram
                        fig_hist = go.Figure(data=[go.Histogram(x=conf_dist, nbinsx=20, name='Confidence Distribution')])
                        fig_hist.update_layout(title="Confidence Distribution", xaxis_title="Confidence Level", yaxis_title="Count")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Confidence vs accuracy
                        high_conf_mask = conf_dist > 0.7
                        med_conf_mask = (conf_dist >= 0.5) & (conf_dist <= 0.7)
                        low_conf_mask = conf_dist < 0.5
                        
                        st.metric("High Confidence (>70%)", f"{high_conf_mask.sum()} signals")
                        st.metric("Medium Confidence (50-70%)", f"{med_conf_mask.sum()} signals")
                        st.metric("Low Confidence (<50%)", f"{low_conf_mask.sum()} signals")
            
            with analysis_tab3:
                st.markdown("**Signal Quality Assessment**")
                
                # Signal strength analysis
                if probabilities is not None:
                    conf_scores = np.max(probabilities, axis=1)
                    
                    # Quality categories
                    very_high = (conf_scores > 0.8).sum()
                    high = ((conf_scores > 0.7) & (conf_scores <= 0.8)).sum()
                    medium = ((conf_scores > 0.6) & (conf_scores <= 0.7)).sum()
                    low = (conf_scores <= 0.6).sum()
                    
                    quality_df = pd.DataFrame({
                        'Quality Level': ['Very High (>80%)', 'High (70-80%)', 'Medium (60-70%)', 'Low (≤60%)'],
                        'Signal Count': [very_high, high, medium, low],
                        'Percentage': [f"{(very_high/len(conf_scores)*100):.1f}%",
                                     f"{(high/len(conf_scores)*100):.1f}%",
                                     f"{(medium/len(conf_scores)*100):.1f}%",
                                     f"{(low/len(conf_scores)*100):.1f}%"]
                    })
                    
                    st.dataframe(quality_df, use_container_width=True)
                    
                    # Quality over time
                    fig_quality = go.Figure()
                    fig_quality.add_trace(go.Scatter(
                        x=filtered_data.index,
                        y=conf_scores[-len(filtered_data):],
                        mode='lines+markers',
                        name='Confidence Over Time',
                        line=dict(color='blue')
                    ))
                    fig_quality.update_layout(title="Signal Confidence Over Time", 
                                            xaxis_title="Time", 
                                            yaxis_title="Confidence Level",
                                            yaxis=dict(range=[0, 1]))
                    st.plotly_chart(fig_quality, use_container_width=True)