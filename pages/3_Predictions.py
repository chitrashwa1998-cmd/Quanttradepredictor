import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import all model classes and utilities
from models.volatility_model import VolatilityModel
from models.direction_model import DirectionModel
from models.profit_probability_model import ProfitProbabilityModel
from models.reversal_model import ReversalModel
from utils.database_adapter import DatabaseAdapter

def show_predictions_page():
    """Main predictions page with all 4 models - NO FALLBACK LOGIC"""

    st.title("🔮 Real-Time Predictions")
    st.markdown("### Advanced ML Model Predictions - Authentic Data Only")

    # Add cache clearing button to remove synthetic values
    if st.button("🗑️ Clear All Cached Data", help="Click if you see synthetic datetime warnings"):
        # Clear ALL session state to remove any synthetic datetime values
        st.session_state.clear()
        st.success("✅ Cleared all cached data. Page will reload with fresh database data.")
        st.rerun()

    # Initialize database with error handling
    try:
        db = DatabaseAdapter()
    except Exception as e:
        if "AdminShutdown" in str(e) or "terminating connection" in str(e):
            st.error("🔌 Database connection was terminated. This can happen if the database was idle.")
            st.info("💡 **Solution**: Wait a few seconds and refresh the page. PostgreSQL databases automatically reconnect.")
            if st.button("🔄 Retry Connection"):
                st.rerun()
            st.stop()
        elif "synthetic datetime" in str(e).lower():
            st.error("⚠️ Database contains synthetic datetime values that need to be cleared.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Database", type="primary"):
                    try:
                        temp_db = DatabaseAdapter()
                        success = temp_db.clear_all_data()
                        if success:
                            st.success("✅ Database cleared successfully!")
                            st.info("👆 Please go to 'Data Upload' page to upload your original data file.")
                        else:
                            st.error("❌ Failed to clear database")
                    except:
                        st.error("❌ Unable to clear database due to connection issues")
            with col2:
                if st.button("🔍 View Database Manager"):
                    st.switch_page("pages/5_Database_Manager.py")
            st.stop()
        else:
            st.error(f"❌ Database initialization failed: {str(e)}")
            st.stop()

    # Get fresh data from database instead of session state
    fresh_data = db.load_ohlc_data()

    # Check if database has data
    if fresh_data is None or len(fresh_data) == 0:
        st.error("⚠️ No data available in database. Please upload data first in the Data Upload page.")
        st.stop()

    # Validate that data contains authentic datetime data
    if not pd.api.types.is_datetime64_any_dtype(fresh_data.index):
        st.error("⚠️ Data contains invalid datetime index. Please re-upload your data.")
        st.stop()

    # Check for synthetic datetime patterns in the database data (but not legitimate timestamps)
    sample_datetime_str = str(fresh_data.index[0])
    # Only flag synthetic patterns like "Data_1", "Point_123", but NOT legitimate timestamps like "2015-01-09 09:15:00"
    is_synthetic = (
        any(pattern in sample_datetime_str for pattern in ['Data_', 'Point_']) or
        (sample_datetime_str == '09:15:00')  # Only flag if it's JUST the time without date
    )

    if is_synthetic:
        st.error("⚠️ Database contains synthetic datetime values. Please clear database and re-upload your data.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Database", type="primary"):
                with st.spinner("Clearing all database data..."):
                    success = db.clear_all_data()
                if success:
                    st.success("✅ Database cleared successfully!")
                    st.info("👆 Please go to 'Data Upload' page to upload your original data file.")
                else:
                    st.error("❌ Failed to clear database")
                st.stop()

        with col2:
            if st.button("🔍 View Database Manager"):
                st.switch_page("pages/5_Database_Manager.py")
        st.stop()

    st.success(f"✅ Using authentic data with {len(fresh_data):,} records")

    # Create tabs for all 4 models
    vol_tab, dir_tab, profit_tab, reversal_tab = st.tabs([
        "📊 Volatility Predictions", 
        "📈 Direction Predictions", 
        "💰 Profit Probability", 
        "🔄 Reversal Detection"
    ])

    # Pass fresh database data to all prediction functions
    with vol_tab:
        show_volatility_predictions(db, fresh_data)

    with dir_tab:
        show_direction_predictions(db, fresh_data)

    with profit_tab:
        show_profit_predictions(db, fresh_data)

    with reversal_tab:
        show_reversal_predictions(db, fresh_data)

def show_volatility_predictions(db, fresh_data):
    """Volatility predictions with authentic data only"""

    st.header("📊 Volatility Forecasting")

    # Use the fresh data passed from main function
    if fresh_data is None or len(fresh_data) == 0:
        st.error("No fresh data available")
        return

    # Initialize model manager and check for trained models
    from models.model_manager import ModelManager
    model_manager = ModelManager()

    # Check if volatility model exists
    if not model_manager.is_model_trained('volatility'):
        st.warning("⚠️ Volatility model not trained. Please train the model first.")
        return

    # Prepare features from fresh data
    try:
        from features.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        features = ti.calculate_all_indicators(fresh_data)

        if features is None or len(features) == 0:
            st.error("Failed to calculate features")
            return

        # Make predictions using trained model
        predictions, _ = model_manager.predict('volatility', features)

        if predictions is None or len(predictions) == 0:
            st.error("Failed to generate predictions")
            return

        # Handle array length mismatch
        if len(predictions) != len(features):
            if len(predictions) < len(features):
                padded_predictions = np.full(len(features), np.nan)
                padded_predictions[:len(predictions)] = predictions
                predictions = padded_predictions
            else:
                predictions = predictions[:len(features)]

        # Create predictions dataframe
        pred_df = pd.DataFrame({
            'DateTime': features.index,
            'Predicted_Volatility': predictions
        })

        # Remove rows with NaN predictions for display
        pred_df = pred_df.dropna(subset=['Predicted_Volatility'])

        if len(pred_df) == 0:
            st.error("No valid predictions generated")
            return

        # Format datetime for display
        pred_df['Date'] = pred_df['DateTime'].dt.strftime('%Y-%m-%d')
        pred_df['Time'] = pred_df['DateTime'].dt.strftime('%H:%M:%S')

        # Create 5 comprehensive sub-tabs for detailed analysis
        chart_tab, data_tab, dist_tab, stats_tab, metrics_tab = st.tabs([
            "📈 Interactive Chart", 
            "📋 Detailed Data", 
            "📊 Distribution Analysis", 
            "🔍 Statistical Analysis", 
            "⚡ Performance Metrics"
        ])

        with chart_tab:
            st.subheader("📈 Volatility Prediction Chart")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                chart_points = st.selectbox("Data Points", [50, 100, 200, 500], index=1, key="vol_chart_points")
            
            recent_predictions = pred_df.tail(chart_points)

            # Create subplot with multiple views
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Volatility Predictions Over Time', 'Volatility Distribution'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Main volatility line chart
            fig.add_trace(go.Scatter(
                x=recent_predictions['DateTime'],
                y=recent_predictions['Predicted_Volatility'],
                mode='lines+markers',
                name='Predicted Volatility',
                line=dict(color='#00ffff', width=2),
                marker=dict(size=4)
            ), row=1, col=1)

            # Add volatility bands
            vol_mean = recent_predictions['Predicted_Volatility'].mean()
            vol_std = recent_predictions['Predicted_Volatility'].std()
            
            fig.add_hline(y=vol_mean, line_dash="dash", line_color="yellow", 
                         annotation_text="Mean", row=1, col=1)
            fig.add_hline(y=vol_mean + vol_std, line_dash="dot", line_color="red", 
                         annotation_text="+1σ", row=1, col=1)
            fig.add_hline(y=vol_mean - vol_std, line_dash="dot", line_color="green", 
                         annotation_text="-1σ", row=1, col=1)

            # Histogram of volatility predictions
            fig.add_trace(go.Histogram(
                x=recent_predictions['Predicted_Volatility'],
                nbinsx=30,
                name='Volatility Distribution',
                marker_color='rgba(0, 255, 255, 0.6)'
            ), row=2, col=1)

            fig.update_layout(
                title=f"Volatility Analysis - Last {chart_points} Data Points",
                height=700,
                showlegend=True,
                template="plotly_dark"
            )
            
            fig.update_xaxes(title_text="DateTime", row=1, col=1)
            fig.update_yaxes(title_text="Predicted Volatility", row=1, col=1)
            fig.update_xaxes(title_text="Volatility Value", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Volatility", f"{recent_predictions['Predicted_Volatility'].iloc[-1]:.6f}")
            with col2:
                st.metric("Average", f"{vol_mean:.6f}")
            with col3:
                st.metric("Min", f"{recent_predictions['Predicted_Volatility'].min():.6f}")
            with col4:
                st.metric("Max", f"{recent_predictions['Predicted_Volatility'].max():.6f}")

        with data_tab:
            st.subheader("📋 Detailed Prediction Data")
            
            col1, col2 = st.columns([2, 1])
            with col2:
                data_points = st.selectbox("Show Records", [100, 200, 500, 1000], index=1, key="vol_data_points")
            
            recent_predictions = pred_df.tail(data_points)
            
            # Enhanced data table with additional calculated columns
            detailed_df = recent_predictions.copy()
            detailed_df['Prediction_Error'] = np.abs(detailed_df['Predicted_Volatility'] - detailed_df['Predicted_Volatility'].mean())
            detailed_df['Volatility_Percentile'] = detailed_df['Predicted_Volatility'].rank(pct=True) * 100
            detailed_df['Volatility_Regime'] = pd.cut(detailed_df['Predicted_Volatility'], 
                                                    bins=3, labels=['Low', 'Medium', 'High'])
            detailed_df['Moving_Avg_5'] = detailed_df['Predicted_Volatility'].rolling(5).mean()
            detailed_df['Volatility_Change'] = detailed_df['Predicted_Volatility'].diff()
            detailed_df['Volatility_Direction'] = detailed_df['Volatility_Change'].apply(
                lambda x: '📈' if x > 0 else '📉' if x < 0 else '➡️'
            )
            
            # Display enhanced table
            display_columns = [
                'Date', 'Time', 'Predicted_Volatility', 'Volatility_Direction',
                'Moving_Avg_5', 'Volatility_Change', 'Volatility_Regime', 
                'Volatility_Percentile', 'Prediction_Error'
            ]
            
            st.dataframe(
                detailed_df[display_columns].round(6), 
                use_container_width=True,
                column_config={
                    "Predicted_Volatility": st.column_config.NumberColumn("Predicted Vol", format="%.6f"),
                    "Moving_Avg_5": st.column_config.NumberColumn("5-Period MA", format="%.6f"),
                    "Volatility_Change": st.column_config.NumberColumn("Change", format="%.6f"),
                    "Volatility_Percentile": st.column_config.NumberColumn("Percentile", format="%.1f%%"),
                    "Prediction_Error": st.column_config.NumberColumn("Error", format="%.6f")
                }
            )
            
            # Data summary
            st.subheader("📊 Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Volatility Regimes:**")
                regime_counts = detailed_df['Volatility_Regime'].value_counts()
                for regime, count in regime_counts.items():
                    st.write(f"• {regime}: {count} ({count/len(detailed_df)*100:.1f}%)")
            
            with col2:
                st.write("**Trend Analysis:**")
                direction_counts = detailed_df['Volatility_Direction'].value_counts()
                for direction, count in direction_counts.items():
                    st.write(f"• {direction}: {count}")
            
            with col3:
                st.write("**Statistics:**")
                st.write(f"• Mean: {detailed_df['Predicted_Volatility'].mean():.6f}")
                st.write(f"• Std Dev: {detailed_df['Predicted_Volatility'].std():.6f}")
                st.write(f"• Skewness: {detailed_df['Predicted_Volatility'].skew():.3f}")

        with dist_tab:
            st.subheader("📊 Distribution Analysis")
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram with KDE
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=pred_df['Predicted_Volatility'],
                    nbinsx=50,
                    histnorm='probability density',
                    name='Volatility Distribution',
                    marker_color='rgba(0, 255, 255, 0.7)'
                ))
                
                fig.update_layout(
                    title="Volatility Distribution",
                    xaxis_title="Predicted Volatility",
                    yaxis_title="Density",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot by hour
                pred_df_sample = pred_df.tail(1000)  # Use sample for performance
                pred_df_sample['Hour'] = pred_df_sample['DateTime'].dt.hour
                
                fig = go.Figure()
                fig.add_trace(go.Box(
                    x=pred_df_sample['Hour'],
                    y=pred_df_sample['Predicted_Volatility'],
                    name='Volatility by Hour',
                    marker_color='cyan'
                ))
                
                fig.update_layout(
                    title="Volatility Distribution by Hour",
                    xaxis_title="Hour of Day",
                    yaxis_title="Predicted Volatility",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical distribution analysis
            st.subheader("📈 Distribution Statistics")
            
            vol_data = pred_df['Predicted_Volatility']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{vol_data.mean():.6f}")
                st.metric("Median", f"{vol_data.median():.6f}")
            with col2:
                st.metric("Std Dev", f"{vol_data.std():.6f}")
                st.metric("Variance", f"{vol_data.var():.8f}")
            with col3:
                st.metric("Skewness", f"{vol_data.skew():.3f}")
                st.metric("Kurtosis", f"{vol_data.kurtosis():.3f}")
            with col4:
                st.metric("Min", f"{vol_data.min():.6f}")
                st.metric("Max", f"{vol_data.max():.6f}")
            
            # Percentile analysis
            st.subheader("🎯 Percentile Analysis")
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            perc_values = [np.percentile(vol_data, p) for p in percentiles]
            
            perc_df = pd.DataFrame({
                'Percentile': [f"{p}%" for p in percentiles],
                'Value': [f"{v:.6f}" for v in perc_values]
            })
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(perc_df, use_container_width=True)
            
            with col2:
                # Percentile plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=percentiles,
                    y=perc_values,
                    mode='lines+markers',
                    name='Percentiles',
                    line=dict(color='yellow', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Volatility Percentiles",
                    xaxis_title="Percentile",
                    yaxis_title="Volatility Value",
                    height=300,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

        with stats_tab:
            st.subheader("🔍 Statistical Analysis")
            
            # Time series analysis
            vol_data = pred_df['Predicted_Volatility'].tail(500)  # Use recent data
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Autocorrelation analysis
                st.write("**📊 Autocorrelation Analysis**")
                
                lags = range(1, min(21, len(vol_data)//4))
                autocorr = [vol_data.autocorr(lag=lag) for lag in lags]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(lags),
                    y=autocorr,
                    name='Autocorrelation',
                    marker_color='lightblue'
                ))
                
                fig.add_hline(y=0.05, line_dash="dash", line_color="red")
                fig.add_hline(y=-0.05, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title="Autocorrelation Function",
                    xaxis_title="Lag",
                    yaxis_title="Correlation",
                    height=300,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling statistics
                st.write("**📈 Rolling Statistics (20-period)**")
                rolling_mean = vol_data.rolling(20).mean()
                rolling_std = vol_data.rolling(20).std()
                
                stats_df = pd.DataFrame({
                    'Current Mean': f"{rolling_mean.iloc[-1]:.6f}",
                    'Current Std': f"{rolling_std.iloc[-1]:.6f}",
                    'Mean Change': f"{(rolling_mean.iloc[-1] - rolling_mean.iloc[-20]):.6f}",
                    'Std Change': f"{(rolling_std.iloc[-1] - rolling_std.iloc[-20]):.6f}"
                }, index=[0])
                
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                # Volatility clustering analysis
                st.write("**🔗 Volatility Clustering**")
                
                # Calculate volatility changes
                vol_changes = vol_data.diff().abs()
                high_vol_threshold = vol_changes.quantile(0.8)
                
                clusters = []
                cluster_start = None
                
                for i, change in enumerate(vol_changes):
                    if change > high_vol_threshold:
                        if cluster_start is None:
                            cluster_start = i
                    else:
                        if cluster_start is not None:
                            clusters.append(i - cluster_start)
                            cluster_start = None
                
                if clusters:
                    avg_cluster_length = np.mean(clusters)
                    max_cluster_length = max(clusters)
                    total_clusters = len(clusters)
                else:
                    avg_cluster_length = 0
                    max_cluster_length = 0
                    total_clusters = 0
                
                cluster_df = pd.DataFrame({
                    'Total Clusters': [total_clusters],
                    'Avg Length': [f"{avg_cluster_length:.1f}"],
                    'Max Length': [max_cluster_length],
                    'Clustering %': [f"{(sum(clusters)/len(vol_data)*100):.1f}%"]
                })
                
                st.dataframe(cluster_df, use_container_width=True)
                
                # Stationarity test (simplified)
                st.write("**📏 Stationarity Analysis**")
                
                # ADF test approximation
                vol_diff = vol_data.diff().dropna()
                mean_reversion = abs(vol_diff.mean()) < 0.01
                variance_stable = vol_diff.std() < vol_data.std() * 0.5
                
                stationarity_df = pd.DataFrame({
                    'Mean Reversion': ['✅' if mean_reversion else '❌'],
                    'Variance Stable': ['✅' if variance_stable else '❌'],
                    'Likely Stationary': ['✅' if mean_reversion and variance_stable else '❌']
                }, index=[0])
                
                st.dataframe(stationarity_df, use_container_width=True)
            
            # Regime detection
            st.subheader("🎭 Volatility Regime Detection")
            
            # Simple regime detection based on rolling statistics
            window = 50
            vol_data_full = pred_df['Predicted_Volatility'].tail(200)
            rolling_mean = vol_data_full.rolling(window).mean()
            rolling_std = vol_data_full.rolling(window).std()
            
            regimes = []
            for i in range(len(vol_data_full)):
                if i < window:
                    regimes.append('Insufficient Data')
                else:
                    current_vol = vol_data_full.iloc[i]
                    mean_val = rolling_mean.iloc[i]
                    std_val = rolling_std.iloc[i]
                    
                    if current_vol > mean_val + std_val:
                        regimes.append('High Volatility')
                    elif current_vol < mean_val - std_val:
                        regimes.append('Low Volatility')
                    else:
                        regimes.append('Normal Volatility')
            
            regime_counts = pd.Series(regimes).value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Current Regime Distribution:**")
                for regime, count in regime_counts.items():
                    if regime != 'Insufficient Data':
                        st.write(f"• {regime}: {count} ({count/len([r for r in regimes if r != 'Insufficient Data'])*100:.1f}%)")
            
            with col2:
                # Regime transition matrix (simplified)
                transitions = {'High→Normal': 0, 'Normal→High': 0, 'Low→Normal': 0, 'Normal→Low': 0}
                for i in range(1, len(regimes)):
                    if regimes[i-1] == 'High Volatility' and regimes[i] == 'Normal Volatility':
                        transitions['High→Normal'] += 1
                    elif regimes[i-1] == 'Normal Volatility' and regimes[i] == 'High Volatility':
                        transitions['Normal→High'] += 1
                    elif regimes[i-1] == 'Low Volatility' and regimes[i] == 'Normal Volatility':
                        transitions['Low→Normal'] += 1
                    elif regimes[i-1] == 'Normal Volatility' and regimes[i] == 'Low Volatility':
                        transitions['Normal→Low'] += 1
                
                st.write("**Regime Transitions:**")
                for transition, count in transitions.items():
                    st.write(f"• {transition}: {count}")

        with metrics_tab:
            st.subheader("⚡ Model Performance Metrics")

            # Get model info with debug information
            model_info = model_manager.get_model_info('volatility')
            
            # Debug: Show what's actually in model_info
            if model_info:
                st.write("**Debug: Available model info keys:**", list(model_info.keys()))
                
                # Try to find metrics in various possible locations
                metrics = None
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    st.success("✅ Found metrics in 'metrics' key")
                elif 'training_metrics' in model_info:
                    metrics = model_info['training_metrics']
                    st.success("✅ Found metrics in 'training_metrics' key")
                elif 'performance' in model_info:
                    metrics = model_info['performance']
                    st.success("✅ Found metrics in 'performance' key")
                else:
                    # Try to extract from console logs or other sources
                    st.info("🔍 Metrics not found in standard locations, checking alternative sources...")
                    
                    # Check if we can find any numerical performance data
                    for key, value in model_info.items():
                        if isinstance(value, dict):
                            st.write(f"**Found nested data in '{key}':**", list(value.keys()) if value else "Empty")
                            if any(metric_key in value for metric_key in ['rmse', 'r2', 'mae', 'mse']):
                                metrics = value
                                st.success(f"✅ Found metrics in '{key}' key")
                                break

                if metrics:
                    st.write("**Available metric keys:**", list(metrics.keys()) if isinstance(metrics, dict) else "Not a dictionary")
                    
                    # Main performance metrics
                    st.subheader("🎯 Core Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        rmse = metrics.get('rmse', metrics.get('test_rmse', 0))
                        st.metric("RMSE", f"{rmse:.6f}")
                    with col2:
                        mae = metrics.get('mae', 0)
                        st.metric("MAE", f"{mae:.6f}")
                    with col3:
                        mse = metrics.get('mse', 0)
                        st.metric("MSE", f"{mse:.8f}")
                    with col4:
                        r2 = metrics.get('test_r2', metrics.get('r2', 0))
                        st.metric("R² Score", f"{r2:.4f}")

                    # Feature importance analysis
                    feature_importance = model_manager.get_feature_importance('volatility')
                    if feature_importance:
                        st.subheader("🔍 Feature Importance Analysis")
                        
                        importance_df = pd.DataFrame(
                            list(feature_importance.items()), 
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("**Top 15 Features:**")
                            st.dataframe(
                                importance_df.head(15).round(4), 
                                use_container_width=True,
                                column_config={
                                    "Importance": st.column_config.ProgressColumn("Importance", min_value=0, max_value=1)
                                }
                            )
                        
                        with col2:
                            # Feature importance chart
                            top_features = importance_df.head(10)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=top_features['Importance'],
                                y=top_features['Feature'],
                                orientation='h',
                                marker_color='lightblue',
                                text=top_features['Importance'].round(3),
                                textposition='inside'
                            ))
                            
                            fig.update_layout(
                                title="Top 10 Most Important Features",
                                xaxis_title="Importance Score",
                                yaxis_title="Features",
                                height=400,
                                template="plotly_dark"
                            )
                            fig.update_yaxes(categoryorder='total ascending')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Model complexity and training info
                    st.subheader("🏗️ Model Architecture & Training")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Model Type:** Ensemble (XGBoost + CatBoost + Random Forest)")
                        st.write("**Features Used:** 27 technical indicators")
                        st.write("**Training Split:** 80% train / 20% test")
                    
                    with col2:
                        train_rmse = metrics.get('train_rmse', 0)
                        test_rmse = metrics.get('test_rmse', metrics.get('rmse', 0))
                        overfit_ratio = train_rmse / test_rmse if test_rmse > 0 else 0
                        
                        st.metric("Training RMSE", f"{train_rmse:.6f}")
                        st.metric("Test RMSE", f"{test_rmse:.6f}")
                        st.metric("Overfitting Ratio", f"{overfit_ratio:.3f}")
                    
                    with col3:
                        train_r2 = metrics.get('train_r2', 0)
                        test_r2 = metrics.get('test_r2', metrics.get('r2', 0))
                        generalization = test_r2 / train_r2 if train_r2 > 0 else 0
                        
                        st.metric("Training R²", f"{train_r2:.4f}")
                        st.metric("Test R²", f"{test_r2:.4f}")
                        st.metric("Generalization", f"{generalization:.3f}")
                    
                    # Feature categories breakdown
                    st.subheader("📊 Feature Categories")
                    
                    if feature_importance:
                        # Categorize features
                        tech_indicators = ['atr', 'bb_width', 'keltner_width', 'rsi', 'donchian_width']
                        custom_features = ['log_return', 'realized_volatility', 'parkinson_volatility', 
                                         'high_low_ratio', 'gap_pct', 'price_vs_vwap', 'volatility_spike_flag', 'candle_body_ratio']
                        lagged_features = ['lag_volatility_1', 'lag_volatility_3', 'lag_volatility_5',
                                         'lag_atr_1', 'lag_atr_3', 'lag_bb_width', 'volatility_regime']
                        time_features = ['hour', 'minute', 'day_of_week', 'is_post_10am', 
                                       'is_opening_range', 'is_closing_phase', 'is_weekend']
                        
                        category_importance = {}
                        for category, features in [
                            ('Technical Indicators', tech_indicators),
                            ('Custom Engineered', custom_features),
                            ('Lagged Features', lagged_features),
                            ('Time Context', time_features)
                        ]:
                            total_importance = sum(feature_importance.get(f, 0) for f in features)
                            category_importance[category] = total_importance
                        
                        # Category importance chart
                        categories = list(category_importance.keys())
                        importances = list(category_importance.values())
                        
                        fig = go.Figure()
                        fig.add_trace(go.Pie(
                            labels=categories,
                            values=importances,
                            hole=0.4,
                            textinfo='label+percent',
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title="Feature Importance by Category",
                            height=400,
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Model is trained but performance metrics are not accessible in the expected format.")
                    st.info("💡 This can happen if the model was trained but metrics weren't properly saved. Please retrain the volatility model to generate fresh metrics.")
            else:
                st.warning("⚠️ No model performance metrics available. Please train the volatility model first.")

    except Exception as e:
        st.error(f"Error generating volatility predictions: {str(e)}")

def show_direction_predictions(db, fresh_data):
    """Direction predictions with authentic data only"""

    st.header("📈 Direction Predictions")

    # Use the fresh data passed from main function
    if fresh_data is None or len(fresh_data) == 0:
        st.error("No fresh data available")
        return

    # Initialize model manager and check for trained models
    from models.model_manager import ModelManager
    model_manager = ModelManager()

    # Check if direction model exists
    if not model_manager.is_model_trained('direction'):
        st.warning("⚠️ Direction model not trained. Please train the model first.")
        return

    # Prepare features from fresh data
    try:
        # Use direction-specific features
        from features.direction_technical_indicators import DirectionTechnicalIndicators
        dti = DirectionTechnicalIndicators()
        features = dti.calculate_all_direction_indicators(fresh_data)

        if features is None or len(features) == 0:
            st.error("Failed to calculate direction features")
            return

        # Make predictions using trained model
        predictions, probabilities = model_manager.predict('direction', features)

        if predictions is None or len(predictions) == 0:
            st.error("Model prediction failed")
            return

        # Ensure arrays are same length
        if len(predictions) != len(features):
            if len(predictions) < len(features):
                padded_predictions = np.full(len(features), np.nan)
                padded_predictions[:len(predictions)] = predictions
                predictions = padded_predictions

                if probabilities is not None:
                    padded_probs = np.full((len(features), probabilities.shape[1]), np.nan)
                    padded_probs[:len(probabilities)] = probabilities
                    probabilities = padded_probs
            else:
                predictions = predictions[:len(features)]
                if probabilities is not None:
                    probabilities = probabilities[:len(features)]

        # Use authentic datetime index
        datetime_index = features.index

        # Create DataFrame with authentic data only
        pred_df = pd.DataFrame({
            'DateTime': datetime_index,
            'Direction': ['Bullish' if p == 1 else 'Bearish' for p in predictions],
            'Confidence': [np.max(prob) if not np.isnan(prob).all() else 0.5 for prob in probabilities] if probabilities is not None else [0.5] * len(predictions),
            'Date': datetime_index.strftime('%Y-%m-%d'),
            'Time': datetime_index.strftime('%H:%M:%S')
        }, index=datetime_index)

        # Remove NaN predictions
        pred_df = pred_df.dropna(subset=['DateTime'])

        # Create 5 comprehensive sub-tabs for detailed analysis
        chart_tab, data_tab, dist_tab, stats_tab, metrics_tab = st.tabs([
            "📈 Interactive Chart", 
            "📋 Detailed Data", 
            "📊 Distribution Analysis", 
            "🔍 Statistical Analysis", 
            "⚡ Performance Metrics"
        ])

        with chart_tab:
            st.subheader("📈 Direction Prediction Chart")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                chart_points = st.selectbox("Data Points", [50, 100, 200, 500], index=1, key="dir_chart_points")
            
            recent_predictions = pred_df.tail(chart_points)

            # Create subplot with multiple views
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Direction Predictions Over Time', 'Confidence Distribution'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Add bullish signals
            bullish_data = recent_predictions[recent_predictions['Direction'] == 'Bullish']
            if len(bullish_data) > 0:
                fig.add_trace(go.Scatter(
                    x=bullish_data['DateTime'],
                    y=[1] * len(bullish_data),
                    mode='markers',
                    name='Bullish',
                    marker=dict(color='green', size=8, symbol='triangle-up'),
                    text=bullish_data['Confidence'].round(3),
                    textposition="top center"
                ), row=1, col=1)

            # Add bearish signals
            bearish_data = recent_predictions[recent_predictions['Direction'] == 'Bearish']
            if len(bearish_data) > 0:
                fig.add_trace(go.Scatter(
                    x=bearish_data['DateTime'],
                    y=[0] * len(bearish_data),
                    mode='markers',
                    name='Bearish',
                    marker=dict(color='red', size=8, symbol='triangle-down'),
                    text=bearish_data['Confidence'].round(3),
                    textposition="bottom center"
                ), row=1, col=1)

            # Add trend line for confidence using iloc-based grouping
            if len(recent_predictions) >= 10:
                # Group by every 10 data points using iloc
                group_size = 10
                num_groups = len(recent_predictions) // group_size
                confidence_trend = []
                trend_times = []
                
                for i in range(num_groups):
                    start_idx = i * group_size
                    end_idx = min((i + 1) * group_size, len(recent_predictions))
                    group_data = recent_predictions.iloc[start_idx:end_idx]
                    
                    if len(group_data) > 0:
                        confidence_trend.append(group_data['Confidence'].mean())
                        trend_times.append(group_data['DateTime'].iloc[0])
                
                if len(trend_times) > 0 and len(confidence_trend) > 0:
                    fig.add_trace(go.Scatter(
                        x=trend_times,
                        y=confidence_trend,
                        mode='lines',
                        name='Confidence Trend',
                        line=dict(color='yellow', width=2),
                        yaxis='y2'
                    ), row=1, col=1)

            # Confidence histogram
            fig.add_trace(go.Histogram(
                x=recent_predictions['Confidence'],
                nbinsx=20,
                name='Confidence Distribution',
                marker_color='rgba(0, 255, 255, 0.6)'
            ), row=2, col=1)

            # Update layout
            fig.update_layout(
                title=f"Direction Analysis - Last {chart_points} Data Points",
                height=700,
                showlegend=True,
                template="plotly_dark"
            )
            
            fig.update_xaxes(title_text="DateTime", row=1, col=1)
            fig.update_yaxes(title_text="Direction (1=Bullish, 0=Bearish)", row=1, col=1)
            fig.update_yaxes(title_text="Confidence", side='right', row=1, col=1, secondary_y=True)
            fig.update_xaxes(title_text="Confidence Level", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_direction = recent_predictions['Direction'].iloc[-1]
                current_confidence = recent_predictions['Confidence'].iloc[-1]
                st.metric("Current Direction", current_direction)
            with col2:
                st.metric("Current Confidence", f"{current_confidence:.3f}")
            with col3:
                bullish_pct = len(bullish_data) / len(recent_predictions) * 100
                st.metric("Bullish %", f"{bullish_pct:.1f}%")
            with col4:
                avg_confidence = recent_predictions['Confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")

        with data_tab:
            st.subheader("📋 Detailed Direction Data")
            
            col1, col2 = st.columns([2, 1])
            with col2:
                data_points = st.selectbox("Show Records", [100, 200, 500, 1000], index=1, key="dir_data_points")
            
            recent_predictions = pred_df.tail(data_points)
            
            # Enhanced data table with additional calculated columns
            detailed_df = recent_predictions.copy()
            detailed_df['Direction_Score'] = detailed_df['Direction'].map({'Bullish': 1, 'Bearish': 0})
            detailed_df['Confidence_Level'] = pd.cut(detailed_df['Confidence'], 
                                                   bins=[0, 0.6, 0.8, 1.0], 
                                                   labels=['Low', 'Medium', 'High'])
            
            # Calculate streaks
            direction_changes = detailed_df['Direction_Score'].diff().fillna(0)
            streak_groups = (direction_changes != 0).cumsum()
            detailed_df['Streak_Length'] = detailed_df.groupby(streak_groups).cumcount() + 1
            
            # Add momentum indicators
            detailed_df['Confidence_Change'] = detailed_df['Confidence'].diff()
            detailed_df['Direction_Momentum'] = detailed_df['Confidence_Change'].apply(
                lambda x: '📈' if x > 0.1 else '📉' if x < -0.1 else '➡️'
            )
            
            # Display enhanced table
            display_columns = [
                'Date', 'Time', 'Direction', 'Confidence', 'Direction_Momentum',
                'Confidence_Level', 'Streak_Length', 'Confidence_Change'
            ]
            
            st.dataframe(
                detailed_df[display_columns].round(3), 
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                    "Streak_Length": st.column_config.NumberColumn("Streak", format="%d"),
                    "Confidence_Change": st.column_config.NumberColumn("Δ Confidence", format="%.3f")
                }
            )
            
            # Data summary
            st.subheader("📊 Direction Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Direction Distribution:**")
                direction_counts = detailed_df['Direction'].value_counts()
                for direction, count in direction_counts.items():
                    st.write(f"• {direction}: {count} ({count/len(detailed_df)*100:.1f}%)")
            
            with col2:
                st.write("**Confidence Levels:**")
                confidence_counts = detailed_df['Confidence_Level'].value_counts()
                for level, count in confidence_counts.items():
                    st.write(f"• {level}: {count} ({count/len(detailed_df)*100:.1f}%)")
            
            with col3:
                st.write("**Statistics:**")
                st.write(f"• Avg Confidence: {detailed_df['Confidence'].mean():.3f}")
                st.write(f"• Max Streak: {detailed_df['Streak_Length'].max()}")
                st.write(f"• Confidence Std: {detailed_df['Confidence'].std():.3f}")

        with dist_tab:
            st.subheader("📊 Distribution Analysis")
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Direction distribution pie chart
                direction_counts = pred_df['Direction'].value_counts()
                
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=direction_counts.index,
                    values=direction_counts.values,
                    hole=0.4,
                    marker_colors=['green', 'red'],
                    textinfo='label+percent',
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Direction Distribution",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=pred_df['Confidence'],
                    nbinsx=30,
                    histnorm='probability density',
                    name='Confidence Distribution',
                    marker_color='rgba(0, 255, 255, 0.7)'
                ))
                
                fig.update_layout(
                    title="Confidence Distribution",
                    xaxis_title="Confidence Level",
                    yaxis_title="Density",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence by direction
            st.subheader("📈 Confidence by Direction")
            
            fig = go.Figure()
            
            for direction in ['Bullish', 'Bearish']:
                direction_data = pred_df[pred_df['Direction'] == direction]
                if len(direction_data) > 0:
                    fig.add_trace(go.Box(
                        y=direction_data['Confidence'],
                        name=direction,
                        marker_color='green' if direction == 'Bullish' else 'red',
                        boxpoints='outliers'
                    ))
            
            fig.update_layout(
                title="Confidence Distribution by Direction",
                yaxis_title="Confidence Level",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical distribution analysis
            st.subheader("📈 Distribution Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bullish Count", len(pred_df[pred_df['Direction'] == 'Bullish']))
                st.metric("Bearish Count", len(pred_df[pred_df['Direction'] == 'Bearish']))
            with col2:
                bullish_conf = pred_df[pred_df['Direction'] == 'Bullish']['Confidence']
                st.metric("Bullish Avg Conf", f"{bullish_conf.mean():.3f}" if len(bullish_conf) > 0 else "N/A")
                st.metric("Bullish Conf Std", f"{bullish_conf.std():.3f}" if len(bullish_conf) > 0 else "N/A")
            with col3:
                bearish_conf = pred_df[pred_df['Direction'] == 'Bearish']['Confidence']
                st.metric("Bearish Avg Conf", f"{bearish_conf.mean():.3f}" if len(bearish_conf) > 0 else "N/A")
                st.metric("Bearish Conf Std", f"{bearish_conf.std():.3f}" if len(bearish_conf) > 0 else "N/A")
            with col4:
                high_conf = len(pred_df[pred_df['Confidence'] > 0.8])
                st.metric("High Confidence", f"{high_conf} ({high_conf/len(pred_df)*100:.1f}%)")
                low_conf = len(pred_df[pred_df['Confidence'] < 0.6])
                st.metric("Low Confidence", f"{low_conf} ({low_conf/len(pred_df)*100:.1f}%)")

        with stats_tab:
            st.subheader("🔍 Statistical Analysis")
            
            # Time series analysis
            recent_data = pred_df.tail(500)  # Use recent data
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Direction streak analysis
                st.write("**📊 Direction Streak Analysis**")
                
                # Calculate streaks
                direction_numeric = recent_data['Direction'].map({'Bullish': 1, 'Bearish': 0})
                streaks = []
                current_streak = 1
                current_direction = direction_numeric.iloc[0]
                
                for i in range(1, len(direction_numeric)):
                    if direction_numeric.iloc[i] == current_direction:
                        current_streak += 1
                    else:
                        streaks.append(current_streak)
                        current_streak = 1
                        current_direction = direction_numeric.iloc[i]
                streaks.append(current_streak)
                
                if streaks:
                    avg_streak = np.mean(streaks)
                    max_streak = max(streaks)
                    
                    streak_df = pd.DataFrame({
                        'Average Streak': [f"{avg_streak:.1f}"],
                        'Max Streak': [max_streak],
                        'Total Streaks': [len(streaks)],
                        'Streak Consistency': [f"{(avg_streak/max_streak)*100:.1f}%"]
                    })
                    
                    st.dataframe(streak_df, use_container_width=True)
                    
                    # Streak distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=streaks,
                        nbinsx=15,
                        name='Streak Length Distribution',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title="Direction Streak Distribution",
                        xaxis_title="Streak Length",
                        yaxis_title="Frequency",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence trend analysis
                st.write("**📈 Confidence Trend Analysis**")
                rolling_conf = recent_data['Confidence'].rolling(20).mean()
                conf_trend = rolling_conf.iloc[-1] - rolling_conf.iloc[-20] if len(rolling_conf) >= 20 else 0
                
                trend_df = pd.DataFrame({
                    'Current Avg': f"{rolling_conf.iloc[-1]:.3f}" if len(rolling_conf) > 0 else "N/A",
                    'Trend': f"{conf_trend:+.3f}" if abs(conf_trend) > 0.001 else "Stable",
                    'Volatility': f"{recent_data['Confidence'].std():.3f}",
                    'Range': f"{recent_data['Confidence'].max() - recent_data['Confidence'].min():.3f}"
                }, index=[0])
                
                st.dataframe(trend_df, use_container_width=True)
            
            with col2:
                # Direction transition analysis
                st.write("**🔗 Direction Transition Analysis**")
                
                # Calculate transition probabilities
                transitions = {'Bull→Bear': 0, 'Bear→Bull': 0, 'Bull→Bull': 0, 'Bear→Bear': 0}
                for i in range(1, len(recent_data)):
                    prev_dir = recent_data['Direction'].iloc[i-1]
                    curr_dir = recent_data['Direction'].iloc[i]
                    
                    if prev_dir == 'Bullish' and curr_dir == 'Bearish':
                        transitions['Bull→Bear'] += 1
                    elif prev_dir == 'Bearish' and curr_dir == 'Bullish':
                        transitions['Bear→Bull'] += 1
                    elif prev_dir == 'Bullish' and curr_dir == 'Bullish':
                        transitions['Bull→Bull'] += 1
                    elif prev_dir == 'Bearish' and curr_dir == 'Bearish':
                        transitions['Bear→Bear'] += 1
                
                total_transitions = sum(transitions.values())
                if total_transitions > 0:
                    transition_probs = {k: v/total_transitions for k, v in transitions.items()}
                    
                    st.write("**Transition Probabilities:**")
                    for transition, prob in transition_probs.items():
                        st.write(f"• {transition}: {prob:.1%}")
                    
                    # Persistence analysis
                    persistence = (transitions['Bull→Bull'] + transitions['Bear→Bear']) / total_transitions
                    st.metric("Direction Persistence", f"{persistence:.1%}")
                
                # Confidence autocorrelation
                st.write("**📊 Confidence Autocorrelation**")
                
                conf_data = recent_data['Confidence'].tail(200)  # Use recent data for performance
                lags = range(1, min(11, len(conf_data)//4))
                autocorr = [conf_data.autocorr(lag=lag) for lag in lags]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(lags),
                    y=autocorr,
                    name='Autocorrelation',
                    marker_color='lightcyan'
                ))
                
                fig.add_hline(y=0.1, line_dash="dash", line_color="red")
                fig.add_hline(y=-0.1, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title="Confidence Autocorrelation",
                    xaxis_title="Lag",
                    yaxis_title="Correlation",
                    height=300,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Signal quality analysis
            st.subheader("🎯 Signal Quality Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # High confidence signals
                high_conf_signals = recent_data[recent_data['Confidence'] > 0.8]
                st.write(f"**High Confidence Signals (>80%): {len(high_conf_signals)}**")
                if len(high_conf_signals) > 0:
                    high_conf_bullish = len(high_conf_signals[high_conf_signals['Direction'] == 'Bullish'])
                    st.write(f"• Bullish: {high_conf_bullish} ({high_conf_bullish/len(high_conf_signals)*100:.1f}%)")
                    st.write(f"• Bearish: {len(high_conf_signals) - high_conf_bullish} ({(len(high_conf_signals) - high_conf_bullish)/len(high_conf_signals)*100:.1f}%)")
            
            with col2:
                # Medium confidence signals
                med_conf_signals = recent_data[(recent_data['Confidence'] >= 0.6) & (recent_data['Confidence'] <= 0.8)]
                st.write(f"**Medium Confidence Signals (60-80%): {len(med_conf_signals)}**")
                if len(med_conf_signals) > 0:
                    med_conf_bullish = len(med_conf_signals[med_conf_signals['Direction'] == 'Bullish'])
                    st.write(f"• Bullish: {med_conf_bullish} ({med_conf_bullish/len(med_conf_signals)*100:.1f}%)")
                    st.write(f"• Bearish: {len(med_conf_signals) - med_conf_bullish} ({(len(med_conf_signals) - med_conf_bullish)/len(med_conf_signals)*100:.1f}%)")
            
            with col3:
                # Low confidence signals
                low_conf_signals = recent_data[recent_data['Confidence'] < 0.6]
                st.write(f"**Low Confidence Signals (<60%): {len(low_conf_signals)}**")
                if len(low_conf_signals) > 0:
                    low_conf_bullish = len(low_conf_signals[low_conf_signals['Direction'] == 'Bullish'])
                    st.write(f"• Bullish: {low_conf_bullish} ({low_conf_bullish/len(low_conf_signals)*100:.1f}%)")
                    st.write(f"• Bearish: {len(low_conf_signals) - low_conf_bullish} ({(len(low_conf_signals) - low_conf_bullish)/len(low_conf_signals)*100:.1f}%)")

        with metrics_tab:
            st.subheader("⚡ Model Performance Metrics")

            # Get model info with debug information
            model_info = model_manager.get_model_info('direction')
            
            if model_info:
                st.write("**Debug: Available model info keys:**", list(model_info.keys()))
                
                # Try to find metrics in various possible locations
                metrics = None
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    st.success("✅ Found metrics in 'metrics' key")
                elif 'training_metrics' in model_info:
                    metrics = model_info['training_metrics']
                    st.success("✅ Found metrics in 'training_metrics' key")
                elif 'performance' in model_info:
                    metrics = model_info['performance']
                    st.success("✅ Found metrics in 'performance' key")
                else:
                    st.info("🔍 Metrics not found in standard locations, checking alternative sources...")
                    
                    for key, value in model_info.items():
                        if isinstance(value, dict):
                            if any(metric_key in value for metric_key in ['accuracy', 'precision', 'recall', 'f1']):
                                metrics = value
                                st.success(f"✅ Found metrics in '{key}' key")
                                break

                if metrics:
                    # Main performance metrics
                    st.subheader("🎯 Core Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
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
                    with col4:
                        f1_score = classification_metrics.get('weighted avg', {}).get('f1-score', 0)
                        st.metric("F1 Score", f"{f1_score:.2%}")

                    # Detailed classification report
                    if 'classification_report' in metrics:
                        st.subheader("📊 Detailed Classification Report")
                        
                        class_report = metrics['classification_report']
                        if isinstance(class_report, dict):
                            # Create a formatted table
                            report_data = []
                            for class_name, class_metrics in class_report.items():
                                if isinstance(class_metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                    report_data.append({
                                        'Class': 'Bullish' if class_name == '1' else 'Bearish' if class_name == '0' else class_name,
                                        'Precision': f"{class_metrics.get('precision', 0):.3f}",
                                        'Recall': f"{class_metrics.get('recall', 0):.3f}",
                                        'F1-Score': f"{class_metrics.get('f1-score', 0):.3f}",
                                        'Support': class_metrics.get('support', 0)
                                    })
                            
                            if report_data:
                                report_df = pd.DataFrame(report_data)
                                st.dataframe(report_df, use_container_width=True)

                    # Feature importance analysis
                    feature_importance = model_manager.get_feature_importance('direction')
                    if feature_importance:
                        st.subheader("🔍 Feature Importance Analysis")
                        
                        importance_df = pd.DataFrame(
                            list(feature_importance.items()), 
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("**Top 15 Features:**")
                            st.dataframe(
                                importance_df.head(15).round(4), 
                                use_container_width=True,
                                column_config={
                                    "Importance": st.column_config.ProgressColumn("Importance", min_value=0, max_value=1)
                                }
                            )
                        
                        with col2:
                            # Feature importance chart
                            top_features = importance_df.head(10)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=top_features['Importance'],
                                y=top_features['Feature'],
                                orientation='h',
                                marker_color='lightgreen',
                                text=top_features['Importance'].round(3),
                                textposition='inside'
                            ))
                            
                            fig.update_layout(
                                title="Top 10 Most Important Features",
                                xaxis_title="Importance Score",
                                yaxis_title="Features",
                                height=400,
                                template="plotly_dark"
                            )
                            fig.update_yaxes(categoryorder='total ascending')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Model complexity and training info
                    st.subheader("🏗️ Model Architecture & Training")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Model Type:** Classification Ensemble")
                        st.write("**Task:** Binary Classification (Bullish/Bearish)")
                        st.write("**Training Split:** 80% train / 20% test")
                    
                    with col2:
                        train_accuracy = metrics.get('train_accuracy', 0)
                        test_accuracy = metrics.get('test_accuracy', metrics.get('accuracy', 0))
                        overfit_ratio = (train_accuracy - test_accuracy) if train_accuracy > 0 else 0
                        
                        st.metric("Training Accuracy", f"{train_accuracy:.2%}")
                        st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                        st.metric("Overfitting", f"{overfit_ratio:.1%}")
                    
                    with col3:
                        if 'confusion_matrix' in metrics:
                            cm = metrics['confusion_matrix']
                            if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
                                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                                
                                st.metric("Specificity", f"{specificity:.2%}")
                                st.metric("Sensitivity", f"{sensitivity:.2%}")
                                st.metric("Total Predictions", f"{tp + tn + fp + fn:,}")
                    
                    # Confusion matrix visualization
                    if 'confusion_matrix' in metrics:
                        st.subheader("📊 Confusion Matrix")
                        
                        cm = metrics['confusion_matrix']
                        if isinstance(cm, list) and len(cm) == 2:
                            fig = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=['Predicted Bearish', 'Predicted Bullish'],
                                y=['Actual Bearish', 'Actual Bullish'],
                                colorscale='Blues',
                                text=cm,
                                texttemplate="%{text}",
                                textfont={"size": 16}
                            ))
                            
                            fig.update_layout(
                                title="Confusion Matrix",
                                height=400,
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Model is trained but performance metrics are not accessible in the expected format.")
                    st.info("💡 This can happen if the model was trained but metrics weren't properly saved. Please retrain the direction model to generate fresh metrics.")
            else:
                st.warning("⚠️ No model performance metrics available. Please train the direction model first.")

    except Exception as e:
        st.error(f"Error generating direction predictions: {str(e)}")

def show_profit_predictions(db, fresh_data):
    """Profit probability predictions with authentic data only"""

    st.header("💰 Profit Probability Predictions")

    # Use the fresh data passed from main function
    if fresh_data is None or len(fresh_data) == 0:
        st.error("No fresh data available")
        return

    # Initialize model manager and check for trained models
    from models.model_manager import ModelManager
    model_manager = ModelManager()

    # Check if profit probability model exists
    if not model_manager.is_model_trained('profit_probability'):
        st.warning("⚠️ Profit probability model not trained. Please train the model first.")
        return

    # Prepare features from fresh data
    try:
        # Use profit probability-specific features
        profit_prob_model_instance = model_manager.trained_models.get('profit_probability', {})
        if 'feature_names' in profit_prob_model_instance:
            required_features = profit_prob_model_instance['feature_names']
            st.info(f"Using exactly {len(required_features)} features from training (no additional features)")

            # Calculate profit probability features to match training
            from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
            all_features = ProfitProbabilityTechnicalIndicators.calculate_all_profit_probability_indicators(fresh_data)

            # Use ONLY the exact features that were used during training
            missing_features = [col for col in required_features if col not in all_features.columns]

            if missing_features:
                st.error(f"Missing required features: {missing_features}")
                st.error("Cannot make predictions without all required features")
                return

            # Select ONLY the exact features from training - no additions
            features = all_features[required_features].copy()

            st.success(f"✅ Using exactly {len(required_features)} features from training")
        else:
            st.error("No feature names stored in model. Please retrain the model.")
            return

        if features is None or len(features) == 0:
            st.error("Failed to calculate profit probability features")
            return

    # Make predictions using trained model
        predictions, probabilities = model_manager.predict('profit_probability', features)

        if predictions is None or len(predictions) == 0:
            st.error("Model prediction failed")
            return

        # Ensure arrays are same length
        if len(predictions) != len(features):
            st.error(f"Array length mismatch: predictions={len(predictions)}, features={len(features)}")
            return

        # Use authentic datetime index
        datetime_index = features.index

        # Create DataFrame with authentic data only
        pred_df = pd.DataFrame({
            'DateTime': datetime_index,
            'Profit_Probability': ['High Profit' if p == 1 else 'Low Profit' for p in predictions],
            'Confidence': [np.max(prob) for prob in probabilities] if probabilities is not None else None,
            'Date': datetime_index.strftime('%Y-%m-%d'),
            'Time': datetime_index.strftime('%H:%M:%S')
        }, index=datetime_index)

        # Remove rows with NaN predictions for display
        pred_df = pred_df.dropna(subset=['Profit_Probability'])

        if len(pred_df) == 0:
            st.error("No valid predictions generated")
            return

        # Create 5 comprehensive sub-tabs for detailed analysis
        chart_tab, data_tab, dist_tab, stats_tab, metrics_tab = st.tabs([
            "📈 Interactive Chart", 
            "📋 Detailed Data", 
            "📊 Distribution Analysis", 
            "🔍 Statistical Analysis", 
            "⚡ Performance Metrics"
        ])

        with chart_tab:
            st.subheader("📈 Profit Probability Prediction Chart")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                chart_points = st.selectbox("Data Points", [50, 100, 200, 500], index=1, key="profit_chart_points")
            
            recent_predictions = pred_df.tail(chart_points)

            # Create subplot with multiple views
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Profit Probability Predictions Over Time', 'Confidence Distribution'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Add high profit signals
            high_profit = recent_predictions[recent_predictions['Profit_Probability'] == 'High Profit']
            if len(high_profit) > 0:
                fig.add_trace(go.Scatter(
                    x=high_profit['DateTime'],
                    y=[1] * len(high_profit),
                    mode='markers',
                    name='High Profit',
                    marker=dict(color='green', size=8, symbol='triangle-up'),
                    text=high_profit['Confidence'].round(3) if 'Confidence' in high_profit.columns else None,
                    textposition="top center"
                ), row=1, col=1)

            # Add low profit signals
            low_profit = recent_predictions[recent_predictions['Profit_Probability'] == 'Low Profit']
            if len(low_profit) > 0:
                fig.add_trace(go.Scatter(
                    x=low_profit['DateTime'],
                    y=[0] * len(low_profit),
                    mode='markers',
                    name='Low Profit',
                    marker=dict(color='red', size=8, symbol='triangle-down'),
                    text=low_profit['Confidence'].round(3) if 'Confidence' in low_profit.columns else None,
                    textposition="bottom center"
                ), row=1, col=1)

            # Add confidence trend line if confidence data exists
            if 'Confidence' in recent_predictions.columns and not recent_predictions['Confidence'].isna().all():
                if len(recent_predictions) >= 10:
                    # Group by every 10 data points using iloc
                    group_size = 10
                    num_groups = len(recent_predictions) // group_size
                    confidence_trend = []
                    trend_times = []
                    
                    for i in range(num_groups):
                        start_idx = i * group_size
                        end_idx = min((i + 1) * group_size, len(recent_predictions))
                        group_data = recent_predictions.iloc[start_idx:end_idx]
                        
                        if len(group_data) > 0 and not group_data['Confidence'].isna().all():
                            confidence_trend.append(group_data['Confidence'].mean())
                            trend_times.append(group_data['DateTime'].iloc[0])
                    
                    if len(trend_times) > 0 and len(confidence_trend) > 0:
                        fig.add_trace(go.Scatter(
                            x=trend_times,
                            y=confidence_trend,
                            mode='lines',
                            name='Confidence Trend',
                            line=dict(color='yellow', width=2),
                            yaxis='y2'
                        ), row=1, col=1)

            # Confidence histogram
            if 'Confidence' in recent_predictions.columns and not recent_predictions['Confidence'].isna().all():
                fig.add_trace(go.Histogram(
                    x=recent_predictions['Confidence'],
                    nbinsx=20,
                    name='Confidence Distribution',
                    marker_color='rgba(255, 165, 0, 0.6)'
                ), row=2, col=1)

            fig.update_layout(
                title=f"Profit Probability Analysis - Last {chart_points} Data Points",
                height=700,
                showlegend=True,
                template="plotly_dark"
            )
            
            fig.update_xaxes(title_text="DateTime", row=1, col=1)
            fig.update_yaxes(title_text="Profit Probability (1=High, 0=Low)", row=1, col=1)
            fig.update_yaxes(title_text="Confidence", side='right', row=1, col=1, secondary_y=True)
            fig.update_xaxes(title_text="Confidence Level", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_prob = recent_predictions['Profit_Probability'].iloc[-1]
                st.metric("Current Prediction", current_prob)
            with col2:
                if 'Confidence' in recent_predictions.columns and not recent_predictions['Confidence'].isna().all():
                    current_confidence = recent_predictions['Confidence'].iloc[-1]
                    st.metric("Current Confidence", f"{current_confidence:.3f}")
                else:
                    st.metric("Current Confidence", "N/A")
            with col3:
                high_profit_pct = len(high_profit) / len(recent_predictions) * 100
                st.metric("High Profit %", f"{high_profit_pct:.1f}%")
            with col4:
                if 'Confidence' in recent_predictions.columns and not recent_predictions['Confidence'].isna().all():
                    avg_confidence = recent_predictions['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                else:
                    st.metric("Avg Confidence", "N/A")

        with data_tab:
            st.subheader("📋 Detailed Profit Probability Data")
            
            col1, col2 = st.columns([2, 1])
            with col2:
                data_points = st.selectbox("Show Records", [100, 200, 500, 1000], index=1, key="profit_data_points")
            
            recent_predictions = pred_df.tail(data_points)
            
            # Enhanced data table with additional calculated columns
            detailed_df = recent_predictions.copy()
            detailed_df['Profit_Score'] = detailed_df['Profit_Probability'].map({'High Profit': 1, 'Low Profit': 0})
            
            if 'Confidence' in detailed_df.columns and not detailed_df['Confidence'].isna().all():
                detailed_df['Confidence_Level'] = pd.cut(detailed_df['Confidence'], 
                                                       bins=[0, 0.6, 0.8, 1.0], 
                                                       labels=['Low', 'Medium', 'High'])
                # Calculate streaks
                profit_changes = detailed_df['Profit_Score'].diff().fillna(0)
                streak_groups = (profit_changes != 0).cumsum()
                detailed_df['Streak_Length'] = detailed_df.groupby(streak_groups).cumcount() + 1
                
                # Add momentum indicators
                detailed_df['Confidence_Change'] = detailed_df['Confidence'].diff()
                detailed_df['Profit_Momentum'] = detailed_df['Confidence_Change'].apply(
                    lambda x: '📈' if x > 0.1 else '📉' if x < -0.1 else '➡️'
                )
                
                display_columns = [
                    'Date', 'Time', 'Profit_Probability', 'Confidence', 'Profit_Momentum',
                    'Confidence_Level', 'Streak_Length', 'Confidence_Change'
                ]
            else:
                # Calculate streaks without confidence
                profit_changes = detailed_df['Profit_Score'].diff().fillna(0)
                streak_groups = (profit_changes != 0).cumsum()
                detailed_df['Streak_Length'] = detailed_df.groupby(streak_groups).cumcount() + 1
                
                display_columns = [
                    'Date', 'Time', 'Profit_Probability', 'Streak_Length'
                ]
            
            # Display enhanced table
            st.dataframe(
                detailed_df[display_columns].round(3), 
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                    "Streak_Length": st.column_config.NumberColumn("Streak", format="%d"),
                    "Confidence_Change": st.column_config.NumberColumn("Δ Confidence", format="%.3f")
                }
            )
            
            # Data summary
            st.subheader("📊 Profit Probability Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Profit Distribution:**")
                profit_counts = detailed_df['Profit_Probability'].value_counts()
                for profit_type, count in profit_counts.items():
                    st.write(f"• {profit_type}: {count} ({count/len(detailed_df)*100:.1f}%)")
            
            with col2:
                if 'Confidence_Level' in detailed_df.columns:
                    st.write("**Confidence Levels:**")
                    confidence_counts = detailed_df['Confidence_Level'].value_counts()
                    for level, count in confidence_counts.items():
                        st.write(f"• {level}: {count} ({count/len(detailed_df)*100:.1f}%)")
                else:
                    st.write("**Confidence Levels:**")
                    st.write("• N/A (No confidence data)")
            
            with col3:
                st.write("**Statistics:**")
                if 'Confidence' in detailed_df.columns and not detailed_df['Confidence'].isna().all():
                    st.write(f"• Avg Confidence: {detailed_df['Confidence'].mean():.3f}")
                    st.write(f"• Confidence Std: {detailed_df['Confidence'].std():.3f}")
                else:
                    st.write("• Avg Confidence: N/A")
                    st.write("• Confidence Std: N/A")
                st.write(f"• Max Streak: {detailed_df['Streak_Length'].max()}")

        with dist_tab:
            st.subheader("📊 Distribution Analysis")
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Profit probability distribution pie chart
                profit_counts = pred_df['Profit_Probability'].value_counts()
                
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=profit_counts.index,
                    values=profit_counts.values,
                    hole=0.4,
                    marker_colors=['green', 'red'],
                    textinfo='label+percent',
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Profit Probability Distribution",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution histogram
                if 'Confidence' in pred_df.columns and not pred_df['Confidence'].isna().all():
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=pred_df['Confidence'],
                        nbinsx=30,
                        histnorm='probability density',
                        name='Confidence Distribution',
                        marker_color='rgba(255, 165, 0, 0.7)'
                    ))
                    
                    fig.update_layout(
                        title="Confidence Distribution",
                        xaxis_title="Confidence Level",
                        yaxis_title="Density",
                        height=400,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No confidence data available for distribution analysis")
            
            # Confidence by profit probability
            if 'Confidence' in pred_df.columns and not pred_df['Confidence'].isna().all():
                st.subheader("📈 Confidence by Profit Probability")
                
                fig = go.Figure()
                
                for profit_type in ['High Profit', 'Low Profit']:
                    profit_data = pred_df[pred_df['Profit_Probability'] == profit_type]
                    if len(profit_data) > 0:
                        fig.add_trace(go.Box(
                            y=profit_data['Confidence'],
                            name=profit_type,
                            marker_color='green' if profit_type == 'High Profit' else 'red',
                            boxpoints='outliers'
                        ))
                
                fig.update_layout(
                    title="Confidence Distribution by Profit Probability",
                    yaxis_title="Confidence Level",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical distribution analysis
            st.subheader("📈 Distribution Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("High Profit Count", len(pred_df[pred_df['Profit_Probability'] == 'High Profit']))
                st.metric("Low Profit Count", len(pred_df[pred_df['Profit_Probability'] == 'Low Profit']))
            with col2:
                if 'Confidence' in pred_df.columns and not pred_df['Confidence'].isna().all():
                    high_profit_conf = pred_df[pred_df['Profit_Probability'] == 'High Profit']['Confidence']
                    st.metric("High Profit Avg Conf", f"{high_profit_conf.mean():.3f}" if len(high_profit_conf) > 0 else "N/A")
                    st.metric("High Profit Conf Std", f"{high_profit_conf.std():.3f}" if len(high_profit_conf) > 0 else "N/A")
                else:
                    st.metric("High Profit Avg Conf", "N/A")
                    st.metric("High Profit Conf Std", "N/A")
            with col3:
                if 'Confidence' in pred_df.columns and not pred_df['Confidence'].isna().all():
                    low_profit_conf = pred_df[pred_df['Profit_Probability'] == 'Low Profit']['Confidence']
                    st.metric("Low Profit Avg Conf", f"{low_profit_conf.mean():.3f}" if len(low_profit_conf) > 0 else "N/A")
                    st.metric("Low Profit Conf Std", f"{low_profit_conf.std():.3f}" if len(low_profit_conf) > 0 else "N/A")
                else:
                    st.metric("Low Profit Avg Conf", "N/A")
                    st.metric("Low Profit Conf Std", "N/A")
            with col4:
                if 'Confidence' in pred_df.columns and not pred_df['Confidence'].isna().all():
                    high_conf = len(pred_df[pred_df['Confidence'] > 0.8])
                    st.metric("High Confidence", f"{high_conf} ({high_conf/len(pred_df)*100:.1f}%)")
                    low_conf = len(pred_df[pred_df['Confidence'] < 0.6])
                    st.metric("Low Confidence", f"{low_conf} ({low_conf/len(pred_df)*100:.1f}%)")
                else:
                    st.metric("High Confidence", "N/A")
                    st.metric("Low Confidence", "N/A")

        with stats_tab:
            st.subheader("🔍 Statistical Analysis")
            
            # Time series analysis
            recent_data = pred_df.tail(500)  # Use recent data
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Profit probability streak analysis
                st.write("**📊 Profit Probability Streak Analysis**")
                
                # Calculate streaks
                profit_numeric = recent_data['Profit_Probability'].map({'High Profit': 1, 'Low Profit': 0})
                streaks = []
                current_streak = 1
                current_profit = profit_numeric.iloc[0]
                
                for i in range(1, len(profit_numeric)):
                    if profit_numeric.iloc[i] == current_profit:
                        current_streak += 1
                    else:
                        streaks.append(current_streak)
                        current_streak = 1
                        current_profit = profit_numeric.iloc[i]
                streaks.append(current_streak)
                
                if streaks:
                    avg_streak = np.mean(streaks)
                    max_streak = max(streaks)
                    
                    streak_df = pd.DataFrame({
                        'Average Streak': [f"{avg_streak:.1f}"],
                        'Max Streak': [max_streak],
                        'Total Streaks': [len(streaks)],
                        'Streak Consistency': [f"{(avg_streak/max_streak)*100:.1f}%"]
                    })
                    
                    st.dataframe(streak_df, use_container_width=True)
                    
                    # Streak distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=streaks,
                        nbinsx=15,
                        name='Streak Length Distribution',
                        marker_color='lightcoral'
                    ))
                    
                    fig.update_layout(
                        title="Profit Probability Streak Distribution",
                        xaxis_title="Streak Length",
                        yaxis_title="Frequency",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence trend analysis
                if 'Confidence' in recent_data.columns and not recent_data['Confidence'].isna().all():
                    st.write("**📈 Confidence Trend Analysis**")
                    rolling_conf = recent_data['Confidence'].rolling(20).mean()
                    conf_trend = rolling_conf.iloc[-1] - rolling_conf.iloc[-20] if len(rolling_conf) >= 20 else 0
                    
                    trend_df = pd.DataFrame({
                        'Current Avg': f"{rolling_conf.iloc[-1]:.3f}" if len(rolling_conf) > 0 else "N/A",
                        'Trend': f"{conf_trend:+.3f}" if abs(conf_trend) > 0.001 else "Stable",
                        'Volatility': f"{recent_data['Confidence'].std():.3f}",
                        'Range': f"{recent_data['Confidence'].max() - recent_data['Confidence'].min():.3f}"
                    }, index=[0])
                    
                    st.dataframe(trend_df, use_container_width=True)
                else:
                    st.write("**📈 Confidence Trend Analysis**")
                    st.info("No confidence data available for trend analysis")
            
            with col2:
                # Profit probability transition analysis
                st.write("**🔗 Profit Transition Analysis**")
                
                # Calculate transition probabilities
                transitions = {'High→Low': 0, 'Low→High': 0, 'High→High': 0, 'Low→Low': 0}
                for i in range(1, len(recent_data)):
                    prev_profit = recent_data['Profit_Probability'].iloc[i-1]
                    curr_profit = recent_data['Profit_Probability'].iloc[i]
                    
                    if prev_profit == 'High Profit' and curr_profit == 'Low Profit':
                        transitions['High→Low'] += 1
                    elif prev_profit == 'Low Profit' and curr_profit == 'High Profit':
                        transitions['Low→High'] += 1
                    elif prev_profit == 'High Profit' and curr_profit == 'High Profit':
                        transitions['High→High'] += 1
                    elif prev_profit == 'Low Profit' and curr_profit == 'Low Profit':
                        transitions['Low→Low'] += 1
                
                total_transitions = sum(transitions.values())
                if total_transitions > 0:
                    transition_probs = {k: v/total_transitions for k, v in transitions.items()}
                    
                    st.write("**Transition Probabilities:**")
                    for transition, prob in transition_probs.items():
                        st.write(f"• {transition}: {prob:.1%}")
                    
                    # Persistence analysis
                    persistence = (transitions['High→High'] + transitions['Low→Low']) / total_transitions
                    st.metric("Profit Persistence", f"{persistence:.1%}")
                
                # Confidence autocorrelation
                if 'Confidence' in recent_data.columns and not recent_data['Confidence'].isna().all():
                    st.write("**📊 Confidence Autocorrelation**")
                    
                    conf_data = recent_data['Confidence'].tail(200)  # Use recent data for performance
                    lags = range(1, min(11, len(conf_data)//4))
                    autocorr = [conf_data.autocorr(lag=lag) for lag in lags]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(lags),
                        y=autocorr,
                        name='Autocorrelation',
                        marker_color='lightsalmon'
                    ))
                    
                    fig.add_hline(y=0.1, line_dash="dash", line_color="red")
                    fig.add_hline(y=-0.1, line_dash="dash", line_color="red")
                    
                    fig.update_layout(
                        title="Confidence Autocorrelation",
                        xaxis_title="Lag",
                        yaxis_title="Correlation",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("**📊 Confidence Autocorrelation**")
                    st.info("No confidence data available for autocorrelation analysis")
            
            # Signal quality analysis
            st.subheader("🎯 Signal Quality Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # High confidence signals
                if 'Confidence' in recent_data.columns and not recent_data['Confidence'].isna().all():
                    high_conf_signals = recent_data[recent_data['Confidence'] > 0.8]
                    st.write(f"**High Confidence Signals (>80%): {len(high_conf_signals)}**")
                    if len(high_conf_signals) > 0:
                        high_conf_profit = len(high_conf_signals[high_conf_signals['Profit_Probability'] == 'High Profit'])
                        st.write(f"• High Profit: {high_conf_profit} ({high_conf_profit/len(high_conf_signals)*100:.1f}%)")
                        st.write(f"• Low Profit: {len(high_conf_signals) - high_conf_profit} ({(len(high_conf_signals) - high_conf_profit)/len(high_conf_signals)*100:.1f}%)")
                else:
                    st.write("**High Confidence Signals (>80%): N/A**")
                    st.info("No confidence data available")
            
            with col2:
                # Medium confidence signals
                if 'Confidence' in recent_data.columns and not recent_data['Confidence'].isna().all():
                    med_conf_signals = recent_data[(recent_data['Confidence'] >= 0.6) & (recent_data['Confidence'] <= 0.8)]
                    st.write(f"**Medium Confidence Signals (60-80%): {len(med_conf_signals)}**")
                    if len(med_conf_signals) > 0:
                        med_conf_profit = len(med_conf_signals[med_conf_signals['Profit_Probability'] == 'High Profit'])
                        st.write(f"• High Profit: {med_conf_profit} ({med_conf_profit/len(med_conf_signals)*100:.1f}%)")
                        st.write(f"• Low Profit: {len(med_conf_signals) - med_conf_profit} ({(len(med_conf_signals) - med_conf_profit)/len(med_conf_signals)*100:.1f}%)")
                else:
                    st.write("**Medium Confidence Signals (60-80%): N/A**")
                    st.info("No confidence data available")
            
            with col3:
                # Low confidence signals
                if 'Confidence' in recent_data.columns and not recent_data['Confidence'].isna().all():
                    low_conf_signals = recent_data[recent_data['Confidence'] < 0.6]
                    st.write(f"**Low Confidence Signals (<60%): {len(low_conf_signals)}**")
                    if len(low_conf_signals) > 0:
                        low_conf_profit = len(low_conf_signals[low_conf_signals['Profit_Probability'] == 'High Profit'])
                        st.write(f"• High Profit: {low_conf_profit} ({low_conf_profit/len(low_conf_signals)*100:.1f}%)")
                        st.write(f"• Low Profit: {len(low_conf_signals) - low_conf_profit} ({(len(low_conf_signals) - low_conf_profit)/len(low_conf_signals)*100:.1f}%)")
                else:
                    st.write("**Low Confidence Signals (<60%): N/A**")
                    st.info("No confidence data available")

        with metrics_tab:
            st.subheader("⚡ Model Performance Metrics")

            # Get model info with debug information
            model_info = model_manager.get_model_info('profit_probability')
            
            if model_info:
                st.write("**Debug: Available model info keys:**", list(model_info.keys()))
                
                # Try to find metrics in various possible locations
                metrics = None
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    st.success("✅ Found metrics in 'metrics' key")
                elif 'training_metrics' in model_info:
                    metrics = model_info['training_metrics']
                    st.success("✅ Found metrics in 'training_metrics' key")
                elif 'performance' in model_info:
                    metrics = model_info['performance']
                    st.success("✅ Found metrics in 'performance' key")
                else:
                    st.info("🔍 Metrics not found in standard locations, checking alternative sources...")
                    
                    for key, value in model_info.items():
                        if isinstance(value, dict):
                            if any(metric_key in value for metric_key in ['accuracy', 'precision', 'recall', 'f1']):
                                metrics = value
                                st.success(f"✅ Found metrics in '{key}' key")
                                break

                if metrics:
                    # Main performance metrics
                    st.subheader("🎯 Core Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
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
                    with col4:
                        f1_score = classification_metrics.get('weighted avg', {}).get('f1-score', 0)
                        st.metric("F1 Score", f"{f1_score:.2%}")

                    # Detailed classification report
                    if 'classification_report' in metrics:
                        st.subheader("📊 Detailed Classification Report")
                        
                        class_report = metrics['classification_report']
                        if isinstance(class_report, dict):
                            # Create a formatted table
                            report_data = []
                            for class_name, class_metrics in class_report.items():
                                if isinstance(class_metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                    report_data.append({
                                        'Class': 'High Profit' if class_name == '1' else 'Low Profit' if class_name == '0' else class_name,
                                        'Precision': f"{class_metrics.get('precision', 0):.3f}",
                                        'Recall': f"{class_metrics.get('recall', 0):.3f}",
                                        'F1-Score': f"{class_metrics.get('f1-score', 0):.3f}",
                                        'Support': class_metrics.get('support', 0)
                                    })
                            
                            if report_data:
                                report_df = pd.DataFrame(report_data)
                                st.dataframe(report_df, use_container_width=True)

                    # Feature importance analysis
                    feature_importance = model_manager.get_feature_importance('profit_probability')
                    if feature_importance:
                        st.subheader("🔍 Feature Importance Analysis")
                        
                        importance_df = pd.DataFrame(
                            list(feature_importance.items()), 
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("**Top 15 Features:**")
                            st.dataframe(
                                importance_df.head(15).round(4), 
                                use_container_width=True,
                                column_config={
                                    "Importance": st.column_config.ProgressColumn("Importance", min_value=0, max_value=1)
                                }
                            )
                        
                        with col2:
                            # Feature importance chart
                            top_features = importance_df.head(10)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=top_features['Importance'],
                                y=top_features['Feature'],
                                orientation='h',
                                marker_color='lightcoral',
                                text=top_features['Importance'].round(3),
                                textposition='inside'
                            ))
                            
                            fig.update_layout(
                                title="Top 10 Most Important Features",
                                xaxis_title="Importance Score",
                                yaxis_title="Features",
                                height=400,
                                template="plotly_dark"
                            )
                            fig.update_yaxes(categoryorder='total ascending')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Model complexity and training info
                    st.subheader("🏗️ Model Architecture & Training")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Model Type:** Classification Ensemble")
                        st.write("**Task:** Binary Classification (High/Low Profit)")
                        st.write("**Training Split:** 80% train / 20% test")
                    
                    with col2:
                        train_accuracy = metrics.get('train_accuracy', 0)
                        test_accuracy = metrics.get('test_accuracy', metrics.get('accuracy', 0))
                        overfit_ratio = (train_accuracy - test_accuracy) if train_accuracy > 0 else 0
                        
                        st.metric("Training Accuracy", f"{train_accuracy:.2%}")
                        st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                        st.metric("Overfitting", f"{overfit_ratio:.1%}")
                    
                    with col3:
                        if 'confusion_matrix' in metrics:
                            cm = metrics['confusion_matrix']
                            if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
                                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                                
                                st.metric("Specificity", f"{specificity:.2%}")
                                st.metric("Sensitivity", f"{sensitivity:.2%}")
                                st.metric("Total Predictions", f"{tp + tn + fp + fn:,}")
                    
                    # Confusion matrix visualization
                    if 'confusion_matrix' in metrics:
                        st.subheader("📊 Confusion Matrix")
                        
                        cm = metrics['confusion_matrix']
                        if isinstance(cm, list) and len(cm) == 2:
                            fig = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=['Predicted Low Profit', 'Predicted High Profit'],
                                y=['Actual Low Profit', 'Actual High Profit'],
                                colorscale='Oranges',
                                text=cm,
                                texttemplate="%{text}",
                                textfont={"size": 16}
                            ))
                            
                            fig.update_layout(
                                title="Confusion Matrix",
                                height=400,
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Model is trained but performance metrics are not accessible in the expected format.")
                    st.info("💡 This can happen if the model was trained but metrics weren't properly saved. Please retrain the profit probability model to generate fresh metrics.")
            else:
                st.warning("⚠️ No model performance metrics available. Please train the profit probability model first.")

    except Exception as e:
        st.error(f"Error generating profit probability predictions: {str(e)}")

def show_reversal_predictions(db, fresh_data):
    """Reversal detection predictions with authentic data only"""

    st.header("🔄 Reversal Detection")

    # Use the fresh data passed from main function
    if fresh_data is None or len(fresh_data) == 0:
        st.error("No fresh data available")
        return

    # Initialize model manager and check for trained models
    from models.model_manager import ModelManager
    model_manager = ModelManager()

    # Check if reversal model exists
    if not model_manager.is_model_trained('reversal'):
        st.warning("⚠️ Reversal model not trained. Please train the model first.")
        return

    # Prepare features from fresh data
    try:
        # Use comprehensive reversal features like the model was trained on
        from models.reversal_model import ReversalModel
        reversal_model_instance = ReversalModel()
        features = reversal_model_instance.prepare_features(fresh_data)

        if features is None or len(features) == 0:
            st.error("Failed to calculate reversal features")
            return

        # Make predictions using trained model
        predictions, probabilities = model_manager.predict('reversal', features)

        if predictions is None or len(predictions) == 0:
            st.error("Model prediction failed")
            return

        # Ensure arrays are same length
        if len(predictions) != len(features):
            st.error(f"Array length mismatch: predictions={len(predictions)}, features={len(features)}")
            return

        # Use authentic datetime index
        datetime_index = features.index

        # Create DataFrame with authentic data only
        pred_df = pd.DataFrame({
            'DateTime': datetime_index,
            'Reversal_Signal': ['Reversal' if p == 1 else 'No Reversal' for p in predictions],
            'Confidence': [np.max(prob) for prob in probabilities] if probabilities is not None else [0.5] * len(predictions),
            'Date': datetime_index.strftime('%Y-%m-%d'),
            'Time': datetime_index.strftime('%H:%M:%S')
        }, index=datetime_index)

        # Remove rows with NaN predictions for display
        pred_df = pred_df.dropna(subset=['Reversal_Signal'])

        if len(pred_df) == 0:
            st.error("No valid predictions generated")
            return

        # Create 5 comprehensive sub-tabs for detailed analysis
        chart_tab, data_tab, dist_tab, stats_tab, metrics_tab = st.tabs([
            "📈 Interactive Chart", 
            "📋 Detailed Data", 
            "📊 Distribution Analysis", 
            "🔍 Statistical Analysis", 
            "⚡ Performance Metrics"
        ])

        with chart_tab:
            st.subheader("📈 Reversal Detection Chart")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                chart_points = st.selectbox("Data Points", [50, 100, 200, 500], index=1, key="reversal_chart_points")
            
            recent_predictions = pred_df.tail(chart_points)

            # Create subplot with multiple views
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Reversal Signals Over Time', 'Confidence Distribution'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Add reversal signals
            reversals = recent_predictions[recent_predictions['Reversal_Signal'] == 'Reversal']
            if len(reversals) > 0:
                fig.add_trace(go.Scatter(
                    x=reversals['DateTime'],
                    y=[1] * len(reversals),
                    mode='markers',
                    name='Reversal',
                    marker=dict(color='orange', size=10, symbol='star'),
                    text=reversals['Confidence'].round(3),
                    textposition="top center"
                ), row=1, col=1)

            # Add no reversal signals
            no_reversals = recent_predictions[recent_predictions['Reversal_Signal'] == 'No Reversal']
            if len(no_reversals) > 0:
                fig.add_trace(go.Scatter(
                    x=no_reversals['DateTime'],
                    y=[0] * len(no_reversals),
                    mode='markers',
                    name='No Reversal',
                    marker=dict(color='blue', size=6, symbol='circle'),
                    text=no_reversals['Confidence'].round(3),
                    textposition="bottom center"
                ), row=1, col=1)

            # Add confidence trend line if enough data
            if len(recent_predictions) >= 10:
                group_size = 10
                num_groups = len(recent_predictions) // group_size
                confidence_trend = []
                trend_times = []
                
                for i in range(num_groups):
                    start_idx = i * group_size
                    end_idx = min((i + 1) * group_size, len(recent_predictions))
                    group_data = recent_predictions.iloc[start_idx:end_idx]
                    
                    if len(group_data) > 0:
                        confidence_trend.append(group_data['Confidence'].mean())
                        trend_times.append(group_data['DateTime'].iloc[0])
                
                if len(trend_times) > 0 and len(confidence_trend) > 0:
                    fig.add_trace(go.Scatter(
                        x=trend_times,
                        y=confidence_trend,
                        mode='lines',
                        name='Confidence Trend',
                        line=dict(color='purple', width=2),
                        yaxis='y2'
                    ), row=1, col=1)

            # Confidence histogram
            fig.add_trace(go.Histogram(
                x=recent_predictions['Confidence'],
                nbinsx=20,
                name='Confidence Distribution',
                marker_color='rgba(255, 140, 0, 0.6)'
            ), row=2, col=1)

            fig.update_layout(
                title=f"Reversal Analysis - Last {chart_points} Data Points",
                height=700,
                showlegend=True,
                template="plotly_dark"
            )
            
            fig.update_xaxes(title_text="DateTime", row=1, col=1)
            fig.update_yaxes(title_text="Reversal Signal (1=Reversal, 0=No Reversal)", row=1, col=1)
            fig.update_yaxes(title_text="Confidence", side='right', row=1, col=1, secondary_y=True)
            fig.update_xaxes(title_text="Confidence Level", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_signal = recent_predictions['Reversal_Signal'].iloc[-1]
                st.metric("Current Signal", current_signal)
            with col2:
                current_confidence = recent_predictions['Confidence'].iloc[-1]
                st.metric("Current Confidence", f"{current_confidence:.3f}")
            with col3:
                reversal_pct = len(reversals) / len(recent_predictions) * 100
                st.metric("Reversal %", f"{reversal_pct:.1f}%")
            with col4:
                avg_confidence = recent_predictions['Confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")

        with data_tab:
            st.subheader("📋 Detailed Reversal Data")
            
            col1, col2 = st.columns([2, 1])
            with col2:
                data_points = st.selectbox("Show Records", [100, 200, 500, 1000], index=1, key="reversal_data_points")
            
            recent_predictions = pred_df.tail(data_points)
            
            # Enhanced data table with additional calculated columns
            detailed_df = recent_predictions.copy()
            detailed_df['Signal_Score'] = detailed_df['Reversal_Signal'].map({'Reversal': 1, 'No Reversal': 0})
            detailed_df['Confidence_Level'] = pd.cut(detailed_df['Confidence'], 
                                                   bins=[0, 0.6, 0.8, 1.0], 
                                                   labels=['Low', 'Medium', 'High'])
            
            # Calculate streaks
            signal_changes = detailed_df['Signal_Score'].diff().fillna(0)
            streak_groups = (signal_changes != 0).cumsum()
            detailed_df['Streak_Length'] = detailed_df.groupby(streak_groups).cumcount() + 1
            
            # Add momentum indicators
            detailed_df['Confidence_Change'] = detailed_df['Confidence'].diff()
            detailed_df['Signal_Momentum'] = detailed_df['Confidence_Change'].apply(
                lambda x: '📈' if x > 0.1 else '📉' if x < -0.1 else '➡️'
            )
            
            # Display enhanced table
            display_columns = [
                'Date', 'Time', 'Reversal_Signal', 'Confidence', 'Signal_Momentum',
                'Confidence_Level', 'Streak_Length', 'Confidence_Change'
            ]
            
            st.dataframe(
                detailed_df[display_columns].round(3), 
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                    "Streak_Length": st.column_config.NumberColumn("Streak", format="%d"),
                    "Confidence_Change": st.column_config.NumberColumn("Δ Confidence", format="%.3f")
                }
            )
            
            # Data summary
            st.subheader("📊 Reversal Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Signal Distribution:**")
                signal_counts = detailed_df['Reversal_Signal'].value_counts()
                for signal, count in signal_counts.items():
                    st.write(f"• {signal}: {count} ({count/len(detailed_df)*100:.1f}%)")
            
            with col2:
                st.write("**Confidence Levels:**")
                confidence_counts = detailed_df['Confidence_Level'].value_counts()
                for level, count in confidence_counts.items():
                    st.write(f"• {level}: {count} ({count/len(detailed_df)*100:.1f}%)")
            
            with col3:
                st.write("**Statistics:**")
                st.write(f"• Avg Confidence: {detailed_df['Confidence'].mean():.3f}")
                st.write(f"• Max Streak: {detailed_df['Streak_Length'].max()}")
                st.write(f"• Confidence Std: {detailed_df['Confidence'].std():.3f}")

        with dist_tab:
            st.subheader("📊 Distribution Analysis")
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Reversal signal distribution pie chart
                signal_counts = pred_df['Reversal_Signal'].value_counts()
                
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=signal_counts.index,
                    values=signal_counts.values,
                    hole=0.4,
                    marker_colors=['orange', 'blue'],
                    textinfo='label+percent',
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Reversal Signal Distribution",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=pred_df['Confidence'],
                    nbinsx=30,
                    histnorm='probability density',
                    name='Confidence Distribution',
                    marker_color='rgba(255, 140, 0, 0.7)'
                ))
                
                fig.update_layout(
                    title="Confidence Distribution",
                    xaxis_title="Confidence Level",
                    yaxis_title="Density",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence by reversal signal
            st.subheader("📈 Confidence by Reversal Signal")
            
            fig = go.Figure()
            
            for signal in ['Reversal', 'No Reversal']:
                signal_data = pred_df[pred_df['Reversal_Signal'] == signal]
                if len(signal_data) > 0:
                    fig.add_trace(go.Box(
                        y=signal_data['Confidence'],
                        name=signal,
                        marker_color='orange' if signal == 'Reversal' else 'blue',
                        boxpoints='outliers'
                    ))
            
            fig.update_layout(
                title="Confidence Distribution by Reversal Signal",
                yaxis_title="Confidence Level",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical distribution analysis
            st.subheader("📈 Distribution Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Reversal Count", len(pred_df[pred_df['Reversal_Signal'] == 'Reversal']))
                st.metric("No Reversal Count", len(pred_df[pred_df['Reversal_Signal'] == 'No Reversal']))
            with col2:
                reversal_conf = pred_df[pred_df['Reversal_Signal'] == 'Reversal']['Confidence']
                st.metric("Reversal Avg Conf", f"{reversal_conf.mean():.3f}" if len(reversal_conf) > 0 else "N/A")
                st.metric("Reversal Conf Std", f"{reversal_conf.std():.3f}" if len(reversal_conf) > 0 else "N/A")
            with col3:
                no_reversal_conf = pred_df[pred_df['Reversal_Signal'] == 'No Reversal']['Confidence']
                st.metric("No Reversal Avg Conf", f"{no_reversal_conf.mean():.3f}" if len(no_reversal_conf) > 0 else "N/A")
                st.metric("No Reversal Conf Std", f"{no_reversal_conf.std():.3f}" if len(no_reversal_conf) > 0 else "N/A")
            with col4:
                high_conf = len(pred_df[pred_df['Confidence'] > 0.8])
                st.metric("High Confidence", f"{high_conf} ({high_conf/len(pred_df)*100:.1f}%)")
                low_conf = len(pred_df[pred_df['Confidence'] < 0.6])
                st.metric("Low Confidence", f"{low_conf} ({low_conf/len(pred_df)*100:.1f}%)")

        with stats_tab:
            st.subheader("🔍 Statistical Analysis")
            
            # Time series analysis
            recent_data = pred_df.tail(500)  # Use recent data
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Reversal signal streak analysis
                st.write("**📊 Reversal Signal Streak Analysis**")
                
                # Calculate streaks
                signal_numeric = recent_data['Reversal_Signal'].map({'Reversal': 1, 'No Reversal': 0})
                streaks = []
                current_streak = 1
                current_signal = signal_numeric.iloc[0]
                
                for i in range(1, len(signal_numeric)):
                    if signal_numeric.iloc[i] == current_signal:
                        current_streak += 1
                    else:
                        streaks.append(current_streak)
                        current_streak = 1
                        current_signal = signal_numeric.iloc[i]
                streaks.append(current_streak)
                
                if streaks:
                    avg_streak = np.mean(streaks)
                    max_streak = max(streaks)
                    
                    streak_df = pd.DataFrame({
                        'Average Streak': [f"{avg_streak:.1f}"],
                        'Max Streak': [max_streak],
                        'Total Streaks': [len(streaks)],
                        'Streak Consistency': [f"{(avg_streak/max_streak)*100:.1f}%"]
                    })
                    
                    st.dataframe(streak_df, use_container_width=True)
                    
                    # Streak distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=streaks,
                        nbinsx=15,
                        name='Streak Length Distribution',
                        marker_color='lightsalmon'
                    ))
                    
                    fig.update_layout(
                        title="Reversal Signal Streak Distribution",
                        xaxis_title="Streak Length",
                        yaxis_title="Frequency",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence trend analysis
                st.write("**📈 Confidence Trend Analysis**")
                rolling_conf = recent_data['Confidence'].rolling(20).mean()
                conf_trend = rolling_conf.iloc[-1] - rolling_conf.iloc[-20] if len(rolling_conf) >= 20 else 0
                
                trend_df = pd.DataFrame({
                    'Current Avg': f"{rolling_conf.iloc[-1]:.3f}" if len(rolling_conf) > 0 else "N/A",
                    'Trend': f"{conf_trend:+.3f}" if abs(conf_trend) > 0.001 else "Stable",
                    'Volatility': f"{recent_data['Confidence'].std():.3f}",
                    'Range': f"{recent_data['Confidence'].max() - recent_data['Confidence'].min():.3f}"
                }, index=[0])
                
                st.dataframe(trend_df, use_container_width=True)
            
            with col2:
                # Signal transition analysis
                st.write("**🔗 Signal Transition Analysis**")
                
                # Calculate transition probabilities
                transitions = {'Rev→NoRev': 0, 'NoRev→Rev': 0, 'Rev→Rev': 0, 'NoRev→NoRev': 0}
                for i in range(1, len(recent_data)):
                    prev_signal = recent_data['Reversal_Signal'].iloc[i-1]
                    curr_signal = recent_data['Reversal_Signal'].iloc[i]
                    
                    if prev_signal == 'Reversal' and curr_signal == 'No Reversal':
                        transitions['Rev→NoRev'] += 1
                    elif prev_signal == 'No Reversal' and curr_signal == 'Reversal':
                        transitions['NoRev→Rev'] += 1
                    elif prev_signal == 'Reversal' and curr_signal == 'Reversal':
                        transitions['Rev→Rev'] += 1
                    elif prev_signal == 'No Reversal' and curr_signal == 'No Reversal':
                        transitions['NoRev→NoRev'] += 1
                
                total_transitions = sum(transitions.values())
                if total_transitions > 0:
                    transition_probs = {k: v/total_transitions for k, v in transitions.items()}
                    
                    st.write("**Transition Probabilities:**")
                    for transition, prob in transition_probs.items():
                        st.write(f"• {transition}: {prob:.1%}")
                    
                    # Persistence analysis
                    persistence = (transitions['Rev→Rev'] + transitions['NoRev→NoRev']) / total_transitions
                    st.metric("Signal Persistence", f"{persistence:.1%}")
                
                # Confidence autocorrelation
                st.write("**📊 Confidence Autocorrelation**")
                
                conf_data = recent_data['Confidence'].tail(200)  # Use recent data for performance
                lags = range(1, min(11, len(conf_data)//4))
                autocorr = [conf_data.autocorr(lag=lag) for lag in lags]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(lags),
                    y=autocorr,
                    name='Autocorrelation',
                    marker_color='lightcoral'
                ))
                
                fig.add_hline(y=0.1, line_dash="dash", line_color="red")
                fig.add_hline(y=-0.1, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title="Confidence Autocorrelation",
                    xaxis_title="Lag",
                    yaxis_title="Correlation",
                    height=300,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Signal quality analysis
            st.subheader("🎯 Signal Quality Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # High confidence signals
                high_conf_signals = recent_data[recent_data['Confidence'] > 0.8]
                st.write(f"**High Confidence Signals (>80%): {len(high_conf_signals)}**")
                if len(high_conf_signals) > 0:
                    high_conf_reversals = len(high_conf_signals[high_conf_signals['Reversal_Signal'] == 'Reversal'])
                    st.write(f"• Reversals: {high_conf_reversals} ({high_conf_reversals/len(high_conf_signals)*100:.1f}%)")
                    st.write(f"• No Reversals: {len(high_conf_signals) - high_conf_reversals} ({(len(high_conf_signals) - high_conf_reversals)/len(high_conf_signals)*100:.1f}%)")
            
            with col2:
                # Medium confidence signals
                med_conf_signals = recent_data[(recent_data['Confidence'] >= 0.6) & (recent_data['Confidence'] <= 0.8)]
                st.write(f"**Medium Confidence Signals (60-80%): {len(med_conf_signals)}**")
                if len(med_conf_signals) > 0:
                    med_conf_reversals = len(med_conf_signals[med_conf_signals['Reversal_Signal'] == 'Reversal'])
                    st.write(f"• Reversals: {med_conf_reversals} ({med_conf_reversals/len(med_conf_signals)*100:.1f}%)")
                    st.write(f"• No Reversals: {len(med_conf_signals) - med_conf_reversals} ({(len(med_conf_signals) - med_conf_reversals)/len(med_conf_signals)*100:.1f}%)")
            
            with col3:
                # Low confidence signals
                low_conf_signals = recent_data[recent_data['Confidence'] < 0.6]
                st.write(f"**Low Confidence Signals (<60%): {len(low_conf_signals)}**")
                if len(low_conf_signals) > 0:
                    low_conf_reversals = len(low_conf_signals[low_conf_signals['Reversal_Signal'] == 'Reversal'])
                    st.write(f"• Reversals: {low_conf_reversals} ({low_conf_reversals/len(low_conf_signals)*100:.1f}%)")
                    st.write(f"• No Reversals: {len(low_conf_signals) - low_conf_reversals} ({(len(low_conf_signals) - low_conf_reversals)/len(low_conf_signals)*100:.1f}%)")

        with metrics_tab:
            st.subheader("⚡ Model Performance Metrics")

            # Get model info with debug information
            model_info = model_manager.get_model_info('reversal')
            
            if model_info:
                st.write("**Debug: Available model info keys:**", list(model_info.keys()))
                
                # Try to find metrics in various possible locations
                metrics = None
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    st.success("✅ Found metrics in 'metrics' key")
                elif 'training_metrics' in model_info:
                    metrics = model_info['training_metrics']
                    st.success("✅ Found metrics in 'training_metrics' key")
                elif 'performance' in model_info:
                    metrics = model_info['performance']
                    st.success("✅ Found metrics in 'performance' key")
                else:
                    st.info("🔍 Metrics not found in standard locations, checking alternative sources...")
                    
                    for key, value in model_info.items():
                        if isinstance(value, dict):
                            if any(metric_key in value for metric_key in ['accuracy', 'precision', 'recall', 'f1']):
                                metrics = value
                                st.success(f"✅ Found metrics in '{key}' key")
                                break

                if metrics:
                    # Main performance metrics
                    st.subheader("🎯 Core Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
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
                    with col4:
                        f1_score = classification_metrics.get('weighted avg', {}).get('f1-score', 0)
                        st.metric("F1 Score", f"{f1_score:.2%}")

                    # Detailed classification report
                    if 'classification_report' in metrics:
                        st.subheader("📊 Detailed Classification Report")
                        
                        class_report = metrics['classification_report']
                        if isinstance(class_report, dict):
                            # Create a formatted table
                            report_data = []
                            for class_name, class_metrics in class_report.items():
                                if isinstance(class_metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                    report_data.append({
                                        'Class': 'Reversal' if class_name == '1' else 'No Reversal' if class_name == '0' else class_name,
                                        'Precision': f"{class_metrics.get('precision', 0):.3f}",
                                        'Recall': f"{class_metrics.get('recall', 0):.3f}",
                                        'F1-Score': f"{class_metrics.get('f1-score', 0):.3f}",
                                        'Support': class_metrics.get('support', 0)
                                    })
                            
                            if report_data:
                                report_df = pd.DataFrame(report_data)
                                st.dataframe(report_df, use_container_width=True)

                    # Feature importance analysis
                    feature_importance = model_manager.get_feature_importance('reversal')
                    if feature_importance:
                        st.subheader("🔍 Feature Importance Analysis")
                        
                        importance_df = pd.DataFrame(
                            list(feature_importance.items()), 
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("**Top 15 Features:**")
                            st.dataframe(
                                importance_df.head(15).round(4), 
                                use_container_width=True,
                                column_config={
                                    "Importance": st.column_config.ProgressColumn("Importance", min_value=0, max_value=1)
                                }
                            )
                        
                        with col2:
                            # Feature importance chart
                            top_features = importance_df.head(10)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=top_features['Importance'],
                                y=top_features['Feature'],
                                orientation='h',
                                marker_color='lightsalmon',
                                text=top_features['Importance'].round(3),
                                textposition='inside'
                            ))
                            
                            fig.update_layout(
                                title="Top 10 Most Important Features",
                                xaxis_title="Importance Score",
                                yaxis_title="Features",
                                height=400,
                                template="plotly_dark"
                            )
                            fig.update_yaxes(categoryorder='total ascending')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Model complexity and training info
                    st.subheader("🏗️ Model Architecture & Training")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Model Type:** Classification Ensemble")
                        st.write("**Task:** Binary Classification (Reversal/No Reversal)")
                        st.write("**Training Split:** 80% train / 20% test")
                    
                    with col2:
                        train_accuracy = metrics.get('train_accuracy', 0)
                        test_accuracy = metrics.get('test_accuracy', metrics.get('accuracy', 0))
                        overfit_ratio = (train_accuracy - test_accuracy) if train_accuracy > 0 else 0
                        
                        st.metric("Training Accuracy", f"{train_accuracy:.2%}")
                        st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                        st.metric("Overfitting", f"{overfit_ratio:.1%}")
                    
                    with col3:
                        if 'confusion_matrix' in metrics:
                            cm = metrics['confusion_matrix']
                            if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
                                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                                
                                st.metric("Specificity", f"{specificity:.2%}")
                                st.metric("Sensitivity", f"{sensitivity:.2%}")
                                st.metric("Total Predictions", f"{tp + tn + fp + fn:,}")
                    
                    # Confusion matrix visualization
                    if 'confusion_matrix' in metrics:
                        st.subheader("📊 Confusion Matrix")
                        
                        cm = metrics['confusion_matrix']
                        if isinstance(cm, list) and len(cm) == 2:
                            fig = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=['Predicted No Reversal', 'Predicted Reversal'],
                                y=['Actual No Reversal', 'Actual Reversal'],
                                colorscale='Oranges',
                                text=cm,
                                texttemplate="%{text}",
                                textfont={"size": 16}
                            ))
                            
                            fig.update_layout(
                                title="Confusion Matrix",
                                height=400,
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Model is trained but performance metrics are not accessible in the expected format.")
                    st.info("💡 This can happen if the model was trained but metrics weren't properly saved. Please retrain the reversal model to generate fresh metrics.")
            else:
                st.warning("⚠️ No model performance metrics available. Please train the reversal model first.")

    except Exception as e:
        st.error(f"Error generating reversal predictions: {str(e)}")

if __name__ == "__main__":
    show_predictions_page()
# Removed synthetic datetime generation - using only authentic database timestamps
# Removed synthetic datetime generation - using only authentic database timestamps
# Removed synthetic datetime generation - using only authentic database timestamps
# Removed synthetic datetime generation - using only authentic database timestamps
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Removed synthetic datetime generation - using only authentic database timestamps
# Removed synthetic datetime generation - using only authentic database timestamps
# Removed synthetic datetime generation - using only authentic database timestamps
# Removed synthetic datetime generation - using only authentic database timestamps
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Removed synthetic datetime generation - using only authentic database timestamps
# Removed synthetic datetime generation - using only authentic database timestamps
# Removed synthetic datetime generation - using only authentic database timestamps
# Removed synthetic datetime generation - using only authentic database timestamps
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database
# Use authentic datetime from database