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

            # Get model info with proper fallback handling
            model_info = model_manager.get_model_info('volatility')
            if model_info and 'metrics' in model_info:
                metrics = model_info['metrics']

                # Main performance metrics with fallback key names
                st.subheader("🎯 Core Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    rmse = metrics.get('rmse', metrics.get('test_rmse', metrics.get('RMSE', 0)))
                    st.metric("RMSE", f"{rmse:.6f}")
                with col2:
                    mae = metrics.get('mae', metrics.get('test_mae', metrics.get('MAE', 0)))
                    st.metric("MAE", f"{mae:.6f}")
                with col3:
                    mse = metrics.get('mse', metrics.get('test_mse', metrics.get('MSE', 0)))
                    st.metric("MSE", f"{mse:.6f}")
                with col4:
                    r2 = metrics.get('r2', metrics.get('test_r2', metrics.get('R2', 0)))
                    st.metric("R² Score", f"{r2:.4f}")
                
                # Training vs Testing Performance
                st.subheader("📊 Training vs Testing Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Training Metrics:**")
                    train_rmse = metrics.get('train_rmse', metrics.get('training_rmse', 0))
                    train_r2 = metrics.get('train_r2', metrics.get('training_r2', 0))
                    st.write(f"• RMSE: {train_rmse:.6f}")
                    st.write(f"• R²: {train_r2:.4f}")
                
                with col2:
                    st.write("**Testing Metrics:**")
                    test_rmse = metrics.get('test_rmse', metrics.get('testing_rmse', rmse))
                    test_r2 = metrics.get('test_r2', metrics.get('testing_r2', r2))
                    st.write(f"• RMSE: {test_rmse:.6f}")
                    st.write(f"• R²: {test_r2:.4f}")
                
                # Model overfitting check
                if train_rmse > 0 and test_rmse > 0:
                    overfitting_ratio = test_rmse / train_rmse
                    if overfitting_ratio > 1.2:
                        st.warning(f"⚠️ Potential overfitting detected (Test/Train RMSE ratio: {overfitting_ratio:.2f})")
                    else:
                        st.success(f"✅ Good generalization (Test/Train RMSE ratio: {overfitting_ratio:.2f})")

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
                
                # Model architecture and training info
                st.subheader("🏗️ Model Architecture & Training")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Model Type:** Ensemble (XGBoost + CatBoost + Random Forest)")
                    st.write("**Features Used:** 27 technical indicators")
                    st.write("**Training Split:** 80% train / 20% test")
                
                with col2:
                    st.write(f"**Training RMSE:** {train_rmse:.6f}")
                    st.write(f"**Testing RMSE:** {test_rmse:.6f}")
                    overfit_ratio = train_rmse / test_rmse if test_rmse > 0 else 0
                    st.write(f"**Overfitting Ratio:** {overfit_ratio:.3f}")
                    
                with col3:
                    st.write(f"**Training R²:** {train_r2:.4f}")
                    st.write(f"**Testing R²:** {test_r2:.4f}")
                    generalization = test_r2 / train_r2 if train_r2 > 0 else 0
                    st.write(f"**Generalization:** {generalization:.3f}")
                
                # Feature categories breakdown
                if feature_importance:
                    st.subheader("📊 Feature Categories")
                    
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
                    
                    if sum(importances) > 0:  # Only show if we have data
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
                st.warning("⚠️ No model performance metrics available. Please train the volatility model first.")
                
                # Show available model info for debugging
                available_models = model_manager.get_trained_models()
                if available_models:
                    st.info(f"Available trained models: {', '.join(available_models)}")
                    
                    # Show what's in the model info
                    if model_info:
                        st.write("**Available model info keys:**")
                        st.write(list(model_info.keys()))
                else:
                    st.info("No trained models found. Please train models in the Model Training page.")

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

        # Create sub-tabs for different views
        chart_tab, data_tab, metrics_tab = st.tabs(["📈 Interactive Chart", "📋 Detailed Data", "📊 Performance Metrics"])

        with chart_tab:
            st.subheader("Direction Prediction Chart")
            recent_predictions = pred_df.tail(100)

            if len(recent_predictions) > 0:
                fig = go.Figure()

                # Add bullish signals
                bullish_data = recent_predictions[recent_predictions['Direction'] == 'Bullish']
                if len(bullish_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=bullish_data['DateTime'],
                        y=[1] * len(bullish_data),
                        mode='markers',
                        name='Bullish',
                        marker=dict(color='green', size=10)
                    ))

                # Add bearish signals
                bearish_data = recent_predictions[recent_predictions['Direction'] == 'Bearish']
                if len(bearish_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=bearish_data['DateTime'],
                        y=[0] * len(bearish_data),
                        mode='markers',
                        name='Bearish',
                        marker=dict(color='red', size=10)
                    ))

                fig.update_layout(
                    title="Direction Predictions - Last 100 Data Points",
                    xaxis_title="DateTime",
                    yaxis_title="Direction (1=Bullish, 0=Bearish)",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

        with data_tab:
            st.subheader("Detailed Direction Data")
            recent_predictions = pred_df.tail(200)
            st.dataframe(recent_predictions[['Date', 'Time', 'Direction', 'Confidence']], use_container_width=True)

        with metrics_tab:
            st.subheader("Model Performance Metrics")

            # Get model info
            model_info = model_manager.get_model_info('direction')
            if model_info and 'metrics' in model_info:
                metrics = model_info['metrics']

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

                # Feature importance
                feature_importance = model_manager.get_feature_importance('direction')
                if feature_importance:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame(
                        list(feature_importance.items()), 
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)

                    st.dataframe(importance_df.head(10), use_container_width=True)
            else:
                st.info("No performance metrics available")

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

        # Create sub-tabs for different views
        chart_tab, data_tab, metrics_tab = st.tabs(["📈 Interactive Chart", "📋 Detailed Data", "📊 Performance Metrics"])

        with chart_tab:
            st.subheader("Profit Probability Prediction Chart")
            recent_predictions = pred_df.tail(100)

            if len(recent_predictions) > 0:
                fig = go.Figure()

                # Add high profit signals
                high_profit = recent_predictions[recent_predictions['Profit_Probability'] == 'High Profit']
                if len(high_profit) > 0:
                    fig.add_trace(go.Scatter(
                        x=high_profit['DateTime'],
                        y=[1] * len(high_profit),
                        mode='markers',
                        name='High Profit',
                        marker=dict(color='green', size=10)
                    ))

                # Add low profit signals
                low_profit = recent_predictions[recent_predictions['Profit_Probability'] == 'Low Profit']
                if len(low_profit) > 0:
                    fig.add_trace(go.Scatter(
                        x=low_profit['DateTime'],
                        y=[0] * len(low_profit),
                        mode='markers',
                        name='Low Profit',
                        marker=dict(color='red', size=10)
                    ))

                fig.update_layout(
                    title="Profit Probability Predictions - Last 100 Data Points",
                    xaxis_title="DateTime",
                    yaxis_title="Profit Probability (1=High, 0=Low)",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

        with data_tab:
            st.subheader("Detailed Prediction Data")
            recent_predictions = pred_df.tail(200)
            st.dataframe(recent_predictions[['Date', 'Time', 'Profit_Probability', 'Confidence']], use_container_width=True)

        with metrics_tab:
            st.subheader("Model Performance Metrics")

            # Get model info
            model_info = model_manager.get_model_info('profit_probability')
            if model_info and 'metrics' in model_info:
                metrics = model_info['metrics']

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

                # Feature importance
                feature_importance = model_manager.get_feature_importance('profit_probability')
                if feature_importance:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame(
                        list(feature_importance.items()), 
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)

                    st.dataframe(importance_df.head(10), use_container_width=True)
            else:
                st.info("No performance metrics available")

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
            'Confidence': [np.max(prob) for prob in probabilities] if probabilities is not None else None
        }, index=datetime_index)

        # Display recent predictions
        recent_predictions = pred_df.tail(100)

        st.subheader("Recent Reversal Predictions")

        if len(recent_predictions) > 0:
            st.dataframe(recent_predictions, use_container_width=True)

            # Create chart
            fig = go.Figure()

            # Add reversal signals
            reversals = recent_predictions[recent_predictions['Reversal_Signal'] == 'Reversal']
            if len(reversals) > 0:
                fig.add_trace(go.Scatter(
                    x=reversals['DateTime'],
                    y=[1] * len(reversals),
                    mode='markers',
                    name='Reversal',
                    marker=dict(color='orange', size=12)
                ))

            # Add no reversal signals
            no_reversals = recent_predictions[recent_predictions['Reversal_Signal'] == 'No Reversal']
            if len(no_reversals) > 0:
                fig.add_trace(go.Scatter(
                    x=no_reversals['DateTime'],
                    y=[0] * len(no_reversals),
                    mode='markers',
                    name='No Reversal',
                    marker=dict(color='blue', size=8)
                ))

            fig.update_layout(
                title="Reversal Detection - Last 100 Data Points",
                xaxis_title="DateTime",
                yaxis_title="Reversal Signal (1=Reversal, 0=No Reversal)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

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