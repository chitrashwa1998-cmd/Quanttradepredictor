import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.realtime_data import IndianMarketData
from utils.database_adapter import DatabaseAdapter
from features.technical_indicators import TechnicalIndicators
from models.xgboost_models import QuantTradingModels
import time
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="Real-time Indian Market", page_icon="📈", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">📈 NIFTY 50 REAL-TIME</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Live Nifty 50 Index Data - 5 Minute Timeframe
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize components
if 'market_data' not in st.session_state:
    st.session_state.market_data = IndianMarketData()

if 'db' not in st.session_state:
    st.session_state.db = DatabaseAdapter()

market_data = st.session_state.market_data

# Market status indicator
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.header("Market Status")

with col2:
    is_open = market_data.is_market_open()
    if is_open:
        st.success("🟢 Market Open")
    else:
        st.error("🔴 Market Closed")

with col3:
    # Get correct Indian Standard Time
    ist_tz = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_tz).strftime("%H:%M:%S IST")
    st.info(f"🕐 {current_time_ist}")

# Nifty 50 Configuration
st.header("Nifty 50 Index Configuration")

col1, col2 = st.columns([2, 1])

with col1:
    st.info("📊 Configured for Nifty 50 Index (^NSEI) - 5 Minute Timeframe")
    selected_symbol = "^NSEI"
    selected_name = "NIFTY 50"

with col2:
    st.metric("Index", "Nifty 50", "NSE")

# Data fetching controls
st.header("Data Configuration")

col1, col2, col3, col4 = st.columns(4)

with col1:
    interval = "5m"  # Fixed to 5 minutes as requested
    st.info("Interval: 5 minutes (Fixed)")

with col2:
    period = st.selectbox(
        "Period:",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=2,  # Default to "1mo" instead of "5d" for better technical indicators
        help="Historical data range - Use 1mo or longer for reliable technical indicators"
    )

with col3:
    auto_refresh = st.checkbox(
        "Live Updates",
        value=True,
        help="Automatically update every 10 seconds during market hours"
    )

with col4:
    if st.button("Fetch Nifty 50 Data", type="primary"):
        st.session_state.fetch_triggered = True

# Current price display
if selected_symbol:
    current_data = market_data.get_current_price(selected_symbol)

    if current_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Price",
                f"₹{current_data['current_price']:.2f}",
                delta=f"{current_data['change']:.2f} ({current_data['change_percent']:.2f}%)"
            )

        with col2:
            st.metric("Previous Close", f"₹{current_data['previous_close']:.2f}")

        with col3:
            st.metric("Volume", f"{current_data['volume']:,}")

        with col4:
            if current_data['market_cap'] > 0:
                market_cap_cr = current_data['market_cap'] / 10000000  # Convert to crores
                st.metric("Market Cap", f"₹{market_cap_cr:.0f} Cr")

# Auto-refresh logic for live updates
refresh_triggered = False

if auto_refresh and is_open:
    # Initialize or check last refresh time using IST
    ist_now = datetime.now(ist_tz)
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = ist_now
        refresh_triggered = True
    else:
        # Check if 10 seconds have passed since last refresh
        time_since_refresh = ist_now - st.session_state.last_refresh_time
        if time_since_refresh.total_seconds() >= 10:
            st.session_state.last_refresh_time = ist_now
            refresh_triggered = True
            st.rerun()  # Immediately trigger rerun

    # Show next refresh countdown
    if not refresh_triggered:
        time_since_refresh = ist_now - st.session_state.last_refresh_time
        next_refresh_in = max(0, 10 - int(time_since_refresh.total_seconds()))

        # Create a placeholder for dynamic countdown
        countdown_placeholder = st.empty()
        countdown_placeholder.info(f"🔄 Next auto-refresh in {next_refresh_in} seconds")

        # Auto-refresh when countdown reaches 0
        if next_refresh_in <= 1:
            time.sleep(1)
            st.rerun()
    else:
        st.info("🔄 Refreshing data...")

elif auto_refresh and not is_open:
    st.info("🕐 Auto-refresh paused - Market is closed")

# Data fetching and display
if st.session_state.get('fetch_triggered', False) or refresh_triggered:

    with st.spinner(f"Fetching real-time data for {selected_name}..."):

        # Fetch data with extended period for technical indicators
        extended_period = "1mo" if period == "5d" else period  # Use longer period for indicator calculation
        df = market_data.fetch_realtime_data(selected_symbol, period=extended_period, interval=interval)

        if df is not None and not df.empty:
            st.success(f"✅ Fetched {len(df)} data points for {selected_name}")

            # Store in session state
            st.session_state.realtime_data = df
            st.session_state.realtime_symbol = selected_symbol

            # Calculate technical indicators with minimum data check
            if len(df) < 50:
                st.warning(f"⚠️ Only {len(df)} data points available. Need at least 50 for reliable technical indicators.")
                st.info("💡 Try selecting '1mo' or '3mo' period for better indicator calculation.")

            tech_indicators = TechnicalIndicators()
            df_with_indicators = tech_indicators.calculate_all_indicators(df)

            # Remove NaN rows that result from indicator calculation
            clean_data_count = len(df_with_indicators.dropna())
            if clean_data_count < 10:
                st.warning(f"⚠️ Only {clean_data_count} clean data points after indicator calculation. Need more historical data.")
            else:
                st.info(f"✅ {clean_data_count} clean data points available for predictions")

            # Display data summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Data Points", len(df))

            with col2:
                date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)

            with col3:
                avg_volume = df['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")

            with col4:
                price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                st.metric("Period Change", f"{price_change:.2f}%")

            # Price chart
            st.header("Price Chart")

            fig = go.Figure()

            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price"
            ))

            fig.update_layout(
                title=f"{selected_name} - {interval} Candlestick Chart (Last Updated: {current_time_ist})",
                xaxis_title="Time",
                yaxis_title="Price (₹)",
                height=500,
                showlegend=False,
                annotations=[
                    dict(
                        x=0.02, y=0.98,
                        xref='paper', yref='paper',
                        text=f"🔴 LIVE" if is_open else "🔴 CLOSED",
                        showarrow=False,
                        font=dict(size=12, color="red" if is_open else "gray"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="red" if is_open else "gray",
                        borderwidth=1
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=True)

            # Volume chart
            st.header("Volume Chart")

            fig_vol = px.bar(
                x=df.index,
                y=df['Volume'],
                title=f"{selected_name} - Volume",
                labels={'x': 'Time', 'y': 'Volume'}
            )

            fig_vol.update_layout(height=300)
            st.plotly_chart(fig_vol, use_container_width=True)

            # ML Predictions Section
            st.header("ML Predictions")

            # Check if models are available in session state (skip database check for speed)
            models_available = False
            model_trainer = None

            if 'model_trainer' in st.session_state and st.session_state.model_trainer and hasattr(st.session_state.model_trainer, 'models') and st.session_state.model_trainer.models:
                models_available = True
                model_trainer = st.session_state.model_trainer
            else:
                # Only try to load models from database if not auto-refreshing to avoid delays
                if not refresh_triggered:
                    try:
                        from models.xgboost_models import QuantTradingModels
                        from utils.database_adapter import DatabaseAdapter

                        db = DatabaseAdapter()
                        trained_models = db.load_trained_models()

                        if trained_models:
                            model_trainer = QuantTradingModels()
                            model_trainer.models = trained_models
                            st.session_state.model_trainer = model_trainer
                            models_available = True
                            st.success("✅ Loaded models from database")

                    except Exception as e:
                        st.warning(f"⚠️ Could not load models from database: {str(e)}")
                else:
                    st.info("⏳ Models loading... (Auto-refresh in progress)")

            if models_available and model_trainer:

                try:
                    # Cache features preparation if data hasn't changed
                    data_hash = hash(str(df_with_indicators.tail(10).values.tobytes()))

                    if not hasattr(st.session_state, 'cached_features_hash') or st.session_state.cached_features_hash != data_hash:
                        # Ensure we have enough data for technical indicators
                        if len(df_with_indicators) < 50:
                            st.warning(f"⚠️ Only {len(df_with_indicators)} data points available. Technical indicators need at least 50 points for reliable calculations.")
                            st.info("💡 Try selecting '1mo' or '3mo' period for better predictions.")
                            
                        # Prepare features for prediction
                        features_df = model_trainer.prepare_features(df_with_indicators)
                        # Remove rows with NaN values
                        clean_features = features_df.dropna()
                        
                        if len(clean_features) < 10:
                            st.error(f"❌ Insufficient clean data after technical indicator calculation: {len(clean_features)} rows. Need at least 10 rows.")
                            st.info("💡 Select a longer time period (1mo or 3mo) to get more historical data for technical indicators.")
                            clean_features = pd.DataFrame()  # Empty dataframe to prevent errors

                        # Cache the results
                        st.session_state.cached_features = clean_features
                        st.session_state.cached_features_hash = data_hash
                    else:
                        # Use cached features
                        clean_features = st.session_state.cached_features

                    if not clean_features.empty and len(clean_features) >= 1:
                        st.success(f"✅ {len(clean_features)} data points ready for prediction")
                        # Get latest data point for prediction
                        latest_features = clean_features.tail(1)

                        st.subheader("Latest Predictions")

                        pred_cols = st.columns(3)

                        available_models = [name for name in ['direction', 'profit_prob', 'trading_signal'] if name in model_trainer.models and model_trainer.models[name] is not None]

                        if not available_models:
                            st.warning("⚠️ No prediction models available")
                        else:
                            # Direction prediction
                            if 'direction' in available_models:
                                with pred_cols[0]:
                                    try:
                                        direction_pred, direction_prob = model_trainer.predict('direction', latest_features)
                                        direction_text = "📈 BUY" if direction_pred[0] == 1 else "📉 SELL"
                                        confidence = direction_prob[0].max() * 100 if direction_prob is not None and len(direction_prob[0]) > 0 else 50

                                        st.metric(
                                            "Direction",
                                            direction_text,
                                            delta=f"{confidence:.1f}% confidence"
                                        )
                                    except Exception as e:
                                        st.error(f"Direction prediction error: {str(e)}")

                            # Profit probability
                            if 'profit_prob' in available_models:
                                with pred_cols[1]:
                                    try:
                                        profit_pred, profit_prob = model_trainer.predict('profit_prob', latest_features)

                                        # Handle different prediction formats
                                        if hasattr(profit_pred[0], '__iter__'):
                                            profit_value = profit_pred[0][0] if len(profit_pred[0]) > 0 else profit_pred[0]
                                        else:
                                            profit_value = profit_pred[0]

                                        if isinstance(profit_value, (int, float)):
                                            if profit_value > 0.5:
                                                profit_text = "✅ PROFIT"
                                            else:
                                                profit_text = "❌ LOSS"
                                            profit_confidence = profit_value * 100
                                        else:
                                            profit_text = "✅ PROFIT" if profit_value == 1 else "❌ LOSS"
                                            profit_confidence = profit_prob[0].max() * 100 if profit_prob is not None and len(profit_prob[0]) > 0 else 50

                                        st.metric(
                                            "Profit Probability",
                                            profit_text,
                                            delta=f"{profit_confidence:.1f}% confidence"
                                        )
                                    except Exception as e:
                                        st.error(f"Profit prediction error: {str(e)}")

                            # Trading signal
                            if 'trading_signal' in available_models:
                                with pred_cols[2]:
                                    try:
                                        signal_pred, signal_prob = model_trainer.predict('trading_signal', latest_features)

                                        # Handle multi-class trading signals (0=sell, 1=hold, 2=buy)
                                        if signal_pred[0] == 2:
                                            signal_text = "🚀 STRONG BUY"
                                        elif signal_pred[0] == 1:
                                            signal_text = "⏸️ HOLD"
                                        else:
                                            signal_text = "📉 SELL"

                                        signal_confidence = signal_prob[0].max() * 100 if signal_prob is not None and len(signal_prob[0]) > 0 else 50

                                        st.metric(
                                            "Trading Signal",
                                            signal_text,
                                            delta=f"{signal_confidence:.1f}% confidence"
                                        )
                                    except Exception as e:
                                        st.error(f"Trading signal error: {str(e)}")

                        # Prediction history table
                        st.subheader("Recent Predictions (Last 30)")

                        if len(clean_features) >= 5:
                            # Simply get the last 30 predictions regardless of time filtering
                            recent_features = clean_features.tail(30)
                            st.info(f"📊 Showing {len(recent_features)} most recent predictions")

                            # Always show auto-refresh status
                            if auto_refresh and is_open:

                            # Always show auto-refresh status
                                if auto_refresh and is_open:
                                    st.info("🔄 Auto-refresh enabled - Data updates every 10 seconds during market hours")
                                elif is_open:
                                    st.info("⏸️ Auto-refresh disabled - Enable for live updates")

                                prediction_data = []

                                for idx, (timestamp, row) in enumerate(recent_features.iterrows()):
                                    # Format timestamp consistently (assume it's already in IST since we converted earlier)
                                    try:
                                        if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                                            ist_timestamp = timestamp.tz_convert('Asia/Kolkata')
                                        else:
                                            # Treat as timezone-naive IST timestamp
                                            ist_timestamp = pd.to_datetime(timestamp)

                                        row_data = {'Timestamp': ist_timestamp.strftime('%Y-%m-%d %H:%M IST')}
                                    except Exception as e:
                                        # Fallback to simple string conversion
                                        row_data = {'Timestamp': str(timestamp)[:19] + ' IST'}

                                    # Get predictions for each available model
                                    single_row = row.to_frame().T

                                    try:
                                        if 'direction' in available_models:
                                            dir_pred, dir_prob = model_trainer.predict('direction', single_row)
                                            row_data['Direction'] = "BUY" if dir_pred[0] == 1 else "SELL"
                                            row_data['Dir_Conf'] = f"{dir_prob[0].max() * 100:.1f}%" if dir_prob is not None and len(dir_prob[0]) > 0 else "N/A"

                                        if 'profit_prob' in available_models:
                                            profit_pred, profit_prob = model_trainer.predict('profit_prob', single_row)
                                            if isinstance(profit_pred[0], (int, float)):
                                                row_data['Profit'] = "YES" if profit_pred[0] > 0.5 else "NO"
                                                row_data['Profit_Conf'] = f"{profit_pred[0] * 100:.1f}%"
                                            else:
                                                row_data['Profit'] = "YES" if profit_pred[0] == 1 else "NO"
                                                row_data['Profit_Conf'] = f"{profit_prob[0].max() * 100:.1f}%" if profit_prob is not None and len(profit_prob[0]) > 0 else "N/A"

                                        if 'trading_signal' in available_models:
                                            signal_pred, signal_prob = model_trainer.predict('trading_signal', single_row)
                                            if signal_pred[0] == 2:
                                                row_data['Signal'] = "STRONG BUY"
                                            elif signal_pred[0] == 1:
                                                row_data['Signal'] = "HOLD"
                                            else:
                                                row_data['Signal'] = "SELL"
                                            row_data['Signal_Conf'] = f"{signal_prob[0].max() * 100:.1f}%" if signal_prob is not None and len(signal_prob[0]) > 0 else "N/A"

                                    except Exception as e:
                                        row_data['Error'] = f"Prediction failed: {str(e)[:30]}"

                                    prediction_data.append(row_data)

                                if prediction_data:
                                    pred_df = pd.DataFrame(prediction_data)
                                    st.dataframe(pred_df, use_container_width=True)
                                else:
                                    st.info("Need at least 5 data points for prediction history")



                            else:
                                st.warning("⚠️ Cannot generate predictions - insufficient technical indicator data")

                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
                    st.info("💡 Make sure you have trained models available")

            else:
                st.warning("⚠️ No trained models available for predictions.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Quick Train Models", type="primary"):
                        if st.session_state.get('data') is not None:
                            with st.spinner("Training essential models for real-time predictions..."):
                                try:
                                    from models.xgboost_models import QuantTradingModels
                                    from features.technical_indicators import TechnicalIndicators

                                    # Use existing data
                                    data = st.session_state.data

                                    # Calculate features
                                    features_data = TechnicalIndicators.calculate_all_indicators(data)
                                    features_data = features_data.dropna()

                                    if len(features_data) >= 100:
                                        # Initialize trainer
                                        model_trainer = QuantTradingModels()

                                        # Prepare data
                                        X = model_trainer.prepare_features(features_data)
                                        targets = model_trainer.create_targets(features_data)

                                        # Train essential models for real-time predictions
                                        essential_models = ['direction', 'trading_signal', 'profit_prob']
                                        trained_count = 0

                                        for model_name in essential_models:
                                            if model_name in targets:
                                                y = targets[model_name]
                                                task_type = 'classification' if model_name in ['direction', 'trading_signal'] else 'regression'

                                                result = model_trainer.train_model(model_name, X, y, task_type)
                                                if result:
                                                    trained_count += 1

                                        if trained_count > 0:
                                            st.session_state.model_trainer = model_trainer
                                            st.success(f"✅ Trained {trained_count} models successfully!")
                                            st.rerun()
                                    else:
                                        st.error("❌ Need at least 100 clean data points for training")

                                except Exception as e:
                                    st.error(f"❌ Training failed: {str(e)}")
                        else:
                            st.error("❌ No data available. Please upload data first.")

                with col2:
                    st.info("💡 Go to **Model Training** page for comprehensive model training with full configuration options.")

            # Historical Predictions Section (separate from ML Predictions)
            if models_available and model_trainer and len(clean_features) >= 10:
                st.header("📊 Historical Predictions Analysis")

                # Date range selector for historical predictions
                col1, col2 = st.columns(2)

                with col1:
                    days_back = st.selectbox(
                        "Select time range:",
                        options=[1, 3, 7, 14, 30],
                        format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}",
                        index=2  # Default to 7 days
                    )

                with col2:
                    end_date = datetime.now(ist_tz).date()
                    start_date = end_date - timedelta(days=days_back)
                    st.info(f"From {start_date} to {end_date}")

                # Filter historical data - ensure timezone-naive comparison
                if hasattr(clean_features.index, 'tz') and clean_features.index.tz is not None:
                    historical_features_naive = clean_features.copy()
                    historical_features_naive.index = historical_features_naive.index.tz_localize(None)
                else:
                    historical_features_naive = clean_features.copy()

                historical_features = historical_features_naive[
                    (historical_features_naive.index.date >= start_date) & 
                    (historical_features_naive.index.date <= end_date)
                ]

                if len(historical_features) > 0:
                    historical_prediction_data = []

                    for idx, (timestamp, row) in enumerate(historical_features.iterrows()):
                        # Format timestamp consistently (assume it's already in IST since we converted earlier)
                        try:
                            if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                                ist_timestamp = timestamp.tz_convert('Asia/Kolkata')
                            else:
                                # Treat as timezone-naive IST timestamp
                                ist_timestamp = pd.to_datetime(timestamp)

                            row_data = {
                                'Date': ist_timestamp.strftime('%Y-%m-%d'),
                                'Time': ist_timestamp.strftime('%H:%M IST'),
                                'Timestamp': ist_timestamp.strftime('%Y-%m-%d %H:%M IST')
                            }
                        except Exception as e:
                            # Fallback to simple string conversion
                            timestamp_str = str(timestamp)[:19]
                            row_data = {
                                'Date': timestamp_str[:10],
                                'Time': timestamp_str[11:16] + ' IST',
                                'Timestamp': timestamp_str + ' IST'
                            }

                        single_row = row.to_frame().T

                        try:
                            if 'direction' in available_models:
                                dir_pred, dir_prob = model_trainer.predict('direction', single_row)
                                row_data['Direction'] = "BUY" if dir_pred[0] == 1 else "SELL"
                                row_data['Dir_Conf'] = f"{dir_prob[0].max() * 100:.1f}%" if dir_prob is not None and len(dir_prob[0]) > 0 else "N/A"

                            if 'profit_prob' in available_models:
                                profit_pred, profit_prob = model_trainer.predict('profit_prob', single_row)
                                if isinstance(profit_pred[0], (int, float)):
                                    row_data['Profit'] = "YES" if profit_pred[0] > 0.5 else "NO"
                                    row_data['Profit_Conf'] = f"{profit_pred[0] * 100:.1f}%"
                                else:
                                    row_data['Profit'] = "YES" if profit_pred[0] == 1 else "NO"
                                    row_data['Profit_Conf'] = f"{profit_prob[0].max() * 100:.1f}%" if profit_prob is not None and len(profit_prob[0]) > 0 else "N/A"

                            if 'trading_signal' in available_models:
                                signal_pred, signal_prob = model_trainer.predict('trading_signal', single_row)
                                if signal_pred[0] == 2:
                                    row_data['Signal'] = "STRONG BUY"
                                elif signal_pred[0] == 1:
                                    row_data['Signal'] = "HOLD"
                                else:
                                    row_data['Signal'] = "SELL"
                                row_data['Signal_Conf'] = f"{signal_prob[0].max() * 100:.1f}%" if signal_prob is not None and len(signal_prob[0]) > 0 else "N/A"

                        except Exception as e:
                            row_data['Error'] = f"Prediction failed: {str(e)[:30]}"

                        historical_prediction_data.append(row_data)

                    if historical_prediction_data:
                        historical_pred_df = pd.DataFrame(historical_prediction_data)

                        # Show all data in a single table
                        st.dataframe(historical_pred_df, use_container_width=True, hide_index=True)

                        # Download option
                        csv_data = historical_pred_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Historical Predictions CSV",
                            csv_data,
                            file_name=f"nifty_predictions_{start_date}_to_{end_date}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning(f"No prediction data available for the selected {days_back} day period.")

            # Real-time Trading Insights
            st.header("Real-time Trading Insights")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Price momentum
                if len(df) >= 5:
                    recent_prices = df['Close'].tail(5)
                    momentum = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100
                    momentum_color = "normal" if abs(momentum) < 0.5 else ("inverse" if momentum > 0 else "off")
                    st.metric(
                        "5-Period Momentum", 
                        f"{momentum:.2f}%",
                        delta=f"{momentum:.2f}%",
                        delta_color=momentum_color
                    )

            with col2:
                # Volume analysis
                if len(df) >= 10:
                    avg_volume = df['Volume'].tail(10).mean()
                    current_volume = df['Volume'].iloc[-1]
                    volume_ratio = (current_volume / avg_volume) * 100
                    st.metric(
                        "Volume vs Avg", 
                        f"{volume_ratio:.0f}%",
                        delta=f"{volume_ratio - 100:.0f}%",
                        delta_color="normal" if volume_ratio > 120 else "off"
                    )

            with col3:
                # Volatility indicator
                if len(df) >= 20:
                    returns = df['Close'].pct_change().tail(20)
                    volatility = returns.std() * 100
                    vol_level = "High" if volatility > 2 else "Medium" if volatility > 1 else "Low"
                    st.metric("20-Period Volatility", f"{volatility:.2f}%", delta=vol_level)

            # Real-time alerts
            if len(df) >= 2:
                st.subheader("Real-time Alerts")
                alerts = []

                # Price breakout alert
                current_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                price_change = ((current_price - prev_price) / prev_price) * 100

                if abs(price_change)> 0.5:
                    direction = "🔴 Sharp Move Up" if price_change > 0 else "🔴 Sharp Move Down"
                    alerts.append(f"{direction}: {price_change:.2f}% in latest candle")

                # Volume spike alert
                if len(df) >= 5:
                    avg_vol = df['Volume'].tail(5).mean()
                    current_vol = df['Volume'].iloc[-1]
                    if current_vol > avg_vol * 1.5:
                        alerts.append(f"🔊 Volume Spike: {((current_vol/avg_vol-1)*100):.0f}% above average")

                # Technical level alerts
                if len(df) >= 20:
                    high_20 = df['High'].tail(20).max()
                    low_20 = df['Low'].tail(20).min()

                    if current_price >= high_20 * 0.999:
                        alerts.append(f"📈 Near 20-period High: ₹{high_20:.2f}")
                    elif current_price <= low_20 * 1.001:
                        alerts.append(f"📉 Near 20-period Low: ₹{low_20:.2f}")

                if alerts:
                    for alert in alerts:
                        st.warning(alert)
                else:
                    st.info("✅ No significant alerts at current levels")

            # Data export options
            st.header("Data Export")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Save to Database"):
                    db = st.session_state.db
                    success = db.save_ohlc_data(df, f"realtime_{selected_symbol}", preserve_full_data=True)
                    if success:
                        st.success("✅ Data saved to database")
                    else:
                        st.error("❌ Failed to save data")

            with col2:
                if st.button("Update Existing Dataset"):
                    if st.session_state.data is not None:
                        # Update main dataset with new data
                        updated_data = market_data.update_dataset_with_realtime(
                            st.session_state.data, 
                            selected_symbol, 
                            interval
                        )
                        st.session_state.data = updated_data
                        st.success("✅ Main dataset updated with new data")
                    else:
                        st.warning("⚠️ No existing dataset to update")

            with col3:
                csv_data = df.to_csv()
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name=f"{selected_symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

        else:
            st.error(f"❌ No data found for {selected_symbol}. Please check the symbol or try again.")

    # Reset trigger
    st.session_state.fetch_triggered = False

# Auto-refresh functionality with JavaScript timer
if auto_refresh and is_open:
    st.markdown("""
    <script>
    // Auto-refresh every 10 seconds during market hours
    if (window.autoRefreshInterval) {
        clearInterval(window.autoRefreshInterval);
    }

    window.autoRefreshInterval = setInterval(function() {
        // Force Streamlit to rerun
        window.parent.postMessage({
            type: 'streamlit:componentReady',
            apiVersion: 1,
        }, '*');

        // Alternative method - simulate button click
        const buttons = window.parent.document.querySelectorAll('[data-testid="stButton"] button');
        const refreshButton = Array.from(buttons).find(btn => btn.textContent.includes('Fetch'));
        if (refreshButton) {
            refreshButton.click();
        }
    }, 10000);
    </script>
    """, unsafe_allow_html=True)

# Also add a manual refresh button that's always visible
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Manual Refresh", key="manual_refresh", type="secondary"):
        st.session_state.fetch_triggered = True
        st.rerun()

# Instructions
with st.expander("📋 Nifty 50 Real-Time Guide", expanded=False):
    st.markdown("""
    ### Nifty 50 Index Trading System

    **Configuration:**
    - **Index**: Nifty 50 (^NSEI) - Top 50 stocks by market cap
    - **Timeframe**: 5-minute candles (Fixed)
    - **Market Hours**: 9:15 AM to 3:30 PM IST, Monday to Friday
    - **Auto Refresh**: Updates every 5 minutes during market hours

    **Features:**
    - Live Nifty 50 price, volume, and market data
    - Technical indicators calculated automatically on 5-min data
    - ML predictions for index movement direction
    - Profit probability analysis for scalping/intraday
    - Trading signal generation with confidence levels

    **ML Predictions Available:**
    - **Direction**: BUY/SELL signal for next 5-minute candle
    - **Profit Probability**: Likelihood of profitable trade
    - **Trading Signal**: Strong BUY/HOLD recommendation

    **Data Export:**
    - Save Nifty 50 data to database for model training
    - Update existing datasets with latest 5-min data
    - Download CSV for external analysis

    **Trading Strategy:**
    - Use ML predictions as guidance for manual trades
    - High confidence signals (>70%) are more reliable
    - Combine with technical indicators for confirmation
    - Always use proper risk management
    """)