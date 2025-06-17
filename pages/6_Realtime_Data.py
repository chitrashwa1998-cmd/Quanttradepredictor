import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.realtime_data import IndianMarketData
from utils.database_adapter import DatabaseAdapter
from features.technical_indicators import TechnicalIndicators
from models.xgboost_models import QuantTradingModels
import time
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="Realtime Data", page_icon="📊", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'market_data' not in st.session_state:
    st.session_state.market_data = IndianMarketData()

if 'db' not in st.session_state:
    st.session_state.db = DatabaseAdapter()

market_data = st.session_state.market_data

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">📊 REALTIME MARKET DATA</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Live Market Analysis & Predictions
    </p>
</div>
""", unsafe_allow_html=True)

# Market Status
col1, col2, col3 = st.columns(3)

with col1:
    market_open = market_data.is_market_open()
    status_color = "#00ff41" if market_open else "#ff0080"
    status_text = "🟢 OPEN" if market_open else "🔴 CLOSED"

    st.markdown(f"""
    <div class="metric-container">
        <h3 style="color: #00ffff;">Market Status</h3>
        <h2 style="color: {status_color};">{status_text}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
    st.markdown(f"""
    <div class="metric-container">
        <h3 style="color: #00ffff;">IST Time</h3>
        <h2 style="color: #ffd700;">{current_time.strftime('%H:%M:%S')}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <h3 style="color: #00ffff;">Trading Day</h3>
        <h2 style="color: #ffd700;">{current_time.strftime('%d %b %Y')}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Stock Selection
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    # Get available symbols
    indian_stocks = market_data.get_indian_stock_symbols()
    nifty_indices = market_data.get_nifty_symbols()

    all_symbols = {**indian_stocks, **nifty_indices}

    selected_name = st.selectbox(
        "Select Stock/Index",
        list(all_symbols.keys()),
        index=0
    )

    selected_symbol = all_symbols[selected_name]

with col2:
    interval = st.selectbox(
        "Data Interval",
        ["5m", "15m", "30m", "1h"],
        index=0
    )

with col3:
    period = st.selectbox(
        "Time Period",
        ["1d", "5d", "1mo"],
        index=1
    )

# Auto-refresh toggle
auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)

if auto_refresh:
    refresh_triggered = st.empty()
    time.sleep(30)
    st.rerun()

# Fetch and display data
try:
    with st.spinner(f"Fetching data for {selected_name}..."):
        df = market_data.fetch_realtime_data(selected_symbol, period=period, interval=interval)

    if df is not None and not df.empty:
        st.success(f"✅ Loaded {len(df)} data points for {selected_name}")

        # Current Price Info
        current_price = df['Close'].iloc[-1]
        previous_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close * 100) if previous_close != 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Price",
                f"₹{current_price:.2f}",
                f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )

        with col2:
            st.metric("High", f"₹{df['High'].iloc[-1]:.2f}")

        with col3:
            st.metric("Low", f"₹{df['Low'].iloc[-1]:.2f}")

        with col4:
            st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}" if 'Volume' in df.columns else "N/A")

        # Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        )])

        current_time_display = datetime.now().strftime('%H:%M:%S IST')

        fig.update_layout(
            title=f"{selected_name} - {interval} Candlestick Chart (Last Updated: {current_time_display})",
            xaxis_title="Time",
            yaxis_title="Price (₹)",
            height=500,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Technical Analysis
        st.subheader("📈 Technical Analysis")

        try:
            # Calculate technical indicators
            tech_df = TechnicalIndicators.calculate_all_indicators(df)

            col1, col2 = st.columns(2)

            with col1:
                if 'rsi' in tech_df.columns:
                    current_rsi = tech_df['rsi'].iloc[-1]
                    if not pd.isna(current_rsi):
                        st.metric("RSI", f"{current_rsi:.2f}")

                        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                        st.info(f"RSI Signal: {rsi_signal}")

            with col2:
                if 'macd_histogram' in tech_df.columns:
                    current_macd = tech_df['macd_histogram'].iloc[-1]
                    if not pd.isna(current_macd):
                        st.metric("MACD Histogram", f"{current_macd:.4f}")

                        macd_signal = "Bullish" if current_macd > 0 else "Bearish"
                        st.info(f"MACD Signal: {macd_signal}")

        except Exception as e:
            st.warning(f"⚠️ Could not calculate technical indicators: {str(e)}")

        # Today's Predictions
        st.subheader("🎯 Today's Predictions")

        try:
            # Try to generate predictions using available models
            from utils.database_adapter import DatabaseAdapter

            db = DatabaseAdapter()
            trained_models = db.load_trained_models()

            if trained_models:
                model_names = list(trained_models.keys())
                selected_model_name = st.selectbox("Select Model for Predictions", model_names)

                if selected_model_name in trained_models:
                    model_info = trained_models[selected_model_name]

                    # Prepare features for prediction
                    if len(tech_df) >= 30:  # Need sufficient data for indicators
                        try:
                            # Get the last 30 data points for prediction
                            recent_data = tech_df.tail(30).copy()

                            # Initialize model trainer
                            model_trainer = QuantTradingModels()

                            # Generate predictions
                            predictions = model_trainer.generate_predictions(
                                recent_data, 
                                selected_model_name
                            )

                            if predictions is not None and not predictions.empty:
                                # Show recent predictions
                                st.success("✅ Predictions generated successfully!")

                                # Display predictions table
                                display_predictions = predictions.tail(10).copy()

                                if 'dir_conf' in display_predictions.columns and 'profit_conf' in display_predictions.columns:
                                    st.markdown("### Recent Predictions")
                                    st.dataframe(
                                        display_predictions[['predicted_direction', 'dir_conf', 'profit_conf', 'predicted_return']].round(4),
                                        use_container_width=True
                                    )

                                    # Trading signals
                                    latest_prediction = display_predictions.iloc[-1]
                                    dir_conf = latest_prediction.get('dir_conf', 0)
                                    profit_conf = latest_prediction.get('profit_conf', 0)

                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric("Direction Confidence", f"{dir_conf:.2f}")

                                    with col2:
                                        st.metric("Profit Confidence", f"{profit_conf:.2f}")

                                    with col3:
                                        # Trading recommendation
                                        if dir_conf > 0.7 and profit_conf > 0.7:
                                            st.success("🟢 STRONG BUY SIGNAL")
                                        elif dir_conf > 0.6 and profit_conf > 0.6:
                                            st.info("🔵 MODERATE BUY SIGNAL")
                                        else:
                                            st.warning("🟡 HOLD/WAIT")

                                else:
                                    st.dataframe(display_predictions.tail(10))

                            else:
                                st.warning("⚠️ Could not generate predictions for current data")

                        except Exception as e:
                            st.error(f"❌ Error generating predictions: {str(e)}")

                    else:
                        st.warning("⚠️ Insufficient data for predictions. Need at least 30 data points.")

            else:
                st.info("ℹ️ No trained models available. Please train models first in the Model Training page.")

        except Exception as e:
            st.warning(f"⚠️ Prediction system not available: {str(e)}")

        # Data Export
        st.subheader("📁 Data Export")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = df.to_csv()
            st.download_button(
                "Download CSV",
                csv,
                f"{selected_name}_{interval}_data.csv",
                "text/csv"
            )

        with col2:
            if st.button("Save to Database"):
                try:
                    db = DatabaseAdapter()
                    success = db.save_dataset(df, f"{selected_name}_{interval}_realtime")
                    if success:
                        st.success("✅ Data saved to database")
                    else:
                        st.error("❌ Failed to save data")
                except Exception as e:
                    st.error(f"❌ Error saving data: {str(e)}")

        with col3:
            if st.button("Refresh Data"):
                st.rerun()

    else:
        st.error(f"❌ Could not fetch data for {selected_name}")
        st.info("This might be due to:")
        st.write("• Market is closed")
        st.write("• Symbol not available")
        st.write("• Network connectivity issues")

except Exception as e:
    st.error(f"❌ Error loading realtime data: {str(e)}")
    st.info("Please try refreshing the page or selecting a different symbol.")