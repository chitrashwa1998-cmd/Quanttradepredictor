
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
    <h1 style="margin:0;">📊 NIFTY 50 REALTIME ANALYSIS</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Live Market Data with ML Predictions
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

# NIFTY 50 Configuration
st.markdown("### 📊 NIFTY 50 - 5 Minute Data")
selected_symbol = "^NSEI"
interval = "5m"

col1, col2 = st.columns(2)

with col1:
    period = st.selectbox(
        "Time Period",
        ["1d", "5d", "1mo"],
        index=1
    )

with col2:
    auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)

if auto_refresh:
    time.sleep(30)
    st.rerun()

# Fetch and display data
try:
    with st.spinner("Fetching NIFTY 50 data..."):
        df = market_data.fetch_realtime_data(selected_symbol, period=period, interval=interval)

    if df is not None and not df.empty:
        st.success(f"✅ Loaded {len(df)} data points for NIFTY 50")

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
            name="NIFTY 50"
        )])

        current_time_display = datetime.now().strftime('%H:%M:%S IST')

        fig.update_layout(
            title=f"NIFTY 50 - 5m Candlestick Chart (Last Updated: {current_time_display})",
            xaxis_title="Time",
            yaxis_title="Price (₹)",
            height=500,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Technical Analysis
        st.subheader("📈 Technical Analysis")

        try:
            # Calculate technical indicators
            tech_df = TechnicalIndicators.calculate_all_indicators(df)

            col1, col2, col3 = st.columns(3)

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

            with col3:
                if 'bb_position' in tech_df.columns:
                    bb_pos = tech_df['bb_position'].iloc[-1]
                    if not pd.isna(bb_pos):
                        st.metric("Bollinger Position", f"{bb_pos:.2f}")
                        bb_signal = "Overbought" if bb_pos > 0.8 else "Oversold" if bb_pos < 0.2 else "Normal"
                        st.info(f"BB Signal: {bb_signal}")

        except Exception as e:
            st.warning(f"⚠️ Could not calculate technical indicators: {str(e)}")

        # ML Predictions Section
        st.subheader("🤖 Machine Learning Predictions")

        try:
            # Initialize model trainer
            model_trainer = QuantTradingModels()
            
            # Prepare data for prediction
            if len(tech_df) >= 30:
                recent_data = tech_df.tail(30).copy()
                
                # Try to generate predictions
                with st.spinner("Generating ML predictions..."):
                    predictions = model_trainer.generate_predictions(recent_data, "nifty_5m_model")
                
                if predictions is not None and not predictions.empty:
                    st.success("✅ ML predictions generated successfully!")
                    
                    # Display recent predictions
                    st.markdown("#### Recent Predictions")
                    display_predictions = predictions.tail(10).copy()
                    
                    # Check if confidence columns exist
                    if 'dir_conf' in display_predictions.columns and 'profit_conf' in display_predictions.columns:
                        # Show predictions with confidence
                        st.dataframe(
                            display_predictions[['predicted_direction', 'dir_conf', 'profit_conf', 'predicted_return']].round(4),
                            use_container_width=True
                        )
                        
                        # Latest prediction analysis
                        latest = display_predictions.iloc[-1]
                        dir_conf = latest.get('dir_conf', 0)
                        profit_conf = latest.get('profit_conf', 0)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Direction Confidence", f"{dir_conf:.3f}")
                            
                        with col2:
                            st.metric("Profit Confidence", f"{profit_conf:.3f}")
                            
                        with col3:
                            predicted_direction = latest.get('predicted_direction', 0)
                            direction_text = "📈 BULLISH" if predicted_direction > 0 else "📉 BEARISH"
                            st.metric("Predicted Direction", direction_text)
                            
                        with col4:
                            predicted_return = latest.get('predicted_return', 0)
                            st.metric("Expected Return", f"{predicted_return:.4f}")
                        
                        # Trading Signal
                        st.markdown("#### Trading Signal")
                        if dir_conf > 0.7 and profit_conf > 0.7:
                            st.success("🟢 STRONG SIGNAL - High confidence for both direction and profit")
                        elif dir_conf > 0.6 and profit_conf > 0.6:
                            st.info("🔵 MODERATE SIGNAL - Good confidence levels")
                        elif dir_conf > 0.5 or profit_conf > 0.5:
                            st.warning("🟡 WEAK SIGNAL - Low confidence, exercise caution")
                        else:
                            st.error("🔴 NO SIGNAL - Very low confidence, avoid trading")
                            
                    else:
                        # Show basic predictions
                        st.dataframe(display_predictions.tail(5))
                        
                else:
                    st.warning("⚠️ Cannot generate predictions - insufficient technical indicator data")
                    
            else:
                st.warning("⚠️ Need at least 30 data points for ML predictions")
                
        except Exception as e:
            st.error(f"❌ Error in ML prediction system: {str(e)}")
            st.info("Tip: Train models first in the Model Training page")

        # Data Management
        st.subheader("📁 Data Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = df.to_csv()
            st.download_button(
                "📥 Download CSV",
                csv,
                f"nifty50_{interval}_data.csv",
                "text/csv"
            )

        with col2:
            if st.button("💾 Save to Database"):
                try:
                    db = DatabaseAdapter()
                    success = db.save_dataset(df, f"nifty50_{interval}_realtime")
                    if success:
                        st.success("✅ Data saved to database")
                    else:
                        st.error("❌ Failed to save data")
                except Exception as e:
                    st.error(f"❌ Error saving data: {str(e)}")

        with col3:
            if st.button("🔄 Refresh Data"):
                st.rerun()

    else:
        st.error("❌ Could not fetch NIFTY 50 data")
        st.info("This might be due to:")
        st.write("• Market is closed")
        st.write("• Network connectivity issues")
        st.write("• API rate limits")

except Exception as e:
    st.error(f"❌ Error loading realtime data: {str(e)}")
    st.info("Please try refreshing the page.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🔄 Auto-refresh available | 📊 Real-time NIFTY 50 analysis | 🤖 ML-powered predictions</p>
</div>
""", unsafe_allow_html=True)
