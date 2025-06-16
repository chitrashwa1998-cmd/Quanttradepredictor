import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.realtime_data import IndianMarketData
from utils.database import TradingDatabase
from features.technical_indicators import TechnicalIndicators
from models.xgboost_models import QuantTradingModels
import time
from datetime import datetime, timedelta

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
    st.session_state.db = TradingDatabase()

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
    current_time = datetime.now().strftime("%H:%M:%S IST")
    st.info(f"🕐 {current_time}")

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
        index=1,
        help="Historical data range"
    )

with col3:
    auto_refresh = st.checkbox(
        "Auto Refresh",
        value=True,
        help="Automatically update Nifty 50 data every 5 minutes"
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

# Data fetching and display
if st.session_state.get('fetch_triggered', False) or auto_refresh:
    
    with st.spinner(f"Fetching real-time data for {selected_name}..."):
        
        # Fetch data
        df = market_data.fetch_realtime_data(selected_symbol, period=period, interval=interval)
        
        if df is not None and not df.empty:
            st.success(f"✅ Fetched {len(df)} data points for {selected_name}")
            
            # Store in session state
            st.session_state.realtime_data = df
            st.session_state.realtime_symbol = selected_symbol
            
            # Calculate technical indicators
            tech_indicators = TechnicalIndicators()
            df_with_indicators = tech_indicators.calculate_all_indicators(df)
            
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
                title=f"{selected_name} - {interval} Candlestick Chart",
                xaxis_title="Time",
                yaxis_title="Price (₹)",
                height=500,
                showlegend=False
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
            
            # Check if models are available
            if 'model_trainer' in st.session_state and st.session_state.model_trainer.models:
                
                try:
                    model_trainer = st.session_state.model_trainer
                    
                    # Prepare features for prediction
                    features_df = model_trainer.prepare_features(df_with_indicators)
                    
                    if not features_df.empty:
                        # Get latest data point for prediction
                        latest_features = features_df.tail(1)
                        
                        st.subheader("Latest Predictions")
                        
                        pred_cols = st.columns(3)
                        
                        # Direction prediction
                        if 'direction' in model_trainer.models:
                            with pred_cols[0]:
                                direction_pred, direction_prob = model_trainer.predict('direction', latest_features)
                                direction_text = "📈 BUY" if direction_pred[0] == 1 else "📉 SELL"
                                confidence = direction_prob[0].max() * 100 if direction_prob[0] is not None else 50
                                
                                st.metric(
                                    "Direction",
                                    direction_text,
                                    delta=f"{confidence:.1f}% confidence"
                                )
                        
                        # Profit probability
                        if 'profit_prob' in model_trainer.models:
                            with pred_cols[1]:
                                profit_pred, profit_prob = model_trainer.predict('profit_prob', latest_features)
                                profit_text = "✅ PROFIT" if profit_pred[0] == 1 else "❌ LOSS"
                                profit_confidence = profit_prob[0].max() * 100 if profit_prob[0] is not None else 50
                                
                                st.metric(
                                    "Profit Probability",
                                    profit_text,
                                    delta=f"{profit_confidence:.1f}% confidence"
                                )
                        
                        # Trading signal
                        if 'trading_signal' in model_trainer.models:
                            with pred_cols[2]:
                                signal_pred, signal_prob = model_trainer.predict('trading_signal', latest_features)
                                signal_text = "🚀 STRONG BUY" if signal_pred[0] == 1 else "⏸️ HOLD/SELL"
                                signal_confidence = signal_prob[0].max() * 100 if signal_prob[0] is not None else 50
                                
                                st.metric(
                                    "Trading Signal",
                                    signal_text,
                                    delta=f"{signal_confidence:.1f}% confidence"
                                )
                        
                        # Prediction history table
                        st.subheader("Recent Predictions")
                        
                        if len(features_df) >= 10:
                            recent_features = features_df.tail(10)
                            
                            prediction_data = []
                            
                            for idx, (timestamp, row) in enumerate(recent_features.iterrows()):
                                row_data = {'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M')}
                                
                                # Get predictions for each available model
                                single_row = row.to_frame().T
                                
                                if 'direction' in model_trainer.models:
                                    dir_pred, dir_prob = model_trainer.predict('direction', single_row)
                                    row_data['Direction'] = "BUY" if dir_pred[0] == 1 else "SELL"
                                    row_data['Dir_Conf'] = f"{dir_prob[0].max() * 100:.1f}%" if dir_prob[0] is not None else "N/A"
                                
                                if 'profit_prob' in model_trainer.models:
                                    profit_pred, profit_prob = model_trainer.predict('profit_prob', single_row)
                                    row_data['Profit'] = "YES" if profit_pred[0] == 1 else "NO"
                                    row_data['Profit_Conf'] = f"{profit_prob[0].max() * 100:.1f}%" if profit_prob[0] is not None else "N/A"
                                
                                prediction_data.append(row_data)
                            
                            pred_df = pd.DataFrame(prediction_data)
                            st.dataframe(pred_df, use_container_width=True)
                    
                    else:
                        st.warning("⚠️ Cannot generate predictions - insufficient technical indicator data")
                
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
                    st.info("💡 Make sure you have trained models available")
            
            else:
                st.warning("⚠️ No trained models available. Please train models first in the Model Training page.")
            
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

# Auto-refresh functionality
if auto_refresh:
    time.sleep(300)  # Refresh every 5 minutes for Nifty 50
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