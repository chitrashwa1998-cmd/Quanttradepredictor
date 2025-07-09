import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
from utils.upstox_client import UpstoxClient, UpstoxWebSocketClient
from utils.database_adapter import DatabaseAdapter
from features.technical_indicators import TechnicalIndicators
from utils.data_processing import DataProcessor
import os

# Initialize components
trading_db = DatabaseAdapter()

st.set_page_config(page_title="Upstox Data", page_icon="📡", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">📡 UPSTOX LIVE DATA CENTER</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Real-time NIFTY 50 Market Data Integration
    </p>
</div>
""", unsafe_allow_html=True)

# Troubleshooting info
with st.expander("🔧 WebSocket Connection Troubleshooting"):
    st.markdown("""
    **If WebSocket keeps disconnecting:**
    
    1. **Check Token Status** - Most disconnections are due to expired tokens
    2. **Click "Refresh Token"** - This clears the old token and prompts for new authentication
    3. **Login to Upstox** - Complete the authentication process
    4. **Verify token validity** - Should show "Token is valid and working!"
    5. **Start WebSocket** - Connection should now work properly
    
    **Common Issues:**
    - ❌ Token expired → Refresh token and re-authenticate
    - ❌ "WebSocket onclose" → Token issue, refresh needed
    - ❌ No data received → Check market hours and token validity
    - ❌ Authorization errors → Get fresh token from Upstox
    
    **Token Validity:** Upstox tokens expire every 24 hours and need regular refresh.
    """)

# Initialize session state
if 'upstox_client' not in st.session_state:
    st.session_state.upstox_client = None
if 'upstox_authenticated' not in st.session_state:
    st.session_state.upstox_authenticated = False
if 'upstox_access_token' not in st.session_state:
    st.session_state.upstox_access_token = None

# Load token from file if it exists and session state is empty
if not st.session_state.upstox_access_token and os.path.exists('.upstox_token'):
    try:
        with open('.upstox_token', 'r') as f:
            token = f.read().strip()
        if token:
            st.session_state.upstox_access_token = token
            # Test if the token is valid
            client = UpstoxClient()
            client.set_access_token(token)
            quote = client.get_live_quote("NSE_INDEX|Nifty 50")
            if quote:
                st.session_state.upstox_authenticated = True
                st.session_state.upstox_client = client
    except Exception:
        pass
if 'websocket_client' not in st.session_state:
    st.session_state.websocket_client = None
if 'websocket_connected' not in st.session_state:
    st.session_state.websocket_connected = False
if 'live_ohlc_data' not in st.session_state:
    st.session_state.live_ohlc_data = pd.DataFrame()
if 'current_tick' not in st.session_state:
    st.session_state.current_tick = None

# Initialize core session state variables that may be accessed
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None

# Check if we just completed authentication
if st.session_state.upstox_authenticated and 'upstox_just_authenticated' not in st.session_state:
    st.session_state.upstox_just_authenticated = True
    st.success("✅ Successfully authenticated with Upstox!")
    st.rerun()
elif 'upstox_just_authenticated' in st.session_state:
    del st.session_state.upstox_just_authenticated

# Authentication Section
st.header("🔐 Upstox Authentication")

if not st.session_state.upstox_authenticated:
    st.markdown("""
    ### Getting Your Access Token
    
    **Step 1:** Click the button below to authenticate with your Upstox account.
    **Step 2:** You'll be redirected to Upstox login page - enter your credentials.
    **Step 3:** After successful login, you'll be brought back here with a valid token.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("🚀 Login to Upstox", type="primary", use_container_width=True):
            try:
                upstox_client = UpstoxClient()
                login_url = upstox_client.get_login_url()
                st.markdown(f'<meta http-equiv="refresh" content="0; url={login_url}">', unsafe_allow_html=True)
                st.success("Redirecting to Upstox login...")
            except Exception as e:
                st.error(f"Error creating login URL: {str(e)}")

    with col2:
        st.info("🔒 Your credentials are stored securely and used only for data fetching.")
        st.markdown("""
        **Need help?**
        - Make sure popup blockers are disabled
        - Use the same browser for the entire process
        - If redirect fails, try refreshing this page
        """)

else:
    # Authenticated UI
    st.success("✅ Connected to Upstox API")

    # Token Status and Management
    st.subheader("🔑 Token Status")
    
    # Test token validity automatically
    token_valid = False
    if st.session_state.upstox_access_token:
        try:
            # Ensure client has the token
            if not upstox_client:
                upstox_client = UpstoxClient()
                upstox_client.set_access_token(st.session_state.upstox_access_token)
            
            quote = upstox_client.get_live_quote("NSE_INDEX|Nifty 50")
            token_valid = quote is not None and isinstance(quote, dict)
        except:
            token_valid = False
    
    # Show token status
    if token_valid:
        st.success("✅ Token is valid and working!")
        st.info(f"📊 Current NIFTY price: ₹{quote.get('ltp', 'N/A')}")
    else:
        st.error("❌ Token is expired or invalid")
        st.warning("🔄 WebSocket connections will fail until you refresh the token")
    
    col_refresh, col_debug = st.columns([1, 1])
    
    with col_refresh:
        if st.button("🔄 Refresh Token", type="primary"):
            # Clear current authentication
            st.session_state.upstox_authenticated = False
            st.session_state.upstox_access_token = None
            st.session_state.upstox_client = None
            st.session_state.websocket_connected = False
            st.session_state.websocket_client = None
            
            # Save empty token to file to clear cached version
            try:
                with open(".upstox_token", "w") as f:
                    f.write("")
                st.success("🗑️ Cleared cached token")
            except:
                pass
            
            st.info("🔄 Please scroll down and click 'Login to Upstox' to get a fresh token")
            st.rerun()
            
        # Add a helpful note about the refresh process
        st.markdown("""
        **When to refresh token:**
        - Token shows as expired/invalid
        - WebSocket keeps disconnecting
        - Getting authorization errors
        """, help="Upstox tokens expire every 24 hours")
    
    with col_debug:
        with st.expander("🔍 Debug Token Information"):
            st.write(f"**Token Available:** {'✅ Yes' if st.session_state.upstox_access_token else '❌ No'}")
            if st.session_state.upstox_access_token:
                st.write(f"**Token Length:** {len(st.session_state.upstox_access_token)}")
                st.write(f"**Token Preview:** {st.session_state.upstox_access_token[:20]}...")
                
                # Auto-save token to file
                try:
                    with open(".upstox_token", "w") as f:
                        f.write(st.session_state.upstox_access_token)
                    st.success("💾 Token auto-saved to file")
                except Exception as e:
                    st.error(f"❌ Could not save token: {str(e)}")
            else:
                st.error("❌ No access token found in session state")

    # Initialize client with stored token
    if st.session_state.upstox_client is None:
        upstox_client = UpstoxClient()
        upstox_client.set_access_token(st.session_state.upstox_access_token)
        st.session_state.upstox_client = upstox_client

    upstox_client = st.session_state.upstox_client

    # WebSocket Real-time Data Section
    st.header("🔴 Real-time WebSocket Data Stream")
    
    # Check token validity before allowing WebSocket
    if not token_valid:
        st.error("❌ Cannot start WebSocket - Token is expired or invalid")
        st.info("💡 Please refresh your token first using the 'Refresh Token' button above")
        st.stop()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if not st.session_state.websocket_connected:
            if st.button("🚀 Start WebSocket Stream", type="primary"):
                with st.spinner("Connecting to WebSocket..."):
                    try:
                        # Double check token before connecting
                        test_quote = upstox_client.get_live_quote("NSE_INDEX|Nifty 50")
                        if not test_quote:
                            st.error("❌ Token test failed - please refresh token first")
                            st.stop()
                        
                        ws_client = UpstoxWebSocketClient(upstox_client)

                        # Add callback to update session state
                        def on_ohlc_update(ohlc_candle):
                            if not st.session_state.live_ohlc_data.empty:
                                new_row = pd.DataFrame([ohlc_candle])
                                new_row.set_index('DateTime', inplace=True)
                                st.session_state.live_ohlc_data = pd.concat([st.session_state.live_ohlc_data, new_row])
                            else:
                                st.session_state.live_ohlc_data = pd.DataFrame([ohlc_candle])
                                st.session_state.live_ohlc_data.set_index('DateTime', inplace=True)

                        ws_client.add_callback(on_ohlc_update)

                        success = ws_client.connect()
                        if success:
                            st.session_state.websocket_client = ws_client
                            st.session_state.websocket_connected = True
                            st.success("✅ WebSocket connected successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to connect WebSocket")
                            st.warning("Common causes: Token expired, network issues, or API rate limits")
                    except Exception as e:
                        st.error(f"❌ WebSocket connection error: {str(e)}")
                        st.info("💡 If you see authorization errors, please refresh your token")
        else:
            st.success("🟢 WebSocket Active")
            
            # Check if WebSocket is still actually connected
            if st.session_state.websocket_client and not st.session_state.websocket_client.is_connected:
                st.warning("⚠️ WebSocket appears to be disconnected")
                st.info("💡 Click 'Stop WebSocket' and try reconnecting")

    with col2:
        if st.session_state.websocket_connected:
            if st.button("⏹️ Stop WebSocket", type="secondary"):
                try:
                    if st.session_state.websocket_client:
                        st.session_state.websocket_client.disconnect()
                    st.session_state.websocket_connected = False
                    st.session_state.websocket_client = None
                    st.success("✅ WebSocket disconnected")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error disconnecting: {str(e)}")

    with col3:
        if st.session_state.websocket_connected and st.session_state.websocket_client:
            # Display current tick info
            current_tick = st.session_state.websocket_client.get_latest_tick()
            current_candle = st.session_state.websocket_client.get_current_ohlc()

            if current_tick:
                st.session_state.current_tick = current_tick

            if st.session_state.current_tick:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Live Price", f"₹{st.session_state.current_tick['ltp']:.2f}")
                with col_b:
                    st.metric("Last Update", st.session_state.current_tick['timestamp'].strftime('%H:%M:%S'))

            if current_candle:
                st.write("**Current 5-min Candle:**")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("O", f"₹{current_candle['Open']:.2f}")
                with col_b:
                    st.metric("H", f"₹{current_candle['High']:.2f}")
                with col_c:
                    st.metric("L", f"₹{current_candle['Low']:.2f}")
                with col_d:
                    st.metric("C", f"₹{current_candle['Close']:.2f}")

    # Live Data Display
    if st.session_state.websocket_connected and not st.session_state.live_ohlc_data.empty:
        st.subheader("📈 Live OHLC Data Stream")

        # Auto-refresh the chart every 5 seconds
        placeholder = st.empty()

        with placeholder.container():
            recent_data = st.session_state.live_ohlc_data.tail(50)

            if len(recent_data) > 0:
                fig = go.Figure(data=go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name="NIFTY 50 Live"
                ))

                fig.update_layout(
                    title="Live NIFTY 50 - Last 50 Candles",
                    xaxis_title="Time",
                    yaxis_title="Price (₹)",
                    height=400,
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show recent candles table
                st.write("**Recent 5-minute Candles:**")
                st.dataframe(recent_data.tail(10), use_container_width=True)

                # Auto-save to database button
                if st.button("💾 Save Live Data to Database"):
                    with st.spinner("Saving live data..."):
                        save_success = trading_db.save_ohlc_data(
                            st.session_state.live_ohlc_data, 
                            "upstox_live_websocket", 
                            preserve_full_data=True
                        )
                        if save_success:
                            st.success(f"✅ Saved {len(st.session_state.live_ohlc_data)} live candles to database!")
                            # Update main session data
                            st.session_state.data = st.session_state.live_ohlc_data
                            st.session_state.data_source = "upstox_websocket"
                            st.session_state.last_data_update = datetime.now()
                        else:
                            st.error("❌ Failed to save live data")

    # Historical Data Fetching Section
    st.header("📊 Historical NIFTY 50 Data Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        data_period = st.selectbox(
            "Historical Data Period",
            [7, 15, 30, 60, 90],
            index=2,
            help="Number of days of historical data to fetch"
        )

    with col2:
        interval = st.selectbox(
            "Candle Interval",
            ["1minute", "30minute", "day", "week", "month"],
            index=1,
            help="Timeframe for OHLC candles"
        )

    with col3:
        auto_update = st.checkbox(
            "Auto-refresh every 5 minutes",
            value=False,
            help="Automatically fetch new data every 5 minutes"
        )

    # Fetch Data Button
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("📡 Fetch NIFTY 50 Data", type="primary"):
            with st.spinner(f"Fetching {data_period} days of {interval} NIFTY 50 data..."):
                try:
                    df = upstox_client.fetch_nifty50_data(days=data_period, interval=interval)

                    if df is not None and len(df) > 0:
                        # Store in session state
                        st.session_state.data = df

                        # Auto-save to database
                        save_success = trading_db.save_ohlc_data(df, "upstox_nifty50", preserve_full_data=True)

                        if save_success:
                            st.success(f"✅ Fetched {len(df)} records and saved to database!")

                            # Auto-calculate technical indicators
                            with st.spinner("Calculating technical indicators..."):
                                features_data = TechnicalIndicators.calculate_all_indicators(df)
                                st.session_state.features = features_data
                                st.success("✅ Technical indicators calculated!")

                                # Set data source flag for other pages
                                st.session_state.data_source = "upstox_live"
                                st.session_state.last_data_update = datetime.now()

                                st.info("🔗 **Ready for AI**: Your live data is now available in Model Training and Predictions pages!")
                        else:
                            st.warning("⚠️ Data fetched but failed to save to database")

                        st.rerun()
                    else:
                        st.error("❌ No data received from Upstox API")

                except Exception as e:
                    st.error(f"❌ Error fetching data: {str(e)}")

    with col2:
        if st.button("🔄 Get Live Quote"):
            with st.spinner("Fetching live NIFTY 50 quote..."):
                try:
                    quote = upstox_client.get_live_quote("NSE_INDEX|Nifty 50")

                    if quote:
                        st.json(quote)
                    else:
                        st.error("❌ Failed to get live quote")

                except Exception as e:
                    st.error(f"❌ Error getting live quote: {str(e)}")

    # Display current data if available
    if st.session_state.data is not None:
        df = st.session_state.data

        st.header("📈 Current Dataset Overview")

        # Data summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Date Range", f"{(df.index.max() - df.index.min()).days} days")
        with col3:
            st.metric("Latest Close", f"₹{df['Close'].iloc[-1]:.2f}")
        with col4:
            daily_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100)
            st.metric("Last Change", f"{daily_change:.2f}%")

        # Chart
        st.subheader("📊 NIFTY 50 Price Chart")

        # Chart controls
        col1, col2 = st.columns(2)
        with col1:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"], key="chart_type")
        with col2:
            show_volume = st.checkbox("Show Volume", value=True, key="show_volume")

        # Create chart
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=('NIFTY 50 Price', 'Volume')
            )
        else:
            fig = go.Figure()

        if chart_type == "Candlestick":
            candlestick = go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="NIFTY 50"
            )

            if show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
        else:
            line_chart = go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='NIFTY 50 Close',
                line=dict(color='#00ffff', width=2)
            )

            if show_volume:
                fig.add_trace(line_chart, row=1, col=1)
            else:
                fig.add_trace(line_chart)

        # Add volume if requested
        if show_volume and 'Volume' in df.columns:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.6)'
            ), row=2, col=1)

        fig.update_layout(
            title="NIFTY 50 Real-time Data",
            xaxis_title="DateTime",
            yaxis_title="Price (₹)",
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("📋 Recent Data")
        st.dataframe(df.tail(20), use_container_width=True)

        # Integration info
        st.info("💡 **Next Steps**: Your Upstox data is now loaded! Go to **Model Training** to train AI models with this live data, or **Predictions** to generate forecasts.")

    # Auto-refresh logic
    if auto_update and st.session_state.upstox_authenticated:
        time.sleep(300)  # 5 minutes
        st.rerun()

# Logout option
if st.session_state.upstox_authenticated:
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚪 Logout from Upstox"):
            # First disconnect WebSocket if active
            if 'websocket_client' in st.session_state and st.session_state.websocket_client:
                try:
                    st.session_state.websocket_client.disconnect()
                except:
                    pass
            
            # Delete the token file first
            try:
                if os.path.exists('.upstox_token'):
                    os.remove('.upstox_token')
                    st.info("🗑️ Token file deleted")
            except Exception as e:
                st.warning(f"⚠️ Error deleting token file: {e}")
            
            # Clear ALL session state (nuclear option)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Reinitialize essential session state
            st.session_state.upstox_authenticated = False
            st.session_state.upstox_client = None
            st.session_state.upstox_access_token = None
            st.session_state.websocket_connected = False
            st.session_state.live_ohlc_data = pd.DataFrame()
            
            # Initialize core session state variables
            st.session_state.data = None
            st.session_state.features = None
            st.session_state.models = {}
            st.session_state.predictions = None
            st.session_state.model_trainer = None
            
            # Show success message
            st.success("✅ Complete logout successful - all sessions cleared")
            st.info("🔄 Page will refresh automatically...")
            
            # Force immediate rerun
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("🔄 Reset WebSocket Connection"):
            # Clear WebSocket related session state
            if 'websocket_client' in st.session_state:
                if st.session_state.websocket_client:
                    st.session_state.websocket_client.disconnect()
                st.session_state.websocket_client = None
            st.session_state.websocket_connected = False
            st.session_state.live_ohlc_data = pd.DataFrame()
            st.session_state.current_tick = None
            st.success("✅ WebSocket connection reset")
            st.rerun()