import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from utils.live_data_manager import LiveDataManager
from utils.live_prediction_pipeline import LivePredictionPipeline
from utils.database_adapter import DatabaseAdapter
import json

# Page configuration
st.set_page_config(page_title="Live Data", page_icon="📡", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_live_data_page():
    """Main live data page."""

    st.markdown("""
    <div class="trading-header">
        <h1 style="margin:0;">📡 LIVE MARKET DATA</h1>
        <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
            Real-time Upstox WebSocket Integration
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for live data manager and prediction pipeline
    if 'live_data_manager' not in st.session_state:
        st.session_state.live_data_manager = None
    if 'live_prediction_pipeline' not in st.session_state:
        st.session_state.live_prediction_pipeline = None
    if 'is_live_connected' not in st.session_state:
        st.session_state.is_live_connected = False
    if 'is_prediction_pipeline_active' not in st.session_state:
        st.session_state.is_prediction_pipeline_active = False

    # Configuration section
    st.header("🔧 Configuration")

    # Create tabs for different features
    config_tab, historical_tab = st.tabs(["🔌 Live Data Config", "📊 Historical Data Fetch"])

    with config_tab:
        col1, col2 = st.columns(2)

    with col1:
        st.subheader("📱 Upstox API Credentials")
        access_token = st.text_input(
            "Access Token", 
            type="password", 
            help="Your Upstox access token",
            key="upstox_access_token"
        )
        api_key = st.text_input(
            "API Key", 
            type="password", 
            help="Your Upstox API key",
            key="upstox_api_key"
        )

    with col2:
        st.subheader("📊 Instrument Configuration")

        # Common instrument keys for Indian market
        popular_instruments = {
            "NIFTY 50": "NSE_INDEX|Nifty 50",
            "BANK NIFTY": "NSE_INDEX|Nifty Bank",
            "RELIANCE": "NSE_EQ|INE002A01018",
            "TCS": "NSE_EQ|INE467B01029",
            "HDFC BANK": "NSE_EQ|INE040A01034",
            "INFOSYS": "NSE_EQ|INE009A01021"
        }

        selected_instruments = st.multiselect(
            "Select Instruments",
            options=list(popular_instruments.keys()),
            default=["NIFTY 50", "BANK NIFTY"],
            help="Choose instruments to subscribe for live data"
        )

        # Custom instrument input
        custom_instrument = st.text_input(
            "Custom Instrument Key",
            help="Enter custom instrument key (e.g., NSE_EQ|INE002A01018)"
        )

        if custom_instrument:
            selected_instruments.append(custom_instrument)

    with historical_tab:
        st.subheader("📈 Fetch Historical Data from Upstox")
        st.write("Fetch historical 1-minute data for Nifty 50 and other instruments using Upstox API")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**API Credentials**")
            hist_access_token = st.text_input(
                "Access Token",
                type="password",
                help="Your Upstox access token for historical data",
                key="hist_access_token"
            )
            hist_api_key = st.text_input(
                "API Key", 
                type="password",
                help="Your Upstox API key for historical data",
                key="hist_api_key"
            )

        with col2:
            st.write("**Instrument Selection**")
            hist_instruments = {
                "NIFTY 50": "NSE_INDEX|Nifty 50",
                "BANK NIFTY": "NSE_INDEX|Nifty Bank",
                "NIFTY IT": "NSE_INDEX|Nifty IT",
                "NIFTY FMCG": "NSE_INDEX|Nifty FMCG",
                "RELIANCE": "NSE_EQ|INE002A01018",
                "TCS": "NSE_EQ|INE467B01029",
                "HDFC BANK": "NSE_EQ|INE040A01034",
                "INFOSYS": "NSE_EQ|INE009A01021"
            }

            selected_hist_instrument = st.selectbox(
                "Select Instrument",
                options=list(hist_instruments.keys()),
                index=0
            )

            custom_hist_instrument = st.text_input(
                "Custom Instrument",
                placeholder="NSE_EQ|INE002A01018"
            )

            if custom_hist_instrument:
                instrument_key = custom_hist_instrument
                display_name = custom_hist_instrument
            else:
                instrument_key = hist_instruments[selected_hist_instrument]
                display_name = selected_hist_instrument

        with col3:
            st.write("**Data Parameters**")
            interval_options = {
                "1 minute": "1minute",
                "5 minutes": "5minute", 
                "15 minutes": "15minute",
                "30 minutes": "30minute",
                "1 hour": "1hour",
                "1 day": "day"
            }

            selected_interval = st.selectbox(
                "Interval",
                options=list(interval_options.keys()),
                index=0  # Default to 1 minute
            )

            days_back = st.number_input(
                "Days Back",
                min_value=1,
                max_value=365,
                value=7,
                help="Number of days of historical data"
            )

        # Fetch button
        if st.button("📥 Fetch Historical Data", type="primary", disabled=not (hist_access_token and hist_api_key)):
            if hist_access_token and hist_api_key:
                with st.spinner(f"Fetching {days_back} days of {selected_interval} data for {display_name}..."):
                    try:
                        # Calculate date range
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=days_back)

                        # Format dates for Upstox API
                        from_date = start_date.strftime('%Y-%m-%d')
                        to_date = end_date.strftime('%Y-%m-%d')

                        # Upstox historical data API endpoint
                        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval_options[selected_interval]}/{to_date}/{from_date}"

                        headers = {
                            "Authorization": f"Bearer {hist_access_token}",
                            "Accept": "application/json"
                        }

                        import requests
                        response = requests.get(url, headers=headers)

                        if response.status_code == 200:
                            data = response.json()

                            if data.get("status") == "success" and "data" in data and "candles" in data["data"]:
                                candles = data["data"]["candles"]

                                if candles:
                                    # Convert to DataFrame
                                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])

                                    # Convert timestamp to datetime
                                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                                    df = df.set_index('timestamp')

                                    # Rename columns to standard format
                                    df = df.rename(columns={
                                        'open': 'Open',
                                        'high': 'High', 
                                        'low': 'Low',
                                        'close': 'Close',
                                        'volume': 'Volume'
                                    })

                                    # Remove unnecessary columns
                                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                                    # Sort by timestamp
                                    df = df.sort_index()

                                    st.success(f"✅ Successfully fetched {len(df)} data points!")

                                    # Display summary
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Records", f"{len(df):,}")
                                    with col2:
                                        st.metric("Date Range", f"{df.index.min().strftime('%Y-%m-%d')}")
                                    with col3:
                                        st.metric("To", f"{df.index.max().strftime('%Y-%m-%d')}")
                                    with col4:
                                        st.metric("Latest Price", f"₹{df['Close'].iloc[-1]:.2f}")

                                    # Show sample data
                                    st.subheader("📊 Sample Data")
                                    st.dataframe(df.head(10), use_container_width=True)

                                    # Download button
                                    csv_data = df.to_csv()
                                    file_name = f"{display_name.replace(' ', '_')}_{interval_options[selected_interval]}_{days_back}days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                                    st.download_button(
                                        label=f"📥 Download {display_name} {selected_interval} Data",
                                        data=csv_data,
                                        file_name=file_name,
                                        mime="text/csv",
                                        use_container_width=True
                                    )

                                    # Option to save to database
                                    if st.button("💾 Save to Database"):
                                        try:
                                            db = DatabaseAdapter()
                                            dataset_name = f"upstox_{display_name.replace(' ', '_').lower()}_{interval_options[selected_interval]}"

                                            if db.save_ohlc_data(df, dataset_name):
                                                st.success(f"✅ Saved historical data to database as '{dataset_name}'")
                                            else:
                                                st.error("❌ Failed to save data to database")
                                        except Exception as e:
                                            st.error(f"❌ Database error: {str(e)}")

                                    # Basic chart
                                    st.subheader("📈 Price Chart")
                                    fig = go.Figure(data=go.Candlestick(
                                        x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'],
                                        name=display_name
                                    ))

                                    fig.update_layout(
                                        title=f"{display_name} - {selected_interval} Chart ({days_back} days)",
                                        xaxis_title="Time",
                                        yaxis_title="Price (₹)",
                                        height=500,
                                        template="plotly_dark"
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                else:
                                    st.warning("⚠️ No candle data returned from API")
                            else:
                                st.error(f"❌ API Error: {data.get('message', 'Unknown error')}")
                        else:
                            st.error(f"❌ HTTP Error {response.status_code}: {response.text}")

                    except Exception as e:
                        st.error(f"❌ Error fetching historical data: {str(e)}")
            else:
                st.warning("⚠️ Please provide both Access Token and API Key")

        # Information section
        st.info("""
        **📋 Upstox Historical Data Features:**
        • Supports 1-minute to daily intervals
        • Up to 1 year of historical data
        • Real-time API integration
        • Direct CSV download
        • Database storage option
        • Interactive charts

        **🔑 API Requirements:**
        • Valid Upstox access token (refreshed daily)
        • Active API subscription for historical data
        """)

    # Continue with live data configuration

    # Connection controls for live data
    st.header("🔌 Live Data Connection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 Connect", type="primary", disabled=not (access_token and api_key)):
            if access_token and api_key:
                try:
                    # Initialize prediction pipeline (includes live data manager)
                    st.session_state.live_prediction_pipeline = LivePredictionPipeline(access_token, api_key)
                    st.session_state.live_data_manager = st.session_state.live_prediction_pipeline.live_data_manager

                    # Start the prediction pipeline
                    if st.session_state.live_prediction_pipeline.start_pipeline():
                        st.session_state.is_live_connected = True
                        st.session_state.is_prediction_pipeline_active = True
                        st.success("✅ Connected to Upstox WebSocket with Prediction Pipeline!")

                        # Wait a moment for connection to establish
                        time.sleep(2)

                        # Subscribe to selected instruments
                        if selected_instruments:
                            instrument_keys = [popular_instruments.get(inst, inst) for inst in selected_instruments]
                            if st.session_state.live_prediction_pipeline.subscribe_instruments(instrument_keys):
                                st.success(f"✅ Subscribed to {len(instrument_keys)} instruments with live predictions")
                            else:
                                st.warning("⚠️ Failed to subscribe to instruments")
                    else:
                        st.error("❌ Failed to start prediction pipeline")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

    with col2:
        if st.button("🔌 Disconnect", disabled=not st.session_state.is_live_connected):
            if st.session_state.live_prediction_pipeline:
                st.session_state.live_prediction_pipeline.stop_pipeline()
                st.session_state.is_live_connected = False
                st.session_state.is_prediction_pipeline_active = False
                st.info("🔌 Disconnected from live data feed and prediction pipeline")

    with col3:
        if st.button("🔄 Refresh Status"):
            st.rerun()

    with col4:
        auto_refresh = st.toggle("🔄 Auto Refresh", value=False)



    # Status dashboard
    if st.session_state.live_prediction_pipeline:
        pipeline_status = st.session_state.live_prediction_pipeline.get_pipeline_status()

        st.header("📊 Live Prediction Pipeline Status")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            status_color = "🟢" if pipeline_status['data_connected'] else "🔴"
            st.metric("Data Connection", f"{status_color} {'Connected' if pipeline_status['data_connected'] else 'Disconnected'}")

        with col2:
            pipeline_color = "🟢" if pipeline_status['pipeline_active'] else "🔴"
            st.metric("Prediction Pipeline", f"{pipeline_color} {'Active' if pipeline_status['pipeline_active'] else 'Inactive'}")

        with col3:
            model_color = "🟢" if pipeline_status['model_ready'] else "🔴"
            st.metric("Direction Model", f"{model_color} {'Ready' if pipeline_status['model_ready'] else 'Not Trained'}")

        with col4:
            st.metric("Subscribed Instruments", pipeline_status['subscribed_instruments'])

        with col5:
            st.metric("Live Predictions", pipeline_status['instruments_with_predictions'])

    # Live data display
    if st.session_state.is_live_connected and st.session_state.live_data_manager:

        # Get tick statistics
        tick_stats = st.session_state.live_data_manager.get_tick_statistics()

        if tick_stats:
            st.header("📈 Live Market Data")

            # Create tabs for different views
            overview_tab, predictions_tab, charts_tab, tick_details_tab, export_tab = st.tabs([
                "📊 Market Overview",
                "🎯 Live Predictions",
                "📈 Live Charts", 
                "🔍 Tick Details",
                "💾 Export Data"
            ])

            with predictions_tab:
                if st.session_state.is_prediction_pipeline_active:
                    st.subheader("🎯 Real-time Direction Predictions")

                    # Get live predictions
                    live_predictions = st.session_state.live_prediction_pipeline.get_latest_predictions()

                    if live_predictions:
                        # Display predictions in a grid
                        for instrument_key, prediction in live_predictions.items():
                            display_name = instrument_key.split('|')[-1] if '|' in instrument_key else instrument_key

                            # Get detailed summary
                            summary = st.session_state.live_prediction_pipeline.get_instrument_summary(instrument_key)

                            with st.container():
                                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

                                with col1:
                                    direction_color = "🟢" if prediction['direction'] == 'Bullish' else "🔴"
                                    st.markdown(f"""
                                    **{direction_color} {display_name}**  
                                    Direction: **{prediction['direction']}**  
                                    Confidence: **{prediction['confidence']:.1%}**
                                    """)

                                with col2:
                                    st.metric("Current Price", f"₹{prediction['current_price']:.2f}")

                                with col3:
                                    st.metric("Volume", f"{prediction['volume']:,}")

                                with col4:
                                    if summary and 'recent_stats' in summary:
                                        stats = summary['recent_stats']
                                        st.markdown(f"""
                                        **Recent Signals (20):**  
                                        Bullish: {stats['bullish_signals']} ({stats['bullish_percentage']:.0f}%)  
                                        Avg Confidence: {stats['average_confidence']:.1%}
                                        """)

                                # Show prediction timestamp
                                time_ago = datetime.now() - prediction['generated_at']
                                st.caption(f"Generated {time_ago.total_seconds():.0f}s ago")

                                st.divider()

                        # Auto-refresh toggle
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 Refresh Predictions"):
                                st.rerun()

                        with col2:
                            auto_refresh_predictions = st.toggle("🔄 Auto Refresh (30s)", value=False, key="auto_refresh_predictions")

                        # Auto-refresh functionality for predictions
                        if auto_refresh_predictions:
                            time.sleep(30)
                            st.rerun()

                    else:
                        st.info("🎯 Prediction pipeline is active but no predictions generated yet. Please wait for sufficient OHLC data to accumulate...")

                        # Show requirements
                        st.write("**Requirements for predictions:**")
                        st.write("• Minimum 100 OHLC data points")
                        st.write("• Direction model must be trained")
                        st.write("• Sufficient tick data for feature calculation")

                else:
                    st.warning("⚠️ Prediction pipeline not active. Please connect to start receiving live predictions.")

            with overview_tab:
                st.subheader("💹 Real-time Price Dashboard")

                # Display live prices in a grid
                cols = st.columns(min(3, len(tick_stats)))

                for i, (instrument, stats) in enumerate(tick_stats.items()):
                    with cols[i % len(cols)]:
                        # Get instrument display name
                        display_name = instrument.split('|')[-1] if '|' in instrument else instrument

                        # Color based on change
                        change_pct = stats.get('change_percent', 0)
                        color = "🟢" if change_pct >= 0 else "🔴"

                        st.markdown(f"""
                        <div class="metric-container">
                            <h4 style="color: #00ffff; margin: 0;">{color} {display_name}</h4>
                            <h2 style="margin: 0.5rem 0; color: #00ff41;">₹{stats['latest_price']:.2f}</h2>
                            <p style="color: #9ca3af; margin: 0;">
                                {change_pct:+.2f}% | Vol: {stats['latest_volume']:,}
                            </p>
                            <p style="color: #6b7280; font-size: 0.8rem; margin: 0;">
                                Ticks: {stats['tick_count']:,}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

            with charts_tab:
                st.subheader("📈 Real-time Price Charts")

                # Select instrument for detailed chart
                if tick_stats:
                    selected_instrument = st.selectbox(
                        "Select Instrument for Chart",
                        options=list(tick_stats.keys()),
                        format_func=lambda x: x.split('|')[-1] if '|' in x else x
                    )

                    if selected_instrument:
                        # Get OHLC data
                        ohlc_data = st.session_state.live_data_manager.get_live_ohlc(selected_instrument)

                        if ohlc_data is not None and len(ohlc_data) > 0:
                            # Create candlestick chart
                            fig = go.Figure(data=go.Candlestick(
                                x=ohlc_data.index,
                                open=ohlc_data['Open'],
                                high=ohlc_data['High'],
                                low=ohlc_data['Low'],
                                close=ohlc_data['Close'],
                                name="Price"
                            ))

                            fig.update_layout(
                                title=f"Live Chart - {selected_instrument.split('|')[-1] if '|' in selected_instrument else selected_instrument}",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                height=500,
                                template="plotly_dark"
                            )

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("📊 Accumulating tick data... Please wait for OHLC chart generation.")

            with tick_details_tab:
                st.subheader("🔍 Detailed Tick Information")

                # Show latest ticks for each instrument
                for instrument, stats in tick_stats.items():
                    with st.expander(f"📊 {instrument.split('|')[-1] if '|' in instrument else instrument} - Latest Tick"):
                        latest_tick = st.session_state.live_data_manager.get_latest_tick(instrument)

                        if latest_tick:
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Price Information:**")
                                st.write(f"• LTP: ₹{latest_tick.get('ltp', 0):.2f}")
                                st.write(f"• Open: ₹{latest_tick.get('open', 0):.2f}")
                                st.write(f"• High: ₹{latest_tick.get('high', 0):.2f}")
                                st.write(f"• Low: ₹{latest_tick.get('low', 0):.2f}")
                                st.write(f"• Close: ₹{latest_tick.get('close', 0):.2f}")

                            with col2:
                                st.write("**Market Data:**")
                                st.write(f"• Volume: {latest_tick.get('volume', 0):,}")
                                st.write(f"• Bid: ₹{latest_tick.get('bid_price', 0):.2f} ({latest_tick.get('bid_qty', 0):,})")
                                st.write(f"• Ask: ₹{latest_tick.get('ask_price', 0):.2f} ({latest_tick.get('ask_qty', 0):,})")
                                st.write(f"• Change: {latest_tick.get('change_percent', 0):+.2f}%")
                                st.write(f"• Timestamp: {latest_tick.get('timestamp', 'N/A')}")

            with export_tab:
                st.subheader("💾 Export Live Data")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Export Options:**")

                    # Select instrument to export
                    export_instrument = st.selectbox(
                        "Select Instrument to Export",
                        options=list(tick_stats.keys()),
                        format_func=lambda x: x.split('|')[-1] if '|' in x else x,
                        key="export_instrument"
                    )

                    export_format = st.radio(
                        "Export Format",
                        ["OHLC Data", "Raw Tick Data"],
                        key="export_format"
                    )

                with col2:
                    st.write("**Export Actions:**")

                    if st.button("📥 Download CSV", type="primary"):
                        if export_instrument:
                            if export_format == "OHLC Data":
                                ohlc_data = st.session_state.live_data_manager.get_live_ohlc(export_instrument)
                                if ohlc_data is not None and len(ohlc_data) > 0:
                                    csv_data = ohlc_data.to_csv()
                                    st.download_button(
                                        label="📥 Download OHLC CSV",
                                        data=csv_data,
                                        file_name=f"live_ohlc_{export_instrument.replace('|', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.warning("No OHLC data available for export")
                            else:
                                # Export raw tick data would require storing individual ticks
                                st.info("Raw tick export feature coming soon!")

                    if st.button("💾 Save to Database"):
                        if export_instrument:
                            try:
                                ohlc_data = st.session_state.live_data_manager.get_live_ohlc(export_instrument)
                                if ohlc_data is not None and len(ohlc_data) > 0:
                                    # Save to database
                                    db = DatabaseAdapter()
                                    dataset_name = f"live_{export_instrument.replace('|', '_')}"
                                    if db.save_ohlc_data(ohlc_data, dataset_name):
                                        st.success(f"✅ Saved live data to database as '{dataset_name}'")
                                    else:
                                        st.error("❌ Failed to save data to database")
                                else:
                                    st.warning("No data available to save")
                            except Exception as e:
                                st.error(f"❌ Error saving to database: {str(e)}")
        else:
            st.info("📡 Connected but no tick data received yet. Please wait...")
            
            # Add debugging information
            if st.session_state.live_data_manager:
                connection_status = st.session_state.live_data_manager.ws_client.get_connection_status()
                
                with st.expander("🔍 Connection Debug Info"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Connection Status:**")
                        st.write(f"• Connected: {connection_status['is_connected']}")
                        st.write(f"• Subscribed: {connection_status['total_instruments']} instruments")
                        st.write(f"• Latest ticks: {connection_status['last_tick_count']}")
                        
                    with col2:
                        st.write("**WebSocket Info:**")
                        if hasattr(st.session_state.live_data_manager.ws_client, 'total_ticks_received'):
                            st.write(f"• Total ticks: {st.session_state.live_data_manager.ws_client.total_ticks_received}")
                        if hasattr(st.session_state.live_data_manager.ws_client, 'close_count'):
                            st.write(f"• Close count: {st.session_state.live_data_manager.ws_client.close_count}")
                        
                        # Check if in market hours
                        is_market_hours = st.session_state.live_data_manager.ws_client._is_market_hours()
                        st.write(f"• Market hours: {'Yes' if is_market_hours else 'No'}")
                
                # Show raw connection details
                st.json(connection_status)
    else:
        st.info("🔌 Please connect to start receiving live market data.")

    # Auto-refresh functionality
    if auto_refresh and st.session_state.is_live_connected:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    show_live_data_page()