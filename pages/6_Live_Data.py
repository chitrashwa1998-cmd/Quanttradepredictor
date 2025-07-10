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

    # Connection controls
    st.header("🔌 Connection Controls")

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
    else:
        st.info("🔌 Please connect to start receiving live market data.")

    # Historical data section (separate from live connection)
    st.header("📊 Historical Data Fetch")
    st.write("Fetch historical data independently (no live connection required)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔧 Custom Instruments")

        # Custom instruments text area
        custom_instruments = st.text_area(
            "Instrument Keys (one per line)",
            value="NSE_EQ|INE002A01018\nNSE_EQ|INE467B01029",
            height=100,
            help="Enter instrument keys, one per line"
        )

        historical_days = st.slider("Days of historical data", 1, 30, 5, key="hist_days")

        interval_option = st.selectbox(
            "Time Interval",
            ["day", "1minute", "5minute", "15minute", "30minute", "60minute"],
            index=0,
            help="Select the candle interval"
        )

        if st.button("📥 Fetch Historical Data"):
            instrument_keys = [key.strip() for key in custom_instruments.split('\n') if key.strip()]

            if instrument_keys and access_token and api_key:
                try:
                    # Initialize historical client if needed
                    if st.session_state.live_data_manager is None:
                        st.session_state.live_data_manager = LiveDataManager(access_token, api_key)

                    # Fetch historical data
                    with st.spinner(f"📥 Fetching {interval_option} historical data from Upstox API..."):
                        results = st.session_state.live_data_manager.fetch_historical_data(
                            instrument_keys, historical_days, interval_option
                        )

                    if results:
                        st.success(f"✅ Fetched {interval_option} data for {len(results)} instruments!")

                        for instrument, data in results.items():
                            st.info(f"📊 {instrument}: {len(data)} {interval_option} candles")

                            # Show sample data
                            st.dataframe(data.head(10), use_container_width=True)
                    else:
                        st.error("❌ Failed to fetch historical data")

                except Exception as e:
                    st.error(f"❌ Historical data fetch error: {str(e)}")
            elif not (access_token and api_key):
                st.error("❌ Please enter Access Token and API Key first")

    with col2:
        # Quick Nifty 50 fetch (Daily)
        st.subheader("⚡ Quick Nifty (Daily)")

        nifty_days = st.slider("Days for Nifty data", 1, 30, 5, key="nifty_days")

        if st.button("📊 Fetch Nifty Daily Data"):
            if access_token and api_key:
                try:
                    # Initialize historical client if needed
                    if st.session_state.live_data_manager is None:
                        st.session_state.live_data_manager = LiveDataManager(access_token, api_key)

                    with st.spinner("📥 Fetching Nifty 50 daily data..."):
                        results = st.session_state.live_data_manager.fetch_nifty_historical_data(nifty_days)

                    if results:
                        st.success(f"✅ Fetched Nifty daily data!")

                        for instrument, data in results.items():
                            name = "Nifty 50" if "Nifty 50" in instrument else "Bank Nifty"
                            st.info(f"📊 {name}: {len(data)} daily candles")

                            # Show sample data
                            st.dataframe(data.head(10), use_container_width=True)
                    else:
                        st.error("❌ Failed to fetch Nifty daily data")

                except Exception as e:
                    st.error(f"❌ Nifty daily fetch error: {str(e)}")
            else:
                st.error("❌ Please enter Access Token and API Key first")

    with col3:
        # Quick Nifty 50 fetch (1-minute)
        st.subheader("🚀 Nifty 1-Minute")

        nifty_1min_days = st.slider("Days for 1-min data", 1, 5, 1, key="nifty_1min_days", 
                                   help="1-minute data is limited to recent days")

        if st.button("⚡ Fetch Nifty 1-Min Data"):
            if access_token and api_key:
                try:
                    # Initialize historical client if needed
                    if st.session_state.live_data_manager is None:
                        st.session_state.live_data_manager = LiveDataManager(access_token, api_key)

                    with st.spinner("📥 Fetching Nifty 50 1-minute data..."):
                        results = st.session_state.live_data_manager.fetch_nifty_1min_historical_data(nifty_1min_days)

                    if results:
                        st.success(f"✅ Fetched Nifty 1-minute data!")

                        for instrument, data in results.items():
                            name = "Nifty 50" if "Nifty 50" in instrument else "Bank Nifty"
                            st.info(f"📊 {name}: {len(data)} 1-minute candles")

                            # Show sample data with timestamp info
                            st.dataframe(data.head(10), use_container_width=True)

                            # Show data range info
                            if len(data) > 0:
                                start_time = data.index.min().strftime('%Y-%m-%d %H:%M:%S')
                                end_time = data.index.max().strftime('%Y-%m-%d %H:%M:%S')
                                st.caption(f"Data range: {start_time} to {end_time}")
                    else:
                        st.error("❌ Failed to fetch Nifty 1-minute data")

                except Exception as e:
                    st.error(f"❌ Nifty 1-minute fetch error: {str(e)}")
            else:
                st.error("❌ Please enter Access Token and API Key first")

    # Auto-refresh functionality
    if auto_refresh and st.session_state.is_live_connected:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    show_live_data_page()