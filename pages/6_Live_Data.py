
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from utils.live_data_manager import LiveDataManager
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
    
    # Initialize session state for live data manager
    if 'live_data_manager' not in st.session_state:
        st.session_state.live_data_manager = None
    if 'is_live_connected' not in st.session_state:
        st.session_state.is_live_connected = False
    
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
                    # Initialize live data manager
                    st.session_state.live_data_manager = LiveDataManager(access_token, api_key)
                    
                    # Connect to WebSocket
                    if st.session_state.live_data_manager.connect():
                        st.session_state.is_live_connected = True
                        st.success("✅ Connected to Upstox WebSocket!")
                        
                        # Wait a moment for connection to establish
                        time.sleep(2)
                        
                        # Subscribe to selected instruments
                        if selected_instruments:
                            instrument_keys = [popular_instruments.get(inst, inst) for inst in selected_instruments]
                            if st.session_state.live_data_manager.subscribe_instruments(instrument_keys):
                                st.success(f"✅ Subscribed to {len(instrument_keys)} instruments")
                            else:
                                st.warning("⚠️ Failed to subscribe to instruments")
                    else:
                        st.error("❌ Failed to connect to Upstox WebSocket")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
    
    with col2:
        if st.button("🔌 Disconnect", disabled=not st.session_state.is_live_connected):
            if st.session_state.live_data_manager:
                st.session_state.live_data_manager.disconnect()
                st.session_state.is_live_connected = False
                st.info("🔌 Disconnected from live data feed")
    
    with col3:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    with col4:
        auto_refresh = st.toggle("🔄 Auto Refresh", value=False)
    
    # Status dashboard
    if st.session_state.live_data_manager:
        status = st.session_state.live_data_manager.get_connection_status()
        
        st.header("📊 Connection Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if status['connected'] else "🔴"
            st.metric("Connection", f"{status_color} {status['status'].title()}")
        
        with col2:
            st.metric("Subscribed Instruments", status['subscribed_instruments'])
        
        with col3:
            st.metric("Total Ticks Received", f"{status['total_ticks_received']:,}")
        
        with col4:
            last_update = status['last_update']
            if last_update:
                time_diff = datetime.now() - last_update
                st.metric("Last Update", f"{time_diff.total_seconds():.1f}s ago")
            else:
                st.metric("Last Update", "Never")
    
    # Live data display
    if st.session_state.is_live_connected and st.session_state.live_data_manager:
        
        # Get tick statistics
        tick_stats = st.session_state.live_data_manager.get_tick_statistics()
        
        if tick_stats:
            st.header("📈 Live Market Data")
            
            # Create tabs for different views
            overview_tab, charts_tab, tick_details_tab, export_tab = st.tabs([
                "📊 Market Overview",
                "📈 Live Charts", 
                "🔍 Tick Details",
                "💾 Export Data"
            ])
            
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
    
    # Auto-refresh functionality
    if auto_refresh and st.session_state.is_live_connected:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    show_live_data_page()
