
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import time
from collections import deque
from utils.upstox_websocket import UpstoxWebSocketClient
from utils.upstox_historical import UpstoxHistoricalClient

class LiveDataManager:
    """Manage real-time tick data and convert to OHLC format."""
    
    def __init__(self, access_token: str, api_key: str):
        """Initialize live data manager."""
        self.ws_client = UpstoxWebSocketClient(access_token, api_key)
        self.historical_client = UpstoxHistoricalClient(access_token, api_key)
        self.tick_buffer = {}  # Store ticks for each instrument
        self.ohlc_data = {}    # Store OHLC data for each instrument
        self.buffer_size = 1000  # Maximum ticks to store per instrument
        
        # Set up callbacks
        self.ws_client.set_callbacks(
            tick_callback=self.on_tick_received,
            error_callback=self.on_error,
            connection_callback=self.on_connection_change
        )
        
        # Status tracking
        self.connection_status = "disconnected"
        self.last_update_time = None
        self.total_ticks_received = 0
        
        # Pre-seeding tracking
        self.is_seeded = {}  # Track which instruments are pre-seeded
        self.seed_data_count = {}  # Track how much historical data loaded
        
    def on_tick_received(self, tick_data: Dict):
        """Handle incoming tick data."""
        try:
            instrument_key = tick_data['instrument_token']
            timestamp = tick_data['timestamp']
            
            # Initialize buffer for new instrument
            if instrument_key not in self.tick_buffer:
                self.tick_buffer[instrument_key] = deque(maxlen=self.buffer_size)
                self.ohlc_data[instrument_key] = pd.DataFrame()
            
            # Add tick to buffer
            self.tick_buffer[instrument_key].append(tick_data)
            self.total_ticks_received += 1
            self.last_update_time = timestamp
            
            # Update OHLC data if we have enough ticks
            if len(self.tick_buffer[instrument_key]) >= 5:
                self.update_ohlc_data(instrument_key)
                
        except Exception as e:
            print(f"Error processing tick: {e}")
    
    def on_error(self, error):
        """Handle WebSocket errors."""
        print(f"WebSocket error: {error}")
        self.connection_status = "error"
    
    def on_connection_change(self, status: str):
        """Handle connection status changes."""
        self.connection_status = status
        print(f"Connection status: {status}")
    
    def update_ohlc_data(self, instrument_key: str, timeframe: str = "5T"):
        """Convert tick data to OHLC format."""
        try:
            ticks = list(self.tick_buffer[instrument_key])
            if not ticks:
                return
            
            # Create DataFrame from ticks
            df = pd.DataFrame(ticks)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Resample to OHLC format
            new_ohlc = df['ltp'].resample(timeframe).ohlc()
            new_ohlc['volume'] = df['volume'].resample(timeframe).sum()
            
            # Remove NaN values
            new_ohlc = new_ohlc.dropna()
            
            if len(new_ohlc) > 0:
                # Rename columns to match existing format
                new_ohlc.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Combine with existing data (could be historical + live)
                if instrument_key in self.ohlc_data and len(self.ohlc_data[instrument_key]) > 0:
                    existing_ohlc = self.ohlc_data[instrument_key]
                    
                    # Combine with new live data
                    combined_ohlc = pd.concat([existing_ohlc, new_ohlc])
                    
                    # Remove duplicate timestamps, keeping the latest
                    combined_ohlc = combined_ohlc[~combined_ohlc.index.duplicated(keep='last')]
                    
                    # Sort by timestamp
                    combined_ohlc = combined_ohlc.sort_index()
                    
                    # Keep reasonable amount of data (100 recent + any pre-seeded)
                    max_rows = 200 if self.is_seeded.get(instrument_key, False) else 100
                    if len(combined_ohlc) > max_rows:
                        combined_ohlc = combined_ohlc.tail(max_rows)
                    
                    self.ohlc_data[instrument_key] = combined_ohlc
                    seeded_info = " (pre-seeded)" if self.is_seeded.get(instrument_key, False) else ""
                    print(f"📈 Live OHLC for {instrument_key}: {len(combined_ohlc)} total rows{seeded_info}")
                else:
                    # First time - store new data
                    self.ohlc_data[instrument_key] = new_ohlc
                    print(f"📈 Initial OHLC for {instrument_key}: {len(new_ohlc)} rows")
                
        except Exception as e:
            print(f"Error updating OHLC data: {e}")
    
    def connect(self) -> bool:
        """Connect to live data feed."""
        return self.ws_client.connect()
    
    def disconnect(self):
        """Disconnect from live data feed."""
        self.ws_client.disconnect()
        self.connection_status = "disconnected"
    
    def subscribe_instruments(self, instrument_keys: List[str], mode: str = "full") -> bool:
        """Subscribe to instruments for live data."""
        return self.ws_client.subscribe(instrument_keys, mode)
    
    def unsubscribe_instruments(self, instrument_keys: List[str]) -> bool:
        """Unsubscribe from instruments."""
        return self.ws_client.unsubscribe(instrument_keys)
    
    def get_live_ohlc(self, instrument_key: str, rows: int = 100) -> Optional[pd.DataFrame]:
        """Get latest OHLC data for an instrument."""
        if instrument_key in self.ohlc_data:
            ohlc = self.ohlc_data[instrument_key]
            return ohlc.tail(rows) if len(ohlc) > 0 else None
        return None
    
    def get_latest_tick(self, instrument_key: str) -> Optional[Dict]:
        """Get the latest tick for an instrument."""
        return self.ws_client.get_latest_tick(instrument_key)
    
    def get_connection_status(self) -> Dict:
        """Get connection status and statistics."""
        return {
            'status': self.connection_status,
            'connected': self.ws_client.is_connected,
            'subscribed_instruments': len(self.ws_client.subscribed_instruments),
            'total_ticks_received': self.total_ticks_received,
            'last_update': self.last_update_time,
            'instruments_with_data': len(self.ohlc_data)
        }
    
    def get_tick_statistics(self) -> Dict:
        """Get tick statistics for all instruments."""
        stats = {}
        for instrument_key, ticks in self.tick_buffer.items():
            if ticks:
                latest_tick = ticks[-1]
                ohlc_rows = len(self.ohlc_data.get(instrument_key, pd.DataFrame()))
                stats[instrument_key] = {
                    'tick_count': len(ticks),
                    'ohlc_rows': ohlc_rows,
                    'latest_price': latest_tick.get('ltp', 0),
                    'latest_volume': latest_tick.get('volume', 0),
                    'change_percent': latest_tick.get('change_percent', 0),
                    'last_update': latest_tick.get('timestamp')
                }
        return stats
    
    

    
    
    def pre_seed_historical_data(self, 
                                   instrument_keys: List[str], 
                                   days_back: int = 5) -> bool:
        """
        Pre-seed OHLC data with historical data from Upstox API.
        
        Args:
            instrument_keys: List of instrument keys to pre-seed
            days_back: Number of days of historical data to fetch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🌱 Pre-seeding historical data for {len(instrument_keys)} instruments...")
            
            success_count = 0
            for instrument_key in instrument_keys:
                print(f"\n📥 Fetching historical data for {instrument_key}...")
                
                # Fetch historical data
                historical_data = self.historical_client.get_historical_data(
                    instrument_key, 
                    interval="5minute", 
                    days_back=days_back
                )
                
                if historical_data is not None and len(historical_data) > 0:
                    # Store historical data
                    self.ohlc_data[instrument_key] = historical_data
                    self.is_seeded[instrument_key] = True
                    self.seed_data_count[instrument_key] = len(historical_data)
                    
                    print(f"✅ Pre-seeded {instrument_key} with {len(historical_data)} historical candles")
                    print(f"   Date range: {historical_data.index.min()} to {historical_data.index.max()}")
                    success_count += 1
                else:
                    print(f"❌ Failed to fetch historical data for {instrument_key}")
                
                # Rate limiting
                time.sleep(0.5)
            
            print(f"\n🌱 Pre-seeding complete: {success_count}/{len(instrument_keys)} instruments seeded")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error during pre-seeding: {e}")
            return False
    
    def pre_seed_nifty_instruments(self, days_back: int = 5) -> bool:
        """Pre-seed common Nifty instruments with historical data."""
        instruments = [
            "NSE_INDEX|Nifty 50",
            "NSE_INDEX|Nifty Bank"
        ]
        return self.pre_seed_historical_data(instruments, days_back)
    
    def get_seeding_status(self) -> Dict:
        """Get information about pre-seeding status."""
        total_seeded = sum(1 for seeded in self.is_seeded.values() if seeded)
        total_seed_rows = sum(self.seed_data_count.values())
        
        return {
            'is_seeded': total_seeded > 0,
            'seed_count': total_seeded,
            'total_seed_rows': total_seed_rows,
            'live_data_available': len(self.ohlc_data) > 0,
            'total_ohlc_rows': sum(len(df) for df in self.ohlc_data.values()),
            'instruments_seeded': [key for key, seeded in self.is_seeded.items() if seeded],
            'instruments_with_data': list(self.ohlc_data.keys()),
            'seed_details': {key: count for key, count in self.seed_data_count.items()}
        }
    
    def clear_historical_data(self, instrument_key: str = None):
        """Clear historical/pre-seeded data for instrument(s)."""
        if instrument_key:
            # Clear specific instrument
            if instrument_key in self.ohlc_data:
                del self.ohlc_data[instrument_key]
            if instrument_key in self.is_seeded:
                del self.is_seeded[instrument_key]
            if instrument_key in self.seed_data_count:
                del self.seed_data_count[instrument_key]
            print(f"🧹 Cleared historical data for {instrument_key}")
        else:
            # Clear all
            self.ohlc_data.clear()
            self.is_seeded.clear() 
            self.seed_data_count.clear()
            print("🧹 Cleared all historical data")
    
    def get_historical_client(self) -> UpstoxHistoricalClient:
        """Get the historical data client for direct access."""
        return self.historical_client
