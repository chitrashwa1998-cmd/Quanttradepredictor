
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import time
from collections import deque
from utils.upstox_websocket import UpstoxWebSocketClient

class LiveDataManager:
    """Manage real-time tick data and convert to OHLC format."""
    
    def __init__(self, access_token: str, api_key: str):
        """Initialize live data manager."""
        self.ws_client = UpstoxWebSocketClient(access_token, api_key)
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
        
        # Seed with historical data for immediate predictions
        self._seed_historical_data()
        
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
        """Convert tick data to OHLC format and seamlessly blend with historical data."""
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
                
                # Seamlessly blend with existing data (historical + live)
                if instrument_key in self.ohlc_data and len(self.ohlc_data[instrument_key]) > 0:
                    existing_ohlc = self.ohlc_data[instrument_key]
                    
                    # Combine historical data with new live data
                    combined_ohlc = pd.concat([existing_ohlc, new_ohlc])
                    
                    # Remove duplicate timestamps, keeping the latest (live data wins)
                    combined_ohlc = combined_ohlc[~combined_ohlc.index.duplicated(keep='last')]
                    
                    # Sort by timestamp
                    combined_ohlc = combined_ohlc.sort_index()
                    
                    # Keep last 500 rows (preserve both historical seed and live data)
                    if len(combined_ohlc) > 500:
                        # Keep some historical data + all recent live data
                        historical_keep = 100  # Keep 100 historical rows
                        live_start = len(combined_ohlc) - 400  # Keep 400 most recent
                        keep_start = max(0, min(historical_keep, live_start))
                        combined_ohlc = combined_ohlc.iloc[keep_start:]
                    
                    self.ohlc_data[instrument_key] = combined_ohlc
                    
                    # Determine if this is historical or live data update
                    is_historical_seed = hasattr(self, '_historical_seed_count') and len(combined_ohlc) <= self._historical_seed_count + 10
                    update_type = "🌱 Historical+Live" if not is_historical_seed else "📈 Live"
                    
                    print(f"{update_type} OHLC for {instrument_key}: {len(combined_ohlc)} total rows")
                else:
                    # First time - store new data (should not happen if seeded)
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
    
    def _seed_historical_data(self):
        """Seed OHLC data with historical data from database for immediate predictions."""
        try:
            from utils.database_adapter import DatabaseAdapter
            
            print("🌱 Seeding historical data for immediate predictions...")
            db = DatabaseAdapter()
            historical_data = db.load_ohlc_data("main_dataset")
            
            if historical_data is not None and len(historical_data) > 100:
                # Take the last 200 rows for seeding (enough for technical indicators)
                seed_data = historical_data.tail(200).copy()
                
                # Ensure proper 5-minute timeframe and standard column names
                if not all(col in seed_data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    # Try to map columns
                    col_mapping = {}
                    for col in seed_data.columns:
                        col_lower = col.lower()
                        if col_lower in ['open', 'o']:
                            col_mapping[col] = 'Open'
                        elif col_lower in ['high', 'h']:
                            col_mapping[col] = 'High'
                        elif col_lower in ['low', 'l']:
                            col_mapping[col] = 'Low'
                        elif col_lower in ['close', 'c']:
                            col_mapping[col] = 'Close'
                        elif col_lower in ['volume', 'vol', 'v']:
                            col_mapping[col] = 'Volume'
                    
                    seed_data = seed_data.rename(columns=col_mapping)
                
                # Ensure we have Volume column
                if 'Volume' not in seed_data.columns:
                    seed_data['Volume'] = 1000  # Default volume
                
                # Resample to 5-minute intervals if needed
                if len(seed_data) > 500:  # If data is too granular, resample
                    seed_data = seed_data.resample('5T').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                
                # Store for Nifty 50 (our main instrument)
                instrument_key = "NSE_INDEX|Nifty 50"
                self.ohlc_data[instrument_key] = seed_data
                
                print(f"✅ Seeded {len(seed_data)} historical OHLC rows for {instrument_key}")
                print(f"📅 Historical data range: {seed_data.index[0]} to {seed_data.index[-1]}")
                
                # Mark as seeded to distinguish from live data
                self._historical_seed_count = len(seed_data)
                
            else:
                print("⚠️ No sufficient historical data found for seeding")
                
        except Exception as e:
            print(f"❌ Error seeding historical data: {e}")
            # Continue without seeding - system will work normally but require wait time

    def bootstrap_ohlc_from_ticks(self, instrument_key: str):
        """Create initial OHLC data from recent ticks for faster predictions."""
        if instrument_key not in self.tick_buffer or len(self.tick_buffer[instrument_key]) < 20:
            return
        
        try:
            # Get all ticks
            ticks = list(self.tick_buffer[instrument_key])
            
            # Create DataFrame
            df = pd.DataFrame(ticks)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Create 1-minute OHLC first, then resample to 5-minute
            ohlc_1m = df['ltp'].resample('1T').ohlc()
            ohlc_1m['volume'] = df['volume'].resample('1T').sum()
            ohlc_1m = ohlc_1m.dropna()
            
            if len(ohlc_1m) >= 5:
                # Resample to 5-minute
                ohlc_5m = ohlc_1m['open'].resample('5T').first()
                ohlc_5m = pd.DataFrame({
                    'Open': ohlc_1m['open'].resample('5T').first(),
                    'High': ohlc_1m['high'].resample('5T').max(),
                    'Low': ohlc_1m['low'].resample('5T').min(),
                    'Close': ohlc_1m['close'].resample('5T').last(),
                    'Volume': ohlc_1m['volume'].resample('5T').sum()
                }).dropna()
                
                if len(ohlc_5m) > 0:
                    self.ohlc_data[instrument_key] = ohlc_5m
                    print(f"🚀 Bootstrapped {len(ohlc_5m)} OHLC rows for {instrument_key}")
                    
        except Exception as e:
            print(f"Error bootstrapping OHLC data: {e}")
    
    def get_seeding_status(self) -> Dict:
        """Get information about historical data seeding."""
        return {
            'is_seeded': hasattr(self, '_historical_seed_count'),
            'seed_count': getattr(self, '_historical_seed_count', 0),
            'live_data_available': len(self.ohlc_data) > 0,
            'total_ohlc_rows': sum(len(df) for df in self.ohlc_data.values()),
            'instruments_seeded': list(self.ohlc_data.keys())
        }
