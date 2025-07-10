
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
        """Convert tick data to OHLC format with proper 5-minute alignment."""
        try:
            ticks = list(self.tick_buffer[instrument_key])
            if not ticks:
                return
            
            # Create DataFrame from ticks
            df = pd.DataFrame(ticks)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Resample to OHLC format starting from market opening time (9:15 AM)
            # This ensures candles align to 9:15, 9:20, 9:25, etc.
            new_ohlc = df['ltp'].resample(timeframe, origin='2024-01-01 09:15:00').ohlc()
            new_ohlc['volume'] = df['volume'].resample(timeframe, origin='2024-01-01 09:15:00').sum()
            
            # Remove NaN values and incomplete candles
            new_ohlc = new_ohlc.dropna()
            
            # Only keep complete 5-minute candles (filter out partial candles)
            if len(new_ohlc) > 0:
                current_time = pd.Timestamp.now()
                # Remove the last candle if it's still forming (within current 5-minute period)
                last_candle_time = new_ohlc.index[-1]
                current_5min_boundary = current_time.floor('5T')
                
                if last_candle_time >= current_5min_boundary:
                    new_ohlc = new_ohlc[:-1]  # Remove incomplete candle
                    
                if len(new_ohlc) > 0:
                    # Rename columns to match existing format
                    new_ohlc.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # Combine with existing live data
                    if instrument_key in self.ohlc_data and len(self.ohlc_data[instrument_key]) > 0:
                        existing_ohlc = self.ohlc_data[instrument_key]
                        
                        # Combine with new live data
                        combined_ohlc = pd.concat([existing_ohlc, new_ohlc])
                        
                        # Remove duplicate timestamps, keeping the latest
                        combined_ohlc = combined_ohlc[~combined_ohlc.index.duplicated(keep='last')]
                        
                        # Sort by timestamp
                        combined_ohlc = combined_ohlc.sort_index()
                        
                        # Keep last 100 rows maximum for live data
                        if len(combined_ohlc) > 100:
                            combined_ohlc = combined_ohlc.tail(100)
                        
                        self.ohlc_data[instrument_key] = combined_ohlc
                        
                        # Only log when we get a complete new candle
                        if len(new_ohlc) > 0:
                            latest_candle_time = combined_ohlc.index[-1]
                            print(f"📈 Complete 5-min candle for {instrument_key}: {latest_candle_time.strftime('%H:%M:%S')} | Total rows: {len(combined_ohlc)}")
                    else:
                        # First time - store new data
                        self.ohlc_data[instrument_key] = new_ohlc
                        if len(new_ohlc) > 0:
                            latest_candle_time = new_ohlc.index[-1]
                            print(f"📈 Initial complete candle for {instrument_key}: {latest_candle_time.strftime('%H:%M:%S')} | Rows: {len(new_ohlc)}")
                
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
    
    

    
    
    def get_seeding_status(self) -> Dict:
        """Get information about live data status."""
        return {
            'is_seeded': False,
            'seed_count': 0,
            'live_data_available': len(self.ohlc_data) > 0,
            'total_ohlc_rows': sum(len(df) for df in self.ohlc_data.values()),
            'instruments_seeded': list(self.ohlc_data.keys())
        }
