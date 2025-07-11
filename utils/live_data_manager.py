
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
                    print(f"📈 Live OHLC for {instrument_key}: {len(combined_ohlc)} total rows")
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
    
    

    
    
    def get_seeding_status(self) -> Dict:
        """Get information about live data status."""
        return {
            'is_seeded': False,
            'seed_count': 0,
            'live_data_available': len(self.ohlc_data) > 0,
            'total_ohlc_rows': sum(len(df) for df in self.ohlc_data.values()),
            'instruments_seeded': list(self.ohlc_data.keys())
        }
    
    def start_simulated_data(self, instruments: List[str]):
        """Start generating simulated tick data for testing when live data is not available."""
        import threading
        import random
        
        def generate_simulated_ticks():
            """Generate realistic simulated tick data."""
            base_prices = {
                'NSE_INDEX|Nifty 50': 24000.0,
                'NSE_INDEX|Nifty Bank': 51000.0,
                'NSE_EQ|INE002A01018': 2800.0,  # Reliance
                'NSE_EQ|INE467B01029': 4200.0   # TCS
            }
            
            current_prices = base_prices.copy()
            
            while self.connection_status == "connected":
                try:
                    for instrument in instruments:
                        if instrument in current_prices:
                            # Generate realistic price movement
                            change_pct = random.uniform(-0.002, 0.002)  # ±0.2% change
                            current_prices[instrument] *= (1 + change_pct)
                            
                            # Create simulated tick
                            tick = {
                                'instrument_token': instrument,
                                'timestamp': datetime.now(),
                                'ltp': current_prices[instrument],
                                'ltq': random.randint(50, 200),
                                'volume': random.randint(1000, 10000),
                                'bid_price': current_prices[instrument] * 0.9999,
                                'ask_price': current_prices[instrument] * 1.0001,
                                'bid_qty': random.randint(10, 100),
                                'ask_qty': random.randint(10, 100),
                                'open': current_prices[instrument],
                                'high': current_prices[instrument],
                                'low': current_prices[instrument],
                                'close': current_prices[instrument],
                                'change': change_pct * 100,
                                'change_percent': change_pct * 100
                            }
                            
                            # Process the simulated tick
                            self.on_tick_received(tick)
                            
                            print(f"🤖 Simulated tick: {instrument.split('|')[-1]} @ ₹{current_prices[instrument]:.2f}")
                    
                    time.sleep(2)  # Generate tick every 2 seconds
                    
                except Exception as e:
                    print(f"❌ Simulated data error: {e}")
                    time.sleep(5)
        
        # Start simulation in background thread
        self.simulation_thread = threading.Thread(target=generate_simulated_ticks, daemon=True)
        self.simulation_thread.start()
        print("🤖 Started simulated tick data generation")
