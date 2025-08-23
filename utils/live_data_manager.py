import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import time
from collections import deque
from utils.upstox_websocket import UpstoxWebSocketClient
import pytz

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
        
        # Continuation tracking
        self.seeded_instruments = {}  # Track which instruments were seeded from database

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

            # Ensure timestamps are in IST
            ist = pytz.timezone('Asia/Kolkata')
            if df['timestamp'].dt.tz is None:
                # If no timezone info, assume UTC and convert to IST
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ist)
            elif df['timestamp'].dt.tz != ist:
                # If different timezone, convert to IST
                df['timestamp'] = df['timestamp'].dt.tz_convert(ist)

            # Convert timezone-aware to timezone-naive for clean timestamps
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            df = df.set_index('timestamp')

            # Get existing data first
            existing_ohlc = self.ohlc_data.get(instrument_key, pd.DataFrame())
            
            # If we have seeded data, use a different approach
            if instrument_key in self.seeded_instruments and len(existing_ohlc) > 0:
                # For seeded instruments, create live candles based on current time
                current_time = df.index[-1]  # Latest tick time
                
                # Round down to nearest 5-minute interval
                current_candle_time = current_time.floor('5T')
                
                # Ensure we're working with the full seeded dataset
                if len(existing_ohlc) < self.seeded_instruments[instrument_key]['seed_count']:
                    # Re-seed if data was lost
                    self.seed_live_data_from_database(instrument_key)
                    existing_ohlc = self.ohlc_data.get(instrument_key, pd.DataFrame())
                
                # Check if we already have a candle for this time period
                if current_candle_time in existing_ohlc.index:
                    # Update existing candle with new tick data
                    existing_row = existing_ohlc.loc[current_candle_time]
                    
                    # Get all ticks for this time period
                    period_ticks = df[df.index >= current_candle_time]
                    if len(period_ticks) > 0:
                        # Update the existing candle
                        updated_high = max(existing_row['High'], period_ticks['ltp'].max())
                        updated_low = min(existing_row['Low'], period_ticks['ltp'].min())
                        updated_close = period_ticks['ltp'].iloc[-1]  # Latest price
                        updated_volume = existing_row['Volume'] + period_ticks['volume'].sum()
                        
                        # Update the candle
                        existing_ohlc.loc[current_candle_time] = {
                            'Open': existing_row['Open'],
                            'High': updated_high,
                            'Low': updated_low,
                            'Close': updated_close,
                            'Volume': updated_volume
                        }
                        
                        self.ohlc_data[instrument_key] = existing_ohlc
                        
                        # Only show update message occasionally to reduce noise
                        if not hasattr(self, '_update_counter'):
                            self._update_counter = {}
                        if instrument_key not in self._update_counter:
                            self._update_counter[instrument_key] = 0
                        self._update_counter[instrument_key] += 1
                        
                        if self._update_counter[instrument_key] % 10 == 0:  # Show every 10th update
                            seed_count = self.seeded_instruments[instrument_key]['seed_count']
                            total_rows = len(existing_ohlc)
                            live_count = max(0, total_rows - seed_count)
                            print(f"📈 Updated candle for {instrument_key}: {total_rows} total rows ({seed_count} seeded + {live_count} live)")
                else:
                    # Create new candle for this time period
                    period_ticks = df[df.index >= current_candle_time]
                    if len(period_ticks) > 0:
                        new_candle = pd.DataFrame({
                            'Open': [period_ticks['ltp'].iloc[0]],
                            'High': [period_ticks['ltp'].max()],
                            'Low': [period_ticks['ltp'].min()],
                            'Close': [period_ticks['ltp'].iloc[-1]],
                            'Volume': [period_ticks['volume'].sum()]
                        }, index=[current_candle_time])
                        
                        # Append to existing data
                        combined_ohlc = pd.concat([existing_ohlc, new_candle])
                        combined_ohlc = combined_ohlc.sort_index()
                        
                        # Keep reasonable limits but protect seeded data
                        if instrument_key in self.seeded_instruments:
                            original_seed_count = self.seeded_instruments[instrument_key]['seed_count']
                            max_rows = max(300, original_seed_count + 50)  # Allow 50 new live candles beyond seed
                            
                            if len(combined_ohlc) > max_rows:
                                # Only trim if we exceed reasonable limits, but keep all seeded data
                                trim_to_rows = max(original_seed_count + 10, max_rows - 20)
                                combined_ohlc = combined_ohlc.tail(trim_to_rows)
                        else:
                            max_rows = 100
                            if len(combined_ohlc) > max_rows:
                                combined_ohlc = combined_ohlc.tail(max_rows)
                        
                        self.ohlc_data[instrument_key] = combined_ohlc
                        
                        seed_count = self.seeded_instruments[instrument_key]['seed_count']
                        total_rows = len(combined_ohlc)
                        live_count = max(0, total_rows - seed_count)
                        print(f"🕐 NEW 5-MINUTE CANDLE CREATED for {instrument_key}: {current_candle_time}")
                        print(f"📈 Total data: {total_rows} rows ({seed_count} seeded + {live_count} live)")
                        print(f"💡 This candle is now ready for prediction processing!")
                
            else:
                # Standard resampling for non-seeded instruments
                new_ohlc = df['ltp'].resample(timeframe).ohlc()
                new_ohlc['volume'] = df['volume'].resample(timeframe).sum()

                # Remove NaN values
                new_ohlc = new_ohlc.dropna()

                if len(new_ohlc) > 0:
                    # Rename columns to match existing format
                    new_ohlc.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

                    # Combine with existing data
                    if len(existing_ohlc) > 0:
                        # Only add new OHLC rows that don't already exist
                        new_timestamps = set(new_ohlc.index)
                        existing_timestamps = set(existing_ohlc.index)
                        truly_new_timestamps = new_timestamps - existing_timestamps

                        if truly_new_timestamps:
                            # Only add truly new data
                            new_data_to_add = new_ohlc.loc[list(truly_new_timestamps)]
                            combined_ohlc = pd.concat([existing_ohlc, new_data_to_add])
                            
                            # Sort by timestamp
                            combined_ohlc = combined_ohlc.sort_index()

                            # Keep standard limit
                            max_rows = 100
                            if len(combined_ohlc) > max_rows:
                                combined_ohlc = combined_ohlc.tail(max_rows)

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

    def subscribe_instruments(self, instrument_keys: List[str], mode: str = "full_d30") -> bool:
        """Subscribe to instruments for live data with full_d30 mode (30 market levels)."""
        # First, try to seed each instrument from database
        for instrument_key in instrument_keys:
            self.seed_live_data_from_database(instrument_key)
        
        # Then subscribe for live updates with full_d30 mode
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

    def get_complete_ohlc_data(self, instrument_key: str) -> Optional[pd.DataFrame]:
        """Get complete OHLC data for an instrument (all seeded + live data)."""
        if instrument_key in self.ohlc_data:
            return self.ohlc_data[instrument_key]
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





    def seed_live_data_from_database(self, instrument_key: str) -> bool:
        """Seed live OHLC data from database for continuation."""
        try:
            from utils.database_adapter import DatabaseAdapter
            
            # Convert instrument key to database dataset name
            # Use livenifty50 as primary, fallback to pre_seed_dataset
            db_test = DatabaseAdapter(use_row_based=True)
            datasets = db_test.get_dataset_list()
            dataset_names = [d['name'] for d in datasets]
            
            if "livenifty50" in dataset_names:
                dataset_name = "livenifty50"
            elif "pre_seed_dataset" in dataset_names:
                dataset_name = "pre_seed_dataset"
            else:
                dataset_name = "livenifty50"
            
            # Use row-based storage for better performance
            db = DatabaseAdapter(use_row_based=True)
            
            # Try row-based first, fallback to blob-based
            try:
                historical_data = db.get_latest_rows(dataset_name, 250)
            except:
                # Fallback to blob-based storage
                db = DatabaseAdapter(use_row_based=False)
                historical_data = db.load_ohlc_data(dataset_name)
                if historical_data is not None and len(historical_data) > 250:
                    historical_data = historical_data.tail(250)
            
            if historical_data is not None and len(historical_data) > 0:
                # Use the most recent data (last 250 rows for performance)
                seed_data = historical_data.tail(250).copy()
                
                # Ensure the data has the correct column names
                if all(col in seed_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    # Store as foundation for live data
                    self.ohlc_data[instrument_key] = seed_data
                    self.seeded_instruments[instrument_key] = {
                        'seed_count': len(seed_data),
                        'seed_date_range': f"{seed_data.index.min()} to {seed_data.index.max()}",
                        'seeded_at': pd.Timestamp.now()
                    }
                    
                    print(f"🌱 SEEDED {instrument_key} with {len(seed_data)} historical OHLC rows from database")
                    print(f"📈 Foundation set: {len(seed_data)} rows ready for live continuation")
                    return True
                else:
                    print(f"⚠️ Database data for {dataset_name} missing required columns")
                    return False
            else:
                print(f"📊 No historical data found for {dataset_name}, starting fresh")
                return False
                
        except Exception as e:
            print(f"❌ Error seeding data for {instrument_key}: {str(e)}")
            return False

    def get_seeding_status(self) -> Dict:
        """Get information about live data status."""
        total_seeded = sum(info['seed_count'] for info in self.seeded_instruments.values())
        return {
            'is_seeded': len(self.seeded_instruments) > 0,
            'seed_count': total_seeded,
            'live_data_available': len(self.ohlc_data) > 0,
            'total_ohlc_rows': sum(len(df) for df in self.ohlc_data.values()),
            'instruments_seeded': list(self.seeded_instruments.keys()),
            'seeding_details': self.seeded_instruments
        }

    def get_new_live_data_only(self, instrument_key: str) -> Optional[pd.DataFrame]:
        """Get only the new live-generated data (excluding seeded data) for an instrument."""
        if instrument_key not in self.ohlc_data:
            return None
            
        current_data = self.ohlc_data[instrument_key]
        
        if instrument_key in self.seeded_instruments:
            # For seeded instruments, return only data beyond the seed count
            seed_count = self.seeded_instruments[instrument_key]['seed_count']
            if len(current_data) > seed_count:
                return current_data.iloc[seed_count:].copy()
            else:
                # No new data generated yet
                return pd.DataFrame()
        else:
            # For non-seeded instruments, all data is new
            return current_data.copy() if current_data is not None else None
    
    @staticmethod
    def get_continuation_dataset_name(instrument_key: str) -> str:
        """Get the dataset name needed for continuation for a given instrument."""
        return "livenifty50"
    
    @staticmethod
    def get_continuation_info() -> Dict[str, str]:
        """Get continuation dataset names for common instruments."""
        common_instruments = {
            "NSE_INDEX|Nifty 50": "livenifty50",
            "NSE_INDEX|Nifty Bank": "liveniftybank", 
            "NSE_EQ|INE002A01018": "livereliance",  # Reliance
            "NSE_EQ|INE467B01029": "livetcs",  # TCS
            "NSE_EQ|INE040A01034": "livehdfcbank",  # HDFC Bank
            "NSE_EQ|INE009A01021": "liveinfosys"   # Infosys
        }
        return common_instruments