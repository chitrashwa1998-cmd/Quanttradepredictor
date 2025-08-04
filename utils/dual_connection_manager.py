
import threading
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime
from utils.upstox_websocket import UpstoxWebSocketClient
from utils.live_data_manager import LiveDataManager

class DualConnectionManager:
    """Manages two separate WebSocket connections for different instrument types."""
    
    def __init__(self, access_token: str, api_key: str):
        """Initialize dual connection manager."""
        self.access_token = access_token
        self.api_key = api_key
        
        # Create two separate WebSocket clients
        self.ws_client_index = UpstoxWebSocketClient(access_token, api_key)
        self.ws_client_futures = UpstoxWebSocketClient(access_token, api_key)
        
        # Create two separate live data managers
        self.live_data_manager_index = LiveDataManager(access_token, api_key)
        self.live_data_manager_futures = LiveDataManager(access_token, api_key)
        
        # Override their WebSocket clients with our separate ones
        self.live_data_manager_index.ws_client = self.ws_client_index
        self.live_data_manager_futures.ws_client = self.ws_client_futures
        
        # Connection status
        self.is_connected = False
        self.connection_status = {
            'index_connected': False,
            'futures_connected': False
        }
        
        # Instrument routing
        self.index_instruments = set()
        self.futures_instruments = set()
        
        # Combined tick data storage
        self.combined_tick_buffer = {}
        self.combined_ohlc_data = {}
        
        # Callbacks
        self.tick_callback = None
        self.error_callback = None
        self.connection_callback = None
        
        # Setup callbacks for both managers
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup callbacks for both live data managers."""
        # Index connection callbacks
        self.live_data_manager_index.ws_client.set_callbacks(
            tick_callback=self._on_index_tick,
            error_callback=self._on_index_error,
            connection_callback=self._on_index_connection
        )
        
        # Futures connection callbacks
        self.live_data_manager_futures.ws_client.set_callbacks(
            tick_callback=self._on_futures_tick,
            error_callback=self._on_futures_error,
            connection_callback=self._on_futures_connection
        )
    
    def _on_index_tick(self, tick_data: Dict):
        """Handle tick data from index connection."""
        try:
            instrument_key = tick_data['instrument_token']
            
            # Store in combined buffer
            if instrument_key not in self.combined_tick_buffer:
                self.combined_tick_buffer[instrument_key] = []
            self.combined_tick_buffer[instrument_key].append(tick_data)
            
            # Forward to index live data manager
            self.live_data_manager_index.on_tick_received(tick_data)
            
            # Forward to main callback if set
            if self.tick_callback:
                self.tick_callback(tick_data)
                
            print(f"📊 Index Tick: {instrument_key.split('|')[-1]} @ ₹{tick_data.get('ltp', 0):.2f}")
            
        except Exception as e:
            print(f"❌ Error processing index tick: {e}")
    
    def _on_futures_tick(self, tick_data: Dict):
        """Handle tick data from futures connection."""
        try:
            instrument_key = tick_data['instrument_token']
            
            # Store in combined buffer
            if instrument_key not in self.combined_tick_buffer:
                self.combined_tick_buffer[instrument_key] = []
            self.combined_tick_buffer[instrument_key].append(tick_data)
            
            # Forward to futures live data manager
            self.live_data_manager_futures.on_tick_received(tick_data)
            
            # Forward to main callback if set
            if self.tick_callback:
                self.tick_callback(tick_data)
                
            # Enhanced logging for futures market depth
            bid_price = tick_data.get('bid_price', 0)
            ask_price = tick_data.get('ask_price', 0)
            bid_qty = tick_data.get('bid_qty', 0)
            ask_qty = tick_data.get('ask_qty', 0)
            total_buy = tick_data.get('total_buy_quantity', 0)
            total_sell = tick_data.get('total_sell_quantity', 0)
            
            print(f"📊 Futures Market Depth: {instrument_key.split('|')[-1]} @ ₹{tick_data.get('ltp', 0):.2f}")
            print(f"   🟢 Bid: ₹{bid_price:.2f}({bid_qty}) | 🔴 Ask: ₹{ask_price:.2f}({ask_qty})")
            print(f"   📈 Total Buy: {total_buy:,} | 📉 Total Sell: {total_sell:,}")
            
        except Exception as e:
            print(f"❌ Error processing futures tick: {e}")
    
    def _on_index_error(self, error):
        """Handle index connection errors."""
        print(f"❌ Index WebSocket error: {error}")
        if self.error_callback:
            self.error_callback(f"Index: {error}")
    
    def _on_futures_error(self, error):
        """Handle futures connection errors."""
        print(f"❌ Futures WebSocket error: {error}")
        if self.error_callback:
            self.error_callback(f"Futures: {error}")
    
    def _on_index_connection(self, status: str):
        """Handle index connection status changes."""
        self.connection_status['index_connected'] = (status == "connected")
        self._update_overall_connection_status()
        print(f"📊 Index connection: {status}")
    
    def _on_futures_connection(self, status: str):
        """Handle futures connection status changes."""
        self.connection_status['futures_connected'] = (status == "connected")
        self._update_overall_connection_status()
        print(f"📈 Futures connection: {status}")
    
    def _update_overall_connection_status(self):
        """Update overall connection status."""
        both_connected = (self.connection_status['index_connected'] and 
                         self.connection_status['futures_connected'])
        
        if both_connected != self.is_connected:
            self.is_connected = both_connected
            if self.connection_callback:
                status = "connected" if both_connected else "disconnected"
                self.connection_callback(status)
    
    def _categorize_instruments(self, instrument_keys: List[str]):
        """Categorize instruments into index and futures groups."""
        index_instruments = []
        futures_instruments = []
        
        for instrument in instrument_keys:
            if 'NSE_INDEX' in instrument:
                index_instruments.append(instrument)
                self.index_instruments.add(instrument)
            elif 'NSE_FO' in instrument or 'FUT' in instrument:
                futures_instruments.append(instrument)
                self.futures_instruments.add(instrument)
            else:
                # Default to index for other instruments
                index_instruments.append(instrument)
                self.index_instruments.add(instrument)
        
        return index_instruments, futures_instruments
    
    def connect(self) -> bool:
        """Connect both WebSocket clients."""
        print("🔄 Connecting dual WebSocket connections...")
        
        # Connect index WebSocket
        index_success = self.live_data_manager_index.connect()
        if index_success:
            print("✅ Index WebSocket connected")
        else:
            print("❌ Index WebSocket connection failed")
        
        # Small delay between connections
        time.sleep(1)
        
        # Connect futures WebSocket
        futures_success = self.live_data_manager_futures.connect()
        if futures_success:
            print("✅ Futures WebSocket connected")
        else:
            print("❌ Futures WebSocket connection failed")
        
        overall_success = index_success and futures_success
        
        if overall_success:
            print("✅ Dual WebSocket connections established successfully")
        else:
            print("⚠️ Partial connection - some features may be limited")
        
        return overall_success
    
    def disconnect(self):
        """Disconnect both WebSocket clients."""
        print("🔌 Disconnecting dual WebSocket connections...")
        
        self.live_data_manager_index.disconnect()
        self.live_data_manager_futures.disconnect()
        
        self.is_connected = False
        self.connection_status = {
            'index_connected': False,
            'futures_connected': False
        }
        
        print("🔌 Dual WebSocket connections disconnected")
    
    def subscribe_instruments(self, instrument_keys: List[str], mode: str = "full") -> bool:
        """Subscribe instruments to appropriate WebSocket connections."""
        print(f"🔄 Subscribing {len(instrument_keys)} instruments via dual connections...")
        
        # Categorize instruments
        index_instruments, futures_instruments = self._categorize_instruments(instrument_keys)
        
        success_count = 0
        total_attempts = 0
        
        # Subscribe index instruments
        if index_instruments:
            print(f"📊 Subscribing {len(index_instruments)} index instruments:")
            for inst in index_instruments:
                print(f"   - {inst.split('|')[-1]}")
            
            if self.live_data_manager_index.subscribe_instruments(index_instruments, mode):
                success_count += len(index_instruments)
                print(f"✅ Index instruments subscribed successfully")
            else:
                print(f"❌ Failed to subscribe index instruments")
            total_attempts += len(index_instruments)
        
        # Subscribe futures instruments
        if futures_instruments:
            print(f"📈 Subscribing {len(futures_instruments)} futures instruments:")
            for inst in futures_instruments:
                print(f"   - {inst.split('|')[-1]}")
            
            if self.live_data_manager_futures.subscribe_instruments(futures_instruments, mode):
                success_count += len(futures_instruments)
                print(f"✅ Futures instruments subscribed successfully")
            else:
                print(f"❌ Failed to subscribe futures instruments")
            total_attempts += len(futures_instruments)
        
        overall_success = success_count == total_attempts
        
        if overall_success:
            print(f"✅ All {total_attempts} instruments subscribed via dual connections")
        else:
            print(f"⚠️ Partial subscription: {success_count}/{total_attempts} instruments")
        
        return overall_success
    
    def get_live_ohlc(self, instrument_key: str, rows: int = 100) -> Optional[object]:
        """Get OHLC data from appropriate manager."""
        if instrument_key in self.index_instruments:
            return self.live_data_manager_index.get_live_ohlc(instrument_key, rows)
        elif instrument_key in self.futures_instruments:
            return self.live_data_manager_futures.get_live_ohlc(instrument_key, rows)
        else:
            # Try both managers
            data = self.live_data_manager_index.get_live_ohlc(instrument_key, rows)
            if data is None:
                data = self.live_data_manager_futures.get_live_ohlc(instrument_key, rows)
            return data
    
    def get_latest_tick(self, instrument_key: str) -> Optional[Dict]:
        """Get latest tick from appropriate manager."""
        if instrument_key in self.index_instruments:
            return self.live_data_manager_index.get_latest_tick(instrument_key)
        elif instrument_key in self.futures_instruments:
            return self.live_data_manager_futures.get_latest_tick(instrument_key)
        else:
            # Try both managers
            tick = self.live_data_manager_index.get_latest_tick(instrument_key)
            if tick is None:
                tick = self.live_data_manager_futures.get_latest_tick(instrument_key)
            return tick
    
    def get_connection_status(self) -> Dict:
        """Get combined connection status."""
        index_status = self.live_data_manager_index.get_connection_status()
        futures_status = self.live_data_manager_futures.get_connection_status()
        
        return {
            'status': 'connected' if self.is_connected else 'disconnected',
            'connected': self.is_connected,
            'index_connected': self.connection_status['index_connected'],
            'futures_connected': self.connection_status['futures_connected'],
            'subscribed_instruments': len(self.index_instruments) + len(self.futures_instruments),
            'index_instruments': len(self.index_instruments),
            'futures_instruments': len(self.futures_instruments),
            'total_ticks_received': (index_status.get('total_ticks_received', 0) + 
                                   futures_status.get('total_ticks_received', 0)),
            'instruments_with_data': (index_status.get('instruments_with_data', 0) + 
                                    futures_status.get('instruments_with_data', 0))
        }
    
    def get_tick_statistics(self) -> Dict:
        """Get combined tick statistics."""
        index_stats = self.live_data_manager_index.get_tick_statistics()
        futures_stats = self.live_data_manager_futures.get_tick_statistics()
        
        # Combine statistics
        combined_stats = {}
        combined_stats.update(index_stats)
        combined_stats.update(futures_stats)
        
        return combined_stats
    
    def set_callbacks(self, 
                     tick_callback: Optional[Callable] = None,
                     error_callback: Optional[Callable] = None,
                     connection_callback: Optional[Callable] = None):
        """Set callback functions."""
        self.tick_callback = tick_callback
        self.error_callback = error_callback
        self.connection_callback = connection_callback
    
    def get_seeding_status(self) -> Dict:
        """Get combined seeding status."""
        index_seeding = self.live_data_manager_index.get_seeding_status()
        futures_seeding = self.live_data_manager_futures.get_seeding_status()
        
        return {
            'is_seeded': index_seeding['is_seeded'] or futures_seeding['is_seeded'],
            'seed_count': index_seeding['seed_count'] + futures_seeding['seed_count'],
            'live_data_available': index_seeding['live_data_available'] or futures_seeding['live_data_available'],
            'total_ohlc_rows': index_seeding['total_ohlc_rows'] + futures_seeding['total_ohlc_rows'],
            'instruments_seeded': index_seeding['instruments_seeded'] + futures_seeding['instruments_seeded'],
            'seeding_details': {**index_seeding['seeding_details'], **futures_seeding['seeding_details']}
        }
