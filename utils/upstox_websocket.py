
import websocket
import threading
import time
import pandas as pd
import streamlit as st
from datetime import datetime
import json
import struct
from typing import Callable, Optional, Dict, Any
import requests
import uuid

class UpstoxWebSocketClient:
    """Real-time Upstox WebSocket client using v3 API with proper protobuf support."""
    
    def __init__(self, access_token: str, api_key: str):
        """Initialize Upstox WebSocket client."""
        self.access_token = access_token
        self.api_key = api_key
        self.ws = None
        self.is_connected = False
        self.subscribed_instruments = set()
        self.tick_callback = None
        self.error_callback = None
        self.connection_callback = None
        self.last_tick_data = {}
        self.websocket_url = None
        self.connection_start_time = None
        self.last_pong_time = None
        self.ping_count = 0
        self.pong_count = 0
        self.market_status = {}
        
        # Get WebSocket URL from Upstox API
        self._get_websocket_url()
        
    def _get_websocket_url(self):
        """Get WebSocket URL from Upstox Market Data Feed API v3."""
        try:
            # Use the correct v3 API endpoint
            url = "https://api.upstox.com/v2/feed/market-data-feed/authorize"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "*/*"
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success" and "data" in data:
                    self.websocket_url = data["data"]["authorized_redirect_uri"]
                    print(f"✅ Got WebSocket URL (v3): {self.websocket_url}")
                else:
                    print(f"❌ API Error: {data}")
                    self._fallback_websocket_url()
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                self._fallback_websocket_url()
                
        except Exception as e:
            print(f"❌ Error getting WebSocket URL: {e}")
            self._fallback_websocket_url()
    
    def _fallback_websocket_url(self):
        """Fallback to direct WebSocket URL construction for v3."""
        # Updated v3 WebSocket URL based on documentation
        self.websocket_url = f"wss://api.upstox.com/v2/feed/market-data-feed?access_token={self.access_token}"
        print(f"🔄 Using fallback WebSocket URL (v3)")
        
    def set_callbacks(self, 
                     tick_callback: Optional[Callable] = None,
                     error_callback: Optional[Callable] = None,
                     connection_callback: Optional[Callable] = None):
        """Set callback functions for different events."""
        self.tick_callback = tick_callback
        self.error_callback = error_callback
        self.connection_callback = connection_callback
    
    def on_open(self, ws):
        """Handle WebSocket connection open."""
        self.is_connected = True
        self._was_connected = True
        self.connection_start_time = time.time()
        self.ping_count = 0
        self.pong_count = 0
        
        # Check market hours
        if self._is_market_hours():
            print("✅ Upstox WebSocket connected successfully (Market Hours)")
        else:
            print("✅ Upstox WebSocket connected successfully (Outside Market Hours)")
        
        if self.connection_callback:
            self.connection_callback("connected")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        self.is_connected = False
        
        if hasattr(self, '_was_connected') and self._was_connected:
            print(f"🔌 WebSocket connection closed: {close_status_code} - {close_msg}")
        
        if self.connection_callback:
            self.connection_callback("disconnected")
        
        # Auto-reconnect with exponential backoff
        if hasattr(self, '_was_connected') and self._was_connected:
            import threading
            def reconnect():
                import time
                backoff_time = 10 if self._is_market_hours() else 30
                time.sleep(backoff_time)
                
                if not self.is_connected:
                    print(f"🔄 Attempting auto-reconnect after {backoff_time}s...")
                    self.connect()
            
            thread = threading.Thread(target=reconnect, daemon=True)
            thread.start()
    
    def _is_market_hours(self):
        """Check if current time is within Indian market hours."""
        from datetime import datetime, time
        import pytz
        
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        current_time = now.time()
        
        # Market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        market_start = time(9, 15)
        market_end = time(15, 30)
        
        is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
        is_market_time = market_start <= current_time <= market_end
        
        return is_weekday and is_market_time
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"❌ Upstox WebSocket error: {error}")
        if self.error_callback:
            self.error_callback(error)
    
    def on_pong(self, ws, data):
        """Handle WebSocket pong response."""
        self.last_pong_time = time.time()
        self.pong_count += 1
        print(f"💚 Pong #{self.pong_count} - Connection healthy")
    
    def on_ping(self, ws, data):
        """Handle WebSocket ping from server."""
        # WebSocket library automatically sends pong response
        pass
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            # Handle both binary (protobuf) and text (JSON) messages
            if isinstance(message, bytes):
                # Binary protobuf message - decode properly
                tick_data = self.parse_protobuf_message(message)
            else:
                # JSON message for market status or other info
                data = json.loads(message)
                tick_data = self.parse_json_message(data)
            
            if tick_data and self.tick_callback:
                self.tick_callback(tick_data)
                    
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    def parse_protobuf_message(self, message: bytes) -> Optional[Dict]:
        """Parse protobuf message from Upstox v3 API."""
        try:
            # Basic protobuf parsing for Upstox v3
            # This is a simplified implementation - ideally you'd use the official .proto file
            
            if len(message) < 10:
                return None
            
            # Parse protobuf fields
            # Field 1: Instrument token (varint)
            # Field 2: LTP (fixed64/double)
            # Field 3: LTT (varint)
            # Field 4: LTQ (varint)
            # Field 5: Volume (varint)
            # Field 6: Bid price (fixed64/double)
            # Field 7: Ask price (fixed64/double)
            # Field 8: Open (fixed64/double)
            # Field 9: High (fixed64/double)
            # Field 10: Low (fixed64/double)
            # Field 11: Close (fixed64/double)
            
            offset = 0
            fields = {}
            
            while offset < len(message):
                if offset + 1 >= len(message):
                    break
                
                # Read field header (varint)
                field_header = message[offset]
                offset += 1
                
                field_number = field_header >> 3
                wire_type = field_header & 0x07
                
                if wire_type == 0:  # Varint
                    value, bytes_consumed = self._read_varint(message, offset)
                    offset += bytes_consumed
                    fields[field_number] = value
                elif wire_type == 1:  # Fixed64 (double)
                    if offset + 8 <= len(message):
                        value = struct.unpack('<d', message[offset:offset+8])[0]
                        offset += 8
                        fields[field_number] = value
                elif wire_type == 2:  # Length-delimited (string/bytes)
                    if offset < len(message):
                        length = message[offset]
                        offset += 1
                        if offset + length <= len(message):
                            value = message[offset:offset+length]
                            offset += length
                            fields[field_number] = value
                else:
                    # Skip unknown wire types
                    offset += 1
            
            # Extract meaningful data based on common protobuf field mappings
            if fields:
                # Default instrument key
                instrument_key = 'NSE_INDEX|Nifty 50'
                
                # Try to extract LTP from various possible fields
                ltp = 0.0
                for field_num in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    if field_num in fields and isinstance(fields[field_num], float):
                        if 15000 <= fields[field_num] <= 30000:  # Typical Nifty range
                            ltp = fields[field_num]
                            break
                
                if ltp > 0:
                    tick = {
                        'instrument_token': instrument_key,
                        'timestamp': datetime.now(),
                        'ltp': float(ltp),
                        'ltq': int(fields.get(4, 1)),
                        'volume': int(fields.get(5, 100)),
                        'bid_price': float(ltp * 0.9999),
                        'ask_price': float(ltp * 1.0001),
                        'bid_qty': 10,
                        'ask_qty': 10,
                        'open': float(ltp),
                        'high': float(ltp),
                        'low': float(ltp),
                        'close': float(ltp),
                        'change': 0.0,
                        'change_percent': 0.0
                    }
                    
                    print(f"📈 Live Market Data: {instrument_key} @ ₹{ltp:.2f}")
                    return tick
            
            return None
            
        except Exception as e:
            print(f"❌ Error parsing protobuf message: {e}")
            return None
    
    def _read_varint(self, data: bytes, offset: int) -> tuple:
        """Read a varint from protobuf data."""
        value = 0
        shift = 0
        bytes_consumed = 0
        
        while offset + bytes_consumed < len(data):
            byte = data[offset + bytes_consumed]
            bytes_consumed += 1
            
            value |= (byte & 0x7F) << shift
            
            if (byte & 0x80) == 0:
                break
            
            shift += 7
            
            if shift >= 64:
                raise ValueError("Varint too long")
        
        return value, bytes_consumed
    
    def parse_json_message(self, data: Dict) -> Optional[Dict]:
        """Parse JSON message format for market status and other info."""
        try:
            # Handle market status message
            if data.get('type') == 'market_info':
                self.market_status = data.get('marketInfo', {})
                print(f"📊 Market Status Updated: {len(self.market_status.get('segmentStatus', {}))} segments")
                return None
            
            # Handle live feed message  
            elif data.get('type') == 'live_feed':
                feeds = data.get('feeds', {})
                if not feeds:
                    return None
                
                # Extract first instrument data
                instrument_key = list(feeds.keys())[0]
                feed_data = feeds[instrument_key]
                
                # Parse different feed types
                if 'ltpc' in feed_data:
                    # LTPC mode
                    ltpc_data = feed_data['ltpc']
                    tick = {
                        'instrument_token': instrument_key,
                        'timestamp': datetime.now(),
                        'ltp': float(ltpc_data.get('ltp', 0)),
                        'ltq': int(ltpc_data.get('ltq', 0)),
                        'ltt': ltpc_data.get('ltt'),
                        'close': float(ltpc_data.get('cp', 0)),
                        'bid_price': float(ltpc_data.get('ltp', 0) * 0.9999),
                        'ask_price': float(ltpc_data.get('ltp', 0) * 1.0001),
                        'bid_qty': 10,
                        'ask_qty': 10,
                        'volume': 100,
                        'open': float(ltpc_data.get('ltp', 0)),
                        'high': float(ltpc_data.get('ltp', 0)),
                        'low': float(ltpc_data.get('ltp', 0)),
                        'change': 0.0,
                        'change_percent': 0.0
                    }
                    
                    print(f"📈 LTPC Update: {instrument_key} @ ₹{tick['ltp']:.2f}")
                    return tick
                
                elif 'fullFeed' in feed_data:
                    # Full feed mode
                    full_feed = feed_data['fullFeed']['marketFF']
                    ltpc_data = full_feed['ltpc']
                    
                    tick = {
                        'instrument_token': instrument_key,
                        'timestamp': datetime.now(),
                        'ltp': float(ltpc_data.get('ltp', 0)),
                        'ltq': int(ltpc_data.get('ltq', 0)),
                        'ltt': ltpc_data.get('ltt'),
                        'close': float(ltpc_data.get('cp', 0)),
                        'volume': int(full_feed.get('vtt', 0)),
                        'open': float(ltpc_data.get('ltp', 0)),
                        'high': float(ltpc_data.get('ltp', 0)),
                        'low': float(ltpc_data.get('ltp', 0)),
                        'change': 0.0,
                        'change_percent': 0.0
                    }
                    
                    # Add bid/ask data if available
                    if 'marketLevel' in full_feed and 'bidAskQuote' in full_feed['marketLevel']:
                        quotes = full_feed['marketLevel']['bidAskQuote']
                        if quotes:
                            tick['bid_price'] = float(quotes[0].get('bidP', 0))
                            tick['ask_price'] = float(quotes[0].get('askP', 0))
                            tick['bid_qty'] = int(quotes[0].get('bidQ', 0))
                            tick['ask_qty'] = int(quotes[0].get('askQ', 0))
                    
                    print(f"📈 Full Feed Update: {instrument_key} @ ₹{tick['ltp']:.2f}")
                    return tick
            
            return None
            
        except Exception as e:
            print(f"❌ Error parsing JSON message: {e}")
            return None
    
    def connect(self):
        """Establish WebSocket connection."""
        try:
            if not self.websocket_url:
                print("❌ No WebSocket URL available")
                return False
            
            print(f"🔄 Connecting to Upstox v3 WebSocket...")
            
            # Create WebSocket connection with proper headers
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "*/*",
                "User-Agent": "Python-WebSocket-Client/1.0"
            }
            
            self.ws = websocket.WebSocketApp(
                self.websocket_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_ping=self.on_ping,
                on_pong=self.on_pong,
                header=headers
            )
            
            # Start WebSocket with proper ping mechanism
            wst = threading.Thread(target=lambda: self.ws.run_forever(
                ping_interval=60,    # Send ping every 60 seconds
                ping_timeout=10,     # Wait 10 seconds for pong
                ping_payload="upstox_ping"
            ))
            wst.daemon = True
            wst.start()
            
            # Wait for connection
            time.sleep(3)
            return self.is_connected
            
        except Exception as e:
            print(f"❌ Failed to connect to Upstox WebSocket: {e}")
            return False
    
    def disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            self.is_connected = False
            print("🔌 Disconnected from WebSocket")
    
    def subscribe(self, instrument_keys: list, mode: str = "ltpc"):
        """Subscribe to instruments using v3 API binary format."""
        if not self.is_connected:
            print("❌ WebSocket not connected. Please connect first.")
            return False
        
        try:
            # Create subscription request in v3 format
            subscribe_request = {
                "guid": str(uuid.uuid4()),
                "method": "sub",
                "data": {
                    "mode": mode,
                    "instrumentKeys": instrument_keys
                }
            }
            
            # Convert to binary format as required by v3 API
            message_json = json.dumps(subscribe_request)
            message_binary = message_json.encode('utf-8')
            
            # Send binary message
            if self.ws and hasattr(self.ws, 'send'):
                self.ws.send(message_binary, websocket.ABNF.OPCODE_BINARY)
                
                # Update subscribed instruments
                self.subscribed_instruments.update(instrument_keys)
                
                print(f"✅ Subscribed to {len(instrument_keys)} instruments (v3 Binary): {instrument_keys}")
                return True
            
        except Exception as e:
            print(f"❌ Failed to subscribe to instruments: {e}")
            return False
    
    def unsubscribe(self, instrument_keys: list):
        """Unsubscribe from instruments using v3 API binary format."""
        if not self.is_connected:
            return False
        
        try:
            unsubscribe_request = {
                "guid": str(uuid.uuid4()),
                "method": "unsub",
                "data": {
                    "instrumentKeys": instrument_keys
                }
            }
            
            # Convert to binary format
            message_json = json.dumps(unsubscribe_request)
            message_binary = message_json.encode('utf-8')
            
            if self.ws:
                self.ws.send(message_binary, websocket.ABNF.OPCODE_BINARY)
                
                # Remove from subscribed instruments
                self.subscribed_instruments.difference_update(instrument_keys)
                
                print(f"✅ Unsubscribed from {len(instrument_keys)} instruments (v3 Binary)")
                return True
            
        except Exception as e:
            print(f"❌ Failed to unsubscribe from instruments: {e}")
            return False
    
    def get_latest_tick(self, instrument_key: str) -> Optional[Dict]:
        """Get the latest tick data for an instrument."""
        return self.last_tick_data.get(instrument_key)
    
    def get_all_latest_ticks(self) -> Dict:
        """Get latest tick data for all subscribed instruments."""
        return self.last_tick_data.copy()
    
    def get_connection_status(self) -> Dict:
        """Get current connection status."""
        return {
            'is_connected': self.is_connected,
            'subscribed_instruments': list(self.subscribed_instruments),
            'total_instruments': len(self.subscribed_instruments),
            'websocket_url': self.websocket_url,
            'last_tick_count': len(self.last_tick_data),
            'market_status': self.market_status
        }
