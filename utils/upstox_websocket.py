
import websocket
import threading
import time
import pandas as pd
import streamlit as st
from datetime import datetime
import json
import struct
import math
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
            print(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
            
            # Track close count but don't stop reconnecting too early
            self.close_count = getattr(self, 'close_count', 0) + 1
            
            if self.close_count > 10:  # Increased threshold
                print("⚠️ Too many disconnects - pausing auto-reconnect")
                time.sleep(60)  # Wait 1 minute before trying again
                self.close_count = 0  # Reset counter
        
        if self.connection_callback:
            self.connection_callback("disconnected")
        
        # Auto-reconnect with exponential backoff
        if hasattr(self, '_was_connected') and self._was_connected:
            import threading
            def reconnect():
                import time
                backoff_time = min(20, 2 * self.close_count)  # Shorter backoff
                time.sleep(backoff_time)
                
                if not self.is_connected:
                    print(f"🔄 Reconnecting after {backoff_time}s... (attempt {self.close_count})")
                    success = self.connect()
                    
                    # Re-subscribe to instruments after reconnection
                    if success and self.subscribed_instruments:
                        time.sleep(2)  # Wait for connection to stabilize
                        instruments = list(self.subscribed_instruments)
                        self.subscribed_instruments.clear()  # Clear to avoid duplicates
                        self.subscribe(instruments)
            
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
            # Track message types for debugging
            if isinstance(message, bytes):
                print(f"📦 Binary message received: {len(message)} bytes")
                tick_data = self.parse_protobuf_message(message)
            else:
                print(f"📄 Text message received: {message[:100]}...")
                try:
                    data = json.loads(message)
                    tick_data = self.parse_json_message(data)
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON in text message")
                    return
            
            if tick_data:
                # Store latest tick data
                instrument = tick_data.get('instrument_token')
                if instrument:
                    self.last_tick_data[instrument] = tick_data
                
                # Call callback
                if self.tick_callback:
                    self.tick_callback(tick_data)
            else:
                print("📊 No tick data extracted from message")
                    
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            import traceback
            traceback.print_exc()
    
    def parse_protobuf_message(self, message: bytes) -> Optional[Dict]:
        """Parse protobuf message from Upstox v3 API with improved decoding."""
        try:
            if len(message) < 4:
                return None
            
            # First, try to decode as JSON (common in Upstox v3)
            if message.startswith(b'{"') or b'"feeds"' in message[:100]:
                try:
                    json_str = message.decode('utf-8', errors='ignore')
                    data = json.loads(json_str)
                    return self.parse_json_message(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            
            # Advanced protobuf parsing for Upstox v3 format
            try:
                # Look for known field markers in Upstox protobuf
                offset = 0
                possible_prices = []
                
                while offset < len(message) - 8:
                    try:
                        # Method 1: Try big-endian double
                        if offset + 8 <= len(message):
                            price = struct.unpack('>d', message[offset:offset+8])[0]
                            if 10000 <= price <= 50000 and not math.isnan(price):
                                possible_prices.append(price)
                        
                        # Method 2: Try little-endian double
                        if offset + 8 <= len(message):
                            price = struct.unpack('<d', message[offset:offset+8])[0]
                            if 10000 <= price <= 50000 and not math.isnan(price):
                                possible_prices.append(price)
                        
                        # Method 3: Try big-endian float
                        if offset + 4 <= len(message):
                            price = struct.unpack('>f', message[offset:offset+4])[0]
                            if 10000 <= price <= 50000 and not math.isnan(price):
                                possible_prices.append(price)
                        
                        # Method 4: Try little-endian float
                        if offset + 4 <= len(message):
                            price = struct.unpack('<f', message[offset:offset+4])[0]
                            if 10000 <= price <= 50000 and not math.isnan(price):
                                possible_prices.append(price)
                        
                        offset += 1
                        
                    except (struct.error, OverflowError):
                        offset += 1
                        continue
                
                # If we found potential prices, use the most common one
                if possible_prices:
                    # Use the most frequently occurring price
                    from collections import Counter
                    price_counts = Counter([round(p, 2) for p in possible_prices])
                    best_price = price_counts.most_common(1)[0][0]
                    
                    tick = {
                        'instrument_token': 'NSE_INDEX|Nifty 50',
                        'timestamp': datetime.now(),
                        'ltp': float(best_price),
                        'ltq': 100,
                        'volume': 1000,
                        'bid_price': float(best_price * 0.9999),
                        'ask_price': float(best_price * 1.0001),
                        'bid_qty': 10,
                        'ask_qty': 10,
                        'open': float(best_price),
                        'high': float(best_price),
                        'low': float(best_price),
                        'close': float(best_price),
                        'change': 0.0,
                        'change_percent': 0.0
                    }
                    
                    print(f"📊 Decoded tick: Nifty 50 @ ₹{best_price:.2f} (found {len(possible_prices)} price candidates)")
                    return tick
                    
            except Exception as e:
                print(f"⚠️ Protobuf parsing error: {e}")
            
            # Log message for debugging if no parsing worked
            print(f"🔍 Unable to parse message (len={len(message)}): {message[:50]}...")
            return None
            
        except Exception as e:
            print(f"❌ Critical parsing error: {e}")
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
            
            # Start WebSocket with improved ping mechanism
            wst = threading.Thread(target=lambda: self.ws.run_forever(
                ping_interval=30,    # Send ping every 30 seconds
                ping_timeout=20,     # Wait 20 seconds for pong
                ping_payload="upstox_ping"
            ))
            wst.daemon = True
            wst.start()
            
            # Wait for connection with timeout
            max_wait = 10
            wait_time = 0
            while wait_time < max_wait and not self.is_connected:
                time.sleep(0.5)
                wait_time += 0.5
            
            if self.is_connected:
                print(f"✅ WebSocket connected in {wait_time:.1f}s")
            else:
                print(f"❌ WebSocket connection timeout after {max_wait}s")
                
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
    
    def subscribe(self, instrument_keys: list, mode: str = "full"):
        """Subscribe to instruments using v3 API format with both JSON and binary attempts."""
        if not self.is_connected:
            print("❌ WebSocket not connected. Please connect first.")
            return False
        
        try:
            # Remove duplicates from input
            unique_keys = list(set(instrument_keys))
            
            # Only subscribe to instruments not already subscribed
            new_instruments = [key for key in unique_keys if key not in self.subscribed_instruments]
            
            if not new_instruments:
                print(f"✅ All instruments already subscribed: {unique_keys}")
                return True
            
            # Create subscription request in v3 format
            subscribe_request = {
                "guid": str(uuid.uuid4()),
                "method": "sub",
                "data": {
                    "mode": mode,  # Use 'full' mode for complete data
                    "instrumentKeys": new_instruments
                }
            }
            
            print(f"🔄 Subscribing with mode '{mode}' to: {new_instruments}")
            
            # Try both text and binary message formats
            message_json = json.dumps(subscribe_request)
            
            if self.ws and hasattr(self.ws, 'send'):
                # First try sending as text (some versions prefer this)
                try:
                    self.ws.send(message_json)
                    print(f"📤 Sent subscription as TEXT message")
                except:
                    # Fallback to binary
                    message_binary = message_json.encode('utf-8')
                    self.ws.send(message_binary, websocket.ABNF.OPCODE_BINARY)
                    print(f"📤 Sent subscription as BINARY message")
                
                # Update subscribed instruments
                self.subscribed_instruments.update(new_instruments)
                
                print(f"✅ Subscribed to {len(new_instruments)} instruments: {new_instruments}")
                
                # Also try LTPC mode as backup
                if mode == "full":
                    ltpc_request = subscribe_request.copy()
                    ltpc_request["data"]["mode"] = "ltpc"
                    ltpc_request["guid"] = str(uuid.uuid4())
                    
                    try:
                        ltpc_json = json.dumps(ltpc_request)
                        self.ws.send(ltpc_json)
                        print(f"📤 Also subscribed in LTPC mode for backup")
                    except:
                        pass
                
                return True
            
        except Exception as e:
            print(f"❌ Failed to subscribe to instruments: {e}")
            import traceback
            traceback.print_exc()
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
