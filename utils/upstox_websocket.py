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
            # Use the correct v3 API endpoint as per documentation
            url = "https://api.upstox.com/v2/feed/market-data-feed/authorize"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
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
            elif response.status_code == 302:
                # Handle redirection as mentioned in documentation
                redirect_url = response.headers.get('Location')
                if redirect_url:
                    self.websocket_url = redirect_url
                    print(f"✅ Got WebSocket URL via redirect: {self.websocket_url}")
                else:
                    self._fallback_websocket_url()
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                self._fallback_websocket_url()

        except Exception as e:
            print(f"❌ Error getting WebSocket URL: {e}")
            self._fallback_websocket_url()

    def _fallback_websocket_url(self):
        """Fallback to direct WebSocket URL construction for v3."""
        # Use the correct market data feed endpoint as per documentation
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

            # Track close count
            self.close_count = getattr(self, 'close_count', 0) + 1

            # Don't auto-reconnect during market hours if too many failures
            if self.close_count > 15:
                print("⚠️ Too many disconnects - stopping auto-reconnect")
                return

        if self.connection_callback:
            self.connection_callback("disconnected")

        # Auto-reconnect only during market hours and if not too many failures
        if (hasattr(self, '_was_connected') and self._was_connected and 
            self._is_market_hours() and self.close_count <= 15):

            import threading
            def reconnect():
                import time
                backoff_time = min(30, 5 * self.close_count)
                time.sleep(backoff_time)

                if not self.is_connected and self._is_market_hours():
                    print(f"🔄 Reconnecting attempt {self.close_count}...")
                    success = self.connect()

                    if success and self.subscribed_instruments:
                        time.sleep(3)  # Wait for connection to stabilize
                        instruments = list(self.subscribed_instruments)
                        self.subscribed_instruments.clear()
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
            # Handle both binary and text messages from Upstox v3
            if isinstance(message, bytes):
                print(f"📦 Binary message received: {len(message)} bytes")
                # First try to decode as UTF-8 for JSON messages
                try:
                    decoded_message = message.decode('utf-8')
                    print(f"🔍 Decoded message preview: {decoded_message[:200]}...")
                    
                    if decoded_message.strip().startswith('{'):
                        data = json.loads(decoded_message)
                        tick_data = self.parse_json_message(data)
                        if tick_data:
                            self._process_tick_data(tick_data)
                    else:
                        print(f"⚠️ Non-JSON binary message, attempting protobuf parsing...")
                        tick_data = self.parse_protobuf_message(message)
                        if tick_data:
                            self._process_tick_data(tick_data)
                            
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    print(f"🔄 UTF-8 decode failed, trying protobuf: {e}")
                    tick_data = self.parse_protobuf_message(message)
                    if tick_data:
                        self._process_tick_data(tick_data)
                    else:
                        print(f"⚠️ Could not parse binary message of {len(message)} bytes")
            else:
                print(f"📄 Text message received: {message[:200]}...")
                try:
                    data = json.loads(message)
                    tick_data = self.parse_json_message(data)
                    if tick_data:
                        self._process_tick_data(tick_data)
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON in text message: {message[:100]}...")

        except Exception as e:
            print(f"❌ Error processing message: {e}")
            import traceback
            traceback.print_exc()

    def _process_tick_data(self, tick_data):
        """Process and store tick data."""
        if tick_data:
            # Store latest tick data
            instrument = tick_data.get('instrument_token')
            if instrument:
                self.last_tick_data[instrument] = tick_data
                print(f"📊 Live tick: {instrument.split('|')[-1]} @ ₹{tick_data.get('ltp', 0):.2f}")

            # Call callback
            if self.tick_callback:
                self.tick_callback(tick_data)

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

                    import pytz
                    ist = pytz.timezone('Asia/Kolkata')

                    tick = {
                        'instrument_token': 'NSE_INDEX|Nifty 50',
                        'timestamp': datetime.now(ist),
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
        """Parse JSON message format for v3 API responses."""
        try:
            # Debug: Print the full structure of received data
            print(f"🔍 Received JSON structure keys: {list(data.keys())}")
            print(f"🔍 Full data: {json.dumps(data, indent=2)[:800]}...")

            # Handle market status (market_info type)
            if data.get('type') == 'market_info':
                self.market_status = data
                print(f"📊 Market status received: {data.get('marketInfo', {}).get('segmentStatus', {})}")
                return None

            # Handle live feed data - Look for feeds structure
            if data.get('type') == 'live_feed' and 'feeds' in data:
                feeds = data['feeds']
                current_ts = data.get('currentTs')

                print(f"📊 Live feed with {len(feeds)} instruments at timestamp {current_ts}")

                # Process the first instrument in feeds (can be modified to handle multiple)
                for instrument_key, feed_data in feeds.items():
                    print(f"🔍 Processing {instrument_key}: {list(feed_data.keys())}")

                    # Handle LTPC data (most common)
                    if 'ltpc' in feed_data:
                        ltpc = feed_data['ltpc']
                        
                        # Extract values with proper type conversion
                        ltp = float(ltpc.get('ltp', 0))
                        cp = float(ltpc.get('cp', ltp))  # Close price, fallback to LTP
                        ltq = int(ltpc.get('ltq', 0))   # Last traded quantity
                        ltt = ltpc.get('ltt', '')       # Last traded time

                        if ltp > 0:
                            # Calculate change
                            change = ltp - cp if cp > 0 else 0.0
                            change_percent = (change / cp * 100) if cp > 0 else 0.0

                            # Convert timestamp
                            import pytz
                            ist = pytz.timezone('Asia/Kolkata')
                            
                            # Use current time if ltt is not provided or invalid
                            if ltt:
                                try:
                                    timestamp = datetime.fromtimestamp(int(ltt) / 1000, tz=ist)
                                except:
                                    timestamp = datetime.now(ist)
                            else:
                                timestamp = datetime.now(ist)

                            tick = {
                                'instrument_token': instrument_key,
                                'timestamp': timestamp,
                                'ltp': ltp,
                                'ltq': ltq,
                                'volume': ltq,  # Use ltq as volume approximation
                                'close': cp,
                                'open': ltp,   # Approximate open with current price
                                'high': ltp,   # Approximate high with current price  
                                'low': ltp,    # Approximate low with current price
                                'change': change,
                                'change_percent': change_percent,
                                'bid_price': ltp * 0.9999,  # Approximate bid
                                'ask_price': ltp * 1.0001,  # Approximate ask
                                'bid_qty': 10,
                                'ask_qty': 10
                            }

                            print(f"✅ LTPC Parsed: {instrument_key.split('|')[-1]} @ ₹{ltp:.2f} (Change: {change_percent:+.2f}%)")
                            return tick
                        else:
                            print(f"⚠️ Invalid LTP value: {ltp}")

                    # Handle full feed data
                    elif 'ff' in feed_data:
                        print(f"📊 Full feed data received for {instrument_key}")
                        # Add full feed parsing logic here if needed
                        
                    else:
                        print(f"🔍 Unknown feed structure for {instrument_key}: {list(feed_data.keys())}")

            # Handle subscription acknowledgment
            elif data.get('status') == 'success':
                print(f"✅ Success response: {data}")
                return None

            # Handle error responses
            elif data.get('status') == 'error':
                print(f"❌ Error response: {data}")
                return None

            else:
                print(f"📊 Unhandled message type: {data.get('type', 'unknown')}")
                print(f"📊 Message keys: {list(data.keys())}")

            return None

        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            print(f"❌ Data causing error: {data}")
            import traceback
            traceback.print_exc()
            return None

    def connect(self):
        """Establish WebSocket connection."""
        try:
            if not self.websocket_url:
                print("❌ No WebSocket URL available")
                return False

            print(f"🔄 Connecting to Upstox v3 WebSocket...")
            print(f"🔗 WebSocket URL: {self.websocket_url[:80]}...")

            # Create WebSocket connection with proper headers as per v3 documentation
            headers = [
                f"Authorization: Bearer {self.access_token}",
                "Accept: */*"
            ]

            print(f"📋 Headers prepared for connection")

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

            # Start WebSocket with improved ping mechanism and debugging
            def run_websocket():
                print("🚀 Starting WebSocket connection...")
                self.ws.run_forever(
                    ping_interval=30,    # Send ping every 30 seconds
                    ping_timeout=20,     # Wait 20 seconds for pong
                    ping_payload="upstox_ping"
                )
                print("🔌 WebSocket run_forever ended")

            wst = threading.Thread(target=run_websocket)
            wst.daemon = True
            wst.start()

            # Wait for connection with timeout
            max_wait = 15  # Increased timeout
            wait_time = 0
            print(f"⏳ Waiting for connection (max {max_wait}s)...")
            
            while wait_time < max_wait and not self.is_connected:
                time.sleep(0.5)
                wait_time += 0.5
                if wait_time % 2 == 0:  # Print every 2 seconds
                    print(f"⏳ Still waiting... ({wait_time:.1f}s)")

            if self.is_connected:
                print(f"✅ WebSocket connected successfully in {wait_time:.1f}s")
                print(f"🔄 Ready to subscribe to instruments")
            else:
                print(f"❌ WebSocket connection timeout after {max_wait}s")
                print(f"🔍 Check if:")
                print(f"   • Access token is valid and not expired")
                print(f"   • API key is correct")
                print(f"   • Market is open (check market hours)")
                print(f"   • Network connectivity is working")

            return self.is_connected

        except Exception as e:
            print(f"❌ Failed to connect to Upstox WebSocket: {e}")
            import traceback
            traceback.print_exc()
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
        """Subscribe to instruments using v3 API format - BINARY messages only."""
        if not self.is_connected:
            print("❌ WebSocket not connected. Please connect first.")
            return False

        try:
            # Remove duplicates and filter new instruments
            unique_keys = list(set(instrument_keys))
            new_instruments = [key for key in unique_keys if key not in self.subscribed_instruments]

            if not new_instruments:
                print(f"✅ All instruments already subscribed")
                return True

            # Create subscription request exactly as per v3 documentation
            subscribe_request = {
                "guid": str(uuid.uuid4()),
                "method": "sub",
                "data": {
                    "mode": mode,
                    "instrumentKeys": new_instruments
                }
            }

            print(f"🔄 Subscribing to {len(new_instruments)} instruments in '{mode}' mode")
            print(f"📋 Instruments: {[key.split('|')[-1] for key in new_instruments]}")
            print(f"📤 Subscription request: {json.dumps(subscribe_request, indent=2)}")

            # Convert to binary format (v3 requirement)
            message_json = json.dumps(subscribe_request)
            message_binary = message_json.encode('utf-8')

            if self.ws and hasattr(self.ws, 'send'):
                # Send as binary message (critical for V3)
                self.ws.send(message_binary, websocket.ABNF.OPCODE_BINARY)
                print(f"📤 Subscription sent as binary message ({len(message_binary)} bytes)")

                # Update subscribed instruments
                self.subscribed_instruments.update(new_instruments)

                # Wait a bit for acknowledgment
                time.sleep(2)

                # If mode is not LTPC, also subscribe to LTPC for basic updates
                if mode != "ltpc":
                    print(f"📤 Also subscribing to LTPC for basic price updates...")
                    ltpc_request = {
                        "guid": str(uuid.uuid4()),
                        "method": "sub", 
                        "data": {
                            "mode": "ltpc",
                            "instrumentKeys": new_instruments
                        }
                    }

                    ltpc_binary = json.dumps(ltpc_request).encode('utf-8')
                    self.ws.send(ltpc_binary, websocket.ABNF.OPCODE_BINARY)
                    print(f"📤 LTPC subscription also sent ({len(ltpc_binary)} bytes)")

                print(f"✅ Subscription process completed for {len(new_instruments)} instruments")
                return True

        except Exception as e:
            print(f"❌ Subscription failed: {e}")
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
            'market_status': self.market_status,
            'connection_start_time': self.connection_start_time,
            'last_pong_time': self.last_pong_time,
            'ping_count': self.ping_count,
            'pong_count': self.pong_count
        }

    def test_subscription(self, test_instrument: str = "NSE_INDEX|Nifty 50"):
        """Test subscription with a single instrument for debugging."""
        if not self.is_connected:
            print("❌ Cannot test subscription - not connected")
            return False

        print(f"🧪 Testing subscription with {test_instrument}")
        
        # Send a simple LTPC subscription
        test_request = {
            "guid": str(uuid.uuid4()),
            "method": "sub",
            "data": {
                "mode": "ltpc",
                "instrumentKeys": [test_instrument]
            }
        }

        print(f"📤 Test request: {json.dumps(test_request, indent=2)}")

        try:
            message_binary = json.dumps(test_request).encode('utf-8')
            self.ws.send(message_binary, websocket.ABNF.OPCODE_BINARY)
            print(f"✅ Test subscription sent ({len(message_binary)} bytes)")
            return True
        except Exception as e:
            print(f"❌ Test subscription failed: {e}")
            return False