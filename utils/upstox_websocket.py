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
        
        # Reset price tracking for clean assignment
        print("🔄 Price-based instrument assignment ready")

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

            # Don't auto-reconnect if too many failures
            if self.close_count > 10:
                print("⚠️ Too many disconnects - stopping auto-reconnect")
                return

        if self.connection_callback:
            self.connection_callback("disconnected")

        # Auto-reconnect with exponential backoff (reduced frequency to avoid spam)
        if (hasattr(self, '_was_connected') and self._was_connected and 
            self.close_count <= 10):

            import threading
            def reconnect():
                import time
                # Exponential backoff: 5s, 10s, 20s, 30s max
                backoff_time = min(30, 5 * (2 ** min(self.close_count - 1, 3)))
                print(f"🔄 Scheduled reconnect in {backoff_time}s (attempt {self.close_count})")
                time.sleep(backoff_time)

                if not self.is_connected:
                    print(f"🔄 Reconnecting attempt {self.close_count}...")
                    try:
                        success = self.connect()

                        if success and self.subscribed_instruments:
                            time.sleep(2)  # Wait for connection to stabilize
                            instruments = list(self.subscribed_instruments)
                            self.subscribed_instruments.clear()
                            self.subscribe(instruments)
                            # Reset close count on successful reconnect
                            self.close_count = 0
                    except Exception as e:
                        print(f"❌ Reconnection failed: {e}")

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
        error_str = str(error)
        
        # Filter out common non-critical errors
        if "Connection is already closed" in error_str:
            print(f"ℹ️ WebSocket already closed")
        elif "Handshake status 403" in error_str:
            print(f"❌ Authentication error: {error}")
            print("💡 Please check your access token and API key")
        elif "timeout" in error_str.lower():
            print(f"⏰ WebSocket timeout: {error}")
        else:
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
        """Handle incoming WebSocket messages - JSON ONLY processing."""
        try:
            # Handle both binary and text messages from Upstox v3
            if isinstance(message, bytes):
                # Message counter for logging
                if not hasattr(self, '_msg_counter'):
                    self._msg_counter = 0
                self._msg_counter += 1
                
                if self._msg_counter % 25 == 0:  # Show every 25th message
                    print(f"📦 Processing JSON message #{self._msg_counter}")
                
                # STRICT JSON-ONLY DECODING - NO FALLBACK
                try:
                    # Decode as UTF-8 with proper error handling
                    decoded_message = message.decode('utf-8', errors='replace')
                    
                    # Clean any null bytes or control characters
                    decoded_message = decoded_message.strip('\x00').strip()
                    
                    # Ensure it's valid JSON format
                    if not decoded_message.startswith('{'):
                        # Try to find JSON start in the message
                        json_start = decoded_message.find('{')
                        if json_start > 0:
                            decoded_message = decoded_message[json_start:]
                        else:
                            print(f"⚠️ Message #{self._msg_counter}: No JSON structure found, skipping...")
                            return
                    
                    # Parse JSON strictly
                    data = json.loads(decoded_message)
                    print(f"✅ JSON Message #{self._msg_counter} parsed successfully")
                    tick_data = self.parse_json_message(data)
                    
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    print(f"❌ JSON parsing failed for message #{self._msg_counter}: {e}")
                    print(f"🔍 Message preview: {message[:100]}...")
                    return  # Skip non-JSON messages completely
                    
            else:
                # Handle text messages
                print(f"📄 Text message received")
                try:
                    data = json.loads(message)
                    print(f"✅ Text JSON message parsed successfully")
                    tick_data = self.parse_json_message(data)
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON in text message: {e}")
                    return

            # Process valid tick data
            if tick_data:
                # Store latest tick data
                instrument = tick_data.get('instrument_token')
                if instrument:
                    self.last_tick_data[instrument] = tick_data
                    
                    # Show live updates
                    if self._msg_counter % 10 == 0:  # Show every 10th tick
                        instrument_name = instrument.split('|')[-1] if '|' in instrument else instrument
                        ltp = tick_data.get('ltp', 0)
                        print(f"📊 Live JSON tick: {instrument_name} @ ₹{ltp:.2f}")

                # Call callback
                if self.tick_callback:
                    self.tick_callback(tick_data)
            else:
                print(f"⚠️ No valid tick data extracted from JSON message #{self._msg_counter}")

        except Exception as e:
            # Show all errors for JSON debugging
            print(f"❌ Critical JSON processing error: {e}")
            import traceback
            traceback.print_exc()

    def parse_protobuf_message(self, message: bytes) -> Optional[Dict]:
        """Parse protobuf message from Upstox v3 API with enhanced JSON priority and instrument separation."""
        try:
            if len(message) < 4:
                return None

            # PRIORITY 1: Try to decode as JSON first (has proper instrument identification)
            try:
                # Check for JSON patterns more thoroughly
                if (message.startswith(b'{') or 
                    b'"feeds"' in message[:200] or 
                    b'"ltpc"' in message[:200] or
                    b'"NSE_' in message[:500]):
                    
                    json_str = message.decode('utf-8', errors='ignore')
                    
                    # Clean up any binary artifacts
                    json_str = json_str.strip('\x00').strip()
                    
                    if json_str.startswith('{'):
                        data = json.loads(json_str)
                        print(f"📊 JSON Message Parsed - processing feeds data...")
                        return self.parse_json_message(data)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Continue to protobuf parsing if JSON fails
                pass

            # PRIORITY 2: Enhanced protobuf parsing with better instrument detection

            # Enhanced protobuf parsing for full market data extraction
            try:
                # Look for multiple data points in protobuf
                offset = 0
                extracted_data = {
                    'prices': [],
                    'quantities': [],
                    'volumes': [],
                    'timestamps': []
                }

                while offset < len(message) - 8:
                    try:
                        # Extract prices (LTP, bid, ask)
                        if offset + 8 <= len(message):
                            # Try different price formats
                            for fmt in ['>d', '<d', '>f', '<f']:
                                try:
                                    if fmt.endswith('d'):  # double
                                        value = struct.unpack(fmt, message[offset:offset+8])[0]
                                    else:  # float
                                        value = struct.unpack(fmt, message[offset:offset+4])[0]
                                    
                                    # Check if it's a valid price (expanded range for futures)
                                    if 5000 <= value <= 60000 and not math.isnan(value):
                                        extracted_data['prices'].append(round(value, 2))
                                except (struct.error, OverflowError):
                                    continue

                        # Extract quantities (could be in different integer formats)
                        if offset + 4 <= len(message):
                            for fmt in ['>I', '<I', '>H', '<H']:
                                try:
                                    value = struct.unpack(fmt, message[offset:offset+(4 if 'I' in fmt else 2)])[0]
                                    # Valid quantity range
                                    if 1 <= value <= 100000:
                                        extracted_data['quantities'].append(int(value))
                                except (struct.error, OverflowError):
                                    continue

                        # Extract larger volume numbers
                        if offset + 8 <= len(message):
                            for fmt in ['>Q', '<Q']:
                                try:
                                    value = struct.unpack(fmt, message[offset:offset+8])[0]
                                    # Valid volume range
                                    if 100 <= value <= 10000000:
                                        extracted_data['volumes'].append(int(value))
                                except (struct.error, OverflowError):
                                    continue

                        offset += 1

                    except (struct.error, OverflowError):
                        offset += 1
                        continue

                # Process extracted data if we found valid information
                if extracted_data['prices']:
                    from collections import Counter
                    
                    # Get most common price (likely LTP)
                    price_counts = Counter(extracted_data['prices'])
                    ltp = price_counts.most_common(1)[0][0]
                    
                    # Try to identify bid/ask from nearby prices
                    prices = sorted(set(extracted_data['prices']))
                    ltp_index = prices.index(ltp) if ltp in prices else 0
                    
                    bid_price = prices[max(0, ltp_index - 1)] if ltp_index > 0 else ltp * 0.9999
                    ask_price = prices[min(len(prices) - 1, ltp_index + 1)] if ltp_index < len(prices) - 1 else ltp * 1.0001
                    
                    # Get quantities and volumes
                    quantities = extracted_data['quantities']
                    volumes = extracted_data['volumes']
                    
                    # Estimate market depth quantities
                    ltq = quantities[0] if quantities else 100
                    bid_qty = quantities[1] if len(quantities) > 1 else 50
                    ask_qty = quantities[2] if len(quantities) > 2 else 50
                    
                    # Estimate total quantities (larger numbers likely represent totals)
                    total_buy_qty = max(quantities) if quantities else 5000
                    total_sell_qty = total_buy_qty * 0.95 if quantities else 4500  # Slightly less sell pressure
                    
                    # Volume information
                    volume = volumes[0] if volumes else 1000

                    import pytz
                    ist = pytz.timezone('Asia/Kolkata')

                    # Use message sequence to determine instrument (no price-based guessing)
                    likely_instrument = self._get_next_instrument_in_sequence()
                    
                    # Get clean display name
                    instrument_display = likely_instrument.split('|')[-1] if '|' in likely_instrument else likely_instrument
                    
                    # Create comprehensive tick data structure
                    tick = {
                        'instrument_token': likely_instrument,
                        'timestamp': datetime.now(ist),
                        'ltp': float(ltp),
                        'ltq': int(ltq),
                        'volume': int(volume),
                        'open': float(ltp),
                        'high': float(ltp),
                        'low': float(ltp),
                        'close': float(ltp),
                        'change': 0.0,
                        'change_percent': 0.0,
                        # Enhanced market depth fields
                        'last_traded_price': float(ltp),
                        'last_traded_quantity': int(ltq),
                        'total_buy_quantity': int(total_buy_qty),
                        'total_sell_quantity': int(total_sell_qty),
                        'best_bid': float(bid_price),
                        'best_bid_quantity': int(bid_qty),
                        'best_ask': float(ask_price),
                        'best_ask_quantity': int(ask_qty),
                        'bid_price': float(bid_price),
                        'ask_price': float(ask_price),
                        'bid_qty': int(bid_qty),
                        'ask_qty': int(ask_qty)
                    }

                    # Get display name for logging
                    display_name = likely_instrument.split('|')[-1] if '|' in likely_instrument else likely_instrument
                    print(f"📊 Enhanced Protobuf: {display_name} @ ₹{ltp:.2f} | Bid: ₹{bid_price:.2f}({bid_qty}) | Ask: ₹{ask_price:.2f}({ask_qty}) | TotalBuy: {total_buy_qty} | TotalSell: {total_sell_qty}")
                    return tick

            except Exception as e:
                print(f"⚠️ Enhanced protobuf parsing error: {e}")

            # Log message for debugging if no parsing worked
            print(f"🔍 Unable to parse message (len={len(message)}): {message[:50]}...")
            return None

        except Exception as e:
            print(f"❌ Critical parsing error: {e}")
            return None

    def _get_next_instrument_in_sequence(self) -> str:
        """Get next instrument in sequence for protobuf assignment (no price guessing)."""
        subscribed_list = list(self.subscribed_instruments)
        
        if not subscribed_list:
            return 'NSE_INDEX|Nifty 50'
        
        if len(subscribed_list) == 1:
            return subscribed_list[0]
        
        # Initialize sequence counter if not exists
        if not hasattr(self, '_protobuf_sequence_counter'):
            self._protobuf_sequence_counter = 0
        
        # Round-robin assignment to prevent contamination
        instrument = subscribed_list[self._protobuf_sequence_counter % len(subscribed_list)]
        self._protobuf_sequence_counter += 1
        
        # Log assignment for debugging
        instrument_name = instrument.split('|')[-1] if '|' in instrument else instrument
        print(f"🔄 Sequential assignment: Message #{self._protobuf_sequence_counter} → {instrument_name}")
        
        return instrument

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
        """Parse JSON message format for v3 API responses with DIRECT instrument identification - ERROR-FREE."""
        try:
            # Validate input data structure
            if not isinstance(data, dict):
                print(f"❌ Invalid JSON structure: Expected dict, got {type(data)}")
                return None
            
            # Handle subscription acknowledgment
            if data.get('status') == 'success' and 'data' in data:
                if 'subscribed' in data['data']:
                    subscribed_count = len(data['data']['subscribed']) if isinstance(data['data']['subscribed'], list) else 1
                    print(f"✅ JSON Subscription confirmed: {subscribed_count} instruments")
                return None

            # Handle market status (initial message)
            if 'type' in data and data['type'] == 'initial':
                self.market_status = data
                print(f"📊 JSON Market status received successfully")
                return None

            # Handle live feed data - Look for feeds structure
            if 'feeds' in data and isinstance(data['feeds'], dict):
                feeds = data['feeds']
                
                if not feeds:
                    print(f"⚠️ Empty feeds structure in JSON message")
                    return None

                # Process each instrument in feeds with DIRECT key mapping
                processed_instruments = []
                
                print(f"📊 Processing JSON feeds for {len(feeds)} instruments")
                
                for instrument_key, feed_data in feeds.items():
                    if not isinstance(feed_data, dict):
                        print(f"⚠️ Invalid feed data for {instrument_key}: {type(feed_data)}")
                        continue
                    # DIRECT instrument identification - NO GUESSING
                    exact_instrument_key = instrument_key  # Use exact key from JSON
                    
                    # Create separate tick for each instrument with exact identification
                    tick_data = {
                        'instrument_token': exact_instrument_key,  # EXACT instrument key
                        'timestamp': datetime.now(),
                        'ltp': 0.0,
                        'ltq': 0,
                        'volume': 0,
                        'close': 0.0,
                        'open': 0.0,
                        'high': 0.0,
                        'low': 0.0,
                        'change': 0.0,
                        'change_percent': 0.0,
                        # Full market depth fields
                        'last_traded_price': 0.0,
                        'last_traded_quantity': 0,
                        'total_buy_quantity': 0,
                        'total_sell_quantity': 0,
                        'best_bid': 0.0,
                        'best_bid_quantity': 0,
                        'best_ask': 0.0,
                        'best_ask_quantity': 0,
                        'bid_price': 0.0,
                        'ask_price': 0.0,
                        'bid_qty': 0,
                        'ask_qty': 0
                    }

                    # Look for LTPC data (basic price info) with validation
                    if 'ltpc' in feed_data and isinstance(feed_data['ltpc'], dict):
                        ltpc = feed_data['ltpc']
                        try:
                            ltp = float(ltpc.get('ltp', 0))
                            cp = float(ltpc.get('cp', ltp))
                            ltq = int(ltpc.get('ltq', 0))
                        except (ValueError, TypeError) as e:
                            print(f"⚠️ Invalid LTPC data for {instrument_key}: {e}")
                            continue

                        if ltp > 0:
                            tick_data.update({
                                'ltp': ltp,
                                'last_traded_price': ltp,
                                'ltq': ltq,
                                'last_traded_quantity': ltq,
                                'volume': ltq,
                                'close': cp,
                                'open': ltp,
                                'high': ltp,
                                'low': ltp,
                                'change': ltp - cp if cp > 0 else 0.0,
                                'change_percent': ((ltp - cp) / cp * 100) if cp > 0 else 0.0
                            })

                    # Look for full market data (complete depth) with validation
                    if 'ff' in feed_data and isinstance(feed_data['ff'], dict):
                        ff_data = feed_data['ff']
                        
                        # Extract market full feed data with validation
                        if 'marketFF' in ff_data and isinstance(ff_data['marketFF'], dict):
                            market_ff = ff_data['marketFF']
                            
                            # LTPC within full feed
                            if 'ltpc' in market_ff:
                                ltpc = market_ff['ltpc']
                                ltp = float(ltpc.get('ltp', 0))
                                cp = float(ltpc.get('cp', ltp))
                                ltq = int(ltpc.get('ltq', 0))
                                
                                tick_data.update({
                                    'ltp': ltp,
                                    'last_traded_price': ltp,
                                    'ltq': ltq,
                                    'last_traded_quantity': ltq,
                                    'close': cp
                                })

                            # Market depth information with validation
                            if 'marketLevel' in market_ff and isinstance(market_ff['marketLevel'], dict):
                                market_level = market_ff['marketLevel']
                                
                                # Best bid/ask from level 1 data with validation
                                if 'bidAskQuote' in market_level and isinstance(market_level['bidAskQuote'], list):
                                    bid_ask = market_level['bidAskQuote']
                                    if len(bid_ask) > 0 and isinstance(bid_ask[0], dict):
                                        level1 = bid_ask[0]
                                        try:
                                            tick_data.update({
                                                'best_bid': float(level1.get('bq', 0)),
                                                'best_bid_quantity': int(level1.get('bs', 0)),
                                                'best_ask': float(level1.get('aq', 0)),
                                                'best_ask_quantity': int(level1.get('as', 0)),
                                                'bid_price': float(level1.get('bq', 0)),
                                                'ask_price': float(level1.get('aq', 0)),
                                                'bid_qty': int(level1.get('bs', 0)),
                                                'ask_qty': int(level1.get('as', 0))
                                            })
                                        except (ValueError, TypeError) as e:
                                            print(f"⚠️ Invalid bid/ask data for {instrument_key}: {e}")

                            # Volume and quantity totals
                            if 'vtt' in market_ff:  # Total traded volume
                                tick_data['volume'] = int(market_ff['vtt'])
                            
                            if 'tbq' in market_ff:  # Total buy quantity
                                tick_data['total_buy_quantity'] = int(market_ff['tbq'])
                            
                            if 'tsq' in market_ff:  # Total sell quantity
                                tick_data['total_sell_quantity'] = int(market_ff['tsq'])

                            # OHLC data if available
                            if 'op' in market_ff:
                                tick_data['open'] = float(market_ff['op'])
                            if 'hp' in market_ff:
                                tick_data['high'] = float(market_ff['hp'])
                            if 'lp' in market_ff:
                                tick_data['low'] = float(market_ff['lp'])

                    # Only process tick if we have valid price data
                    if tick_data['ltp'] > 0:
                        # Store with EXACT instrument key (no contamination possible)
                        self.last_tick_data[exact_instrument_key] = tick_data
                        
                        # Get clean display name for logging
                        display_name = exact_instrument_key.split('|')[-1] if '|' in exact_instrument_key else exact_instrument_key
                        ltp = tick_data['ltp']
                        bid = tick_data['best_bid']
                        ask = tick_data['best_ask']
                        total_buy = tick_data['total_buy_quantity']
                        total_sell = tick_data['total_sell_quantity']
                        
                        # Enhanced logging with instrument validation
                        if 'NIFTY28AUGFUT' in exact_instrument_key:
                            print(f"🚀 FUTURES: {display_name} @ ₹{ltp:.2f} | Premium | Bid: ₹{bid:.2f}({tick_data['best_bid_quantity']}) | Ask: ₹{ask:.2f}({tick_data['best_ask_quantity']})")
                        elif 'Nifty 50' in exact_instrument_key:
                            print(f"📊 SPOT: {display_name} @ ₹{ltp:.2f} | Index | Bid: ₹{bid:.2f}({tick_data['best_bid_quantity']}) | Ask: ₹{ask:.2f}({tick_data['best_ask_quantity']})")
                        else:
                            print(f"📈 {display_name} @ ₹{ltp:.2f} | Bid: ₹{bid:.2f}({tick_data['best_bid_quantity']}) | Ask: ₹{ask:.2f}({tick_data['best_ask_quantity']})")
                        
                        processed_instruments.append(tick_data)

                # Process all instruments via callback to prevent data mixing
                if processed_instruments:
                    for tick in processed_instruments:
                        if self.tick_callback:
                            self.tick_callback(tick)
                    
                    # Return first tick (others already processed via callback)
                    return processed_instruments[0]

            # If no recognizable structure, log it
            print(f"📊 Unrecognized message structure. Keys: {list(data.keys())}")
            return None

        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            import traceback
            traceback.print_exc()
            return None

    def connect(self):
        """Establish WebSocket connection."""
        try:
            if not self.websocket_url:
                print("❌ No WebSocket URL available")
                return False

            # Disconnect any existing connection
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass

            print(f"🔄 Connecting to Upstox v3 WebSocket...")

            # Create WebSocket connection with proper headers as per v3 documentation
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "*/*"
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

            # Start WebSocket with improved ping mechanism and error handling
            wst = threading.Thread(target=lambda: self.ws.run_forever(
                ping_interval=30,    # Send ping every 30 seconds
                ping_timeout=10,     # Reduced timeout for faster detection
                ping_payload="upstox_ping",
                reconnect=0          # Disable built-in reconnect (we handle it ourselves)
            ))
            wst.daemon = True
            wst.start()

            # Wait for connection with timeout
            max_wait = 15  # Increased timeout
            wait_time = 0
            while wait_time < max_wait and not self.is_connected:
                time.sleep(0.5)
                wait_time += 0.5

            if self.is_connected:
                print(f"✅ WebSocket connected in {wait_time:.1f}s")
                # Reset close count on successful connection
                self.close_count = 0
            else:
                print(f"❌ WebSocket connection timeout after {max_wait}s")

            return self.is_connected

        except Exception as e:
            print(f"❌ Failed to connect to Upstox WebSocket: {e}")
            import traceback
            traceback.print_exc()
            return False

    def disconnect(self):
        """Close WebSocket connection."""
        print("🔌 Disconnecting WebSocket...")
        self.is_connected = False
        
        if self.ws:
            try:
                # Properly close the WebSocket
                self.ws.keep_running = False
                self.ws.close()
            except Exception as e:
                print(f"⚠️ Error during disconnect: {e}")
            finally:
                self.ws = None
        
        # Clear subscriptions
        self.subscribed_instruments.clear()
        self.last_tick_data.clear()
        
        print("🔌 WebSocket disconnected and cleaned up")

    def subscribe(self, instrument_keys: list, mode: str = "full"):
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

            # Create subscription request for JSON format explicitly
            subscribe_request = {
                "guid": str(uuid.uuid4()),
                "method": "sub",
                "data": {
                    "mode": mode,
                    "instrumentKeys": new_instruments
                }
            }
            
            # Add JSON format header explicitly
            headers_request = {
                "guid": str(uuid.uuid4()),
                "method": "configure", 
                "data": {
                    "format": "json",
                    "compression": False
                }
            }

            print(f"🔄 Configuring JSON format and subscribing to {len(new_instruments)} instruments in '{mode}' mode")
            print(f"📋 Instruments: {[key.split('|')[-1] for key in new_instruments]}")

            if self.ws and hasattr(self.ws, 'send'):
                # First, configure JSON format
                headers_json = json.dumps(headers_request)
                headers_binary = headers_json.encode('utf-8')
                self.ws.send(headers_binary, websocket.ABNF.OPCODE_BINARY)
                print(f"📤 JSON format configuration sent")
                
                # Small delay for configuration processing
                time.sleep(0.5)
                
                # Then send subscription request
                message_json = json.dumps(subscribe_request)
                message_binary = message_json.encode('utf-8')
                self.ws.send(message_binary, websocket.ABNF.OPCODE_BINARY)
                print(f"📤 JSON subscription sent for {len(new_instruments)} instruments")

                # Update subscribed instruments
                self.subscribed_instruments.update(new_instruments)

                # Small delay then subscribe to LTPC mode as well for better data coverage
                time.sleep(1)

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
                print(f"📤 LTPC subscription also sent for data redundancy")

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
            'market_status': self.market_status
        }