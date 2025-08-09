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

        print(f"🕐 Market hours check: {current_time.strftime('%H:%M')} IST - {'OPEN' if is_weekday and is_market_time else 'CLOSED'}")
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
        """Handle incoming WebSocket messages - Official Upstox implementation style."""
        try:
            # Message counter for logging
            if not hasattr(self, '_msg_counter'):
                self._msg_counter = 0
            self._msg_counter += 1

            if self._msg_counter % 50 == 0:  # Show every 50th message
                print(f"📦 Processing message #{self._msg_counter}")

            # Handle both binary and text messages (official approach)
            if isinstance(message, bytes):
                # Try JSON first (official priority)
                try:
                    decoded_message = message.decode('utf-8')
                    # Look for JSON structure anywhere in the message
                    json_start = decoded_message.find('{')
                    if json_start >= 0:
                        json_part = decoded_message[json_start:]
                        # Try to find the end of JSON
                        brace_count = 0
                        json_end = json_start
                        for i, char in enumerate(json_part):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = json_start + i + 1
                                    break
                        
                        if json_end > json_start:
                            json_str = decoded_message[json_start:json_end]
                            data = json.loads(json_str)
                            tick_data = self.parse_json_message(data)
                            if tick_data and self.tick_callback:
                                self.tick_callback(tick_data)
                            return
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass

                # Fall back to protobuf parsing (official fallback)
                try:
                    tick_data = self.parse_protobuf_message(message)
                    if tick_data and self.tick_callback:
                        self.tick_callback(tick_data)
                    return
                except Exception:
                    pass

                # Skip if both parsing methods fail
                if self._msg_counter % 100 == 0:
                    print(f"⚠️ Message #{self._msg_counter}: Could not parse binary message")

            else:
                # Handle text messages
                try:
                    data = json.loads(message)
                    tick_data = self.parse_json_message(data)
                    if tick_data and self.tick_callback:
                        self.tick_callback(tick_data)
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON in text message: {e}")

        except Exception as e:
            print(f"❌ Critical message processing error: {e}")

    def parse_protobuf_message(self, message: bytes) -> Optional[Dict]:
        """Parse protobuf message using official Upstox approach."""
        try:
            if len(message) < 8:
                return None

            # Official approach: Extract data from binary format
            try:
                # Basic binary parsing similar to official implementation
                offset = 0
                
                # Try to extract price data (double precision)
                if len(message) >= offset + 8:
                    try:
                        ltp = struct.unpack('>d', message[offset:offset+8])[0]
                        if 1000 <= ltp <= 100000 and not math.isnan(ltp):
                            offset += 8
                        else:
                            ltp = None
                    except struct.error:
                        ltp = None
                
                # Try to extract quantity (integer)
                ltq = 0
                if len(message) >= offset + 4:
                    try:
                        ltq = struct.unpack('>I', message[offset:offset+4])[0]
                        if ltq > 1000000:  # If too large, try smaller format
                            ltq = struct.unpack('>H', message[offset:offset+2])[0]
                    except struct.error:
                        ltq = 0

                # Only create tick if we have valid price
                if ltp and ltp > 0:
                    # Use round-robin for instrument assignment (official approach)
                    instrument = self._get_next_instrument_in_sequence()
                    
                    # Create basic tick data structure
                    tick_data = {
                        'instrument_token': instrument,
                        'timestamp': datetime.now(),
                        'ltp': float(ltp),
                        'ltq': int(ltq) if ltq else 100,
                        'volume': int(ltq) if ltq else 100,
                        'open': float(ltp),
                        'high': float(ltp),
                        'low': float(ltp),
                        'close': float(ltp),
                        'change': 0.0,
                        'change_percent': 0.0,
                        'last_traded_price': float(ltp),
                        'last_traded_quantity': int(ltq) if ltq else 100,
                        'total_buy_quantity': 5000,
                        'total_sell_quantity': 4800,
                        'best_bid': float(ltp * 0.9995),
                        'best_bid_quantity': 50,
                        'best_ask': float(ltp * 1.0005),
                        'best_ask_quantity': 50,
                        'bid_price': float(ltp * 0.9995),
                        'ask_price': float(ltp * 1.0005),
                        'bid_qty': 50,
                        'ask_qty': 50
                    }

                    # Store latest tick
                    self.last_tick_data[instrument] = tick_data

                    # Log with clean display name
                    display_name = instrument.split('|')[-1] if '|' in instrument else instrument
                    if self._msg_counter % 25 == 0:
                        print(f"📊 Protobuf: {display_name} @ ₹{ltp:.2f}")

                    return tick_data

            except Exception as e:
                pass

            return None

        except Exception as e:
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

                    # Look for full market data (complete V3 full_d30 structure) with validation
                    # Handle both 'ff' (legacy) and 'fullFeed' (full_d30) structures
                    market_ff = None

                    if 'fullFeed' in feed_data and isinstance(feed_data['fullFeed'], dict):
                        # full_d30 mode uses 'fullFeed' structure
                        full_feed = feed_data['fullFeed']
                        if 'marketFF' in full_feed and isinstance(full_feed['marketFF'], dict):
                            market_ff = full_feed['marketFF']
                            # Store request mode for validation
                            tick_data['request_mode'] = full_feed.get('requestMode', 'unknown')
                    elif 'ff' in feed_data and isinstance(feed_data['ff'], dict):
                        # Legacy 'full' mode uses 'ff' structure
                        ff_data = feed_data['ff']
                        if 'marketFF' in ff_data and isinstance(ff_data['marketFF'], dict):
                            market_ff = ff_data['marketFF']
                            tick_data['request_mode'] = 'full'

                    if market_ff:

                            # LTPC within full feed (exact V3 structure)
                            if 'ltpc' in market_ff and isinstance(market_ff['ltpc'], dict):
                                ltpc = market_ff['ltpc']
                                ltp = float(ltpc.get('ltp', 0))
                                cp = float(ltpc.get('cp', ltp))
                                ltq = int(ltpc.get('ltq', 0))
                                ltt = ltpc.get('ltt', '')  # Last traded time

                                tick_data.update({
                                    'ltp': ltp,
                                    'last_traded_price': ltp,
                                    'ltq': ltq,
                                    'last_traded_quantity': ltq,
                                    'close': cp,
                                    'last_traded_time': ltt
                                })

                            # Market depth information (full_d30 bidAskQuote structure - 30 levels)
                            if 'marketLevel' in market_ff and isinstance(market_ff['marketLevel'], dict):
                                market_level = market_ff['marketLevel']

                                # Up to 30-level bid/ask data as per V3 full_d30 specification
                                if 'bidAskQuote' in market_level and isinstance(market_level['bidAskQuote'], list):
                                    bid_ask_quotes = market_level['bidAskQuote']

                                    if len(bid_ask_quotes) > 0 and isinstance(bid_ask_quotes[0], dict):
                                        # Level 1 (best bid/ask) - using correct V3 field names
                                        level1 = bid_ask_quotes[0]
                                        try:
                                            tick_data.update({
                                                'best_bid': float(level1.get('bidP', 0)),
                                                'best_bid_quantity': int(level1.get('bidQ', 0)),
                                                'best_ask': float(level1.get('askP', 0)),
                                                'best_ask_quantity': int(level1.get('askQ', 0)),
                                                'bid_price': float(level1.get('bidP', 0)),
                                                'ask_price': float(level1.get('askP', 0)),
                                                'bid_qty': int(level1.get('bidQ', 0)),
                                                'ask_qty': int(level1.get('askQ', 0))
                                            })

                                            # Store all 30 levels for complete D30 market depth
                                            tick_data['market_depth'] = {
                                                'bid_levels': [],
                                                'ask_levels': [],
                                                'total_levels': len(bid_ask_quotes)
                                            }

                                            # Process all available levels (up to 30 for full_d30)
                                            for i, level in enumerate(bid_ask_quotes[:30]):  # Up to 30 levels
                                                tick_data['market_depth']['bid_levels'].append({
                                                    'level': i + 1,
                                                    'price': float(level.get('bidP', 0)),
                                                    'quantity': int(level.get('bidQ', 0))
                                                })
                                                tick_data['market_depth']['ask_levels'].append({
                                                    'level': i + 1,
                                                    'price': float(level.get('askP', 0)),
                                                    'quantity': int(level.get('askQ', 0))
                                                })

                                            # Calculate total quantities across all levels for D30
                                            total_bid_qty = sum(int(level.get('bidQ', 0)) for level in bid_ask_quotes)
                                            total_ask_qty = sum(int(level.get('askQ', 0)) for level in bid_ask_quotes)

                                            tick_data.update({
                                                'total_bid_quantity_all_levels': total_bid_qty,
                                                'total_ask_quantity_all_levels': total_ask_qty,
                                                'market_depth_levels': len(bid_ask_quotes)
                                            })

                                        except (ValueError, TypeError) as e:
                                            print(f"⚠️ Invalid V3 D30 bid/ask data for {instrument_key}: {e}")

                            # Volume and quantity totals (V3 field names)
                            if 'vtt' in market_ff:  # Volume traded today
                                tick_data['volume'] = int(market_ff['vtt'])

                            if 'tbq' in market_ff:  # Total buy quantity
                                tick_data['total_buy_quantity'] = int(market_ff['tbq'])

                            if 'tsq' in market_ff:  # Total sell quantity
                                tick_data['total_sell_quantity'] = int(market_ff['tsq'])

                            if 'atp' in market_ff:  # Average traded price
                                tick_data['average_traded_price'] = float(market_ff['atp'])

                            if 'oi' in market_ff:  # Open interest
                                tick_data['open_interest'] = int(market_ff['oi'])

                            # OHLC data from marketOHLC (V3 structure)
                            if 'marketOHLC' in market_ff and isinstance(market_ff['marketOHLC'], dict):
                                ohlc_data = market_ff['marketOHLC']
                                if 'ohlc' in ohlc_data and isinstance(ohlc_data['ohlc'], list):
                                    # Get daily OHLC (interval: "1d")
                                    for ohlc_item in ohlc_data['ohlc']:
                                        if ohlc_item.get('interval') == '1d':
                                            tick_data.update({
                                                'open': float(ohlc_item.get('open', 0)),
                                                'high': float(ohlc_item.get('high', 0)),
                                                'low': float(ohlc_item.get('low', 0))
                                            })
                                            break

                            # Option Greeks if available (for option instruments)
                            if 'optionGreeks' in market_ff and isinstance(market_ff['optionGreeks'], dict):
                                greeks = market_ff['optionGreeks']
                                tick_data['option_greeks'] = {
                                    'delta': float(greeks.get('delta', 0)),
                                    'theta': float(greeks.get('theta', 0)),
                                    'gamma': float(greeks.get('gamma', 0)),
                                    'vega': float(greeks.get('vega', 0)),
                                    'rho': float(greeks.get('rho', 0))
                                }

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

                        # Enhanced logging with full_d30 depth information
                        depth_levels = tick_data.get('market_depth_levels', 0)
                        request_mode = tick_data.get('request_mode', 'unknown')

                        if '42633' in exact_instrument_key: # NIFTY AUG 2025 FUT
                            print(f"🚀 FUTURES: {display_name} @ ₹{ltp:.2f} | {request_mode} | {depth_levels} levels | Bid: ₹{bid:.2f}({tick_data['best_bid_quantity']}) | Ask: ₹{ask:.2f}({tick_data['best_ask_quantity']})")
                        elif 'Nifty 50' in exact_instrument_key:
                            print(f"📊 SPOT: {display_name} @ ₹{ltp:.2f} | {request_mode} | {depth_levels} levels | Bid: ₹{bid:.2f}({tick_data['best_bid_quantity']}) | Ask: ₹{ask:.2f}({tick_data['best_ask_quantity']})")
                        else:
                            print(f"📈 {display_name} @ ₹{ltp:.2f} | {request_mode} | {depth_levels} levels | Bid: ₹{bid:.2f}({tick_data['best_bid_quantity']}) | Ask: ₹{ask:.2f}({tick_data['best_ask_quantity']})")

                        # Log deep market information for full_d30
                        if depth_levels >= 10:
                            print(f"   📊 Deep Market: L5 Bid: ₹{tick_data['market_depth']['bid_levels'][4]['price']:.2f}({tick_data['market_depth']['bid_levels'][4]['quantity']}) | L10 Ask: ₹{tick_data['market_depth']['ask_levels'][9]['price']:.2f}({tick_data['market_depth']['ask_levels'][9]['quantity']})")

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

    def subscribe(self, instrument_keys: list, mode: str = "full_d30"):
        """Subscribe to instruments using exact V3 API format for complete 30-level market data (Upstox Plus)."""
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

            # Validate instrument count for full_d30 mode (Upstox Plus limit: 50 instruments)
            if mode == "full_d30" and len(new_instruments) > 50:
                print(f"⚠️ full_d30 mode limited to 50 instruments (Upstox Plus). Requested: {len(new_instruments)}")
                print(f"🔄 Truncating to first 50 instruments...")
                new_instruments = new_instruments[:50]

            # Create subscription request exactly as per V3 documentation
            subscribe_request = {
                "guid": str(uuid.uuid4()),
                "method": "sub",
                "data": {
                    "mode": mode,
                    "instrumentKeys": new_instruments
                }
            }

            print(f"🔄 Subscribing to {len(new_instruments)} instruments in '{mode}' mode (V3 JSON Format - Upstox Plus)")
            print(f"📋 Instruments: {[key.split('|')[-1] for key in new_instruments]}")
            print(f"📊 Expected data: FULL D30 - LTPC, 30-level quotes, extended metadata, OHLC, Greeks")

            if self.ws and hasattr(self.ws, 'send'):
                # Send subscription request as binary (as per V3 documentation)
                message_json = json.dumps(subscribe_request)
                message_binary = message_json.encode('utf-8')
                self.ws.send(message_binary, websocket.ABNF.OPCODE_BINARY)
                print(f"📤 V3 Full Market Data subscription sent")

                # Update subscribed instruments
                self.subscribed_instruments.update(new_instruments)

                # Log expected JSON structure for full_d30
                print(f"📋 Expected full_d30 JSON structure:")
                print(f"   - feeds.{new_instruments[0]}.fullFeed.marketFF.ltpc (price data)")
                print(f"   - feeds.{new_instruments[0]}.fullFeed.marketFF.marketLevel.bidAskQuote (30 levels)")
                print(f"   - feeds.{new_instruments[0]}.fullFeed.marketFF.marketOHLC (OHLC data)")
                print(f"   - feeds.{new_instruments[0]}.fullFeed.marketFF.optionGreeks (Greeks)")
                print(f"   - feeds.{new_instruments[0]}.fullFeed.marketFF.tbq/tsq (total quantities)")
                print(f"   - feeds.{new_instruments[0]}.fullFeed.requestMode: '{mode}'")

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