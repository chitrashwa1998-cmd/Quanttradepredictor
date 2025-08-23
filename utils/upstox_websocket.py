import asyncio
import json
import ssl
import websockets
import requests
import threading
import pandas as pd
import streamlit as st
from datetime import datetime
import struct
import math
import time
from typing import Callable, Optional, Dict, Any
from google.protobuf.json_format import MessageToDict

# Import the generated protobuf classes (we'll create this file)
try:
    import MarketDataFeedV3_pb2 as pb
except ImportError:
    print("⚠️ MarketDataFeedV3_pb2 not found. Creating protobuf classes...")
    pb = None

class UpstoxWebSocketClient:
    """Real-time Upstox WebSocket client using official implementation with asyncio and protobuf."""

    def __init__(self, access_token: str, api_key: str):
        """Initialize Upstox WebSocket client."""
        self.access_token = access_token
        self.api_key = api_key
        self.websocket = None
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
        self._asyncio_loop = None
        self._websocket_task = None
        self._stop_event = None

        # Message counter for debugging
        self._msg_counter = 0

    def set_callbacks(self,
                     tick_callback: Optional[Callable] = None,
                     error_callback: Optional[Callable] = None,
                     connection_callback: Optional[Callable] = None):
        """Set callback functions for different events."""
        self.tick_callback = tick_callback
        self.error_callback = error_callback
        self.connection_callback = connection_callback

    def get_market_data_feed_authorize_v3(self):
        """Get authorization for market data feed - Official implementation."""
        try:
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.access_token}'
            }
            url = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
            api_response = requests.get(url=url, headers=headers)

            if api_response.status_code == 200:
                response_data = api_response.json()
                if response_data.get("status") == "success" and "data" in response_data:
                    self.websocket_url = response_data["data"]["authorized_redirect_uri"]
                    print(f"✅ Got Official V3 WebSocket URL: {self.websocket_url}")
                    return response_data
                else:
                    print(f"❌ API Error: {response_data}")
                    return None
            else:
                print(f"❌ HTTP Error {api_response.status_code}: {api_response.text}")
                return None

        except Exception as e:
            print(f"❌ Error getting WebSocket URL: {e}")
            return None

    def decode_protobuf(self, buffer):
        """Decode protobuf message - Official V3 implementation only."""
        try:
            # Use only the official generated protobuf classes as per V3 documentation
            if pb and hasattr(pb, 'FeedResponse'):
                feed_response = pb.FeedResponse()
                feed_response.ParseFromString(buffer)

                # Debug: Print what we got
                if self._msg_counter % 25 == 0:
                    print(f"🔍 Official V3 Protobuf parsed - Type: {feed_response.type}, Feeds count: {len(feed_response.feeds)}")
                    if feed_response.feeds:
                        print(f"🔍 Official V3 Feed keys: {list(feed_response.feeds.keys())}")

                return feed_response
            else:
                # If protobuf classes not available, return None (no fallback as per official docs)
                if self._msg_counter % 25 == 0:
                    print(f"❌ pb.FeedResponse not available - install protobuf classes")
                return None

        except Exception as e:
            if self._msg_counter % 25 == 0:
                print(f"❌ Official V3 Protobuf decode error: {e}")
            return None

    

    async def fetch_market_data(self):
        """Fetch market data using WebSocket - Official implementation style."""
        try:
            # Create default SSL context (official approach)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Get market data feed authorization
            response = self.get_market_data_feed_authorize_v3()
            if not response or not self.websocket_url:
                print("❌ Failed to get WebSocket authorization")
                return

            print(f"🔄 Connecting to official Upstox WebSocket...")

            # Connect to the WebSocket with SSL context (official approach)
            async with websockets.connect(self.websocket_url, ssl=ssl_context) as websocket:
                self.websocket = websocket
                self.is_connected = True
                self.connection_start_time = time.time()

                print('✅ Official WebSocket Connection established')

                if self.connection_callback:
                    self.connection_callback("connected")

                await asyncio.sleep(1)  # Wait for 1 second (official timing)

                # Subscribe to instruments if any are pending
                if self.subscribed_instruments:
                    await self._send_subscription(list(self.subscribed_instruments))

                # Continuously receive and decode data from WebSocket (official loop)
                while self._stop_event and not self._stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        await self._process_message(message)

                    except asyncio.TimeoutError:
                        # Check if this is during market hours
                        import pytz
                        ist = pytz.timezone('Asia/Kolkata')
                        current_time = datetime.now(ist)
                        current_hour = current_time.hour
                        current_minute = current_time.minute
                        current_weekday = current_time.weekday()

                        market_start = 9 * 60 + 15  # 9:15 AM
                        market_end = 15 * 60 + 30   # 3:30 PM
                        current_minutes = current_hour * 60 + current_minute

                        is_trading_hours = (current_weekday < 5 and
                                          market_start <= current_minutes <= market_end)

                        if is_trading_hours:
                            print(f"⚠️ MARKET OPEN but no tick data! Time: {current_time.strftime('%H:%M:%S IST')}")
                            print(f"🔍 Subscribed instruments: {list(self.subscribed_instruments)}")
                            print(f"📊 Last tick count: {len(self.last_tick_data)}")
                        else:
                            print(f"⏰ WebSocket timeout (Market closed: {current_time.strftime('%H:%M:%S IST')})")

                        await websocket.ping()

                    except websockets.exceptions.ConnectionClosed:
                        print("🔌 WebSocket connection closed")
                        break

                    except Exception as e:
                        print(f"❌ Error in message loop: {e}")
                        break

        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            if self.error_callback:
                self.error_callback(e)
        finally:
            self.is_connected = False
            if self.connection_callback:
                self.connection_callback("disconnected")

    async def _process_message(self, message):
        """Process incoming message - Official V3 implementation only."""
        try:
            self._msg_counter += 1

            # Debug logging
            if self._msg_counter % 25 == 0:
                print(f"📦 Processing official V3 message #{self._msg_counter}")
                print(f"🔍 Message size: {len(message)} bytes")

            # Decode using official protobuf only (no fallbacks)
            decoded_data = self.decode_protobuf(message)
            if not decoded_data:
                if self._msg_counter % 25 == 0:
                    print(f"⚠️ No decoded data from official V3 protobuf")
                return

            # Process feeds using official protobuf structure only
            if hasattr(decoded_data, 'feeds') and decoded_data.feeds:
                # Use MessageToDict for official protobuf objects (V3 documentation approach)
                if pb and hasattr(decoded_data, 'DESCRIPTOR'):
                    try:
                        data_dict = MessageToDict(decoded_data)
                        if 'feeds' in data_dict and data_dict['feeds']:
                            await self._process_feeds(data_dict['feeds'])
                        else:
                            if self._msg_counter % 25 == 0:
                                print(f"📋 Official V3 protobuf parsed but no feeds data")
                    except Exception as e:
                        if self._msg_counter % 25 == 0:
                            print(f"❌ Official V3 MessageToDict error: {e}")
                else:
                    if self._msg_counter % 25 == 0:
                        print(f"⚠️ Received non-protobuf object")
            else:
                if self._msg_counter % 100 == 0:
                    print(f"📋 Official V3 message received but no feeds")

        except Exception as e:
            print(f"❌ Official V3 message processing error: {e}")

    async def _process_feeds(self, feeds_data):
        """Process feeds data from official protobuf structure."""
        try:
            for instrument_key, feed_data in feeds_data.items():
                tick_data = self._create_tick_from_feed(instrument_key, feed_data)
                if tick_data and self.tick_callback:
                    # Call tick callback in thread-safe manner
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.tick_callback, tick_data
                    )

        except Exception as e:
            print(f"❌ Feeds processing error: {e}")

    def _create_tick_from_feed(self, instrument_key, feed_data):
        """Create tick data from feed structure - Optimized for full_d30 mode."""
        try:
            tick_data = {
                'instrument_token': instrument_key,
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
                'ask_qty': 0,
                'market_depth_levels': 0,
                'all_bid_prices': [],
                'all_ask_prices': [],
                'all_bid_quantities': [],
                'all_ask_quantities': []
            }

            # Extract data based on feed structure (full_d30 mode prioritized)
            if isinstance(feed_data, dict):
                
                # **PRIORITY 1: Handle full_d30 structure (main path)**
                if 'fullFeed' in feed_data:
                    full_feed = feed_data['fullFeed']
                    
                    # Determine market data type (equity/futures vs index)
                    market_data = None
                    feed_type = None
                    
                    if 'marketFF' in full_feed:
                        market_data = full_feed['marketFF']
                        feed_type = 'marketFF'
                    elif 'indexFF' in full_feed:
                        market_data = full_feed['indexFF']
                        feed_type = 'indexFF'
                    
                    if market_data:
                        # **Extract LTPC data (core pricing)**
                        if 'ltpc' in market_data:
                            ltpc = market_data['ltpc']
                            tick_data.update({
                                'ltp': float(ltpc.get('ltp', 0)),
                                'ltq': int(ltpc.get('ltq', 0)),
                                'close': float(ltpc.get('cp', 0)),
                                'last_traded_price': float(ltpc.get('ltp', 0)),
                                'last_traded_quantity': int(ltpc.get('ltq', 0))
                            })

                        # **Extract full_d30 specific: 30-level market depth**
                        if 'marketLevel' in market_data:
                            market_level = market_data['marketLevel']
                            
                            # Get 30 levels of bid/ask quotes (full_d30 feature)
                            if 'bidAskQuote' in market_level:
                                quotes = market_level['bidAskQuote']
                                if quotes and len(quotes) > 0:
                                    # Store all 30 levels for full_d30
                                    bid_prices = []
                                    ask_prices = []
                                    bid_quantities = []
                                    ask_quantities = []
                                    
                                    for quote in quotes:
                                        bid_prices.append(float(quote.get('bP', 0)))
                                        ask_prices.append(float(quote.get('aP', 0)))
                                        bid_quantities.append(int(quote.get('bQ', 0)))
                                        ask_quantities.append(int(quote.get('aQ', 0)))
                                    
                                    tick_data.update({
                                        'market_depth_levels': len(quotes),
                                        'all_bid_prices': bid_prices,
                                        'all_ask_prices': ask_prices,
                                        'all_bid_quantities': bid_quantities,
                                        'all_ask_quantities': ask_quantities,
                                        'best_bid': bid_prices[0] if bid_prices else 0.0,
                                        'best_ask': ask_prices[0] if ask_prices else 0.0,
                                        'best_bid_quantity': bid_quantities[0] if bid_quantities else 0,
                                        'best_ask_quantity': ask_quantities[0] if ask_quantities else 0
                                    })

                        # **Extract additional full_d30 metadata**
                        # Volume and trading data
                        if 'vtt' in market_data:
                            tick_data['volume'] = int(market_data['vtt'])
                        if 'atp' in market_data:
                            tick_data['avg_traded_price'] = float(market_data['atp'])
                        if 'tbq' in market_data:
                            tick_data['total_buy_quantity'] = int(market_data['tbq'])
                        if 'tsq' in market_data:
                            tick_data['total_sell_quantity'] = int(market_data['tsq'])
                        
                        # OHLC data
                        if 'op' in market_data:
                            tick_data['open'] = float(market_data['op'])
                        if 'hp' in market_data:
                            tick_data['high'] = float(market_data['hp'])
                        if 'lp' in market_data:
                            tick_data['low'] = float(market_data['lp'])
                        
                        # Derivatives specific data
                        if 'oi' in market_data:
                            tick_data['open_interest'] = float(market_data['oi'])
                        if 'poi' in market_data:
                            tick_data['prev_open_interest'] = float(market_data['poi'])
                        
                        # **Option Greeks (full_d30 includes this)**
                        if 'optionGreeks' in market_data:
                            greeks = market_data['optionGreeks']
                            tick_data.update({
                                'delta': float(greeks.get('delta', 0)),
                                'theta': float(greeks.get('theta', 0)),
                                'gamma': float(greeks.get('gamma', 0)),
                                'vega': float(greeks.get('vega', 0)),
                                'rho': float(greeks.get('rho', 0)),
                                'iv': float(greeks.get('iv', 0))
                            })

                # **PRIORITY 2: Handle direct LTPC structure (fallback for ltpc mode)**
                elif 'ltpc' in feed_data:
                    ltpc = feed_data['ltpc']
                    tick_data.update({
                        'ltp': float(ltpc.get('ltp', 0)),
                        'ltq': int(ltpc.get('ltq', 0)),
                        'close': float(ltpc.get('cp', 0)),
                        'last_traded_price': float(ltpc.get('ltp', 0)),
                        'last_traded_quantity': int(ltpc.get('ltq', 0))
                    })

            else:
                # **Handle protobuf object format (non-dict)**
                if hasattr(feed_data, 'fullFeed'):
                    full_feed = feed_data.fullFeed
                    market_data = None
                    
                    if hasattr(full_feed, 'marketFF'):
                        market_data = full_feed.marketFF
                    elif hasattr(full_feed, 'indexFF'):
                        market_data = full_feed.indexFF
                    
                    if market_data and hasattr(market_data, 'ltpc'):
                        ltpc = market_data.ltpc
                        tick_data.update({
                            'ltp': float(getattr(ltpc, 'ltp', 0)),
                            'ltq': int(getattr(ltpc, 'ltq', 0)),
                            'close': float(getattr(ltpc, 'cp', 0))
                        })
                        
                elif hasattr(feed_data, 'ltpc'):
                    ltpc = feed_data.ltpc
                    tick_data.update({
                        'ltp': float(getattr(ltpc, 'ltp', 0)),
                        'ltq': int(getattr(ltpc, 'ltq', 0)),
                        'close': float(getattr(ltpc, 'cp', 0))
                    })

            # **Enhanced validation and logging for full_d30**
            if tick_data['ltp'] > 0:
                self.last_tick_data[instrument_key] = tick_data

                # Enhanced logging for full_d30 data
                display_name = instrument_key.split('|')[-1] if '|' in instrument_key else instrument_key
                ltp = tick_data['ltp']
                volume = tick_data['volume']
                depth_levels = tick_data.get('market_depth_levels', 0)
                
                # Log full_d30 specific data every few ticks
                if self._msg_counter % 10 == 0:
                    print(f"✅ FULL_D30: {display_name} @ ₹{ltp:.2f} | Vol: {volume:,} | Depth: {depth_levels}/30 levels")
                    if depth_levels > 0:
                        best_bid = tick_data.get('best_bid', 0)
                        best_ask = tick_data.get('best_ask', 0)
                        print(f"   📊 Best Bid: ₹{best_bid:.2f} | Best Ask: ₹{best_ask:.2f} | Spread: ₹{best_ask-best_bid:.2f}")

                return tick_data
            else:
                # **Enhanced debugging for full_d30 structure issues**
                display_name = instrument_key.split('|')[-1] if '|' in instrument_key else instrument_key
                
                if self._msg_counter % 25 == 0:
                    print(f"⚠️ NO LTP in FULL_D30 for {display_name}")
                    if isinstance(feed_data, dict):
                        print(f"🔍 Root keys: {list(feed_data.keys())}")
                        
                        if 'fullFeed' in feed_data:
                            full_feed = feed_data['fullFeed']
                            if isinstance(full_feed, dict):
                                print(f"🔍 FullFeed keys: {list(full_feed.keys())}")
                                
                                # Check market data structure
                                for feed_key in ['marketFF', 'indexFF']:
                                    if feed_key in full_feed and isinstance(full_feed[feed_key], dict):
                                        market_data = full_feed[feed_key]
                                        print(f"🔍 {feed_key} keys: {list(market_data.keys())}")
                                        
                                        if 'ltpc' in market_data:
                                            ltpc = market_data['ltpc']
                                            print(f"🔍 LTPC keys: {list(ltpc.keys()) if isinstance(ltpc, dict) else 'Not dict'}")

            return None

        except Exception as e:
            display_name = instrument_key.split('|')[-1] if '|' in instrument_key else instrument_key
            print(f"❌ FULL_D30 tick creation error for {display_name}: {e}")
            return None

    async def _send_subscription(self, instrument_keys, mode="full_d30"):
        """Send subscription request - Official V3 implementation with full_d30 mode."""
        try:
            if not self.websocket or not self.is_connected:
                return False

            # Official V3 subscription format exactly as per documentation
            subscription_request = {
                "guid": "tribex_full_d30_sub",
                "method": "sub",
                "data": {
                    "mode": mode,
                    "instrumentKeys": instrument_keys
                }
            }

            print(f"🔄 Official V3 FULL_D30 subscription for {len(instrument_keys)} instruments")
            print(f"📋 Mode: {mode} (30 market levels + complete data)")
            print(f"📋 Instruments: {[key.split('|')[-1] for key in instrument_keys]}")

            # Send subscription as JSON string over WebSocket (official V3 method)
            message = json.dumps(subscription_request)
            await self.websocket.send(message)

            print(f"📤 FULL_D30 subscription message sent successfully")
            return True

        except Exception as e:
            print(f"❌ FULL_D30 subscription error: {e}")
            return False

    def _get_next_instrument_in_sequence(self):
        """Get next instrument for round-robin assignment."""
        subscribed_list = list(self.subscribed_instruments)
        if not subscribed_list:
            return 'NSE_INDEX|Nifty 50'

        if len(subscribed_list) == 1:
            return subscribed_list[0]

        if not hasattr(self, '_sequence_counter'):
            self._sequence_counter = 0

        instrument = subscribed_list[self._sequence_counter % len(subscribed_list)]
        self._sequence_counter += 1

        return instrument

    def connect(self):
        """Establish WebSocket connection using official implementation."""
        try:
            if self.is_connected:
                return True

            print(f"🔄 Starting official Upstox WebSocket client...")

            # Create asyncio event loop in separate thread (official approach)
            def run_websocket():
                try:
                    self._asyncio_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._asyncio_loop)

                    self._stop_event = asyncio.Event()

                    # Run the official websocket client
                    self._asyncio_loop.run_until_complete(self.fetch_market_data())

                except Exception as e:
                    print(f"❌ Asyncio loop error: {e}")
                finally:
                    if self._asyncio_loop:
                        self._asyncio_loop.close()

            # Start WebSocket in separate thread
            self._websocket_thread = threading.Thread(target=run_websocket, daemon=True)
            self._websocket_thread.start()

            # Wait for connection
            max_wait = 15
            wait_time = 0
            while wait_time < max_wait and not self.is_connected:
                time.sleep(0.5)
                wait_time += 0.5

            if self.is_connected:
                print(f"✅ Official WebSocket connected in {wait_time:.1f}s")
            else:
                print(f"❌ Official WebSocket connection timeout after {max_wait}s")

            return self.is_connected

        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    def disconnect(self):
        """Close WebSocket connection."""
        print("🔌 Disconnecting official WebSocket...")

        try:
            # Signal stop to asyncio loop
            if self._stop_event and self._asyncio_loop:
                try:
                    if self._asyncio_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            self._stop_event.set(),
                            self._asyncio_loop
                        )
                        # Wait briefly for the signal to be processed
                        future.result(timeout=2.0)
                except Exception as e:
                    print(f"⚠️ Error setting stop event: {e}")

            # Close websocket connection
            if self.websocket and self._asyncio_loop:
                try:
                    if self._asyncio_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            self.websocket.close(),
                            self._asyncio_loop
                        )
                        # Wait briefly for close to complete
                        future.result(timeout=2.0)
                except Exception as e:
                    print(f"⚠️ Error closing websocket: {e}")

            self.is_connected = False

            # Wait for thread cleanup
            if hasattr(self, '_websocket_thread') and self._websocket_thread.is_alive():
                self._websocket_thread.join(timeout=5.0)

        except Exception as e:
            print(f"⚠️ Error during disconnect: {e}")

        # Clear data after operations complete
        self.subscribed_instruments.clear()
        self.last_tick_data.clear()
        self.websocket = None
        self._asyncio_loop = None
        self._stop_event = None

        print("🔌 Official WebSocket disconnected")

    def subscribe(self, instrument_keys: list, mode: str = "full_d30"):
        """Subscribe to instruments using full_d30 mode for complete market data."""
        if not instrument_keys:
            return False

        try:
            # Force full_d30 mode for complete market data (30 levels + all features)
            subscription_mode = "full_d30"
            
            # Remove duplicates and add to subscribed set
            unique_keys = list(set(instrument_keys))

            # Clear existing subscriptions to avoid conflicts
            self.subscribed_instruments.clear()
            self.subscribed_instruments.update(unique_keys)

            print(f"🎯 FULL_D30 Subscription Request:")
            print(f"   - Mode: {subscription_mode} (30 market levels + full data)")
            print(f"   - Instruments: {len(unique_keys)}")
            for key in unique_keys:
                display_name = key.split('|')[-1] if '|' in key else key
                print(f"   - {key} ({display_name})")

            # If connected, send subscription immediately
            if self.is_connected and self._asyncio_loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._send_subscription(unique_keys, subscription_mode),
                        self._asyncio_loop
                    )
                    result = future.result(timeout=15.0)

                    if result:
                        print(f"✅ FULL_D30 subscription successful for {len(unique_keys)} instruments")
                    else:
                        print(f"❌ FULL_D30 subscription failed")

                    return result
                except Exception as e:
                    print(f"❌ FULL_D30 async subscription error: {e}")
                    return False
            else:
                print(f"🔄 FULL_D30 instruments queued for subscription: {len(unique_keys)}")
                return True

        except Exception as e:
            print(f"❌ FULL_D30 subscription failed: {e}")
            return False

    def unsubscribe(self, instrument_keys: list):
        """Unsubscribe from instruments."""
        try:
            self.subscribed_instruments.difference_update(instrument_keys)

            # Remove from tick data
            for key in instrument_keys:
                self.last_tick_data.pop(key, None)

            print(f"✅ Unsubscribed from {len(instrument_keys)} instruments")
            return True

        except Exception as e:
            print(f"❌ Unsubscribe error: {e}")
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
            'implementation': 'Official Upstox asyncio/websockets'
        }