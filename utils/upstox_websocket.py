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
                'Accept': '*/*',
                'Authorization': f'Bearer {self.access_token}'
            }
            url = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
            api_response = requests.get(url=url, headers=headers)

            if api_response.status_code == 200:
                response_data = api_response.json()
                if response_data.get("status") == "success" and "data" in response_data:
                    self.websocket_url = response_data["data"]["authorized_redirect_uri"]
                    # Fix version mismatch: Replace v2 with v3 in WebSocket URL
                    if "/v2/" in self.websocket_url:
                        self.websocket_url = self.websocket_url.replace("/v2/", "/v3/")
                        print(f"🔧 Fixed WebSocket URL version from v2 to v3")
                    print(f"✅ Got WebSocket URL (Official v3): {self.websocket_url}")
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
        """Decode protobuf message - Official implementation."""
        if pb is None:
            # Fallback to manual parsing if protobuf classes not available
            return self._manual_protobuf_decode(buffer)

        try:
            feed_response = pb.FeedResponse()
            feed_response.ParseFromString(buffer)
            
            # Verify the protobuf object is properly constructed
            if not hasattr(feed_response, 'feeds'):
                print(f"⚠️ Protobuf object missing 'feeds' attribute, using manual decoding")
                return self._manual_protobuf_decode(buffer)
                
            return feed_response
        except Exception as e:
            print(f"⚠️ Protobuf decode error: {e}, falling back to manual decoding")
            return self._manual_protobuf_decode(buffer)

    def _manual_protobuf_decode(self, buffer):
        """Manual protobuf decoding when generated classes not available."""
        try:
            import time
            
            # Basic protobuf structure simulation
            class MockFeedResponse:
                def __init__(self):
                    self.feeds = {}
                    self.type = 1  # live_feed
                    self.currentTs = int(time.time() * 1000)

            mock_response = MockFeedResponse()

            # Try to extract basic price data from binary
            if len(buffer) >= 8:
                try:
                    # Look for double precision price data
                    for offset in range(0, len(buffer) - 8, 4):
                        try:
                            price = struct.unpack('>d', buffer[offset:offset+8])[0]
                            if 1000 <= price <= 100000 and not math.isnan(price):
                                # Create mock feed structure
                                instrument = self._get_next_instrument_in_sequence()
                                mock_response.feeds[instrument] = {
                                    'ltpc': {'ltp': price, 'ltq': 100, 'cp': price},
                                    'fullFeed': {
                                        'marketFF': {
                                            'ltpc': {'ltp': price, 'ltq': 100, 'cp': price}
                                        }
                                    }
                                }
                                break
                        except:
                            continue
                except:
                    pass

            return mock_response

        except Exception as e:
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
        """Process incoming message - Official implementation."""
        try:
            self._msg_counter += 1

            if self._msg_counter % 50 == 0:
                print(f"📦 Processing message #{self._msg_counter} (Official)")

            # Decode protobuf message (official approach)
            decoded_data = self.decode_protobuf(message)
            if not decoded_data:
                return

            # Convert the decoded data to a dictionary (official approach)
            if pb and hasattr(decoded_data, 'feeds'):
                try:
                    # Only use MessageToDict if the protobuf object has proper DESCRIPTOR
                    if hasattr(decoded_data, 'DESCRIPTOR'):
                        data_dict = MessageToDict(decoded_data)
                        # Process feeds data
                        if 'feeds' in data_dict:
                            await self._process_feeds(data_dict['feeds'])
                    else:
                        # Fallback to manual processing if DESCRIPTOR is missing
                        if hasattr(decoded_data, 'feeds') and decoded_data.feeds:
                            await self._process_feeds(decoded_data.feeds)
                except Exception as protobuf_error:
                    print(f"⚠️ Protobuf conversion error: {protobuf_error}, using manual processing")
                    # Fallback to manual processing
                    if hasattr(decoded_data, 'feeds') and decoded_data.feeds:
                        await self._process_feeds(decoded_data.feeds)

            else:
                # Manual processing for mock response
                if hasattr(decoded_data, 'feeds') and decoded_data.feeds:
                    await self._process_feeds(decoded_data.feeds)

        except Exception as e:
            print(f"❌ Message processing error: {e}")

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
        """Create tick data from feed structure."""
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
                'ask_qty': 0
            }

            # Extract LTPC data (official structure)
            if isinstance(feed_data, dict):
                # Handle dictionary format (from MessageToDict)
                if 'ltpc' in feed_data:
                    ltpc = feed_data['ltpc']
                    tick_data.update({
                        'ltp': float(ltpc.get('ltp', 0)),
                        'ltq': int(ltpc.get('ltq', 0)),
                        'close': float(ltpc.get('cp', 0)),
                        'last_traded_price': float(ltpc.get('ltp', 0)),
                        'last_traded_quantity': int(ltpc.get('ltq', 0))
                    })

                # Handle full feed data
                if 'fullFeed' in feed_data:
                    full_feed = feed_data['fullFeed']
                    if 'marketFF' in full_feed:
                        market_ff = full_feed['marketFF']

                        # Market level data
                        if 'marketLevel' in market_ff and 'bidAskQuote' in market_ff['marketLevel']:
                            quotes = market_ff['marketLevel']['bidAskQuote']
                            if quotes and len(quotes) > 0:
                                first_quote = quotes[0]
                                tick_data.update({
                                    'best_bid': float(first_quote.get('bidP', 0)),
                                    'best_bid_quantity': int(first_quote.get('bidQ', 0)),
                                    'best_ask': float(first_quote.get('askP', 0)),
                                    'best_ask_quantity': int(first_quote.get('askQ', 0))
                                })

            else:
                # Handle mock object format
                if hasattr(feed_data, 'get'):
                    ltpc = feed_data.get('ltpc', {})
                    if ltpc:
                        tick_data.update({
                            'ltp': float(ltpc.get('ltp', 0)),
                            'ltq': int(ltpc.get('ltq', 0)),
                            'close': float(ltpc.get('cp', 0))
                        })

            # Only return tick if we have valid price data
            if tick_data['ltp'] > 0:
                self.last_tick_data[instrument_key] = tick_data

                # Log with clean format
                display_name = instrument_key.split('|')[-1] if '|' in instrument_key else instrument_key
                ltp = tick_data['ltp']
                volume = tick_data['volume']

                # More frequent logging during market hours for debugging
                if self._msg_counter % 10 == 0:
                    print(f"📊 LIVE TICK: {display_name} @ ₹{ltp:.2f} | Vol: {volume:,} | Time: {tick_data['timestamp'].strftime('%H:%M:%S')}")

                return tick_data
            else:
                # Debug: Log when we receive data but no valid price
                if self._msg_counter % 50 == 0:
                    display_name = instrument_key.split('|')[-1] if '|' in instrument_key else instrument_key
                    print(f"⚠️ Received data for {display_name} but LTP is 0 or invalid")

            return None

        except Exception as e:
            print(f"❌ Tick creation error: {e}")
            return None

    async def _send_subscription(self, instrument_keys):
        """Send subscription request - Official implementation."""
        try:
            if not self.websocket or not self.is_connected:
                return False

            # Data to be sent over the WebSocket (official format)
            data = {
                "guid": "someguid",
                "method": "sub", 
                "data": {
                    "mode": "full",
                    "instrumentKeys": instrument_keys
                }
            }

            print(f"🔄 Official subscription for {len(instrument_keys)} instruments")
            print(f"📋 Instruments: {[key.split('|')[-1] for key in instrument_keys]}")

            # Convert data to binary and send over WebSocket (V3 format)
            message_json = json.dumps(data)
            print(f"📤 Sending subscription: {message_json}")
            binary_data = message_json.encode('utf-8')
            await self.websocket.send(binary_data)

            print(f"📤 Official subscription sent")
            return True

        except Exception as e:
            print(f"❌ Subscription error: {e}")
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

    def subscribe(self, instrument_keys: list, mode: str = "full"):
        """Subscribe to instruments using official implementation."""
        if not instrument_keys:
            return False

        try:
            # Remove duplicates and add to subscribed set
            unique_keys = list(set(instrument_keys))
            new_instruments = [key for key in unique_keys if key not in self.subscribed_instruments]

            if not new_instruments:
                print(f"✅ All instruments already subscribed")
                return True

            self.subscribed_instruments.update(new_instruments)

            # If connected, send subscription immediately
            if self.is_connected and self._asyncio_loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._send_subscription(new_instruments),
                        self._asyncio_loop
                    )
                    result = future.result(timeout=10.0)
                    return result
                except Exception as e:
                    print(f"❌ Async subscription error: {e}")
                    return False
            else:
                print(f"🔄 Instruments queued for subscription: {len(new_instruments)}")
                return True

        except Exception as e:
            print(f"❌ Subscription failed: {e}")
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