import os
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import time
import json
from typing import Optional, Dict, Any, List
import urllib.parse
import websocket
import threading
import queue
from collections import defaultdict

class UpstoxClient:
    """Upstox API client for fetching real-time and historical OHLC data."""

    def __init__(self):
        self.api_key = os.getenv('UPSTOX_API_KEY')
        self.api_secret = os.getenv('UPSTOX_API_SECRET')
        self.redirect_uri = f"https://{os.getenv('REPLIT_DEV_DOMAIN', 'localhost')}/"
        self.base_url = "https://api.upstox.com/v2"
        self.access_token = None

        if not self.api_key or not self.api_secret:
            raise ValueError("UPSTOX_API_KEY and UPSTOX_API_SECRET must be set in environment variables")

    def get_login_url(self) -> str:
        """Generate the Upstox OAuth login URL."""
        params = {
            'response_type': 'code',
            'client_id': self.api_key,
            'redirect_uri': self.redirect_uri,
            'state': 'upstox_auth'
        }

        login_url = f"https://api.upstox.com/v2/login/authorization/dialog?" + urllib.parse.urlencode(params)
        return login_url

    def exchange_code_for_token(self, authorization_code: str) -> bool:
        """Exchange authorization code for access token."""
        try:
            url = f"{self.base_url}/login/authorization/token"

            data = {
                'code': authorization_code,
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code'
            }

            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }

            response = requests.post(url, data=data, headers=headers)

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')

                # Store token in session state for persistence
                if 'upstox_access_token' not in st.session_state:
                    st.session_state.upstox_access_token = self.access_token

                return True
            else:
                st.error(f"Token exchange failed: {response.text}")
                return False

        except Exception as e:
            st.error(f"Error exchanging code for token: {str(e)}")
            return False

    def set_access_token(self, token: str):
        """Set the access token manually."""
        self.access_token = token

    def get_nifty50_instruments(self) -> List[Dict]:
        """Get NIFTY 50 instrument list."""
        # NIFTY 50 index instrument key
        nifty50_instruments = [
            {
                'instrument_key': 'NSE_INDEX|Nifty 50',
                'name': 'NIFTY 50',
                'exchange': 'NSE_INDEX'
            }
        ]

        return nifty50_instruments

    def get_historical_data(self, instrument_key: str, interval: str = "5minute", 
                          from_date: str = None, to_date: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLC data from Upstox.

        Args:
            instrument_key: Instrument identifier (e.g., 'NSE_INDEX|Nifty 50')
            interval: Candle interval ('1minute', '5minute', '15minute', '30minute', '1hour', '1day')
            from_date: Start date in 'YYYY-MM-DD' format
            to_date: End date in 'YYYY-MM-DD' format
        """
        if not self.access_token:
            st.error("Access token not available. Please authenticate first.")
            return None

        try:
            # Set default dates if not provided
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"

            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()

                if data.get('status') == 'success' and 'data' in data:
                    candles = data['data']['candles']

                    if not candles:
                        st.warning("No data received from Upstox API")
                        return None

                    # Convert to DataFrame
                    df = pd.DataFrame(candles, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'])

                    # Convert DateTime to proper format
                    df['DateTime'] = pd.to_datetime(df['DateTime'])
                    df.set_index('DateTime', inplace=True)

                    # Remove OpenInterest column and keep only OHLCV
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                    # Sort by datetime
                    df.sort_index(inplace=True)

                    return df
                else:
                    st.error(f"API Error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                st.error(f"HTTP Error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
            return None

    def get_live_quote(self, instrument_key: str) -> Optional[Dict]:
        """Get live market quote for an instrument."""
        if not self.access_token:
            st.error("Access token not available. Please authenticate first.")
            return None

        try:
            url = f"{self.base_url}/market-quote/quotes"

            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }

            params = {
                'instrument_key': instrument_key
            }

            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if data.get('status') == 'success' and 'data' in data:
                    return data['data'][instrument_key]
                else:
                    st.error(f"API Error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                st.error(f"HTTP Error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            st.error(f"Error fetching live quote: {str(e)}")
            return None

    def fetch_nifty50_data(self, days: int = 30, interval: str = "5minute") -> Optional[pd.DataFrame]:
        """
        Convenience method to fetch NIFTY 50 data.

        Args:
            days: Number of days of historical data
            interval: Candle interval
        """
        instrument_key = "NSE_INDEX|Nifty 50"

        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        return self.get_historical_data(
            instrument_key=instrument_key,
            interval=interval,
            from_date=from_date,
            to_date=to_date
        )

    def is_authenticated(self) -> bool:
        """Check if the client is authenticated."""
        return self.access_token is not None

    def get_websocket_url(self) -> Optional[str]:
        """Get WebSocket URL for real-time data streaming."""
        if not self.access_token:
            print("❌ Access token not available. Please authenticate first.")
            return None

        try:
            url = f"{self.base_url}/feed/authorize"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }

            print(f"🔍 Requesting WebSocket URL from: {url}")
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                print(f"📡 WebSocket API Response: {data}")
                
                if data.get('status') == 'success' and 'data' in data:
                    ws_url = data['data']['authorizedRedirectUri']
                    print(f"✅ WebSocket URL obtained: {ws_url}")
                    return ws_url
                else:
                    error_msg = data.get('message', 'Unknown error')
                    print(f"❌ WebSocket authorization failed: {error_msg}")
                    return None
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error getting WebSocket URL: {str(e)}")
            return None


class UpstoxWebSocketClient:
    """WebSocket client for real-time NIFTY 50 data streaming."""

    def __init__(self, upstox_client: UpstoxClient):
        self.upstox_client = upstox_client
        self.ws = None
        self.ws_url = None
        self.is_connected = False
        self.tick_queue = queue.Queue()
        self.ohlc_builder = OHLCBuilder()
        self.callbacks = []
        
    def add_callback(self, callback):
        """Add callback function to receive OHLC updates."""
        self.callbacks.append(callback)
        
    def connect(self) -> bool:
        """Connect to Upstox WebSocket."""
        try:
            print("🔍 Getting WebSocket URL...")
            self.ws_url = self.upstox_client.get_websocket_url()
            if not self.ws_url:
                print("❌ Failed to get WebSocket URL")
                return False

            print(f"🚀 Connecting to WebSocket: {self.ws_url}")
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                header={
                    'Authorization': f'Bearer {self.upstox_client.access_token}'
                }
            )

            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()

            # Wait a bit to see if connection succeeds
            time.sleep(2)
            
            if self.is_connected:
                print("✅ WebSocket connected successfully!")
                return True
            else:
                print("❌ WebSocket connection failed")
                return False

        except Exception as e:
            print(f"❌ Error connecting to WebSocket: {str(e)}")
            return False

    def on_open(self, ws):
        """WebSocket connection opened."""
        self.is_connected = True
        print("🔗 WebSocket connection opened successfully!")
        
        # Subscribe to NIFTY 50 with proper authentication
        subscribe_message = {
            "guid": "someguid",
            "method": "sub",
            "data": {
                "mode": "full",
                "instrumentKeys": ["NSE_INDEX|Nifty 50"]
            }
        }
        
        print(f"📡 Sending subscription message: {subscribe_message}")
        try:
            ws.send(json.dumps(subscribe_message))
            print("✅ Subscription message sent successfully")
        except Exception as e:
            print(f"❌ Error sending subscription: {str(e)}")
            self.is_connected = False

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'feed':
                feeds = data.get('feeds', {})
                
                for instrument_key, feed_data in feeds.items():
                    if instrument_key == 'NSE_INDEX|Nifty 50':
                        tick = {
                            'instrument_key': instrument_key,
                            'ltp': feed_data.get('ltp', 0),
                            'volume': feed_data.get('volume', 0),
                            'timestamp': datetime.now()
                        }
                        
                        # Add tick to queue for processing
                        self.tick_queue.put(tick)
                        
                        # Process tick for OHLC building
                        ohlc_candle = self.ohlc_builder.process_tick(tick)
                        
                        # If a 5-minute candle is complete, notify callbacks
                        if ohlc_candle:
                            for callback in self.callbacks:
                                callback(ohlc_candle)

        except Exception as e:
            print(f"Error processing WebSocket message: {str(e)}")

    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        st.error(f"WebSocket error: {str(error)}")

    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed."""
        self.is_connected = False
        print(f"🔌 WebSocket disconnected - Status: {close_status_code}, Message: {close_msg}")
        
        # Check for authentication errors
        if close_status_code == 1006:
            print("❌ WebSocket closed abnormally - likely authentication issue")
            print("💡 Try refreshing your access token on the Upstox Data page")
        elif close_status_code == 1000:
            print("✅ WebSocket closed normally")
        elif close_status_code == 4001:
            print("❌ WebSocket authentication failed - invalid access token")
        elif close_status_code == 4003:
            print("❌ WebSocket authorization failed - token expired")
        else:
            print(f"⚠️ WebSocket closed with code: {close_status_code}")
            
        # Clear connection state
        self.ws = None

    def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws:
            self.ws.close()
            self.is_connected = False

    def get_latest_tick(self) -> Optional[Dict]:
        """Get the latest tick from queue."""
        try:
            return self.tick_queue.get_nowait()
        except queue.Empty:
            return None

    def get_current_ohlc(self) -> Optional[Dict]:
        """Get current 5-minute OHLC candle in progress."""
        return self.ohlc_builder.get_current_candle()


class OHLCBuilder:
    """Build 5-minute OHLC candles from real-time ticks."""

    def __init__(self):
        self.current_candle = None
        self.candle_start_time = None
        self.ticks_in_candle = []
        self.completed_candles = []

    def process_tick(self, tick: Dict) -> Optional[Dict]:
        """Process a tick and return completed OHLC candle if ready."""
        current_time = tick['timestamp']
        
        # Determine 5-minute window
        minute = current_time.minute
        candle_minute = (minute // 5) * 5
        window_start = current_time.replace(minute=candle_minute, second=0, microsecond=0)
        window_end = window_start + timedelta(minutes=5)

        # If this is a new candle period
        if self.candle_start_time != window_start:
            completed_candle = None
            
            # Complete previous candle if exists
            if self.current_candle and self.ticks_in_candle:
                completed_candle = self._finalize_candle()
                self.completed_candles.append(completed_candle)

            # Start new candle
            self._start_new_candle(window_start, tick)
            
            return completed_candle
        else:
            # Add tick to current candle
            self._update_current_candle(tick)
            return None

    def _start_new_candle(self, window_start: datetime, tick: Dict):
        """Start a new 5-minute candle."""
        self.candle_start_time = window_start
        self.current_candle = {
            'DateTime': window_start,
            'Open': tick['ltp'],
            'High': tick['ltp'],
            'Low': tick['ltp'],
            'Close': tick['ltp'],
            'Volume': tick['volume'],
            'tick_count': 1
        }
        self.ticks_in_candle = [tick]

    def _update_current_candle(self, tick: Dict):
        """Update current candle with new tick."""
        if self.current_candle:
            self.current_candle['High'] = max(self.current_candle['High'], tick['ltp'])
            self.current_candle['Low'] = min(self.current_candle['Low'], tick['ltp'])
            self.current_candle['Close'] = tick['ltp']
            self.current_candle['Volume'] += tick['volume']
            self.current_candle['tick_count'] += 1
            self.ticks_in_candle.append(tick)

    def _finalize_candle(self) -> Dict:
        """Finalize and return completed candle."""
        if self.current_candle:
            return {
                'DateTime': self.current_candle['DateTime'],
                'Open': self.current_candle['Open'],
                'High': self.current_candle['High'],
                'Low': self.current_candle['Low'],
                'Close': self.current_candle['Close'],
                'Volume': self.current_candle['Volume'],
                'tick_count': self.current_candle['tick_count']
            }
        return None

    def get_current_candle(self) -> Optional[Dict]:
        """Get current candle in progress."""
        return self.current_candle.copy() if self.current_candle else None

    def get_completed_candles_df(self) -> pd.DataFrame:
        """Get all completed candles as DataFrame."""
        if not self.completed_candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.completed_candles)
        df.set_index('DateTime', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Keep only OHLCV
        return df