
import upstox_client
import websocket
import threading
import time
import pandas as pd
import streamlit as st
from datetime import datetime
import json
import struct
from typing import Callable, Optional, Dict, Any

class UpstoxWebSocketClient:
    """Real-time Upstox WebSocket client using official SDK."""
    
    def __init__(self, access_token: str, api_key: str):
        """Initialize Upstox WebSocket client with official SDK."""
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
        
        # Configure Upstox API client
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        self.api_client = upstox_client.ApiClient(configuration)
        
        # Get WebSocket URL
        self._get_websocket_url()
        
    def _get_websocket_url(self):
        """Get WebSocket URL from Upstox API."""
        try:
            websocket_api = upstox_client.WebSocketApi(self.api_client)
            api_response = websocket_api.get_market_data_feed_authorize(api_version='2.0')
            if hasattr(api_response, 'data') and hasattr(api_response.data, 'authorized_redirect_uri'):
                self.websocket_url = api_response.data.authorized_redirect_uri
                print(f"✅ Got WebSocket URL: {self.websocket_url}")
            else:
                print("❌ Failed to get WebSocket URL from API response")
        except Exception as e:
            print(f"❌ Error getting WebSocket URL: {e}")
            # Fallback to direct URL construction
            self.websocket_url = f"wss://ws-api.upstox.com/v3/ws/market-data-feed/marketdata?access_token={self.access_token}"
        
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
        print("✅ Upstox WebSocket connected successfully")
        if self.connection_callback:
            self.connection_callback("connected")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        self.is_connected = False
        print("❌ Upstox WebSocket connection closed")
        if self.connection_callback:
            self.connection_callback("disconnected")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"❌ Upstox WebSocket error: {error}")
        if self.error_callback:
            self.error_callback(error)
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            # Handle both binary (protobuf) and text (JSON) messages
            if isinstance(message, bytes):
                # Binary protobuf message - parse manually
                tick_data = self.parse_binary_message(message)
            else:
                # JSON message
                data = json.loads(message)
                tick_data = self.parse_json_message(data)
            
            if tick_data and self.tick_callback:
                self.tick_callback(tick_data)
                    
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def parse_binary_message(self, message: bytes) -> Optional[Dict]:
        """Parse binary protobuf message (simplified parsing)."""
        try:
            # This is a simplified parser - in production, you'd use proper protobuf
            # For now, we'll try to extract basic information
            if len(message) < 10:
                return None
            
            # Basic tick data structure (this is simplified)
            tick = {
                'instrument_token': 'binary_data',
                'timestamp': datetime.now(),
                'ltp': 0,
                'ltq': 0,
                'volume': 0,
                'bid_price': 0,
                'ask_price': 0,
                'bid_qty': 0,
                'ask_qty': 0,
                'open': 0,
                'high': 0,
                'low': 0,
                'close': 0,
                'change': 0,
                'change_percent': 0
            }
            
            return tick
            
        except Exception as e:
            print(f"Error parsing binary message: {e}")
            return None
    
    def parse_json_message(self, data: Dict) -> Optional[Dict]:
        """Parse JSON message format."""
        try:
            # Handle different message types
            if data.get('type') == 'feed':
                feeds = data.get('feeds', {})
                if not feeds:
                    return None
                
                # Extract first instrument data
                instrument_key = list(feeds.keys())[0]
                feed_data = feeds[instrument_key]
                
                tick = {
                    'instrument_token': instrument_key,
                    'timestamp': datetime.now(),
                    'ltp': feed_data.get('ltp', 0),
                    'ltq': feed_data.get('ltq', 0),
                    'ltt': feed_data.get('ltt'),
                    'bid_price': feed_data.get('bp1', 0),
                    'ask_price': feed_data.get('ap1', 0),
                    'bid_qty': feed_data.get('bq1', 0),
                    'ask_qty': feed_data.get('aq1', 0),
                    'volume': feed_data.get('vol', 0),
                    'open': feed_data.get('open', 0),
                    'high': feed_data.get('high', 0),
                    'low': feed_data.get('low', 0),
                    'close': feed_data.get('close', 0),
                    'change': feed_data.get('chng', 0),
                    'change_percent': feed_data.get('chngPer', 0)
                }
                
                # Store latest tick data
                self.last_tick_data[instrument_key] = tick
                return tick
            
            return None
            
        except Exception as e:
            print(f"Error parsing JSON message: {e}")
            return None
    
    def connect(self):
        """Establish WebSocket connection."""
        try:
            if not self.websocket_url:
                print("❌ No WebSocket URL available")
                return False
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.websocket_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket in a separate thread
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            
            # Wait for connection
            time.sleep(2)
            return self.is_connected
            
        except Exception as e:
            print(f"Failed to connect to Upstox WebSocket: {e}")
            return False
    
    def disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            self.is_connected = False
    
    def subscribe(self, instrument_keys: list, mode: str = "full"):
        """Subscribe to instruments for live data."""
        if not self.is_connected:
            print("❌ WebSocket not connected. Please connect first.")
            return False
        
        try:
            # Create subscription message
            subscribe_request = {
                "guid": "someguid",
                "method": "sub",
                "data": {
                    "mode": mode,
                    "instrumentKeys": instrument_keys
                }
            }
            
            # Send subscription through WebSocket
            if self.ws:
                self.ws.send(json.dumps(subscribe_request))
                
                # Update subscribed instruments
                self.subscribed_instruments.update(instrument_keys)
                
                print(f"✅ Subscribed to {len(instrument_keys)} instruments")
                return True
            
        except Exception as e:
            print(f"Failed to subscribe to instruments: {e}")
            return False
    
    def unsubscribe(self, instrument_keys: list):
        """Unsubscribe from instruments."""
        if not self.is_connected:
            return False
        
        try:
            unsubscribe_request = {
                "guid": "someguid",
                "method": "unsub",
                "data": {
                    "instrumentKeys": instrument_keys
                }
            }
            
            if self.ws:
                self.ws.send(json.dumps(unsubscribe_request))
                
                # Remove from subscribed instruments
                self.subscribed_instruments.difference_update(instrument_keys)
                
                print(f"✅ Unsubscribed from {len(instrument_keys)} instruments")
                return True
            
        except Exception as e:
            print(f"Failed to unsubscribe from instruments: {e}")
            return False
    
    def get_latest_tick(self, instrument_key: str) -> Optional[Dict]:
        """Get the latest tick data for an instrument."""
        return self.last_tick_data.get(instrument_key)
    
    def get_all_latest_ticks(self) -> Dict:
        """Get latest tick data for all subscribed instruments."""
        return self.last_tick_data.copy()
