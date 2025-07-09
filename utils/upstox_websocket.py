
import websocket
import json
import threading
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Callable, Optional, Dict, Any
import ssl

class UpstoxWebSocketClient:
    """Real-time Upstox WebSocket client for live market data."""
    
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
        
        # WebSocket URL for Upstox
        self.ws_url = "wss://ws-api.upstox.com/v3/ws/market-data-feed/marketdata"
        
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
        print(f"❌ Upstox WebSocket connection closed: {close_msg}")
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
            # Parse the binary message (Upstox sends protobuf format)
            # For this example, we'll assume JSON format for simplicity
            data = json.loads(message)
            
            # Process tick data
            if data.get('type') == 'feed':
                tick_data = self.parse_tick_data(data)
                if tick_data and self.tick_callback:
                    self.tick_callback(tick_data)
                    
        except json.JSONDecodeError:
            # Handle binary/protobuf data if needed
            pass
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def parse_tick_data(self, data: Dict) -> Optional[Dict]:
        """Parse raw tick data into standardized format."""
        try:
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
                'change': feed_data.get('change', 0),
                'change_percent': feed_data.get('change_percent', 0)
            }
            
            # Store latest tick data
            self.last_tick_data[instrument_key] = tick
            return tick
            
        except Exception as e:
            print(f"Error parsing tick data: {e}")
            return None
    
    def connect(self):
        """Establish WebSocket connection."""
        try:
            # Create WebSocket connection with headers
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Api-Key': self.api_key
            }
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                header=headers,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={'sslopt': {"cert_reqs": ssl.CERT_NONE}}
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to Upstox WebSocket: {e}")
            return False
    
    def disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.is_connected = False
    
    def subscribe(self, instrument_keys: list, mode: str = "full"):
        """Subscribe to instruments for live data."""
        if not self.is_connected:
            print("❌ WebSocket not connected. Please connect first.")
            return False
        
        try:
            # Prepare subscription message
            subscription_data = {
                "action": "subscribe",
                "correlationId": f"sub_{int(time.time())}",
                "data": {
                    "mode": mode,  # "ltpc", "full", "quote"
                    "instrumentKeys": instrument_keys
                }
            }
            
            # Send subscription request
            self.ws.send(json.dumps(subscription_data))
            
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
            unsubscription_data = {
                "action": "unsubscribe",
                "correlationId": f"unsub_{int(time.time())}",
                "data": {
                    "instrumentKeys": instrument_keys
                }
            }
            
            self.ws.send(json.dumps(unsubscription_data))
            
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
