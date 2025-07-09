
import upstox_client
import threading
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Callable, Optional, Dict, Any
import json

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
        
        # Configure Upstox API client
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        self.api_client = upstox_client.ApiClient(configuration)
        self.websocket_api = upstox_client.WebSocketApi(self.api_client)
        
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
    
    def on_close(self, ws):
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
            # The official SDK handles protobuf parsing automatically
            # Message should already be parsed into Python dict/object
            if hasattr(message, 'feeds') and message.feeds:
                tick_data = self.parse_tick_data(message)
                if tick_data and self.tick_callback:
                    self.tick_callback(tick_data)
            elif isinstance(message, dict):
                # Handle dict format messages
                tick_data = self.parse_tick_data_dict(message)
                if tick_data and self.tick_callback:
                    self.tick_callback(tick_data)
                    
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def parse_tick_data(self, message) -> Optional[Dict]:
        """Parse tick data from SDK message object."""
        try:
            feeds = message.feeds if hasattr(message, 'feeds') else {}
            if not feeds:
                return None
            
            # Extract first instrument data
            instrument_key = list(feeds.keys())[0]
            feed_data = feeds[instrument_key]
            
            tick = {
                'instrument_token': instrument_key,
                'timestamp': datetime.now(),
                'ltp': getattr(feed_data, 'ltp', 0),
                'ltq': getattr(feed_data, 'ltq', 0),
                'ltt': getattr(feed_data, 'ltt', None),
                'bid_price': getattr(feed_data, 'bp1', 0),
                'ask_price': getattr(feed_data, 'ap1', 0),
                'bid_qty': getattr(feed_data, 'bq1', 0),
                'ask_qty': getattr(feed_data, 'aq1', 0),
                'volume': getattr(feed_data, 'vol', 0),
                'open': getattr(feed_data, 'open', 0),
                'high': getattr(feed_data, 'high', 0),
                'low': getattr(feed_data, 'low', 0),
                'close': getattr(feed_data, 'close', 0),
                'change': getattr(feed_data, 'chng', 0),
                'change_percent': getattr(feed_data, 'chngPer', 0)
            }
            
            # Store latest tick data
            self.last_tick_data[instrument_key] = tick
            return tick
            
        except Exception as e:
            print(f"Error parsing tick data: {e}")
            return None
    
    def parse_tick_data_dict(self, data: Dict) -> Optional[Dict]:
        """Parse tick data from dictionary format."""
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
                'change': feed_data.get('chng', 0),
                'change_percent': feed_data.get('chngPer', 0)
            }
            
            # Store latest tick data
            self.last_tick_data[instrument_key] = tick
            return tick
            
        except Exception as e:
            print(f"Error parsing dict tick data: {e}")
            return None
    
    def connect(self):
        """Establish WebSocket connection using official SDK."""
        try:
            # Get WebSocket URL from the API
            websocket_api_instance = upstox_client.WebSocketApi(self.api_client)
            
            # Configure WebSocket callbacks
            def on_message_wrapper(ws, message):
                self.on_message(ws, message)
            
            def on_open_wrapper(ws):
                self.on_open(ws)
            
            def on_close_wrapper(ws):
                self.on_close(ws)
            
            def on_error_wrapper(ws, error):
                self.on_error(ws, error)
            
            # Start WebSocket connection
            self.ws = websocket_api_instance.get_market_data_feed_authorize(
                api_version='2.0',
                on_message=on_message_wrapper,
                on_open=on_open_wrapper,
                on_close=on_close_wrapper,
                on_error=on_error_wrapper
            )
            
            return True
            
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
        """Subscribe to instruments for live data using official SDK."""
        if not self.is_connected:
            print("❌ WebSocket not connected. Please connect first.")
            return False
        
        try:
            # Convert mode to SDK format
            mode_mapping = {
                "ltpc": "ltpc",
                "full": "full", 
                "quote": "quote"
            }
            sdk_mode = mode_mapping.get(mode, "full")
            
            # Subscribe using the official SDK method
            subscribe_request = {
                "action": "subscribe",
                "data": {
                    "mode": sdk_mode,
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
        """Unsubscribe from instruments using official SDK."""
        if not self.is_connected:
            return False
        
        try:
            unsubscribe_request = {
                "action": "unsubscribe",
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
