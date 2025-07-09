
#!/usr/bin/env python3
"""
Test WebSocket implementation for Upstox real-time data
"""

import os
import sys
import time
from utils.upstox_client import UpstoxClient, UpstoxWebSocketClient

def test_websocket():
    """Test WebSocket connection and data streaming."""
    
    # Check if authentication tokens are available
    if not os.getenv('UPSTOX_API_KEY') or not os.getenv('UPSTOX_API_SECRET'):
        print("❌ Upstox API credentials not set in environment variables")
        return
    
    print("🔧 Initializing Upstox client...")
    
    try:
        # Initialize Upstox client
        upstox_client = UpstoxClient()
        
        # Try to get access token from a temporary file
        token_file = ".upstox_token"
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                if token:
                    upstox_client.set_access_token(token)
                    print(f"✅ Using token from file: {token[:20]}...")
                else:
                    print("⚠️ Token file is empty")
                    print("💡 Go to the Upstox Data page and click 'Save Token for Console' first")
                    return
            except Exception as e:
                print(f"⚠️ Error reading token file: {e}")
                return
        else:
            print("⚠️ No token file found (.upstox_token)")
            print("💡 Go to the Upstox Data page and click 'Save Token for Console' first")
            return
        
        print("🔗 Creating WebSocket client...")
        ws_client = UpstoxWebSocketClient(upstox_client)
        
        # Add callback to print received OHLC data
        def on_ohlc_received(ohlc_candle):
            print(f"📊 New 5-min candle: {ohlc_candle}")
        
        ws_client.add_callback(on_ohlc_received)
        
        print("🚀 Connecting to WebSocket...")
        success = ws_client.connect()
        
        if success:
            print("✅ WebSocket connected! Streaming data...")
            print("Press Ctrl+C to stop")
            
            # Keep running and show live ticks
            try:
                while True:
                    tick = ws_client.get_latest_tick()
                    if tick:
                        print(f"💰 Live tick: ₹{tick['ltp']:.2f} at {tick['timestamp'].strftime('%H:%M:%S')}")
                    
                    current_candle = ws_client.get_current_ohlc()
                    if current_candle:
                        print(f"📈 Current candle: O:{current_candle['Open']:.2f} H:{current_candle['High']:.2f} L:{current_candle['Low']:.2f} C:{current_candle['Close']:.2f}")
                    
                    time.sleep(2)
                    
            except KeyboardInterrupt:
                print("\n⏹️ Stopping WebSocket...")
                ws_client.disconnect()
                print("✅ WebSocket disconnected")
        else:
            print("❌ Failed to connect WebSocket")
            
    except Exception as e:
        print(f"❌ Error during WebSocket test: {str(e)}")

if __name__ == "__main__":
    test_websocket()
