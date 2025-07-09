
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
        
        # Method 1: Try to get access token from file
        token_file = ".upstox_token"
        token = None
        
        print(f"🔍 Looking for token file: {os.path.abspath(token_file)}")
        
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                    
                if token:
                    print(f"✅ Found token in file: {token[:20]}...")
                else:
                    print("⚠️ Token file is empty")
                    
            except Exception as e:
                print(f"⚠️ Error reading token file: {e}")
        
        # Method 2: Manual token input if file method fails
        if not token:
            print("\n" + "="*60)
            print("🔑 TOKEN INPUT REQUIRED")
            print("="*60)
            print("Since the token file wasn't found, please:")
            print("1. Go to your Streamlit app")
            print("2. Navigate to 'Upstox Data' page")
            print("3. Authenticate and get your token")
            print("4. Copy the token that appears in the debug section")
            print("5. Paste it below:")
            print()
            
            token = input("📋 Paste your Upstox access token here: ").strip()
            
            if not token:
                print("❌ No token provided. Exiting.")
                return
                
            # Save token to file for future use
            try:
                with open(token_file, 'w') as f:
                    f.write(token)
                print(f"💾 Token saved to {token_file} for future use")
            except Exception as e:
                print(f"⚠️ Could not save token to file: {e}")
        
        # Set the token
        upstox_client.set_access_token(token)
        print(f"🔗 Using access token: {token[:20]}...")
        
        # Test API connectivity first
        print("🧪 Testing API connectivity...")
        try:
            quote = upstox_client.get_live_quote("NSE_INDEX|Nifty 50")
            if quote:
                print("✅ API test successful!")
                print(f"📊 Current NIFTY price: ₹{quote.get('ltp', 'N/A')}")
            else:
                print("❌ API test failed - invalid token or API issue")
                return
        except Exception as e:
            print(f"❌ API test failed: {e}")
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
                tick_count = 0
                while True:
                    tick = ws_client.get_latest_tick()
                    if tick:
                        tick_count += 1
                        print(f"💰 Live tick #{tick_count}: Price=₹{tick['ltp']:.2f}, Time={tick['timestamp'].strftime('%H:%M:%S')}")
                    
                    # Show current OHLC candle in progress
                    current_candle = ws_client.get_current_ohlc()
                    if current_candle and tick_count % 10 == 0:  # Show every 10 ticks
                        print(f"📈 Current candle: O={current_candle['Open']:.2f} H={current_candle['High']:.2f} L={current_candle['Low']:.2f} C={current_candle['Close']:.2f}")
                    
                    time.sleep(1)  # Check every second
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping WebSocket...")
                ws_client.disconnect()
                print("✅ WebSocket disconnected")
        else:
            print("❌ Failed to connect to WebSocket")
            print("💡 Check your token validity and try again")
            
    except Exception as e:
        print(f"❌ Error in WebSocket test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_websocket()
