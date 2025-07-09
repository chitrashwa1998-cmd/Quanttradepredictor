#!/usr/bin/env python3
"""
Fix Upstox token authentication and WebSocket connection issues
"""

import os
import requests
import json
from datetime import datetime, timedelta
from utils.upstox_client import UpstoxClient, UpstoxWebSocketClient

def check_token_validity():
    """Check if current token is valid by making a test API call."""
    try:
        # Try to read existing token
        token_file = ".upstox_token"
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                token = f.read().strip()
            
            if token:
                print(f"🔍 Found existing token: {token[:20]}...")
                
                # Test the token with a simple API call
                upstox_client = UpstoxClient()
                upstox_client.set_access_token(token)
                
                # Test API connectivity
                quote = upstox_client.get_live_quote("NSE_INDEX|Nifty 50")
                if quote:
                    print("✅ Token is valid and working!")
                    return True, token
                else:
                    print("❌ Token is expired or invalid")
                    return False, None
            else:
                print("⚠️ Token file is empty")
                return False, None
        else:
            print("⚠️ No token file found")
            return False, None
            
    except Exception as e:
        print(f"❌ Error checking token: {e}")
        return False, None

def save_token_to_file(token):
    """Save token to file for persistence."""
    try:
        with open(".upstox_token", 'w') as f:
            f.write(token)
        print(f"💾 Token saved to .upstox_token")
        return True
    except Exception as e:
        print(f"❌ Error saving token: {e}")
        return False

def get_fresh_token():
    """Get a fresh token from user input."""
    print("\n" + "="*80)
    print("🔑 UPSTOX TOKEN REFRESH REQUIRED")
    print("="*80)
    print()
    print("Your Upstox access token has expired. To get a new token:")
    print()
    print("1. 🌐 Go to your Streamlit app in the browser")
    print("2. 📡 Click on 'Upstox Data' in the sidebar")
    print("3. 🚀 Click 'Login to Upstox' button")
    print("4. 🔐 Complete the authentication process")
    print("5. 📋 Look for the access token in the debug section")
    print("6. 📄 Copy the token and paste it below")
    print()
    print("The token looks like: eyJ0eXAiOiJKV1QiLCJr...")
    print()
    
    token = input("📋 Paste your new Upstox access token here: ").strip()
    
    if not token:
        print("❌ No token provided")
        return None
    
    if not token.startswith("eyJ"):
        print("⚠️ Token format looks incorrect. Tokens usually start with 'eyJ'")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            return None
    
    # Save token to file
    save_token_to_file(token)
    
    return token

def test_websocket_connection(token):
    """Test WebSocket connection with the given token."""
    try:
        print("\n🧪 Testing WebSocket connection...")
        
        # Initialize client
        upstox_client = UpstoxClient()
        upstox_client.set_access_token(token)
        
        # Test WebSocket URL generation
        ws_url = upstox_client.get_websocket_url()
        if not ws_url:
            print("❌ Failed to get WebSocket URL")
            return False
        
        print(f"✅ WebSocket URL obtained: {ws_url}")
        
        # Test WebSocket connection
        ws_client = UpstoxWebSocketClient(upstox_client)
        
        # Add a simple callback
        def on_data(data):
            print(f"📊 Received data: {data}")
        
        ws_client.add_callback(on_data)
        
        # Try to connect
        success = ws_client.connect()
        
        if success:
            print("✅ WebSocket connection successful!")
            print("🎯 WebSocket is now ready for real-time data streaming")
            
            # Wait a bit to see if we get any data
            import time
            print("⏳ Waiting 10 seconds to check for incoming data...")
            time.sleep(10)
            
            # Check if we're still connected
            if ws_client.is_connected:
                print("✅ WebSocket is stable and connected")
                ws_client.disconnect()
                return True
            else:
                print("❌ WebSocket disconnected during test")
                return False
        else:
            print("❌ WebSocket connection failed")
            return False
            
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")
        return False

def fix_upstox_token():
    """Main function to fix Upstox token issues."""
    print("🔧 TribexAlpha Upstox Token Fix Tool")
    print("=" * 50)
    
    # Check current token
    is_valid, current_token = check_token_validity()
    
    if is_valid:
        print("🎉 Your token is working fine!")
        print("The WebSocket disconnection might be due to a different issue.")
        
        # Test WebSocket anyway
        ws_success = test_websocket_connection(current_token)
        if ws_success:
            print("✅ WebSocket is working perfectly!")
        else:
            print("❌ WebSocket still has issues - this might be a network or API problem")
    else:
        print("🔄 Token needs to be refreshed")
        
        # Get new token
        new_token = get_fresh_token()
        if new_token:
            # Test the new token
            upstox_client = UpstoxClient()
            upstox_client.set_access_token(new_token)
            
            # Test API first
            quote = upstox_client.get_live_quote("NSE_INDEX|Nifty 50")
            if quote:
                print("✅ New token is valid!")
                print(f"📊 Current NIFTY price: ₹{quote.get('ltp', 'N/A')}")
                
                # Test WebSocket
                ws_success = test_websocket_connection(new_token)
                if ws_success:
                    print("\n🎉 SUCCESS! Your Upstox connection is now fixed!")
                    print("💡 You can now use the WebSocket in your Streamlit app")
                else:
                    print("❌ WebSocket still has issues")
            else:
                print("❌ New token is invalid")
        else:
            print("❌ Token refresh cancelled")
    
    print("\n" + "="*80)
    print("🔄 To use the fixed token in your Streamlit app:")
    print("1. Restart your Streamlit app")
    print("2. Go to Upstox Data page")
    print("3. The token should now work for WebSocket connections")
    print("="*80)

if __name__ == "__main__":
    fix_upstox_token()