#!/usr/bin/env python3
"""
Debug Upstox authentication and token issues
"""

import os
import requests
import json
from utils.upstox_client import UpstoxClient

def test_current_token():
    """Test the current token from file"""
    try:
        with open('.upstox_token', 'r') as f:
            token = f.read().strip()
        
        if not token:
            print("❌ Token file is empty")
            return False
            
        print(f"📋 Found token: {token[:30]}...")
        
        # Test with a simple API call
        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }
        
        url = "https://api.upstox.com/v2/market-quote/quotes"
        params = {'instrument_key': 'NSE_INDEX|Nifty 50'}
        
        response = requests.get(url, headers=headers, params=params)
        
        print(f"🔍 API Response Status: {response.status_code}")
        print(f"📄 Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Token is valid!")
                return True
        
        print("❌ Token is expired/invalid")
        return False
        
    except Exception as e:
        print(f"❌ Error testing token: {e}")
        return False

def check_environment():
    """Check if environment variables are set"""
    api_key = os.getenv('UPSTOX_API_KEY')
    api_secret = os.getenv('UPSTOX_API_SECRET')
    
    print(f"🔑 API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"🔐 API Secret: {'✅ Set' if api_secret else '❌ Missing'}")
    
    if api_key:
        print(f"📋 API Key preview: {api_key[:10]}...")
    
    return api_key and api_secret

def test_login_url():
    """Test if login URL can be generated"""
    try:
        client = UpstoxClient()
        login_url = client.get_login_url()
        print(f"🔗 Login URL: {login_url}")
        return True
    except Exception as e:
        print(f"❌ Error generating login URL: {e}")
        return False

def main():
    print("🔧 Upstox Authentication Debug Tool")
    print("=" * 50)
    
    # Check environment
    print("\n1. Checking environment variables...")
    env_ok = check_environment()
    
    # Test current token
    print("\n2. Testing current token...")
    token_ok = test_current_token()
    
    # Test login URL generation
    print("\n3. Testing login URL generation...")
    login_ok = test_login_url()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 SUMMARY")
    print(f"Environment: {'✅ OK' if env_ok else '❌ ISSUE'}")
    print(f"Token: {'✅ VALID' if token_ok else '❌ EXPIRED'}")
    print(f"Login URL: {'✅ OK' if login_ok else '❌ ISSUE'}")
    
    if not token_ok:
        print("\n💡 SOLUTION:")
        print("1. Go to your Streamlit app")
        print("2. Click 'Upstox Data' in sidebar")
        print("3. Click 'Refresh Token' button")
        print("4. Click 'Login to Upstox' button")
        print("5. Complete authentication process")
        print("6. Check that you're redirected back to the app")
        
        if not env_ok:
            print("\n⚠️  ISSUE: Environment variables missing")
            print("Please ensure UPSTOX_API_KEY and UPSTOX_API_SECRET are set")

if __name__ == "__main__":
    main()