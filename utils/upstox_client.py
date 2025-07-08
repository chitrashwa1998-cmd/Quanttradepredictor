
import os
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import time
import json
from typing import Optional, Dict, Any, List
import urllib.parse

class UpstoxClient:
    """Upstox API client for fetching real-time and historical OHLC data."""
    
    def __init__(self):
        self.api_key = os.getenv('UPSTOX_API_KEY')
        self.api_secret = os.getenv('UPSTOX_API_SECRET')
        self.redirect_uri = f"https://{os.getenv('REPLIT_DEV_DOMAIN', 'localhost')}/oauth2callback"
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
    
    def test_connection(self) -> bool:
        """Test the API connection."""
        if not self.access_token:
            return False
        
        try:
            # Test by fetching user profile
            url = f"{self.base_url}/user/profile"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers)
            return response.status_code == 200
            
        except Exception:
            return False
