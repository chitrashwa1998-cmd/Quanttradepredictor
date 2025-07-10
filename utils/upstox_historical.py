
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time

class UpstoxHistoricalClient:
    """Client for fetching historical data from Upstox API."""
    
    def __init__(self, access_token: str, api_key: str):
        """Initialize Upstox historical client."""
        self.access_token = access_token
        self.api_key = api_key
        self.base_url = "https://api.upstox.com/v2"
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
        
    def get_historical_data(self, 
                          instrument_key: str,
                          interval: str = "5minute",
                          days_back: int = 5) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLC data from Upstox API.
        
        Args:
            instrument_key: Upstox instrument key (e.g., "NSE_INDEX|Nifty 50")
            interval: Data interval (1minute, 5minute, 30minute, 1day)
            days_back: Number of days to fetch data for
            
        Returns:
            DataFrame with OHLC data or None if failed
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Format dates for API
            to_date_str = to_date.strftime("%Y-%m-%d")
            from_date_str = from_date.strftime("%Y-%m-%d")
            
            # Construct API URL
            url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date_str}/{from_date_str}"
            
            print(f"📥 Fetching historical data from Upstox API...")
            print(f"   Instrument: {instrument_key}")
            print(f"   Interval: {interval}")
            print(f"   Date Range: {from_date_str} to {to_date_str}")
            
            # Make API request
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success" and "data" in data:
                    candles = data["data"]["candles"]
                    
                    if candles:
                        # Convert to DataFrame
                        df = pd.DataFrame(candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                        
                        # Convert timestamp to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                        
                        # Drop OI column and ensure numeric types
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        df = df.astype(float)
                        
                        # Sort by timestamp
                        df = df.sort_index()
                        
                        print(f"✅ Fetched {len(df)} historical candles")
                        print(f"   Date Range: {df.index.min()} to {df.index.max()}")
                        
                        return df
                    else:
                        print("❌ No candle data in API response")
                        return None
                else:
                    print(f"❌ API Error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return None
    
    def get_nifty_50_historical(self, days_back: int = 5) -> Optional[pd.DataFrame]:
        """Convenience method to fetch Nifty 50 historical data."""
        return self.get_historical_data("NSE_INDEX|Nifty 50", "5minute", days_back)
    
    def get_bank_nifty_historical(self, days_back: int = 5) -> Optional[pd.DataFrame]:
        """Convenience method to fetch Bank Nifty historical data."""
        return self.get_historical_data("NSE_INDEX|Nifty Bank", "5minute", days_back)
    
    def get_multiple_instruments_historical(self, 
                                          instruments: Dict[str, str],
                                          days_back: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple instruments.
        
        Args:
            instruments: Dict of {display_name: instrument_key}
            days_back: Number of days to fetch
            
        Returns:
            Dict of {instrument_key: DataFrame}
        """
        results = {}
        
        for display_name, instrument_key in instruments.items():
            print(f"\n📊 Fetching {display_name}...")
            
            data = self.get_historical_data(instrument_key, "5minute", days_back)
            if data is not None:
                results[instrument_key] = data
                print(f"✅ {display_name}: {len(data)} candles")
            else:
                print(f"❌ {display_name}: Failed to fetch")
            
            # Rate limiting - wait between requests
            time.sleep(0.5)
        
        return results
    
    def get_instrument_list(self) -> Optional[pd.DataFrame]:
        """Fetch complete instrument list from Upstox API."""
        try:
            url = f"{self.base_url}/market-quote/instruments"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                # This will be a large CSV file
                df = pd.read_csv(response.content.decode('utf-8'))
                print(f"✅ Fetched {len(df)} instruments")
                return df
            else:
                print(f"❌ Error fetching instruments: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching instrument list: {e}")
            return None
    
    def search_instruments(self, search_term: str) -> Optional[pd.DataFrame]:
        """Search for instruments by name."""
        instruments = self.get_instrument_list()
        
        if instruments is not None:
            # Search in instrument name and trading symbol
            mask = (
                instruments['instrument_name'].str.contains(search_term, case=False, na=False) |
                instruments['trading_symbol'].str.contains(search_term, case=False, na=False)
            )
            results = instruments[mask]
            
            print(f"🔍 Found {len(results)} instruments matching '{search_term}'")
            return results
        
        return None
