try:
    import yfinance as yf
    import requests
    YF_AVAILABLE = True
except ImportError:
    yf = None
    requests = None
    YF_AVAILABLE = False
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit as st
from typing import Optional, Dict, List
import requests

class IndianMarketData:
    """Real-time Indian market data fetcher using Yahoo Finance"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_indian_stock_symbols(self) -> Dict[str, str]:
        """Common Indian stock symbols with their Yahoo Finance tickers"""
        return {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS', 
            'INFY': 'INFY.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'SBIN': 'SBIN.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'ITC': 'ITC.NS',
            'KOTAKBANK': 'KOTAKBANK.NS',
            'LT': 'LT.NS',
            'AXISBANK': 'AXISBANK.NS',
            'MARUTI': 'MARUTI.NS',
            'ASIANPAINT': 'ASIANPAINT.NS',
            'TITAN': 'TITAN.NS',
            'BAJFINANCE': 'BAJFINANCE.NS',
            'HCLTECH': 'HCLTECH.NS',
            'WIPRO': 'WIPRO.NS',
            'ULTRACEMCO': 'ULTRACEMCO.NS',
            'NESTLEIND': 'NESTLEIND.NS'
        }
    
    def get_nifty_symbols(self) -> Dict[str, str]:
        """Nifty index symbols"""
        return {
            'NIFTY50': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'NIFTYNEXT50': '^NSMIDCP',
            'NIFTYIT': '^CNXIT'
        }
    
    def fetch_realtime_data(self, symbol: str, period: str = "5d", interval: str = "5m") -> Optional[pd.DataFrame]:
        """
        Fetch real-time OHLC data for Indian stocks
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        """
        if yf is None:
            print("Creating demo Nifty 50 data for visualization...")
            return self._generate_demo_nifty_data(period=period)
            
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol, session=self.session)
            
            # Fetch historical data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data found for symbol: {symbol}, using demo data")
                return self._generate_demo_nifty_data(period=period)
            
            # Clean and format data
            data = data.dropna()
            
            # Keep only OHLCV columns if they exist
            available_cols = []
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for col in required_cols:
                if col in data.columns:
                    available_cols.append(col)
            
            if available_cols:
                data = data[available_cols]
            
            # Convert timezone-aware index to timezone-naive for consistency
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_convert('Asia/Kolkata').tz_localize(None)
            
            # Add metadata
            data.attrs = {'symbol': symbol, 'last_updated': datetime.now()}
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}, using demo data")
            return self._generate_demo_nifty_data(period=period)
    
    def _generate_demo_nifty_data(self, period: str = "5d") -> pd.DataFrame:
        """Generate realistic demo Nifty 50 data for visualization"""
        
        # Determine number of periods
        period_map = {
            "1d": 78,    # 78 5-minute candles in a trading day
            "5d": 390,   # 5 trading days
            "1mo": 1560, # ~20 trading days
            "3mo": 4680, # ~60 trading days
            "6mo": 9360, # ~120 trading days
            "1y": 18720  # ~240 trading days
        }
        
        num_periods = period_map.get(period, 390)
        
        # Generate time series (5-minute intervals, only during market hours)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=max(1, num_periods // 78))
        
        # Create 5-minute intervals during market hours (9:15 AM to 3:30 PM IST)
        dates = []
        current = start_time.replace(hour=9, minute=15, second=0, microsecond=0)
        
        while len(dates) < num_periods and current <= end_time:
            # Only add times during market hours
            if current.weekday() < 5 and 9*60+15 <= current.hour*60+current.minute <= 15*60+30:
                dates.append(current)
            
            current += timedelta(minutes=5)
            # Skip to next day after market close
            if current.hour*60+current.minute > 15*60+30:
                current = current.replace(hour=9, minute=15) + timedelta(days=1)
        
        # Generate realistic Nifty 50 price data
        np.random.seed(42)  # For reproducible demo data
        base_price = 24500  # Realistic Nifty 50 level
        
        # Generate price walk
        returns = np.random.normal(0, 0.001, len(dates))  # Small volatility for 5-min data
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.95))  # Prevent extreme drops
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC based on close price
            volatility = close * 0.002  # 0.2% volatility
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = prices[i-1] if i > 0 else close
            
            # Ensure OHLC logic
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume (higher during market open/close)
            hour = date.hour
            if hour in [9, 10, 14, 15]:  # Higher volume at open/close
                volume = np.random.randint(800000, 1200000)
            else:
                volume = np.random.randint(400000, 800000)
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data, index=dates[:len(data)])
        df.index.name = 'Datetime'
        
        # Add metadata
        df.attrs = {
            'symbol': '^NSEI',
            'last_updated': datetime.now(),
            'demo_mode': True
        }
        
        return df
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current market price and basic info"""
        if yf is None:
            print("yfinance library not available. Please install it to use real-time data features.")
            return None
            
        try:
            ticker = yf.Ticker(symbol, session=self.session)
            info = ticker.info
            
            current_data = {
                'symbol': symbol,
                'current_price': info.get('regularMarketPrice', 0),
                'previous_close': info.get('regularMarketPreviousClose', 0),
                'change': 0,
                'change_percent': 0,
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'company_name': info.get('longName', symbol)
            }
            
            if current_data['previous_close'] > 0:
                current_data['change'] = current_data['current_price'] - current_data['previous_close']
                current_data['change_percent'] = (current_data['change'] / current_data['previous_close']) * 100
            
            return current_data
            
        except Exception as e:
            print(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def update_dataset_with_realtime(self, existing_df: pd.DataFrame, symbol: str, 
                                   interval: str = "5m") -> pd.DataFrame:
        """
        Update existing dataset with latest real-time data
        
        Args:
            existing_df: Existing OHLC dataframe
            symbol: Stock symbol to fetch new data for
            interval: Data interval
        """
        try:
            # Get the last timestamp from existing data
            last_timestamp = existing_df.index.max()
            
            # Calculate how much new data we need
            now = datetime.now()
            time_diff = now - last_timestamp
            
            # Determine period based on time difference
            if time_diff.days > 5:
                period = "1mo"
            elif time_diff.days > 1:
                period = "5d"
            else:
                period = "1d"
            
            # Fetch new data
            new_data = self.fetch_realtime_data(symbol, period=period, interval=interval)
            
            if new_data is None or new_data.empty:
                print("No new data available")
                return existing_df
            
            # Filter only new data points
            new_data = new_data[new_data.index > last_timestamp]
            
            if new_data.empty:
                print("No new data points found")
                return existing_df
            
            # Combine with existing data
            updated_df = pd.concat([existing_df, new_data])
            updated_df = updated_df.sort_index()
            
            # Remove duplicates
            updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
            
            print(f"Added {len(new_data)} new data points")
            return updated_df
            
        except Exception as e:
            print(f"Error updating dataset: {str(e)}")
            return existing_df
    
    def is_market_open(self) -> bool:
        """Check if Indian market is currently open"""
        try:
            now = datetime.now()
            
            # Indian market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # Check if it's a weekday and within market hours
            is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
            is_market_hours = market_open <= now <= market_close
            
            return is_weekday and is_market_hours
            
        except Exception:
            return False
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists and has data"""
        try:
            ticker = yf.Ticker(symbol, session=self.session)
            info = ticker.info
            return 'regularMarketPrice' in info or 'currentPrice' in info
        except:
            return False
    
    def search_indian_stocks(self, query: str) -> List[Dict]:
        """Search for Indian stocks by name or symbol"""
        symbols = self.get_indian_stock_symbols()
        results = []
        
        query_lower = query.lower()
        
        for name, symbol in symbols.items():
            if query_lower in name.lower() or query_lower in symbol.lower():
                current_data = self.get_current_price(symbol)
                if current_data:
                    results.append({
                        'name': name,
                        'symbol': symbol,
                        'price': current_data['current_price'],
                        'change_percent': current_data['change_percent']
                    })
        
        return results[:10]  # Return top 10 matches