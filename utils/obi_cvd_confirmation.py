
import pandas as pd
import numpy as np
from typing import Dict, Optional, Deque
from collections import deque
from datetime import datetime
import pytz

class OBICVDConfirmation:
    """Order Book Imbalance and Cumulative Volume Delta confirmation for live predictions."""
    
    def __init__(self, cvd_reset_minutes: int = 30, obi_window_seconds: int = 60):
        """
        Initialize OBI+CVD confirmation module.
        
        Args:
            cvd_reset_minutes: Reset CVD accumulation every N minutes
            obi_window_seconds: Window for averaging OBI values
        """
        self.cvd_reset_minutes = cvd_reset_minutes
        self.obi_window_seconds = obi_window_seconds
        
        # Storage for each instrument
        self.instrument_data = {}  # Store CVD and OBI data per instrument
        self.last_cvd_reset = {}   # Track when CVD was last reset
        
    def _initialize_instrument(self, instrument_key: str):
        """Initialize data storage for a new instrument."""
        if instrument_key not in self.instrument_data:
            self.instrument_data[instrument_key] = {
                'cvd': 0.0,  # Cumulative Volume Delta
                'obi_history': deque(maxlen=self.obi_window_seconds * 2),  # Store OBI values with timestamps
                'last_price': 0.0,
                'last_volume': 0,
                'tick_count': 0
            }
            self.last_cvd_reset[instrument_key] = datetime.now()
    
    def calculate_obi(self, tick_data: Dict) -> Optional[float]:
        """
        Calculate Order Book Imbalance from tick data.
        
        OBI = (Bid Quantity - Ask Quantity) / (Bid Quantity + Ask Quantity)
        Range: -1 (only sellers) to +1 (only buyers)
        """
        try:
            # Extract bid/ask quantities from tick data
            bid_qty = tick_data.get('best_bid_quantity', 0) or tick_data.get('bid_qty', 0)
            ask_qty = tick_data.get('best_ask_quantity', 0) or tick_data.get('ask_qty', 0)
            
            # Also check for total buy/sell quantities if available
            total_buy = tick_data.get('total_buy_quantity', 0)
            total_sell = tick_data.get('total_sell_quantity', 0)
            
            # Use total quantities if available, otherwise use best bid/ask
            if total_buy > 0 and total_sell > 0:
                bid_qty = total_buy
                ask_qty = total_sell
            
            if bid_qty <= 0 and ask_qty <= 0:
                return None
            
            # Calculate OBI
            total_qty = bid_qty + ask_qty
            if total_qty == 0:
                return 0.0
            
            obi = (bid_qty - ask_qty) / total_qty
            return max(-1.0, min(1.0, obi))  # Clamp to [-1, 1]
            
        except Exception as e:
            print(f"❌ Error calculating OBI: {e}")
            return None
    
    def calculate_cvd_increment(self, tick_data: Dict, instrument_key: str) -> Optional[float]:
        """
        Calculate CVD increment from current tick.
        
        CVD = ∑(Buy Volume - Sell Volume)
        Uses price comparison to determine if volume is buy or sell initiated.
        """
        try:
            current_price = tick_data.get('ltp', 0) or tick_data.get('last_traded_price', 0)
            current_volume = tick_data.get('ltq', 0) or tick_data.get('last_traded_quantity', 0)
            
            if current_price <= 0 or current_volume <= 0:
                return None
            
            # Get previous price for comparison
            prev_data = self.instrument_data.get(instrument_key, {})
            prev_price = prev_data.get('last_price', current_price)
            
            # Classify volume as buy or sell based on price movement
            if current_price > prev_price:
                # Price up = buyer initiated
                volume_delta = current_volume
            elif current_price < prev_price:
                # Price down = seller initiated
                volume_delta = -current_volume
            else:
                # Price unchanged = use bid/ask comparison
                bid_price = tick_data.get('best_bid', 0) or tick_data.get('bid_price', 0)
                ask_price = tick_data.get('best_ask', 0) or tick_data.get('ask_price', 0)
                
                if ask_price > 0 and current_price >= ask_price:
                    volume_delta = current_volume  # Trade at ask = buy
                elif bid_price > 0 and current_price <= bid_price:
                    volume_delta = -current_volume  # Trade at bid = sell
                else:
                    volume_delta = 0  # Neutral
            
            # Update last price
            if instrument_key in self.instrument_data:
                self.instrument_data[instrument_key]['last_price'] = current_price
                self.instrument_data[instrument_key]['last_volume'] = current_volume
            
            return volume_delta
            
        except Exception as e:
            print(f"❌ Error calculating CVD increment: {e}")
            return None
    
    def update_confirmation(self, instrument_key: str, tick_data: Dict) -> Dict:
        """
        Update OBI and CVD for an instrument and return confirmation analysis.
        """
        try:
            # Initialize instrument if needed
            self._initialize_instrument(instrument_key)
            
            current_time = datetime.now()
            instrument_data = self.instrument_data[instrument_key]
            
            # Calculate OBI
            obi_value = self.calculate_obi(tick_data)
            if obi_value is not None:
                # Store OBI with timestamp
                instrument_data['obi_history'].append({
                    'timestamp': current_time,
                    'obi': obi_value
                })
            
            # Calculate CVD increment
            cvd_increment = self.calculate_cvd_increment(tick_data, instrument_key)
            if cvd_increment is not None:
                instrument_data['cvd'] += cvd_increment
            
            # Check if CVD should be reset
            time_since_reset = current_time - self.last_cvd_reset[instrument_key]
            if time_since_reset.total_seconds() > (self.cvd_reset_minutes * 60):
                print(f"🔄 Resetting CVD for {instrument_key} after {self.cvd_reset_minutes} minutes")
                instrument_data['cvd'] = 0.0
                self.last_cvd_reset[instrument_key] = current_time
            
            # Calculate averaged OBI
            recent_obi_values = [
                entry['obi'] for entry in instrument_data['obi_history']
                if (current_time - entry['timestamp']).total_seconds() <= self.obi_window_seconds
            ]
            
            avg_obi = np.mean(recent_obi_values) if recent_obi_values else 0.0
            
            # Increment tick counter
            instrument_data['tick_count'] += 1
            
            # Return confirmation analysis
            return self._analyze_confirmation(instrument_key, obi_value, avg_obi, instrument_data['cvd'])
            
        except Exception as e:
            print(f"❌ Error updating OBI+CVD confirmation for {instrument_key}: {e}")
            return {
                'error': str(e),
                'obi_current': 0.0,
                'obi_average': 0.0,
                'cvd_current': 0.0,
                'confirmation': 'Error'
            }
    
    def _analyze_confirmation(self, instrument_key: str, current_obi: float, avg_obi: float, current_cvd: float) -> Dict:
        """Analyze OBI and CVD to provide confirmation signals."""
        try:
            # OBI Analysis
            if avg_obi > 0.3:
                obi_signal = 'Strong Bullish'
            elif avg_obi > 0.1:
                obi_signal = 'Bullish'
            elif avg_obi < -0.3:
                obi_signal = 'Strong Bearish'
            elif avg_obi < -0.1:
                obi_signal = 'Bearish'
            else:
                obi_signal = 'Neutral'
            
            # CVD Analysis
            if current_cvd > 1000:
                cvd_signal = 'Strong Buying'
            elif current_cvd > 100:
                cvd_signal = 'Buying'
            elif current_cvd < -1000:
                cvd_signal = 'Strong Selling'
            elif current_cvd < -100:
                cvd_signal = 'Selling'
            else:
                cvd_signal = 'Neutral'
            
            # Combined Confirmation
            if ('Bullish' in obi_signal and 'Buying' in cvd_signal):
                combined_confirmation = 'Strong Bullish'
            elif ('Bearish' in obi_signal and 'Selling' in cvd_signal):
                combined_confirmation = 'Strong Bearish'
            elif ('Bullish' in obi_signal or 'Buying' in cvd_signal):
                combined_confirmation = 'Moderate Bullish'
            elif ('Bearish' in obi_signal or 'Selling' in cvd_signal):
                combined_confirmation = 'Moderate Bearish'
            else:
                combined_confirmation = 'Neutral'
            
            return {
                'obi_current': float(current_obi) if current_obi is not None else 0.0,
                'obi_average': float(avg_obi),
                'obi_signal': obi_signal,
                'cvd_current': float(current_cvd),
                'cvd_signal': cvd_signal,
                'combined_confirmation': combined_confirmation,
                'tick_count': self.instrument_data[instrument_key]['tick_count'],
                'last_update': datetime.now().strftime('%H:%M:%S')
            }
            
        except Exception as e:
            print(f"❌ Error analyzing confirmation: {e}")
            return {
                'obi_current': 0.0,
                'obi_average': 0.0,
                'obi_signal': 'Error',
                'cvd_current': 0.0,
                'cvd_signal': 'Error',
                'combined_confirmation': 'Error'
            }
    
    def get_confirmation_status(self, instrument_key: str) -> Optional[Dict]:
        """Get current confirmation status for an instrument."""
        if instrument_key not in self.instrument_data:
            return None
        
        data = self.instrument_data[instrument_key]
        current_time = datetime.now()
        
        # Calculate current averaged OBI
        recent_obi_values = [
            entry['obi'] for entry in data['obi_history']
            if (current_time - entry['timestamp']).total_seconds() <= self.obi_window_seconds
        ]
        
        avg_obi = np.mean(recent_obi_values) if recent_obi_values else 0.0
        
        return self._analyze_confirmation(instrument_key, 
                                        recent_obi_values[-1] if recent_obi_values else 0.0,
                                        avg_obi, 
                                        data['cvd'])
    
    def reset_instrument(self, instrument_key: str):
        """Reset all data for an instrument."""
        if instrument_key in self.instrument_data:
            del self.instrument_data[instrument_key]
        if instrument_key in self.last_cvd_reset:
            del self.last_cvd_reset[instrument_key]
        print(f"🔄 Reset OBI+CVD data for {instrument_key}")
