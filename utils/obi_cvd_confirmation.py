
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
        self.last_obi_reset = {}   # Track when OBI rolling average was last reset
        self.last_cvd_rolling_reset = {}  # Track when CVD rolling average was last reset
        
    def _initialize_instrument(self, instrument_key: str):
        """Initialize data storage for a new instrument."""
        if instrument_key not in self.instrument_data:
            self.instrument_data[instrument_key] = {
                'cvd': 0.0,  # Cumulative Volume Delta (total accumulation)
                'obi_history': deque(maxlen=self.obi_window_seconds * 3),  # Store OBI values with timestamps
                'cvd_history': deque(maxlen=120 * 2),  # Store CVD increments for 2-minute rolling average
                'current_obi': 0.0,  # Current tick OBI value
                'current_cvd_increment': 0.0,  # Current tick CVD increment
                'rolling_obi_1min': deque(maxlen=60 * 2),  # 1-minute rolling OBI values
                'rolling_cvd_2min': deque(maxlen=120 * 2),  # 2-minute rolling CVD values
                'last_price': 0.0,
                'last_volume': 0,
                'tick_count': 0
            }
            self.last_cvd_reset[instrument_key] = datetime.now()
            self.last_obi_reset[instrument_key] = datetime.now()
            self.last_cvd_rolling_reset[instrument_key] = datetime.now()
    
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
            
            # Calculate current OBI
            current_obi = self.calculate_obi(tick_data)
            if current_obi is not None:
                instrument_data['current_obi'] = current_obi
                
                # Store OBI with timestamp for historical tracking
                instrument_data['obi_history'].append({
                    'timestamp': current_time,
                    'obi': current_obi
                })
                
                # Add to 1-minute rolling OBI
                instrument_data['rolling_obi_1min'].append({
                    'timestamp': current_time,
                    'obi': current_obi
                })
            
            # Calculate current CVD increment
            current_cvd_increment = self.calculate_cvd_increment(tick_data, instrument_key)
            if current_cvd_increment is not None:
                instrument_data['current_cvd_increment'] = current_cvd_increment
                instrument_data['cvd'] += current_cvd_increment
                
                # Add to CVD history for rolling average
                instrument_data['cvd_history'].append({
                    'timestamp': current_time,
                    'cvd_increment': current_cvd_increment
                })
                
                # Add to 2-minute rolling CVD
                instrument_data['rolling_cvd_2min'].append({
                    'timestamp': current_time,
                    'cvd_increment': current_cvd_increment
                })
            
            # Check if CVD total should be reset (every 30 minutes)
            time_since_cvd_reset = current_time - self.last_cvd_reset[instrument_key]
            if time_since_cvd_reset.total_seconds() > (self.cvd_reset_minutes * 60):
                print(f"🔄 Resetting total CVD for {instrument_key} after {self.cvd_reset_minutes} minutes")
                instrument_data['cvd'] = 0.0
                self.last_cvd_reset[instrument_key] = current_time
            
            # Check if OBI rolling average should be reset (every 1 minute)
            time_since_obi_reset = current_time - self.last_obi_reset[instrument_key]
            if time_since_obi_reset.total_seconds() > 60:  # Reset every 1 minute
                print(f"🔄 Resetting OBI rolling average for {instrument_key} after 1 minute")
                instrument_data['rolling_obi_1min'].clear()
                self.last_obi_reset[instrument_key] = current_time
            
            # Check if CVD rolling average should be reset (every 2 minutes)
            time_since_cvd_rolling_reset = current_time - self.last_cvd_rolling_reset[instrument_key]
            if time_since_cvd_rolling_reset.total_seconds() > 120:  # Reset every 2 minutes
                print(f"🔄 Resetting CVD rolling average for {instrument_key} after 2 minutes")
                instrument_data['rolling_cvd_2min'].clear()
                self.last_cvd_rolling_reset[instrument_key] = current_time
            
            # Calculate rolling averages
            # OBI 1-minute rolling average
            recent_obi_1min = [
                entry['obi'] for entry in instrument_data['rolling_obi_1min']
                if (current_time - entry['timestamp']).total_seconds() <= 60
            ]
            rolling_avg_obi_1min = np.mean(recent_obi_1min) if recent_obi_1min else 0.0
            
            # CVD 2-minute rolling average
            recent_cvd_2min = [
                entry['cvd_increment'] for entry in instrument_data['rolling_cvd_2min']
                if (current_time - entry['timestamp']).total_seconds() <= 120
            ]
            rolling_avg_cvd_2min = np.mean(recent_cvd_2min) if recent_cvd_2min else 0.0
            
            # Legacy 60-second OBI average for existing analysis
            recent_obi_values = [
                entry['obi'] for entry in instrument_data['obi_history']
                if (current_time - entry['timestamp']).total_seconds() <= self.obi_window_seconds
            ]
            legacy_avg_obi = np.mean(recent_obi_values) if recent_obi_values else 0.0
            
            # Increment tick counter
            instrument_data['tick_count'] += 1
            
            # Return comprehensive analysis
            return self._analyze_granular_confirmation(
                instrument_key, 
                current_obi if current_obi is not None else 0.0, 
                rolling_avg_obi_1min,
                current_cvd_increment if current_cvd_increment is not None else 0.0,
                rolling_avg_cvd_2min,
                instrument_data['cvd'],
                legacy_avg_obi
            )
            
        except Exception as e:
            print(f"❌ Error updating OBI+CVD confirmation for {instrument_key}: {e}")
            return {
                'error': str(e),
                'obi_current': 0.0,
                'obi_rolling_1min': 0.0,
                'cvd_current_increment': 0.0,
                'cvd_rolling_2min': 0.0,
                'cvd_total': 0.0,
                'confirmation': 'Error'
            }
    
    def _analyze_granular_confirmation(self, instrument_key: str, current_obi: float, 
                                          rolling_obi_1min: float, current_cvd_increment: float,
                                          rolling_cvd_2min: float, total_cvd: float, legacy_avg_obi: float) -> Dict:
        """Analyze granular OBI and CVD to provide detailed confirmation signals."""
        try:
            # Handle None values for current_obi
            if current_obi is None:
                current_obi = 0.0
                
            # Handle None values for current_cvd_increment
            if current_cvd_increment is None:
                current_cvd_increment = 0.0
                
            # Current OBI Analysis (instantaneous)
            if current_obi > 0.5:
                current_obi_signal = 'Strong Bullish'
            elif current_obi > 0.2:
                current_obi_signal = 'Bullish'
            elif current_obi < -0.5:
                current_obi_signal = 'Strong Bearish'
            elif current_obi < -0.2:
                current_obi_signal = 'Bearish'
            else:
                current_obi_signal = 'Neutral'
            
            # Rolling OBI 1-minute Analysis
            if rolling_obi_1min > 0.3:
                rolling_obi_signal = 'Strong Bullish'
            elif rolling_obi_1min > 0.1:
                rolling_obi_signal = 'Bullish'
            elif rolling_obi_1min < -0.3:
                rolling_obi_signal = 'Strong Bearish'
            elif rolling_obi_1min < -0.1:
                rolling_obi_signal = 'Bearish'
            else:
                rolling_obi_signal = 'Neutral'
            
            # Current CVD Increment Analysis (tick-by-tick)
            if current_cvd_increment > 500:
                current_cvd_signal = 'Strong Buying'
            elif current_cvd_increment > 50:
                current_cvd_signal = 'Buying'
            elif current_cvd_increment < -500:
                current_cvd_signal = 'Strong Selling'
            elif current_cvd_increment < -50:
                current_cvd_signal = 'Selling'
            else:
                current_cvd_signal = 'Neutral'
            
            # Rolling CVD 2-minute Analysis
            if rolling_cvd_2min > 200:
                rolling_cvd_signal = 'Strong Buying'
            elif rolling_cvd_2min > 20:
                rolling_cvd_signal = 'Buying'
            elif rolling_cvd_2min < -200:
                rolling_cvd_signal = 'Strong Selling'
            elif rolling_cvd_2min < -20:
                rolling_cvd_signal = 'Selling'
            else:
                rolling_cvd_signal = 'Neutral'
            
            # Total CVD Analysis (cumulative)
            if total_cvd > 1000:
                total_cvd_signal = 'Strong Buying'
            elif total_cvd > 100:
                total_cvd_signal = 'Buying'
            elif total_cvd < -1000:
                total_cvd_signal = 'Strong Selling'
            elif total_cvd < -100:
                total_cvd_signal = 'Selling'
            else:
                total_cvd_signal = 'Neutral'
            
            # Combined Confirmation based on rolling averages
            if ('Bullish' in rolling_obi_signal and 'Buying' in rolling_cvd_signal):
                combined_confirmation = 'Strong Bullish'
            elif ('Bearish' in rolling_obi_signal and 'Selling' in rolling_cvd_signal):
                combined_confirmation = 'Strong Bearish'
            elif ('Bullish' in rolling_obi_signal or 'Buying' in rolling_cvd_signal):
                combined_confirmation = 'Moderate Bullish'
            elif ('Bearish' in rolling_obi_signal or 'Selling' in rolling_cvd_signal):
                combined_confirmation = 'Moderate Bearish'
            else:
                combined_confirmation = 'Neutral'
            
            return {
                # Current tick values
                'obi_current': float(current_obi) if current_obi is not None else 0.0,
                'obi_current_signal': current_obi_signal,
                'cvd_current_increment': float(current_cvd_increment) if current_cvd_increment is not None else 0.0,
                'cvd_current_signal': current_cvd_signal,
                
                # Rolling averages
                'obi_rolling_1min': float(rolling_obi_1min),
                'obi_rolling_signal': rolling_obi_signal,
                'cvd_rolling_2min': float(rolling_cvd_2min),
                'cvd_rolling_signal': rolling_cvd_signal,
                
                # Total accumulation
                'cvd_total': float(total_cvd),
                'cvd_total_signal': total_cvd_signal,
                
                # Legacy support
                'obi_average': float(legacy_avg_obi),  # Keep for backward compatibility
                'cvd_current': float(total_cvd),  # Legacy field
                'obi_signal': rolling_obi_signal,  # Use rolling for legacy
                'cvd_signal': rolling_cvd_signal,  # Use rolling for legacy
                
                # Overall confirmation
                'combined_confirmation': combined_confirmation,
                'tick_count': self.instrument_data[instrument_key]['tick_count'],
                'last_update': datetime.now().strftime('%H:%M:%S')
            }
            
        except Exception as e:
            print(f"❌ Error analyzing granular confirmation: {e}")
            return {
                'obi_current': 0.0,
                'obi_current_signal': 'Error',
                'cvd_current_increment': 0.0,
                'cvd_current_signal': 'Error',
                'obi_rolling_1min': 0.0,
                'obi_rolling_signal': 'Error',
                'cvd_rolling_2min': 0.0,
                'cvd_rolling_signal': 'Error',
                'cvd_total': 0.0,
                'cvd_total_signal': 'Error',
                'combined_confirmation': 'Error'
            }
    
    def get_confirmation_status(self, instrument_key: str) -> Optional[Dict]:
        """Get current confirmation status for an instrument."""
        if instrument_key not in self.instrument_data:
            return None
        
        data = self.instrument_data[instrument_key]
        current_time = datetime.now()
        
        # Calculate current values
        current_obi = data.get('current_obi', 0.0)
        current_cvd_increment = data.get('current_cvd_increment', 0.0)
        total_cvd = data.get('cvd', 0.0)
        
        # Calculate rolling averages
        # OBI 1-minute rolling average
        recent_obi_1min = [
            entry['obi'] for entry in data.get('rolling_obi_1min', [])
            if (current_time - entry['timestamp']).total_seconds() <= 60
        ]
        rolling_avg_obi_1min = np.mean(recent_obi_1min) if recent_obi_1min else 0.0
        
        # CVD 2-minute rolling average
        recent_cvd_2min = [
            entry['cvd_increment'] for entry in data.get('rolling_cvd_2min', [])
            if (current_time - entry['timestamp']).total_seconds() <= 120
        ]
        rolling_avg_cvd_2min = np.mean(recent_cvd_2min) if recent_cvd_2min else 0.0
        
        # Legacy 60-second OBI average for existing analysis
        recent_obi_values = [
            entry['obi'] for entry in data.get('obi_history', [])
            if (current_time - entry['timestamp']).total_seconds() <= self.obi_window_seconds
        ]
        legacy_avg_obi = np.mean(recent_obi_values) if recent_obi_values else 0.0
        
        return self._analyze_granular_confirmation(
            instrument_key, 
            current_obi if current_obi is not None else 0.0, 
            rolling_avg_obi_1min,
            current_cvd_increment if current_cvd_increment is not None else 0.0,
            rolling_avg_cvd_2min,
            total_cvd,
            legacy_avg_obi
        )
    
    def reset_instrument(self, instrument_key: str):
        """Reset all data for an instrument."""
        if instrument_key in self.instrument_data:
            del self.instrument_data[instrument_key]
        if instrument_key in self.last_cvd_reset:
            del self.last_cvd_reset[instrument_key]
        print(f"🔄 Reset OBI+CVD data for {instrument_key}")
