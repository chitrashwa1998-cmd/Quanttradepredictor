
import pandas as pd
import numpy as np
from typing import Dict, Optional, Deque, List
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
        self.last_cvd_reset = {}   # Track when CVD was last reset (30-min)
        self.last_cvd_hourly_reset = {}   # Track when CVD hourly was last reset (1-hour)
        self.last_obi_reset = {}   # Track when OBI 1-min rolling average was last reset
        self.last_obi_2min_reset = {}   # Track when OBI 2-min rolling average was last reset
        self.last_cvd_rolling_reset = {}  # Track when CVD rolling average was last reset
        
        # CVD Delta tracking for short timeframes
        self.last_cvd_1min_reset = {}   # Track when CVD 1-min was last reset
        self.last_cvd_2min_reset = {}   # Track when CVD 2-min was last reset  
        self.last_cvd_5min_reset = {}   # Track when CVD 5-min was last reset
        
        # Advanced liquidity analysis parameters
        self.wall_detection_multiplier = 3.0  # k value for wall detection
        self.wall_absorption_threshold = 0.7  # 70% execution threshold
        self.depth_history_size = 50  # Store last 50 depth snapshots for analysis
        
    def _initialize_instrument(self, instrument_key: str):
        """Initialize data storage for a new instrument."""
        if instrument_key not in self.instrument_data:
            self.instrument_data[instrument_key] = {
                'cvd': 0.0,  # Cumulative Volume Delta (30-minute accumulation)
                'cvd_hourly': 0.0,  # Cumulative Volume Delta (1-hour accumulation)
                'cvd_daily': 0.0,  # Cumulative Volume Delta (daily accumulation)
                'obi_history': deque(maxlen=self.obi_window_seconds * 3),  # Store OBI values with timestamps
                'cvd_history': deque(maxlen=120 * 2),  # Store CVD increments for 2-minute rolling average
                'current_obi': 0.0,  # Current tick OBI value
                'current_cvd_increment': 0.0,  # Current tick CVD increment
                'rolling_obi_1min': deque(maxlen=60 * 2),  # 1-minute rolling OBI values
                'rolling_obi_2min': deque(maxlen=120 * 2),  # 2-minute rolling OBI values
                'rolling_cvd_2min': deque(maxlen=120 * 2),  # 2-minute rolling CVD values
                
                # CVD Delta tracking for short timeframes
                'cvd_delta_1min': 0.0,  # CVD accumulation for 1-minute delta
                'cvd_delta_2min': 0.0,  # CVD accumulation for 2-minute delta
                'cvd_delta_5min': 0.0,  # CVD accumulation for 5-minute delta
                'cvd_delta_1min_prev': 0.0,  # Previous 1-min CVD for delta calculation
                'cvd_delta_2min_prev': 0.0,  # Previous 2-min CVD for delta calculation
                'cvd_delta_5min_prev': 0.0,  # Previous 5-min CVD for delta calculation
                
                # Advanced liquidity analysis storage
                'depth_history': deque(maxlen=self.depth_history_size),  # Store order book snapshots
                'detected_walls': {},  # Track detected walls by price level
                'wall_execution_history': {},  # Track execution against walls
                'liquidity_slope_bid': 0.0,  # Bid side liquidity slope
                'liquidity_slope_ask': 0.0,  # Ask side liquidity slope
                'liquidity_delta_bid': 0.0,  # Liquidity change on bid side
                'liquidity_delta_ask': 0.0,  # Liquidity change on ask side
                'absorption_ratios': {},  # Absorption ratios by price level
                'reload_walls': {},  # Track reload wall behavior
                
                'last_price': 0.0,
                'last_volume': 0,
                'tick_count': 0
            }
            self.last_cvd_reset[instrument_key] = datetime.now()
            self.last_cvd_hourly_reset[instrument_key] = datetime.now()
            self.last_obi_reset[instrument_key] = datetime.now()
            self.last_obi_2min_reset[instrument_key] = datetime.now()
            self.last_cvd_rolling_reset[instrument_key] = datetime.now()
            
            # Initialize CVD Delta reset timers
            self.last_cvd_1min_reset[instrument_key] = datetime.now()
            self.last_cvd_2min_reset[instrument_key] = datetime.now()
            self.last_cvd_5min_reset[instrument_key] = datetime.now()
    
    def calculate_obi(self, tick_data: Dict) -> Optional[float]:
        """
        Calculate Order Book Imbalance from tick data - ONLY for NSE_FO|53001.
        
        Enhanced to use 5-level market depth when available.
        OBI = (Bid Quantity - Ask Quantity) / (Bid Quantity + Ask Quantity)
        Range: -1 (only sellers) to +1 (only buyers)
        """
        try:
            # Strict check - only process if we have valid futures tick data
            instrument_token = tick_data.get('instrument_token', '')
            if '53001' not in str(instrument_token):
                print(f"⚠️ OBI calculation skipped - not from 53001 contract: {instrument_token}")
                return None
            
            # Check if 30-level depth data is available (enhanced data)
            total_bid_30_levels = tick_data.get('total_bid_quantity_30_levels', 0)
            total_ask_30_levels = tick_data.get('total_ask_quantity_30_levels', 0)
            
            if total_bid_30_levels > 0 and total_ask_30_levels > 0:
                # Use 30-level aggregated data for enhanced OBI accuracy
                bid_qty = total_bid_30_levels
                ask_qty = total_ask_30_levels
                print(f"🔍 Using 30-level OBI from 53001: Bid={bid_qty}, Ask={ask_qty}")
            else:
                # Only use Level 1 data from 53001 - NO FALLBACK
                bid_qty = tick_data.get('best_bid_quantity', 0) or tick_data.get('bid_qty', 0)
                ask_qty = tick_data.get('best_ask_quantity', 0) or tick_data.get('ask_qty', 0)
                
                # Also check for total buy/sell quantities if available
                total_buy = tick_data.get('total_buy_quantity', 0)
                total_sell = tick_data.get('total_sell_quantity', 0)
                
                # Use total quantities if available, otherwise use best bid/ask
                if total_buy > 0 and total_sell > 0:
                    bid_qty = total_buy
                    ask_qty = total_sell
            
            # Strict validation - must have valid bid/ask data from 53001
            if bid_qty <= 0 and ask_qty <= 0:
                print(f"⚠️ No valid bid/ask data from 53001 - waiting for data...")
                return None
            
            # Calculate OBI
            total_qty = bid_qty + ask_qty
            if total_qty == 0:
                return 0.0
            
            obi = (bid_qty - ask_qty) / total_qty
            return max(-1.0, min(1.0, obi))  # Clamp to [-1, 1]
            
        except Exception as e:
            print(f"❌ Error calculating OBI from 53001: {e}")
            return None
    
    def detect_liquidity_walls(self, depth_levels: List[Dict]) -> Dict:
        """
        Detect liquidity walls using abnormal level detection.
        Formula: Wall(p) = 1 if Size(p) > k * Median(Size(p-2:p+2))
        """
        try:
            if not depth_levels or len(depth_levels) < 5:
                return {'bid_walls': [], 'ask_walls': []}
            
            bid_walls = []
            ask_walls = []
            
            # Process bid levels
            bid_sizes = [level['bid_quantity'] for level in depth_levels if level['bid_quantity'] > 0]
            if len(bid_sizes) >= 5:
                for i in range(2, len(bid_sizes) - 2):
                    neighbor_start = max(0, i - 2)
                    neighbor_end = min(len(bid_sizes), i + 3)
                    neighbors = bid_sizes[neighbor_start:neighbor_end]
                    median_neighbor = np.median(neighbors)
                    
                    if bid_sizes[i] > self.wall_detection_multiplier * median_neighbor and median_neighbor > 0:
                        wall_info = {
                            'level': i + 1,
                            'price': depth_levels[i]['bid_price'],
                            'size': bid_sizes[i],
                            'median_neighbor': median_neighbor,
                            'strength': bid_sizes[i] / median_neighbor,
                            'type': 'support_wall'
                        }
                        bid_walls.append(wall_info)
            
            # Process ask levels
            ask_sizes = [level['ask_quantity'] for level in depth_levels if level['ask_quantity'] > 0]
            if len(ask_sizes) >= 5:
                for i in range(2, len(ask_sizes) - 2):
                    neighbor_start = max(0, i - 2)
                    neighbor_end = min(len(ask_sizes), i + 3)
                    neighbors = ask_sizes[neighbor_start:neighbor_end]
                    median_neighbor = np.median(neighbors)
                    
                    if ask_sizes[i] > self.wall_detection_multiplier * median_neighbor and median_neighbor > 0:
                        wall_info = {
                            'level': i + 1,
                            'price': depth_levels[i]['ask_price'],
                            'size': ask_sizes[i],
                            'median_neighbor': median_neighbor,
                            'strength': ask_sizes[i] / median_neighbor,
                            'type': 'resistance_wall'
                        }
                        ask_walls.append(wall_info)
            
            return {
                'bid_walls': bid_walls,
                'ask_walls': ask_walls,
                'total_walls': len(bid_walls) + len(ask_walls)
            }
            
        except Exception as e:
            print(f"❌ Error detecting liquidity walls: {e}")
            return {'bid_walls': [], 'ask_walls': [], 'total_walls': 0}
    
    def calculate_order_book_slope(self, depth_levels: List[Dict]) -> Dict:
        """
        Calculate order book slope (shape of liquidity).
        Formula: Slope = Σ(di - d̄)(Qi - Q̄) / Σ(di - d̄)²
        """
        try:
            if not depth_levels or len(depth_levels) < 3:
                return {'bid_slope': 0.0, 'ask_slope': 0.0, 'slope_asymmetry': 0.0}
            
            # Calculate bid slope
            bid_distances = []
            bid_quantities = []
            
            for i, level in enumerate(depth_levels):
                if level['bid_quantity'] > 0:
                    bid_distances.append(i + 1)  # Distance from best bid
                    bid_quantities.append(level['bid_quantity'])
            
            bid_slope = 0.0
            if len(bid_distances) >= 3:
                d_mean = np.mean(bid_distances)
                q_mean = np.mean(bid_quantities)
                
                numerator = sum((d - d_mean) * (q - q_mean) for d, q in zip(bid_distances, bid_quantities))
                denominator = sum((d - d_mean) ** 2 for d in bid_distances)
                
                if denominator > 0:
                    bid_slope = numerator / denominator
            
            # Calculate ask slope
            ask_distances = []
            ask_quantities = []
            
            for i, level in enumerate(depth_levels):
                if level['ask_quantity'] > 0:
                    ask_distances.append(i + 1)  # Distance from best ask
                    ask_quantities.append(level['ask_quantity'])
            
            ask_slope = 0.0
            if len(ask_distances) >= 3:
                d_mean = np.mean(ask_distances)
                q_mean = np.mean(ask_quantities)
                
                numerator = sum((d - d_mean) * (q - q_mean) for d, q in zip(ask_distances, ask_quantities))
                denominator = sum((d - d_mean) ** 2 for d in ask_distances)
                
                if denominator > 0:
                    ask_slope = numerator / denominator
            
            # Calculate slope asymmetry
            slope_asymmetry = bid_slope - ask_slope
            
            return {
                'bid_slope': float(bid_slope),
                'ask_slope': float(ask_slope),
                'slope_asymmetry': float(slope_asymmetry),
                'bid_slope_interpretation': self._interpret_slope(bid_slope, 'bid'),
                'ask_slope_interpretation': self._interpret_slope(ask_slope, 'ask')
            }
            
        except Exception as e:
            print(f"❌ Error calculating order book slope: {e}")
            return {'bid_slope': 0.0, 'ask_slope': 0.0, 'slope_asymmetry': 0.0}
    
    def _interpret_slope(self, slope: float, side: str) -> str:
        """Interpret the meaning of order book slope."""
        if abs(slope) < 0.1:
            return f"{side}_uniform_distribution"
        elif slope < -0.5:
            return f"{side}_steep_near_price"  # Aggressive defense
        elif slope > 0.5:
            return f"{side}_shallow_far_from_price"  # Passive defense
        elif slope < 0:
            return f"{side}_moderate_near_concentration"
        else:
            return f"{side}_moderate_far_concentration"
    
    def calculate_liquidity_delta(self, current_depth: List[Dict], previous_depth: List[Dict]) -> Dict:
        """
        Calculate liquidity delta (change in depth over time).
        Formula: ΔQi(t) = Qi(t) - Qi(t-1)
        """
        try:
            if not current_depth or not previous_depth:
                return {'delta_bid': 0.0, 'delta_ask': 0.0, 'net_liquidity_change': 0.0}
            
            delta_bid_total = 0.0
            delta_ask_total = 0.0
            
            # Create price-level mapping for comparison
            prev_bid_map = {level['bid_price']: level['bid_quantity'] for level in previous_depth if level['bid_price'] > 0}
            prev_ask_map = {level['ask_price']: level['ask_quantity'] for level in previous_depth if level['ask_price'] > 0}
            
            # Calculate bid side changes
            for level in current_depth:
                if level['bid_price'] > 0:
                    current_qty = level['bid_quantity']
                    previous_qty = prev_bid_map.get(level['bid_price'], 0)
                    delta = current_qty - previous_qty
                    
                    if delta > 0:  # Liquidity added
                        delta_bid_total += delta
            
            # Calculate ask side changes
            for level in current_depth:
                if level['ask_price'] > 0:
                    current_qty = level['ask_quantity']
                    previous_qty = prev_ask_map.get(level['ask_price'], 0)
                    delta = current_qty - previous_qty
                    
                    if delta > 0:  # Liquidity added
                        delta_ask_total += delta
            
            net_liquidity_change = delta_bid_total - delta_ask_total
            
            return {
                'delta_bid': float(delta_bid_total),
                'delta_ask': float(delta_ask_total),
                'net_liquidity_change': float(net_liquidity_change),
                'liquidity_sentiment': self._interpret_liquidity_delta(delta_bid_total, delta_ask_total)
            }
            
        except Exception as e:
            print(f"❌ Error calculating liquidity delta: {e}")
            return {'delta_bid': 0.0, 'delta_ask': 0.0, 'net_liquidity_change': 0.0}
    
    def _interpret_liquidity_delta(self, delta_bid: float, delta_ask: float) -> str:
        """Interpret liquidity delta changes."""
        if delta_bid > delta_ask * 1.5:
            return "buyers_adding_liquidity"
        elif delta_ask > delta_bid * 1.5:
            return "sellers_adding_liquidity"
        elif delta_bid < 0 and delta_ask < 0:
            return "liquidity_pulling_both_sides"
        elif delta_bid > 0 and delta_ask > 0:
            return "liquidity_adding_both_sides"
        else:
            return "neutral_liquidity_change"
    
    def calculate_absorption_ratio(self, instrument_key: str, price_level: float, executed_volume: float, initial_size: float) -> float:
        """
        Calculate liquidity absorption ratio.
        Formula: Absorption(p) = ExecutedVolume(p) / InitialSize(p)
        """
        try:
            if initial_size <= 0:
                return 0.0
            
            absorption = executed_volume / initial_size
            return min(1.0, max(0.0, absorption))  # Clamp between 0 and 1
            
        except Exception as e:
            print(f"❌ Error calculating absorption ratio: {e}")
            return 0.0
    
    def detect_wall_reload(self, instrument_key: str, wall_price: float, current_size: float, executed_volume: float) -> Dict:
        """
        Detect wall reload behavior (iceberg orders).
        Combines wall detection + liquidity delta analysis.
        """
        try:
            if instrument_key not in self.instrument_data:
                return {'is_reload_wall': False, 'reload_strength': 0.0}
            
            data = self.instrument_data[instrument_key]
            
            # Check if this price level has reload wall history
            if wall_price not in data['reload_walls']:
                data['reload_walls'][wall_price] = {
                    'initial_size': current_size,
                    'total_executed': 0.0,
                    'reload_count': 0,
                    'last_size': current_size
                }
            
            wall_data = data['reload_walls'][wall_price]
            
            # Update execution tracking
            wall_data['total_executed'] += executed_volume
            
            # Check for reload behavior
            size_maintained_ratio = current_size / wall_data['initial_size'] if wall_data['initial_size'] > 0 else 0
            execution_ratio = wall_data['total_executed'] / wall_data['initial_size'] if wall_data['initial_size'] > 0 else 0
            
            # Reload detected if:
            # 1. Significant volume executed against wall
            # 2. Wall size maintained above threshold
            is_reload_wall = (execution_ratio > 0.3 and size_maintained_ratio > 0.8)
            
            if is_reload_wall:
                wall_data['reload_count'] += 1
            
            wall_data['last_size'] = current_size
            
            return {
                'is_reload_wall': is_reload_wall,
                'reload_strength': float(wall_data['reload_count']),
                'size_maintained_ratio': float(size_maintained_ratio),
                'execution_ratio': float(execution_ratio),
                'total_reloads': wall_data['reload_count']
            }
            
        except Exception as e:
            print(f"❌ Error detecting wall reload: {e}")
            return {'is_reload_wall': False, 'reload_strength': 0.0}

    def calculate_cvd_increment(self, tick_data: Dict, instrument_key: str) -> Optional[float]:
        """
        Calculate CVD increment from current tick - ONLY for NSE_FO|53001.
        
        CVD = ∑(Buy Volume - Sell Volume)
        Uses price comparison to determine if volume is buy or sell initiated.
        """
        try:
            # Strict check - only process if we have valid 53001 tick data
            instrument_token = tick_data.get('instrument_token', '')
            if '53001' not in str(instrument_token) and '53001' not in str(instrument_key):
                print(f"⚠️ CVD calculation skipped - not from 53001 contract: {instrument_key}")
                return None
            
            current_price = tick_data.get('ltp', 0) or tick_data.get('last_traded_price', 0)
            current_volume = tick_data.get('ltq', 0) or tick_data.get('last_traded_quantity', 0)
            
            if current_price <= 0 or current_volume <= 0:
                print(f"⚠️ No valid price/volume data from 53001 - waiting for data...")
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
        STRICT: Only processes NSE_FO|53001 data - NO FALLBACK.
        """
        try:
            # Strict validation - ONLY process 53001 data
            if '53001' not in str(instrument_key):
                print(f"❌ OBI+CVD update rejected - not 53001 instrument: {instrument_key}")
                return {
                    'error': f'Only 53001 instrument supported, got: {instrument_key}',
                    'obi_current': 0.0,
                    'obi_rolling_1min': 0.0,
                    'cvd_current_increment': 0.0,
                    'cvd_rolling_2min': 0.0,
                    'cvd_total': 0.0,
                    'confirmation': 'Waiting for 53001 data'
                }
            
            # Additional validation on tick data
            tick_instrument = tick_data.get('instrument_token', '')
            if '53001' not in str(tick_instrument):
                print(f"❌ OBI+CVD tick rejected - not from 53001: {tick_instrument}")
                return {
                    'error': f'Tick not from 53001, got: {tick_instrument}',
                    'obi_current': 0.0,
                    'obi_rolling_1min': 0.0,
                    'cvd_current_increment': 0.0,
                    'cvd_rolling_2min': 0.0,
                    'cvd_total': 0.0,
                    'confirmation': 'Waiting for 53001 data'
                }
            
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
                
                # Add to 2-minute rolling OBI
                instrument_data['rolling_obi_2min'].append({
                    'timestamp': current_time,
                    'obi': current_obi
                })
            
            # Calculate current CVD increment
            current_cvd_increment = self.calculate_cvd_increment(tick_data, instrument_key)
            if current_cvd_increment is not None:
                instrument_data['current_cvd_increment'] = current_cvd_increment
                
                # Update all CVD accumulations
                instrument_data['cvd'] += current_cvd_increment  # 30-minute CVD
                instrument_data['cvd_hourly'] += current_cvd_increment  # 1-hour CVD
                instrument_data['cvd_daily'] += current_cvd_increment  # Daily CVD
                
                # Update CVD Delta accumulations for short timeframes
                instrument_data['cvd_delta_1min'] += current_cvd_increment  # 1-minute CVD delta
                instrument_data['cvd_delta_2min'] += current_cvd_increment  # 2-minute CVD delta
                instrument_data['cvd_delta_5min'] += current_cvd_increment  # 5-minute CVD delta
                
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
            
            # Advanced liquidity analysis
            advanced_liquidity = {}
            depth_levels = tick_data.get('market_depth_levels', [])
            
            if depth_levels and len(depth_levels) >= 5:
                # Store current depth snapshot
                depth_snapshot = {
                    'timestamp': current_time,
                    'depth_levels': depth_levels.copy()
                }
                instrument_data['depth_history'].append(depth_snapshot)
                
                # 1. Detect liquidity walls
                wall_analysis = self.detect_liquidity_walls(depth_levels)
                advanced_liquidity['walls'] = wall_analysis
                
                # 2. Calculate order book slope
                slope_analysis = self.calculate_order_book_slope(depth_levels)
                instrument_data['liquidity_slope_bid'] = slope_analysis['bid_slope']
                instrument_data['liquidity_slope_ask'] = slope_analysis['ask_slope']
                advanced_liquidity['slopes'] = slope_analysis
                
                # 3. Calculate liquidity delta (if we have previous depth)
                if len(instrument_data['depth_history']) >= 2:
                    previous_depth = instrument_data['depth_history'][-2]['depth_levels']
                    delta_analysis = self.calculate_liquidity_delta(depth_levels, previous_depth)
                    instrument_data['liquidity_delta_bid'] = delta_analysis['delta_bid']
                    instrument_data['liquidity_delta_ask'] = delta_analysis['delta_ask']
                    advanced_liquidity['liquidity_delta'] = delta_analysis
                
                # 4. Track wall execution and absorption
                executed_volume = tick_data.get('ltq', 0)
                current_price = tick_data.get('ltp', 0)
                
                for wall in wall_analysis['bid_walls'] + wall_analysis['ask_walls']:
                    wall_price = wall['price']
                    initial_size = wall['size']
                    
                    # Calculate absorption ratio
                    absorption = self.calculate_absorption_ratio(instrument_key, wall_price, executed_volume, initial_size)
                    instrument_data['absorption_ratios'][wall_price] = absorption
                    
                    # Check for reload behavior
                    reload_analysis = self.detect_wall_reload(instrument_key, wall_price, initial_size, executed_volume)
                    wall['reload_analysis'] = reload_analysis
                
                advanced_liquidity['absorption_ratios'] = dict(instrument_data['absorption_ratios'])
                
                print(f"🔍 Advanced liquidity analysis for 53001: {len(wall_analysis['bid_walls'])} bid walls, {len(wall_analysis['ask_walls'])} ask walls")
            else:
                advanced_liquidity = {
                    'walls': {'bid_walls': [], 'ask_walls': [], 'total_walls': 0},
                    'slopes': {'bid_slope': 0.0, 'ask_slope': 0.0, 'slope_asymmetry': 0.0},
                    'liquidity_delta': {'delta_bid': 0.0, 'delta_ask': 0.0, 'net_liquidity_change': 0.0},
                    'absorption_ratios': {}
                }
            
            # Check if CVD total should be reset (every 30 minutes)
            time_since_cvd_reset = current_time - self.last_cvd_reset[instrument_key]
            if time_since_cvd_reset.total_seconds() > (self.cvd_reset_minutes * 60):
                print(f"🔄 Resetting 30-min CVD for {instrument_key} after {self.cvd_reset_minutes} minutes")
                instrument_data['cvd'] = 0.0
                self.last_cvd_reset[instrument_key] = current_time
            
            # Check if CVD hourly should be reset (every 1 hour)
            time_since_cvd_hourly_reset = current_time - self.last_cvd_hourly_reset[instrument_key]
            if time_since_cvd_hourly_reset.total_seconds() > 3600:  # 1 hour = 3600 seconds
                print(f"🔄 Resetting 1-hour CVD for {instrument_key} after 1 hour")
                instrument_data['cvd_hourly'] = 0.0
                self.last_cvd_hourly_reset[instrument_key] = current_time
            
            # Check if CVD daily should be reset (at market close or start of new day)
            # Reset daily CVD at 3:30 PM IST (market close) or if it's a new trading day
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            current_time_ist = current_time.astimezone(ist)
            
            # Check if it's after 3:30 PM and we haven't reset today
            if current_time_ist.hour >= 15 and current_time_ist.minute >= 30:
                last_reset_ist = self.last_cvd_hourly_reset[instrument_key].astimezone(ist)
                if last_reset_ist.date() < current_time_ist.date():
                    print(f"🔄 Resetting daily CVD for {instrument_key} at market close")
                    instrument_data['cvd_daily'] = 0.0
                    # Update last reset to prevent multiple resets on same day
                    self.last_cvd_hourly_reset[instrument_key] = current_time
            
            # Check if OBI 1-minute rolling average should be reset (every 1 minute)
            time_since_obi_reset = current_time - self.last_obi_reset[instrument_key]
            if time_since_obi_reset.total_seconds() > 60:  # Reset every 1 minute
                print(f"🔄 Resetting OBI 1-min rolling average for {instrument_key} after 1 minute")
                instrument_data['rolling_obi_1min'].clear()
                self.last_obi_reset[instrument_key] = current_time
            
            # Check if OBI 2-minute rolling average should be reset (every 2 minutes)
            time_since_obi_2min_reset = current_time - self.last_obi_2min_reset[instrument_key]
            if time_since_obi_2min_reset.total_seconds() > 120:  # Reset every 2 minutes
                print(f"🔄 Resetting OBI 2-min rolling average for {instrument_key} after 2 minutes")
                instrument_data['rolling_obi_2min'].clear()
                self.last_obi_2min_reset[instrument_key] = current_time
            
            # Check if CVD rolling average should be reset (every 2 minutes)
            time_since_cvd_rolling_reset = current_time - self.last_cvd_rolling_reset[instrument_key]
            if time_since_cvd_rolling_reset.total_seconds() > 120:  # Reset every 2 minutes
                print(f"🔄 Resetting CVD rolling average for {instrument_key} after 2 minutes")
                instrument_data['rolling_cvd_2min'].clear()
                self.last_cvd_rolling_reset[instrument_key] = current_time
            
            # Check if CVD Delta 1-minute should be reset (every 1 minute)
            time_since_cvd_1min_reset = current_time - self.last_cvd_1min_reset[instrument_key]
            if time_since_cvd_1min_reset.total_seconds() > 60:  # Reset every 1 minute
                print(f"🔄 Resetting CVD Delta 1-min for {instrument_key} after 1 minute")
                # Store current value as previous for delta calculation
                instrument_data['cvd_delta_1min_prev'] = instrument_data['cvd_delta_1min']
                instrument_data['cvd_delta_1min'] = 0.0
                self.last_cvd_1min_reset[instrument_key] = current_time
            
            # Check if CVD Delta 2-minute should be reset (every 2 minutes)
            time_since_cvd_2min_reset = current_time - self.last_cvd_2min_reset[instrument_key]
            if time_since_cvd_2min_reset.total_seconds() > 120:  # Reset every 2 minutes
                print(f"🔄 Resetting CVD Delta 2-min for {instrument_key} after 2 minutes")
                # Store current value as previous for delta calculation
                instrument_data['cvd_delta_2min_prev'] = instrument_data['cvd_delta_2min']
                instrument_data['cvd_delta_2min'] = 0.0
                self.last_cvd_2min_reset[instrument_key] = current_time
            
            # Check if CVD Delta 5-minute should be reset (every 5 minutes)
            time_since_cvd_5min_reset = current_time - self.last_cvd_5min_reset[instrument_key]
            if time_since_cvd_5min_reset.total_seconds() > 300:  # Reset every 5 minutes
                print(f"🔄 Resetting CVD Delta 5-min for {instrument_key} after 5 minutes")
                # Store current value as previous for delta calculation
                instrument_data['cvd_delta_5min_prev'] = instrument_data['cvd_delta_5min']
                instrument_data['cvd_delta_5min'] = 0.0
                self.last_cvd_5min_reset[instrument_key] = current_time
            
            # Calculate rolling averages
            # OBI 1-minute rolling average
            recent_obi_1min = [
                entry['obi'] for entry in instrument_data['rolling_obi_1min']
                if (current_time - entry['timestamp']).total_seconds() <= 60
            ]
            rolling_avg_obi_1min = np.mean(recent_obi_1min) if recent_obi_1min else 0.0
            
            # OBI 2-minute rolling average
            recent_obi_2min = [
                entry['obi'] for entry in instrument_data['rolling_obi_2min']
                if (current_time - entry['timestamp']).total_seconds() <= 120
            ]
            rolling_avg_obi_2min = np.mean(recent_obi_2min) if recent_obi_2min else 0.0
            
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
            
            # Return comprehensive analysis including advanced liquidity features
            return self._analyze_granular_confirmation(
                instrument_key, 
                current_obi if current_obi is not None else 0.0, 
                rolling_avg_obi_1min,
                rolling_avg_obi_2min,
                current_cvd_increment if current_cvd_increment is not None else 0.0,
                rolling_avg_cvd_2min,
                instrument_data['cvd'],
                instrument_data['cvd_hourly'],
                instrument_data['cvd_daily'],
                legacy_avg_obi,
                instrument_data['cvd_delta_1min'],
                instrument_data['cvd_delta_2min'],
                instrument_data['cvd_delta_5min'],
                advanced_liquidity
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
                                          rolling_obi_1min: float, rolling_obi_2min: float, current_cvd_increment: float,
                                          rolling_cvd_2min: float, total_cvd: float, cvd_hourly: float, cvd_daily: float, legacy_avg_obi: float,
                                          cvd_delta_1min: float, cvd_delta_2min: float, cvd_delta_5min: float, advanced_liquidity: Dict = None) -> Dict:
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
                rolling_obi_1min_signal = 'Strong Bullish'
            elif rolling_obi_1min > 0.1:
                rolling_obi_1min_signal = 'Bullish'
            elif rolling_obi_1min < -0.3:
                rolling_obi_1min_signal = 'Strong Bearish'
            elif rolling_obi_1min < -0.1:
                rolling_obi_1min_signal = 'Bearish'
            else:
                rolling_obi_1min_signal = 'Neutral'
            
            # Rolling OBI 2-minute Analysis
            if rolling_obi_2min > 0.25:
                rolling_obi_2min_signal = 'Strong Bullish'
            elif rolling_obi_2min > 0.08:
                rolling_obi_2min_signal = 'Bullish'
            elif rolling_obi_2min < -0.25:
                rolling_obi_2min_signal = 'Strong Bearish'
            elif rolling_obi_2min < -0.08:
                rolling_obi_2min_signal = 'Bearish'
            else:
                rolling_obi_2min_signal = 'Neutral'
            
            # Current CVD Increment Analysis (tick-by-tick)
            if current_cvd_increment > 500:
                current_cvd_signal = 'Strong Buying'
            elif current_cvd_increment > 100:
                current_cvd_signal = 'Buying'
            elif current_cvd_increment < -500:
                current_cvd_signal = 'Strong Selling'
            elif current_cvd_increment < -100:
                current_cvd_signal = 'Selling'
            else:
                current_cvd_signal = 'Neutral'
            
            # Rolling CVD 2-minute Analysis
            if rolling_cvd_2min > 150:
                rolling_cvd_signal = 'Strong Buying'
            elif rolling_cvd_2min > 50:
                rolling_cvd_signal = 'Buying'
            elif rolling_cvd_2min < -150:
                rolling_cvd_signal = 'Strong Selling'
            elif rolling_cvd_2min < -50:
                rolling_cvd_signal = 'Selling'
            else:
                rolling_cvd_signal = 'Neutral'
            
            # Total CVD Analysis (30-minute cumulative)
            if total_cvd > 1000:
                total_cvd_signal = 'Strong Buying'
            elif total_cvd > 500:
                total_cvd_signal = 'Buying'
            elif total_cvd < -1000:
                total_cvd_signal = 'Strong Selling'
            elif total_cvd < -500:
                total_cvd_signal = 'Selling'
            else:
                total_cvd_signal = 'Neutral'
            
            # Hourly CVD Analysis (1-hour cumulative)
            if cvd_hourly > 2500:
                hourly_cvd_signal = 'Strong Buying'
            elif cvd_hourly > 1000:
                hourly_cvd_signal = 'Buying'
            elif cvd_hourly < -2500:
                hourly_cvd_signal = 'Strong Selling'
            elif cvd_hourly < -1000:
                hourly_cvd_signal = 'Selling'
            else:
                hourly_cvd_signal = 'Neutral'
            
            # Daily CVD Analysis (full day cumulative)
            if cvd_daily > 10000:
                daily_cvd_signal = 'Strong Buying'
            elif cvd_daily > 5000:
                daily_cvd_signal = 'Buying'
            elif cvd_daily < -10000:
                daily_cvd_signal = 'Strong Selling'
            elif cvd_daily < -5000:
                daily_cvd_signal = 'Selling'
            else:
                daily_cvd_signal = 'Neutral'
            
            # CVD Delta Analysis for short timeframes (momentum detection)
            # 1-minute CVD Delta
            if cvd_delta_1min > 300:
                cvd_delta_1min_signal = 'Strong Buying Momentum'
            elif cvd_delta_1min > 150:
                cvd_delta_1min_signal = 'Buying Momentum'
            elif cvd_delta_1min < -300:
                cvd_delta_1min_signal = 'Strong Selling Momentum'
            elif cvd_delta_1min < -150:
                cvd_delta_1min_signal = 'Selling Momentum'
            else:
                cvd_delta_1min_signal = 'Neutral Momentum'
            
            # 2-minute CVD Delta
            if cvd_delta_2min > 800:
                cvd_delta_2min_signal = 'Strong Buying Momentum'
            elif cvd_delta_2min > 400:
                cvd_delta_2min_signal = 'Buying Momentum'
            elif cvd_delta_2min < -800:
                cvd_delta_2min_signal = 'Strong Selling Momentum'
            elif cvd_delta_2min < -400:
                cvd_delta_2min_signal = 'Selling Momentum'
            else:
                cvd_delta_2min_signal = 'Neutral Momentum'
            
            # 5-minute CVD Delta
            if cvd_delta_5min > 1500:
                cvd_delta_5min_signal = 'Strong Buying Momentum'
            elif cvd_delta_5min > 800:
                cvd_delta_5min_signal = 'Buying Momentum'
            elif cvd_delta_5min < -1500:
                cvd_delta_5min_signal = 'Strong Selling Momentum'
            elif cvd_delta_5min < -800:
                cvd_delta_5min_signal = 'Selling Momentum'
            else:
                cvd_delta_5min_signal = 'Neutral Momentum'
            
            # Combined Confirmation based on rolling averages
            if (('Bullish' in rolling_obi_1min_signal or 'Bullish' in rolling_obi_2min_signal) and 'Buying' in rolling_cvd_signal):
                combined_confirmation = 'Strong Bullish'
            elif (('Bearish' in rolling_obi_1min_signal or 'Bearish' in rolling_obi_2min_signal) and 'Selling' in rolling_cvd_signal):
                combined_confirmation = 'Strong Bearish'
            elif ('Bullish' in rolling_obi_1min_signal or 'Bullish' in rolling_obi_2min_signal or 'Buying' in rolling_cvd_signal):
                combined_confirmation = 'Moderate Bullish'
            elif ('Bearish' in rolling_obi_1min_signal or 'Bearish' in rolling_obi_2min_signal or 'Selling' in rolling_cvd_signal):
                combined_confirmation = 'Moderate Bearish'
            else:
                combined_confirmation = 'Neutral'
            
            # Prepare advanced liquidity summary
            liquidity_summary = {}
            if advanced_liquidity:
                walls = advanced_liquidity.get('walls', {})
                slopes = advanced_liquidity.get('slopes', {})
                liquidity_delta = advanced_liquidity.get('liquidity_delta', {})
                absorption_ratios = advanced_liquidity.get('absorption_ratios', {})
                
                liquidity_summary = {
                    # Liquidity Walls
                    'bid_walls_count': len(walls.get('bid_walls', [])),
                    'ask_walls_count': len(walls.get('ask_walls', [])),
                    'total_walls': walls.get('total_walls', 0),
                    'strongest_bid_wall': max(walls.get('bid_walls', []), key=lambda x: x.get('strength', 0), default={}),
                    'strongest_ask_wall': max(walls.get('ask_walls', []), key=lambda x: x.get('strength', 0), default={}),
                    
                    # Order Book Slope
                    'bid_slope': slopes.get('bid_slope', 0.0),
                    'ask_slope': slopes.get('ask_slope', 0.0),
                    'slope_asymmetry': slopes.get('slope_asymmetry', 0.0),
                    'bid_slope_interpretation': slopes.get('bid_slope_interpretation', 'neutral'),
                    'ask_slope_interpretation': slopes.get('ask_slope_interpretation', 'neutral'),
                    
                    # Liquidity Delta
                    'liquidity_delta_bid': liquidity_delta.get('delta_bid', 0.0),
                    'liquidity_delta_ask': liquidity_delta.get('delta_ask', 0.0),
                    'net_liquidity_change': liquidity_delta.get('net_liquidity_change', 0.0),
                    'liquidity_sentiment': liquidity_delta.get('liquidity_sentiment', 'neutral'),
                    
                    # Absorption Analysis
                    'avg_absorption_ratio': np.mean(list(absorption_ratios.values())) if absorption_ratios else 0.0,
                    'high_absorption_levels': [price for price, ratio in absorption_ratios.items() if ratio > 0.7],
                    'low_absorption_levels': [price for price, ratio in absorption_ratios.items() if ratio < 0.3],
                    
                    # Wall Reload Detection
                    'reload_walls_detected': sum(1 for wall in walls.get('bid_walls', []) + walls.get('ask_walls', []) 
                                               if wall.get('reload_analysis', {}).get('is_reload_wall', False)),
                }
                
                # Generate overall liquidity signal
                wall_signal = 'bullish' if liquidity_summary['bid_walls_count'] > liquidity_summary['ask_walls_count'] else 'bearish' if liquidity_summary['ask_walls_count'] > liquidity_summary['bid_walls_count'] else 'neutral'
                slope_signal = 'bullish' if slopes.get('slope_asymmetry', 0) > 0.1 else 'bearish' if slopes.get('slope_asymmetry', 0) < -0.1 else 'neutral'
                delta_signal = liquidity_delta.get('liquidity_sentiment', 'neutral')
                
                liquidity_summary['overall_liquidity_signal'] = f"{wall_signal}_{slope_signal}_{delta_signal}"
            
            return {
                # Current tick values
                'obi_current': float(current_obi) if current_obi is not None else 0.0,
                'obi_current_signal': current_obi_signal,
                'cvd_current_increment': float(current_cvd_increment) if current_cvd_increment is not None else 0.0,
                'cvd_current_signal': current_cvd_signal,
                
                # Rolling averages
                'obi_rolling_1min': float(rolling_obi_1min),
                'obi_rolling_1min_signal': rolling_obi_1min_signal,
                'obi_rolling_2min': float(rolling_obi_2min),
                'obi_rolling_2min_signal': rolling_obi_2min_signal,
                'cvd_rolling_2min': float(rolling_cvd_2min),
                'cvd_rolling_signal': rolling_cvd_signal,
                
                # Total accumulation
                'cvd_total': float(total_cvd),
                'cvd_total_signal': total_cvd_signal,
                'cvd_hourly': float(cvd_hourly),
                'cvd_hourly_signal': hourly_cvd_signal,
                'cvd_daily': float(cvd_daily),
                'cvd_daily_signal': daily_cvd_signal,
                
                # CVD Delta momentum (short timeframes)
                'cvd_delta_1min': float(cvd_delta_1min),
                'cvd_delta_1min_signal': cvd_delta_1min_signal,
                'cvd_delta_2min': float(cvd_delta_2min),
                'cvd_delta_2min_signal': cvd_delta_2min_signal,
                'cvd_delta_5min': float(cvd_delta_5min),
                'cvd_delta_5min_signal': cvd_delta_5min_signal,
                
                # Advanced Liquidity Features
                'liquidity_walls': liquidity_summary.get('total_walls', 0),
                'bid_walls': liquidity_summary.get('bid_walls_count', 0),
                'ask_walls': liquidity_summary.get('ask_walls_count', 0),
                'order_book_slope_bid': liquidity_summary.get('bid_slope', 0.0),
                'order_book_slope_ask': liquidity_summary.get('ask_slope', 0.0),
                'slope_asymmetry': liquidity_summary.get('slope_asymmetry', 0.0),
                'liquidity_delta_net': liquidity_summary.get('net_liquidity_change', 0.0),
                'absorption_ratio_avg': liquidity_summary.get('avg_absorption_ratio', 0.0),
                'reload_walls': liquidity_summary.get('reload_walls_detected', 0),
                'liquidity_signal': liquidity_summary.get('overall_liquidity_signal', 'neutral_neutral_neutral'),
                
                # Advanced liquidity interpretations
                'wall_analysis': liquidity_summary.get('strongest_bid_wall', {}),
                'slope_interpretation': f"{liquidity_summary.get('bid_slope_interpretation', 'neutral')}_{liquidity_summary.get('ask_slope_interpretation', 'neutral')}",
                'liquidity_sentiment': liquidity_summary.get('liquidity_sentiment', 'neutral'),
                
                # Legacy support
                'obi_average': float(legacy_avg_obi),  # Keep for backward compatibility
                'cvd_current': float(total_cvd),  # Legacy field
                'obi_signal': rolling_obi_1min_signal,  # Use 1-min rolling for legacy
                'obi_rolling_signal': rolling_obi_1min_signal,  # Keep for existing UI
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
                'obi_rolling_1min_signal': 'Error',
                'obi_rolling_2min': 0.0,
                'obi_rolling_2min_signal': 'Error',
                'obi_rolling_signal': 'Error',
                'cvd_rolling_2min': 0.0,
                'cvd_rolling_signal': 'Error',
                'cvd_total': 0.0,
                'cvd_total_signal': 'Error',
                'cvd_delta_1min': 0.0,
                'cvd_delta_1min_signal': 'Error',
                'cvd_delta_2min': 0.0,
                'cvd_delta_2min_signal': 'Error',
                'cvd_delta_5min': 0.0,
                'cvd_delta_5min_signal': 'Error',
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
        cvd_hourly = data.get('cvd_hourly', 0.0)
        cvd_daily = data.get('cvd_daily', 0.0)
        
        # Calculate rolling averages
        # OBI 1-minute rolling average
        recent_obi_1min = [
            entry['obi'] for entry in data.get('rolling_obi_1min', [])
            if (current_time - entry['timestamp']).total_seconds() <= 60
        ]
        rolling_avg_obi_1min = np.mean(recent_obi_1min) if recent_obi_1min else 0.0
        
        # OBI 2-minute rolling average
        recent_obi_2min = [
            entry['obi'] for entry in data.get('rolling_obi_2min', [])
            if (current_time - entry['timestamp']).total_seconds() <= 120
        ]
        rolling_avg_obi_2min = np.mean(recent_obi_2min) if recent_obi_2min else 0.0
        
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
        
        # Get latest advanced liquidity analysis if depth history available
        advanced_liquidity = {}
        if len(data.get('depth_history', [])) > 0:
            latest_depth = data['depth_history'][-1]['depth_levels']
            
            # Regenerate advanced analysis
            wall_analysis = self.detect_liquidity_walls(latest_depth)
            slope_analysis = self.calculate_order_book_slope(latest_depth)
            
            advanced_liquidity = {
                'walls': wall_analysis,
                'slopes': slope_analysis,
                'liquidity_delta': {'delta_bid': data.get('liquidity_delta_bid', 0.0), 'delta_ask': data.get('liquidity_delta_ask', 0.0), 'net_liquidity_change': 0.0},
                'absorption_ratios': data.get('absorption_ratios', {})
            }

        return self._analyze_granular_confirmation(
            instrument_key, 
            current_obi if current_obi is not None else 0.0, 
            rolling_avg_obi_1min,
            rolling_avg_obi_2min,
            current_cvd_increment if current_cvd_increment is not None else 0.0,
            rolling_avg_cvd_2min,
            total_cvd,
            cvd_hourly,
            cvd_daily,
            legacy_avg_obi,
            data.get('cvd_delta_1min', 0.0),
            data.get('cvd_delta_2min', 0.0),
            data.get('cvd_delta_5min', 0.0),
            advanced_liquidity
        )
    
    def reset_instrument(self, instrument_key: str):
        """Reset all data for an instrument."""
        if instrument_key in self.instrument_data:
            del self.instrument_data[instrument_key]
        if instrument_key in self.last_cvd_reset:
            del self.last_cvd_reset[instrument_key]
        print(f"🔄 Reset OBI+CVD data for {instrument_key}")
