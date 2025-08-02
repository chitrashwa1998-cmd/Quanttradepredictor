import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math

class BlackScholesCalculator:
    """Black-Scholes option pricing calculator for live predictions."""

    def __init__(self, risk_free_rate: float = 0.055, dividend_yield: float = 0.012):
        """
        Initialize Black-Scholes calculator.

        Args:
            risk_free_rate: RBI repo rate (default 5.50%)
            dividend_yield: Dividend yield for Nifty 50 (default 1.2%)
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

        # Common Nifty 50 strike price ranges (adjust based on current levels)
        self.strike_offsets = [-500, -400, -300, -200, -100, -50, 0, 50, 100, 200, 300, 400, 500]

    def get_nifty_expiry_dates(self) -> List[datetime]:
        """Get next 3 weekly Thursday expiry dates for Nifty options."""
        today = datetime.now().date()
        today_weekday = today.weekday()
        
        # Calculate days until next Thursday
        days_until_thursday = (3 - today_weekday) % 7
        
        expiry_dates = []
        
        for i in range(3):
            # Calculate expiry date for the next Thursday
            expiry_date = today + timedelta(days=days_until_thursday + i * 7)
            expiry_dates.append(datetime.combine(expiry_date, datetime.min.time()))  # Convert to datetime

        return expiry_dates

    def calculate_time_to_expiry(self, expiry_date: datetime) -> float:
        """Calculate time to expiry in years."""
        now = datetime.now()
        if expiry_date <= now:
            return 0.001  # Minimum time to avoid division by zero

        days_to_expiry = (expiry_date - now).days
        hours_to_expiry = (expiry_date - now).seconds / 3600

        # Convert to years (assuming 365 days per year)
        time_to_expiry = (days_to_expiry + hours_to_expiry / 24) / 365
        return max(time_to_expiry, 0.001)  # Minimum time

    def calculate_d1_d2(self, S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0

        d1 = (np.log(S / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2

    def calculate_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate Black-Scholes call option price."""
        if T <= 0:
            return max(S - K, 0)  # Intrinsic value

        d1, d2 = self.calculate_d1_d2(S, K, T, sigma)

        call_price = (S * np.exp(-self.dividend_yield * T) * norm.cdf(d1) - 
                     K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2))

        return max(call_price, 0)

    def calculate_put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate Black-Scholes put option price."""
        if T <= 0:
            return max(K - S, 0)  # Intrinsic value

        d1, d2 = self.calculate_d1_d2(S, K, T, sigma)

        put_price = (K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - 
                    S * np.exp(-self.dividend_yield * T) * norm.cdf(-d1))

        return max(put_price, 0)

    def calculate_greeks(self, S: float, K: float, T: float, sigma: float) -> Dict[str, float]:
        """Calculate option Greeks."""
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        d1, d2 = self.calculate_d1_d2(S, K, T, sigma)

        # Delta (Call)
        delta_call = np.exp(-self.dividend_yield * T) * norm.cdf(d1)

        # Gamma (same for calls and puts)
        gamma = (np.exp(-self.dividend_yield * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

        # Theta (Call)
        theta_call = ((-S * norm.pdf(d1) * sigma * np.exp(-self.dividend_yield * T)) / (2 * np.sqrt(T)) -
                     self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2) +
                     self.dividend_yield * S * np.exp(-self.dividend_yield * T) * norm.cdf(d1)) / 365

        # Vega (same for calls and puts)
        vega = S * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100

        # Rho (Call)
        rho_call = K * T * np.exp(-self.risk_free_rate * T) * norm.cdf(d2) / 100

        return {
            'delta': delta_call,
            'gamma': gamma,
            'theta': theta_call,
            'vega': vega,
            'rho': rho_call
        }

    def generate_strike_prices(self, current_price: float) -> List[int]:
        """Generate relevant strike prices around current price."""
        base_price = round(current_price / 50) * 50  # Round to nearest 50
        strikes = []

        for offset in self.strike_offsets:
            strike = base_price + offset
            if strike > 0:  # Ensure positive strike prices
                strikes.append(int(strike))

        return sorted(strikes)

    def calculate_index_fair_value(self, current_price: float, volatility: float) -> Dict[str, any]:
        """Calculate fair value metrics for the index itself."""
        expiry_dates = self.get_nifty_expiry_dates()

        if not expiry_dates:
            return {'error': 'No valid expiry dates found'}

        nearest_expiry = expiry_dates[0]
        T = self.calculate_time_to_expiry(nearest_expiry)

        # Calculate theoretical forward price
        forward_price = current_price * np.exp((self.risk_free_rate - self.dividend_yield) * T)

        # Calculate cost of carry
        cost_of_carry = self.risk_free_rate - self.dividend_yield

        # Calculate volatility-adjusted fair value range
        vol_adjustment = volatility * np.sqrt(T) * current_price
        fair_value_range = {
            'lower': current_price - vol_adjustment,
            'fair': forward_price,
            'upper': current_price + vol_adjustment
        }

        return {
            'current_price': current_price,
            'forward_price': forward_price,
            'cost_of_carry': cost_of_carry,
            'fair_value_range': fair_value_range,
            'time_to_expiry_days': T * 365,
            'volatility_used': volatility,
            'risk_free_rate': self.risk_free_rate,
            'dividend_yield': self.dividend_yield
        }

    def calculate_options_fair_values(self, current_price: float, volatility: float) -> Dict[str, any]:
        """Calculate fair values for multiple strike prices and expiries."""
        expiry_dates = self.get_nifty_expiry_dates()
        strike_prices = self.generate_strike_prices(current_price)

        if not expiry_dates:
            return {'error': 'No valid expiry dates found'}

        results = {
            'current_price': current_price,
            'volatility_used': volatility,
            'risk_free_rate': self.risk_free_rate,
            'dividend_yield': self.dividend_yield,
            'timestamp': datetime.now().isoformat(),
            'expiries': {}
        }

        for expiry_date in expiry_dates:
            T = self.calculate_time_to_expiry(expiry_date)
            expiry_str = expiry_date.strftime('%Y-%m-%d')

            results['expiries'][expiry_str] = {
                'expiry_date': expiry_str,
                'days_to_expiry': T * 365,
                'strikes': {}
            }

            for strike in strike_prices:
                # Calculate option prices
                call_price = self.calculate_call_price(current_price, strike, T, volatility)
                put_price = self.calculate_put_price(current_price, strike, T, volatility)

                # Calculate Greeks for the call option
                greeks = self.calculate_greeks(current_price, strike, T, volatility)

                # Calculate moneyness
                moneyness = current_price / strike
                itm_otm = 'ITM' if strike < current_price else 'ATM' if strike == current_price else 'OTM'

                # Calculate intrinsic and time value
                call_intrinsic = max(current_price - strike, 0)
                put_intrinsic = max(strike - current_price, 0)
                call_time_value = call_price - call_intrinsic
                put_time_value = put_price - put_intrinsic

                results['expiries'][expiry_str]['strikes'][strike] = {
                    'strike_price': strike,
                    'moneyness': moneyness,
                    'type': itm_otm,
                    'call': {
                        'fair_value': call_price,
                        'intrinsic_value': call_intrinsic,
                        'time_value': call_time_value
                    },
                    'put': {
                        'fair_value': put_price,
                        'intrinsic_value': put_intrinsic,
                        'time_value': put_time_value
                    },
                    'greeks': greeks
                }

        return results

    def get_quick_summary(self, current_price: float, volatility: float) -> Dict[str, any]:
        """Get a quick summary of key option fair values."""
        expiry_dates = self.get_nifty_expiry_dates()
        if not expiry_dates:
            return {'error': 'No valid expiry dates found'}

        nearest_expiry = expiry_dates[0]
        T = self.calculate_time_to_expiry(nearest_expiry)

        # Get ATM and nearby strikes
        atm_strike = round(current_price / 50) * 50
        strikes = [atm_strike - 100, atm_strike - 50, atm_strike, atm_strike + 50, atm_strike + 100]

        summary = {
            'current_price': current_price,
            'volatility': volatility,
            'nearest_expiry': nearest_expiry.strftime('%Y-%m-%d'),
            'days_to_expiry': round(T * 365, 1),
            'options': []
        }

        for strike in strikes:
            if strike > 0:
                call_price = self.calculate_call_price(current_price, strike, T, volatility)
                put_price = self.calculate_put_price(current_price, strike, T, volatility)

                option_type = 'ATM' if strike == atm_strike else ('ITM' if strike < current_price else 'OTM')

                summary['options'].append({
                    'strike': strike,
                    'type': option_type,
                    'call_fair_value': round(call_price, 2),
                    'put_fair_value': round(put_price, 2)
                })

        return summary