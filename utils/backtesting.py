import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Backtester:
    """Simple backtesting framework for trading strategies."""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
    
    def run_backtest(self, df: pd.DataFrame, signals: pd.Series, prices: pd.Series) -> Dict:
        """Run backtest based on trading signals."""
        
        # Ensure signals and prices are aligned
        aligned_data = pd.DataFrame({
            'price': prices,
            'signal': signals
        }).dropna()
        
        if len(aligned_data) == 0:
            return {'error': 'No valid data for backtesting'}
        
        # Initialize tracking variables
        portfolio_value = [self.initial_capital]
        cash = self.initial_capital
        shares = 0
        trades = []
        
        # Signal mapping: 0=Sell, 1=Hold, 2=Buy
        prev_signal = 1  # Start with hold
        
        for i, (date, row) in enumerate(aligned_data.iterrows()):
            current_price = row['price']
            current_signal = row['signal']
            
            # Execute trades based on signal changes
            if current_signal != prev_signal:
                if current_signal == 2 and prev_signal != 2:  # Buy signal
                    if cash > current_price:
                        shares_to_buy = int(cash / current_price)
                        cost = shares_to_buy * current_price * (1 + self.commission)
                        
                        if cost <= cash:
                            cash -= cost
                            shares += shares_to_buy
                            trades.append({
                                'date': date,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': current_price,
                                'cost': cost
                            })
                
                elif current_signal == 0 and shares > 0:  # Sell signal
                    proceeds = shares * current_price * (1 - self.commission)
                    cash += proceeds
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'shares': shares,
                        'price': current_price,
                        'proceeds': proceeds
                    })
                    shares = 0
            
            # Calculate current portfolio value
            current_value = cash + (shares * current_price)
            portfolio_value.append(current_value)
            prev_signal = current_signal
        
        # Final liquidation
        if shares > 0:
            final_price = aligned_data['price'].iloc[-1]
            final_proceeds = shares * final_price * (1 - self.commission)
            cash += final_proceeds
            trades.append({
                'date': aligned_data.index[-1],
                'action': 'SELL',
                'shares': shares,
                'price': final_price,
                'proceeds': final_proceeds
            })
        
        final_value = cash
        
        # Calculate performance metrics
        returns = pd.Series(portfolio_value[1:], index=aligned_data.index).pct_change().dropna()
        
        # Buy and hold benchmark
        buy_hold_return = (aligned_data['price'].iloc[-1] / aligned_data['price'].iloc[0]) - 1
        strategy_return = (final_value / self.initial_capital) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        portfolio_series = pd.Series(portfolio_value[1:], index=aligned_data.index)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        profitable_trades = [t for t in trades if t['action'] == 'SELL']
        if len(profitable_trades) > 1:
            trade_returns = []
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            for sell_trade in sell_trades:
                # Find corresponding buy trade
                buy_trade = None
                for bt in reversed(buy_trades):
                    if bt['date'] <= sell_trade['date']:
                        buy_trade = bt
                        break
                
                if buy_trade:
                    trade_return = (sell_trade['proceeds'] - buy_trade['cost']) / buy_trade['cost']
                    trade_returns.append(trade_return)
            
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0
        else:
            win_rate = 0
        
        self.results = {
            'portfolio_value': portfolio_series,
            'trades': trades,
            'final_value': final_value,
            'total_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': strategy_return - buy_hold_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len([t for t in trades if t['action'] == 'BUY']),
            'profitable_trades': sum(1 for r in (trade_returns if 'trade_returns' in locals() else []) if r > 0)
        }
        
        return self.results
    
    def create_performance_chart(self, df: pd.DataFrame, signals: pd.Series) -> go.Figure:
        """Create performance visualization chart."""
        
        if not self.results or 'portfolio_value' not in self.results:
            fig = go.Figure()
            fig.add_annotation(text="No backtest results available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price and Signals', 'Portfolio Value', 'Drawdown'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Price chart with signals
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add buy/sell signals
        buy_signals = signals[signals == 2]
        sell_signals = signals[signals == 0]
        
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index, 
                    y=df.loc[buy_signals.index, 'Close'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
        
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index, 
                    y=df.loc[sell_signals.index, 'Close'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='Sell Signal'
                ),
                row=1, col=1
            )
        
        # Portfolio value
        portfolio_series = self.results['portfolio_value']
        fig.add_trace(
            go.Scatter(x=portfolio_series.index, y=portfolio_series.values, 
                      name='Portfolio Value', line=dict(color='green')),
            row=2, col=1
        )
        
        # Buy and hold comparison
        initial_price = df['Close'].iloc[0]
        buy_hold_values = df['Close'] / initial_price * self.initial_capital
        fig.add_trace(
            go.Scatter(x=df.index, y=buy_hold_values, 
                      name='Buy & Hold', line=dict(color='orange', dash='dash')),
            row=2, col=1
        )
        
        # Drawdown
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, 
                      name='Drawdown %', fill='tonexty', line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(height=800, title_text="Backtesting Results")
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
        
        return fig
