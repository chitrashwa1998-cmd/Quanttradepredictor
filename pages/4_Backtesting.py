import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from utils.backtesting import Backtester

st.set_page_config(page_title="Backtesting", page_icon="📈", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">📈 BACKTEST ENGINE</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Strategy Performance Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Check if data and models are available
if st.session_state.data is None:
    st.warning("⚠️ No data loaded. Please go to the **Data Upload** page first.")
    st.stop()

if not st.session_state.models:
    st.warning("⚠️ No trained models found. Please go to the **Model Training** page first.")
    st.stop()

df = st.session_state.data
models = st.session_state.models
model_trainer = st.session_state.model_trainer

# Available models for backtesting
trading_models = [name for name, info in models.items() 
                 if info is not None and name in ['trading_signal', 'direction', 'profit_prob']]

if not trading_models:
    st.error("❌ No trading models found. Please train 'Trading Signal', 'Direction', or 'Profit Probability' models first.")
    st.stop()

st.header("Backtesting Configuration")

# Backtesting parameters
col1, col2, col3 = st.columns(3)

with col1:
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)

with col2:
    commission = st.number_input("Commission Rate (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.01) / 100

with col3:
    selected_model = st.selectbox(
        "Select Trading Model",
        trading_models,
        format_func=lambda x: x.replace('_', ' ').title()
    )

# Strategy selection
st.subheader("Strategy Configuration")

# Initialize variables with default values
confidence_threshold = 0.7
prob_threshold = 0.7
stop_loss_pct = 5
take_profit_pct = 10

if selected_model == 'trading_signal':
    st.info("Using direct trading signals from the model (Buy/Sell/Hold)")
    strategy_type = 'direct_signals'
    
elif selected_model == 'direction':
    st.subheader("Direction-Based Strategy")
    strategy_type = st.selectbox(
        "Strategy Type",
        ["Simple Direction", "Direction with Confidence", "Direction with Stop Loss"]
    )
    
    if strategy_type == "Direction with Confidence":
        confidence_threshold = st.slider("Minimum Confidence Threshold", 0.5, 0.95, 0.7, 0.05)
    elif strategy_type == "Direction with Stop Loss":
        stop_loss_pct = st.slider("Stop Loss %", 1, 10, 5, 1)
        take_profit_pct = st.slider("Take Profit %", 1, 20, 10, 1)

else:  # profit_prob
    st.info("Using profit probability predictions for trade entry")
    prob_threshold = st.slider("Minimum Profit Probability", 0.5, 0.95, 0.7, 0.05)
    strategy_type = 'profit_prob'

# Backtesting period
st.subheader("Backtesting Period")

col1, col2 = st.columns(2)

with col1:
    backtest_period = st.selectbox(
        "Select Period",
        ["Last 6 months", "Last year", "Last 2 years", "All data"]
    )

# Date range filtering

if backtest_period == "Last 6 months":
    start_date = df.index.max() - timedelta(days=180)
elif backtest_period == "Last year":
    start_date = df.index.max() - timedelta(days=365)
elif backtest_period == "Last 2 years":
    start_date = df.index.max() - timedelta(days=730)
else:
    start_date = df.index.min()

df_backtest = df[df.index >= start_date]

with col2:
    st.info(f"**Backtesting Period**: {start_date.strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    st.info(f"**Total Days**: {len(df_backtest)} trading days")

# Run backtest
st.header("Run Backtest")

if st.button("🚀 Run Backtest", type="primary"):
    
    if st.session_state.features is None:
        st.error("Features not available. Please calculate technical indicators first.")
        st.stop()
    
    with st.spinner("Running backtest..."):
        
        # Get features for the backtesting period
        features_backtest = st.session_state.features[st.session_state.features.index >= start_date]
        
        try:
            # Generate predictions
            predictions, probabilities = model_trainer.predict(selected_model, features_backtest)
            
            # Convert predictions to trading signals
            if selected_model == 'trading_signal':
                signals = pd.Series(predictions, index=features_backtest.index)
                
            elif selected_model == 'direction':
                if strategy_type == "Simple Direction":
                    # Convert direction to signals: 1=Up -> 2=Buy, 0=Down -> 0=Sell
                    signals = pd.Series([2 if p == 1 else 0 for p in predictions], index=features_backtest.index)
                
                elif strategy_type == "Direction with Confidence":
                    max_probs = np.max(probabilities, axis=1) if probabilities is not None else np.ones(len(predictions))
                    signals = []
                    for i, (pred, conf) in enumerate(zip(predictions, max_probs)):
                        if conf >= confidence_threshold:
                            signals.append(2 if pred == 1 else 0)  # Buy if up, Sell if down
                        else:
                            signals.append(1)  # Hold if low confidence
                    signals = pd.Series(signals, index=features_backtest.index)
                
                else:  # Direction with Stop Loss
                    # This would require more complex logic - simplified here
                    signals = pd.Series([2 if p == 1 else 0 for p in predictions], index=features_backtest.index)
            
            else:  # profit_prob
                if probabilities is not None:
                    prob_positive = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
                    signals = pd.Series([2 if p >= prob_threshold else 1 for p in prob_positive], index=features_backtest.index)
                else:
                    signals = pd.Series([2 if p == 1 else 1 for p in predictions], index=features_backtest.index)
            
            # Initialize backtester
            backtester = Backtester(initial_capital=initial_capital, commission=commission)
            
            # Run backtest
            results = backtester.run_backtest(df_backtest, signals, df_backtest['Close'])
            
            if 'error' in results:
                st.error(f"Backtesting error: {results['error']}")
            else:
                # Store results in session state
                st.session_state.backtest_results = results
                st.session_state.backtest_signals = signals
                
                st.success("✅ Backtest completed successfully!")
        
        except Exception as e:
            st.error(f"Error during backtesting: {str(e)}")

# Display backtest results
if 'backtest_results' in st.session_state:
    results = st.session_state.backtest_results
    
    st.header("📊 Backtest Results")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return", 
            f"{results['total_return']:.2%}",
            delta=f"{results['excess_return']:.2%} vs B&H"
        )
    
    with col2:
        st.metric("Final Portfolio Value", f"${results['final_value']:,.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
    
    with col4:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", results['total_trades'])
    
    with col2:
        st.metric("Win Rate", f"{results['win_rate']:.1%}")
    
    with col3:
        st.metric("Buy & Hold Return", f"{results['buy_hold_return']:.2%}")
    
    with col4:
        st.metric("Volatility", f"{results['volatility']:.2%}")
    
    # Performance visualization
    st.subheader("📈 Performance Chart")
    
    if 'backtest_signals' in st.session_state:
        signals = st.session_state.backtest_signals
        temp_backtester = Backtester(initial_capital=initial_capital, commission=commission)
        fig = temp_backtester.create_performance_chart(df_backtest, signals)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade history
    st.subheader("📋 Trade History")
    
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df['Date'] = pd.to_datetime(trades_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Show recent trades
        st.dataframe(trades_df.tail(20), use_container_width=True)
        
        # Trade analysis
        st.subheader("🔍 Trade Analysis")
        
        buy_trades = [t for t in results['trades'] if t['action'] == 'BUY']
        sell_trades = [t for t in results['trades'] if t['action'] == 'SELL']
        
        if buy_trades and sell_trades:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Buy Trades Distribution**")
                buy_prices = [t['price'] for t in buy_trades]
                fig_buy = go.Figure(data=[go.Histogram(x=buy_prices, name='Buy Prices')])
                fig_buy.update_layout(title="Distribution of Buy Prices", height=300)
                st.plotly_chart(fig_buy, use_container_width=True)
            
            with col2:
                st.markdown("**Sell Trades Distribution**")
                sell_prices = [t['price'] for t in sell_trades]
                fig_sell = go.Figure(data=[go.Histogram(x=sell_prices, name='Sell Prices')])
                fig_sell.update_layout(title="Distribution of Sell Prices", height=300)
                st.plotly_chart(fig_sell, use_container_width=True)
    else:
        st.info("No trades executed during the backtesting period.")
    
    # Risk analysis
    st.subheader("⚠️ Risk Analysis")
    
    if 'portfolio_value' in results:
        portfolio_series = results['portfolio_value']
        
        # Rolling metrics
        rolling_window = min(30, len(portfolio_series) // 4)
        rolling_returns = portfolio_series.pct_change().rolling(rolling_window).mean() * 252
        rolling_vol = portfolio_series.pct_change().rolling(rolling_window).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_vol
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns.values,
                name='Rolling Return',
                line=dict(color='blue')
            ))
            fig_rolling.update_layout(title="Rolling Annualized Returns", height=300)
            st.plotly_chart(fig_rolling, use_container_width=True)
        
        with col2:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name='Rolling Volatility',
                line=dict(color='red')
            ))
            fig_vol.update_layout(title="Rolling Annualized Volatility", height=300)
            st.plotly_chart(fig_vol, use_container_width=True)
    
    # Strategy comparison
    st.subheader("🔄 Strategy vs Benchmark")
    
    comparison_data = {
        'Metric': ['Total Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        'Strategy': [
            f"{results['total_return']:.2%}",
            f"{results['volatility']:.2%}",
            f"{results['sharpe_ratio']:.2f}",
            f"{results['max_drawdown']:.2%}"
        ],
        'Buy & Hold': [
            f"{results['buy_hold_return']:.2%}",
            f"{df_backtest['Close'].pct_change().std() * np.sqrt(252):.2%}",
            f"{(df_backtest['Close'].pct_change().mean() * 252) / (df_backtest['Close'].pct_change().std() * np.sqrt(252)):.2f}",
            "N/A"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Export results
    st.subheader("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Trade History"):
            if results['trades']:
                trades_csv = pd.DataFrame(results['trades']).to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=trades_csv,
                    file_name=f"trade_history_{selected_model}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("Download Performance Data"):
            if 'portfolio_value' in results:
                perf_data = pd.DataFrame({
                    'Date': results['portfolio_value'].index,
                    'Portfolio_Value': results['portfolio_value'].values,
                    'Returns': results['portfolio_value'].pct_change().values
                })
                perf_csv = perf_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=perf_csv,
                    file_name=f"performance_data_{selected_model}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

else:
    st.info("👆 Configure your backtesting parameters and click 'Run Backtest' to start the analysis.")

# Strategy optimization section
st.header("🔧 Strategy Optimization")

st.markdown("""
**Potential Improvements:**
- Parameter optimization for better performance
- Risk management rules (stop-loss, position sizing)
- Portfolio diversification across multiple signals
- Dynamic rebalancing based on market conditions
- Transaction cost analysis
- Out-of-sample testing for validation
""")

if st.button("🔄 Run Parameter Optimization"):
    st.info("Parameter optimization feature would test different combinations of strategy parameters to find optimal settings.")

# Final notes
st.markdown("---")
st.warning("""
**Important Disclaimers:**
- Past performance does not guarantee future results
- This is a simplified backtesting framework for educational purposes
- Real trading involves additional costs, slippage, and market impact
- Consider transaction costs, taxes, and regulatory requirements
- Always validate strategies on out-of-sample data before live trading
""")
