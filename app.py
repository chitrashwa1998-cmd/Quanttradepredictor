
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="THE QUANT TRADING ENGINE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None

# Header with professional styling
st.markdown("""
<div class="trading-header">
    <h1>QUANT TRADING ENGINE</h1>
    <p style="font-size: 1.3rem; margin: 1rem 0 0 0; opacity: 0.9; font-weight: 300;">
        Advanced Machine Learning for Quantitative Trading
    </p>
</div>
""", unsafe_allow_html=True)

# System Status Dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    data_status = "🟢 LOADED" if st.session_state.data is not None else "🔴 NO DATA"
    st.markdown(f"""
    <div class="metric-container">
        <h3 style="color: #00d4ff; margin: 0;">DATA STATUS</h3>
        <h2 style="margin: 0.5rem 0;">{data_status}</h2>
        <p style="color: #9ca3af; margin: 0;">
            {len(st.session_state.data) if st.session_state.data is not None else 0} records
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    model_count = len([m for m in st.session_state.models.values() if m is not None])
    st.markdown(f"""
    <div class="metric-container">
        <h3 style="color: #00ff88; margin: 0;">MODELS TRAINED</h3>
        <h2 style="margin: 0.5rem 0;">{model_count}/7</h2>
        <p style="color: #9ca3af; margin: 0;">
            Active ML Models
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    pred_status = "🟢 READY" if st.session_state.predictions is not None else "⚠️ PENDING"
    st.markdown(f"""
    <div class="metric-container">
        <h3 style="color: #ffaa00; margin: 0;">PREDICTIONS</h3>
        <h2 style="margin: 0.5rem 0;">{pred_status}</h2>
        <p style="color: #9ca3af; margin: 0;">
            Real-time Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-container glow-animation">
        <h3 style="color: #00ffa3; margin: 0;">SYSTEM</h3>
        <h2 style="margin: 0.5rem 0;">ONLINE</h2>
        <p style="color: #b8b9cf; margin: 0;">
            All Systems Go
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main Content Area
st.markdown("""
<div class="chart-container" style="margin: 2rem 0;">
    <h2 style="color: #00ffa3; margin-bottom: 1rem;">ADVANCED PREDICTION CAPABILITIES</h2>
</div>
""", unsafe_allow_html=True)

# Features Grid
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="chart-container">
        <h3 style="color: #00d4ff;">🔮 MACHINE LEARNING MODELS</h3>
        <ul style="color: #e6e8eb; font-family: 'JetBrains Mono', monospace;">
            <li><strong>Direction Prediction</strong> - Price movement forecasting</li>
            <li><strong>Magnitude Analysis</strong> - Percentage change estimation</li>
            <li><strong>Profit Probability</strong> - Trade success likelihood</li>
            <li><strong>Volatility Forecasting</strong> - Risk assessment</li>
            <li><strong>Trend Classification</strong> - Market condition analysis</li>
            <li><strong>Reversal Detection</strong> - Turn point identification</li>
            <li><strong>Signal Generation</strong> - Buy/Sell/Hold recommendations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="chart-container">
        <h3 style="color: #00ff88;">⚙️ SYSTEM FEATURES</h3>
        <ul style="color: #e6e8eb; font-family: 'JetBrains Mono', monospace;">
            <li><strong>Real-time Processing</strong> - Live market analysis</li>
            <li><strong>Advanced Backtesting</strong> - Historical performance</li>
            <li><strong>Risk Management</strong> - Portfolio optimization</li>
            <li><strong>Technical Indicators</strong> - 50+ built-in indicators</li>
            <li><strong>Database Management</strong> - Persistent data storage</li>
            <li><strong>Multi-timeframe</strong> - Cross-temporal analysis</li>
            <li><strong>Performance Metrics</strong> - Comprehensive reporting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Getting Started Section
st.markdown("""
<div style="background: rgba(0, 212, 255, 0.05); border: 1px solid #00d4ff; 
     border-radius: 12px; padding: 2rem; margin: 2rem 0;">
    <h2 style="color: #00d4ff; margin-bottom: 1rem;">🚀 MISSION CONTROL</h2>
    <p style="font-size: 1.1rem; color: #e6e8eb; margin-bottom: 1.5rem;">
        Initialize your quantitative trading system in 4 simple steps:
    </p>
</div>
""", unsafe_allow_html=True)

# Steps
steps = [
    ("📊", "DATA UPLOAD", "Load your OHLC market data", "Upload CSV files with historical price data"),
    ("🔬", "MODEL TRAINING", "Train ML prediction models", "Configure and train XGBoost models"),
    ("🎯", "PREDICTIONS", "Generate trading signals", "Real-time market analysis and forecasting"),
    ("📈", "BACKTESTING", "Evaluate performance", "Test strategies on historical data")
]

cols = st.columns(4)
for i, (icon, title, subtitle, desc) in enumerate(steps):
    with cols[i]:
        st.markdown(f"""
        <div class="metric-container" style="text-align: center; min-height: 200px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
            <h3 style="color: #00d4ff; margin-bottom: 0.5rem;">{title}</h3>
            <p style="color: #00ff88; margin-bottom: 1rem; font-weight: 600;">{subtitle}</p>
            <p style="color: #9ca3af; font-size: 0.9rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Data Requirements
st.markdown("""
<div class="chart-container" style="margin: 2rem 0;">
    <h3 style="color: #ffaa00;">📋 DATA REQUIREMENTS</h3>
    <div style="font-family: 'JetBrains Mono', monospace; color: #e6e8eb;">
        <p><strong>Required Columns:</strong></p>
        <ul>
            <li><code>Date/Datetime</code> - Timestamp column</li>
            <li><code>Open</code> - Opening price</li>
            <li><code>High</code> - Highest price</li>
            <li><code>Low</code> - Lowest price</li>
            <li><code>Close</code> - Closing price</li>
            <li><code>Volume</code> - Trading volume (optional)</li>
        </ul>
        <p><strong>Supported Formats:</strong> CSV, Excel, JSON</p>
        <p><strong>Minimum Records:</strong> 500+ for optimal model training</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Enhancement
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
     border-radius: 10px; margin-bottom: 2rem;">
    <h2 style="color: white; margin: 0;">⚡ CONTROL PANEL</h2>
</div>
""", unsafe_allow_html=True)

# Display current data info if available
if st.session_state.data is not None:
    st.sidebar.markdown("""
    <div style="background: rgba(0, 255, 136, 0.1); border: 1px solid #00ff88; 
         border-radius: 8px; padding: 1rem; margin: 1rem 0;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">📊 DATASET STATUS</h4>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; color: #e6e8eb;">
            <p><strong>Records:</strong> {len(st.session_state.data):,}</p>
            <p><strong>From:</strong> {st.session_state.data.index.min().strftime('%Y-%m-%d')}</p>
            <p><strong>To:</strong> {st.session_state.data.index.max().strftime('%Y-%m-%d')}</p>
            <p><strong>Columns:</strong> {len(st.session_state.data.columns)}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display model status if available
if st.session_state.models:
    st.sidebar.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid #00d4ff; 
         border-radius: 8px; padding: 1rem; margin: 1rem 0;">
        <h4 style="color: #00d4ff; margin-bottom: 1rem;">🤖 MODEL STATUS</h4>
    """, unsafe_allow_html=True)
    
    for model_name, model in st.session_state.models.items():
        status = "✅ TRAINED" if model is not None else "⏳ PENDING"
        color = "#00ff88" if model is not None else "#ffaa00"
        st.sidebar.markdown(f"""
        <div style="display: flex; justify-content: space-between; color: {color}; 
             font-family: 'JetBrains Mono', monospace; margin: 0.5rem 0;">
            <span>{model_name.replace('_', ' ').title()}</span>
            <span>{status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #9ca3af; font-family: 'JetBrains Mono', monospace;">
    <p>THE QUANT TRADING ENGINE</p>
    <p>v2.0 | Built with ⚡</p>
</div>
""", unsafe_allow_html=True)
