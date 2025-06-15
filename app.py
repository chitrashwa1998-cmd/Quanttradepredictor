import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Quantitative Trading Model System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Main page
st.title("📈 Quantitative Trading Model System")
st.markdown("---")

st.markdown("""
## Welcome to the Comprehensive Trading Model System

This application provides a complete quantitative trading framework using XGBoost for multi-target stock market predictions.

### Available Predictions:
1. **Direction Prediction** - Predict if price will go up or down
2. **Magnitude of Move** - Estimate the percentage change
3. **Probability of Profit** - Score the likelihood of profitable trades
4. **Volatility Forecasting** - Predict future volatility
5. **Trend vs Sideways** - Classify market conditions
6. **Reversal Points** - Identify potential trend reversals
7. **Buy/Sell/Hold Signals** - Generate trading recommendations

### How to Use:
1. **Data Upload**: Upload your OHLC data (CSV format)
2. **Model Training**: Train XGBoost models for different prediction tasks
3. **Predictions**: View model predictions and analysis
4. **Backtesting**: Evaluate strategy performance

### Data Format Requirements:
Your CSV file should contain the following columns:
- `Date` or `Datetime`: Date/time column
- `Open`: Opening price
- `High`: Highest price
- `Low`: Lowest price
- `Close`: Closing price
- `Volume` (optional): Trading volume

Navigate to the **Data Upload** page to get started!
""")

# Sidebar information
st.sidebar.title("Navigation")
st.sidebar.markdown("""
Use the pages in the sidebar to navigate through the application:

📊 **Data Upload** - Load your OHLC data
🔬 **Model Training** - Train prediction models
🎯 **Predictions** - View model results
📈 **Backtesting** - Evaluate performance
""")

# Display current data info if available
if st.session_state.data is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Dataset")
    st.sidebar.info(f"""
    **Rows**: {len(st.session_state.data)}
    **Date Range**: {st.session_state.data.index.min().strftime('%Y-%m-%d')} to {st.session_state.data.index.max().strftime('%Y-%m-%d')}
    **Columns**: {', '.join(st.session_state.data.columns)}
    """)

# Display model status if available
if st.session_state.models:
    st.sidebar.markdown("### Trained Models")
    for model_name in st.session_state.models.keys():
        st.sidebar.success(f"✅ {model_name}")

st.sidebar.markdown("---")
st.sidebar.markdown("*Built with Streamlit & XGBoost*")
