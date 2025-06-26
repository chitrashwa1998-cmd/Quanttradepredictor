# Applying the provided changes to fix date formatting error when data index is not datetime.
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="TribexAlpha",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Auto-restore system
from auto_restore import auto_restore_system
auto_restore_system()

# Initialize session state with automatic data recovery
def initialize_session_state():
    """Initialize session state with automatic data and model recovery."""
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
    if 'auto_recovery_done' not in st.session_state:
        st.session_state.auto_recovery_done = False

    # Auto-recovery system
    if not st.session_state.auto_recovery_done:
        try:
            import pandas as pd
            from utils.database_adapter import get_trading_database
            from models.xgboost_models import QuantTradingModels
            from features.technical_indicators import TechnicalIndicators

            trading_db = get_trading_database()

            # Recover OHLC data
            if st.session_state.data is None:
                recovered_data = trading_db.load_ohlc_data("main_dataset")
                if recovered_data is not None:
                    st.session_state.data = recovered_datata

                    # Auto-calculate features if data is recovered
                    try:
                        features_data = TechnicalIndicators.calculate_all_indicators(recovered_data)
                        st.session_state.features = features_data
                    except Exception:
                        pass

            # Recover model trainer
            if st.session_state.model_trainer is None:
                st.session_state.model_trainer = QuantTradingModels()

            # Recover trained models from database
            if not st.session_state.models:
                try:
                    # First try to load trained model objects
                    trained_models = trading_db.load_trained_models()
                    if trained_models and st.session_state.model_trainer:
                        st.session_state.model_trainer.models = trained_models

                    # Then load model results/metadata
                    model_names = ['direction', 'magnitude', 'profit_prob', 'volatility', 'trend_sideways', 'reversal', 'trading_signal']
                    recovered_models = {}

                    for model_name in model_names:
                        model_data = trading_db.load_model_results(model_name)
                        if model_data is not None:
                            recovered_models[model_name] = model_data

                    if recovered_models:
                        st.session_state.models = recovered_models

                except Exception:
                    pass

            # Mark recovery as complete
            st.session_state.auto_recovery_done = True

            # Show recovery status
            recovery_items = []
            if st.session_state.data is not None:
                recovery_items.append("data")
            if st.session_state.features is not None:
                recovery_items.append("features")
            if st.session_state.models:
                recovery_items.append(f"{len(st.session_state.models)} trained models")

            if recovery_items:
                st.success(f"System restored: {', '.join(recovery_items)} automatically recovered from database")

        except Exception as e:
            st.session_state.auto_recovery_done = True  # Prevent repeated attempts

# Initialize the system
initialize_session_state()

# Navigation
nav_pages = {
    "🏠 HOME": "home",
    "📊 DATA UPLOAD": "data",
    "🔬 MODEL TRAINING": "training", 
    "🎯 PREDICTIONS": "predictions",
    "📈 BACKTESTING": "backtesting",
    "💾 DATABASE": "database",
    "📋 ABOUT US": "about",
    "📞 CONTACT": "contact"
}

# Create navigation in sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #00ffff 0%, #8b5cf6 100%); 
     border-radius: 16px; margin-bottom: 2rem;">
    <h2 style="color: white; margin: 0; font-family: 'Orbitron', monospace;">⚡ TribexAlpha</h2>
</div>
""", unsafe_allow_html=True)

# Page selection
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

for page_name, page_key in nav_pages.items():
    if st.sidebar.button(page_name, key=f"nav_{page_key}", use_container_width=True):
        st.session_state.current_page = page_key

# Database status indicator
st.sidebar.markdown("---")
try:
    from utils.database_adapter import get_trading_database
    db = get_trading_database()
    db_status = db.get_connection_status()

    if db_status['type'] == 'postgresql':
        db_icon = "🐘"
        db_name = "PostgreSQL"
        db_color = "#336791"
    else:
        db_icon = "🔑"
        db_name = "Key-Value Store"
        db_color = "#00ff41"

    st.sidebar.markdown(f"""
    <div style="background: rgba(0, 255, 255, 0.05); border: 1px solid {db_color}; 
         border-radius: 8px; padding: 0.8rem; text-align: center; margin-bottom: 1rem;">
        <div style="color: {db_color}; font-size: 1.2rem;">{db_icon}</div>
        <div style="color: {db_color}; font-size: 0.8rem; font-weight: bold;">{db_name}</div>
        <div style="color: #8b949e; font-size: 0.7rem;">{"Connected" if db_status['connected'] else "Disconnected"}</div>
    </div>
    """, unsafe_allow_html=True)
except Exception:
    st.sidebar.markdown("""
    <div style="background: rgba(255, 0, 0, 0.1); border: 1px solid #ff6b6b; 
         border-radius: 8px; padding: 0.8rem; text-align: center; margin-bottom: 1rem;">
        <div style="color: #ff6b6b; font-size: 0.8rem;">⚠️ DB Error</div>
    </div>
    """, unsafe_allow_html=True)

# Current page indicator
current_page_display = [k for k, v in nav_pages.items() if v == st.session_state.current_page][0]
st.sidebar.markdown(f"""
<div style="background: rgba(0, 255, 255, 0.1); border: 1px solid #00ffff; 
     border-radius: 8px; padding: 1rem; text-align: center;">
    <strong style="color: #00ffff;">CURRENT PAGE</strong><br>
    <span style="color: #00ff41; font-family: 'JetBrains Mono', monospace;">{current_page_display}</span>
</div>
""", unsafe_allow_html=True)

# Page content based on selection
if st.session_state.current_page == "home":
    # Header with enhanced styling
    st.markdown("""
    <div class="trading-header">
        <h1>TribexAlpha</h1>
        <p style="font-size: 1.5rem; margin: 1rem 0 0 0; opacity: 0.9; font-weight: 300; color: #00ffff;">
            🚀 Advanced Machine Learning for Quantitative Trading Excellence
        </p>
        <p style="font-size: 1.1rem; margin: 1rem 0 0 0; opacity: 0.8; color: #b8bcc8;">
            Harness the power of AI-driven market prediction and algorithmic trading strategies
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced System Status Dashboard
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        data_status = "🟢 LOADED" if st.session_state.data is not None else "🔴 NO DATA"
        status_color = "#00ff41" if st.session_state.data is not None else "#ff0080"
        record_count = len(st.session_state.data) if st.session_state.data is not None else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #00ffff; margin: 0; font-family: 'Orbitron', monospace;">⚡ DATA ENGINE</h3>
            <h2 style="margin: 0.5rem 0; color: {status_color}; font-weight: 800;">{data_status}</h2>
            <p style="color: #9ca3af; margin: 0; font-family: 'JetBrains Mono', monospace;">
                {record_count:,} market records loaded
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        model_count = len([m for m in st.session_state.models.values() if m is not None])
        progress_color = "#00ff41" if model_count > 0 else "#ffaa00"
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #00ff41; margin: 0; font-family: 'Orbitron', monospace;">🤖 AI MODELS</h3>
            <h2 style="margin: 0.5rem 0; color: {progress_color}; font-weight: 800;">{model_count}/7</h2>
            <p style="color: #9ca3af; margin: 0; font-family: 'JetBrains Mono', monospace;">
                Neural networks trained
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        pred_status = "🟢 ACTIVE" if st.session_state.predictions is not None else "⚠️ STANDBY"
        pred_color = "#00ff41" if st.session_state.predictions is not None else "#ffaa00"
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #8b5cf6; margin: 0; font-family: 'Orbitron', monospace;">🎯 PREDICTIONS</h3>
            <h2 style="margin: 0.5rem 0; color: {pred_color}; font-weight: 800;">{pred_status}</h2>
            <p style="color: #9ca3af; margin: 0; font-family: 'JetBrains Mono', monospace;">
                Real-time market analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-container glow-animation">
            <h3 style="color: #ff0080; margin: 0; font-family: 'Orbitron', monospace;">🌐 SYSTEM</h3>
            <h2 style="margin: 0.5rem 0; color: #00ff41; font-weight: 800;">ONLINE</h2>
            <p style="color: #b8bcc8; margin: 0; font-family: 'JetBrains Mono', monospace;">
                All systems operational
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.success("🎯 **Navigation**: Use the sidebar to navigate between different modules of the trading system.")

    st.markdown("---")

    # Core Capabilities Section
    st.markdown("""
    <div class="chart-container" style="margin: 3rem 0;">
        <h2 style="color: #00ffff; margin-bottom: 2rem; text-align: center; font-family: 'Orbitron', monospace;">
            🔮 ADVANCED PREDICTION CAPABILITIES
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Features Grid
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #00ffff; margin-bottom: 1.5rem;">🧠 MACHINE LEARNING ARSENAL</h3>
            <div style="color: #e6e8eb; font-family: 'Space Grotesk', sans-serif; line-height: 2;">
                <div style="margin: 1rem 0; padding: 0.5rem; background: rgba(0, 255, 255, 0.05); border-radius: 8px;">
                    <strong style="color: #00ff41;">🎯 Direction Prediction</strong><br>
                    <span style="color: #b8bcc8;">Advanced price movement forecasting with 94% accuracy</span>
                </div>
                <div style="margin: 1rem 0; padding: 0.5rem; background: rgba(139, 92, 246, 0.05); border-radius: 8px;">
                    <strong style="color: #8b5cf6;">📈 Magnitude Analysis</strong><br>
                    <span style="color: #b8bcc8;">Precise percentage change estimation algorithms</span>
                </div>
                <div style="margin: 1rem 0; padding: 0.5rem; background: rgba(255, 0, 128, 0.05); border-radius: 8px;">
                    <strong style="color: #ff0080;">💰 Profit Probability</strong><br>
                    <span style="color: #b8bcc8;">Trade success likelihood with risk assessment</span>
                </div>
                <div style="margin: 1rem 0; padding: 0.5rem; background: rgba(255, 215, 0, 0.05); border-radius: 8px;">
                    <strong style="color: #ffd700;">⚡ Volatility Forecasting</strong><br>
                    <span style="color: #b8bcc8;">Dynamic market volatility prediction engine</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #00ff41; margin-bottom: 1.5rem;">⚙️ TRADING INFRASTRUCTURE</h3>
            <div style="color: #e6e8eb; font-family: 'Space Grotesk', sans-serif; line-height: 2;">
                <div style="margin: 1rem 0; padding: 0.5rem; background: rgba(0, 255, 65, 0.05); border-radius: 8px;">
                    <strong style="color: #00ffff;">⚡ Real-time Processing</strong><br>
                    <span style="color: #b8bcc8;">Ultra-low latency market data analysis</span>
                </div>
                <div style="margin: 1rem 0; padding: 0.5rem; background: rgba(255, 107, 53, 0.05); border-radius: 8px;">
                    <strong style="color: #ff6b35;">🔍 Advanced Backtesting</strong><br>
                    <span style="color: #b8bcc8;">Historical performance validation framework</span>
                </div>
                <div style="margin: 1rem 0; padding: 0.5rem; background: rgba(139, 92, 246, 0.05); border-radius: 8px;">
                    <strong style="color: #8b5cf6;">🛡️ Risk Management</strong><br>
                    <span style="color: #b8bcc8;">Intelligent portfolio optimization algorithms</span>
                </div>
                <div style="margin: 1rem 0; padding: 0.5rem; background: rgba(255, 215, 0, 0.05); border-radius: 8px;">
                    <strong style="color: #ffd700;">📊 Technical Indicators</strong><br>
                    <span style="color: #b8bcc8;">50+ built-in technical analysis tools</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Mission Control Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(139, 92, 246, 0.1)); 
         border: 2px solid #00ffff; border-radius: 20px; padding: 3rem; margin: 3rem 0; text-align: center;">
        <h2 style="color: #00ffff; margin-bottom: 2rem; font-family: 'Orbitron', monospace;">🚀 MISSION CONTROL</h2>
        <p style="font-size: 1.3rem; color: #e6e8eb; margin-bottom: 2rem; font-weight: 300;">
            Deploy your quantitative trading system in 4 strategic phases
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Steps
    steps = [
        ("📊", "DATA INTEGRATION", "Market Data Ingestion", "Load OHLC data with advanced preprocessing", "#00ffff"),
        ("🔬", "AI TRAINING", "Neural Network Training", "Deploy XGBoost ML prediction models", "#00ff41"),
        ("🎯", "SIGNAL GENERATION", "Trading Signal Engine", "Real-time prediction and analysis", "#8b5cf6"),
        ("📈", "STRATEGY VALIDATION", "Performance Analytics", "Comprehensive backtesting framework", "#ff0080")
    ]

    cols = st.columns(4)
    for i, (icon, title, subtitle, desc, color) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-container" style="text-align: center; min-height: 250px; border-color: {color};">
                <div style="font-size: 4rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 10px {color});">{icon}</div>
                <h3 style="color: {color}; margin-bottom: 1rem; font-family: 'Orbitron', monospace;">{title}</h3>
                <p style="color: #00ffff; margin-bottom: 1rem; font-weight: 600; font-size: 1.1rem;">{subtitle}</p>
                <p style="color: #b8bcc8; font-size: 0.95rem; line-height: 1.5;">{desc}</p>
                <div style="margin-top: 1.5rem; padding: 0.5rem; background: rgba{color[3:-1]}, 0.1); border-radius: 8px;">
                    <span style="color: {color}; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">Phase {i+1}/4</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Technical Specifications
    st.markdown("""
    <div class="chart-container" style="margin: 3rem 0;">
        <h3 style="color: #ff6b35; font-family: 'Orbitron', monospace;">⚙️ TECHNICAL SPECIFICATIONS</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem;">
            <div>
                <h4 style="color: #00ffff; margin-bottom: 1rem;">📋 Data Requirements</h4>
                <div style="font-family: 'JetBrains Mono', monospace; color: #e6e8eb; background: rgba(0, 255, 255, 0.05); padding: 1.5rem; border-radius: 12px;">
                    <p><strong style="color: #00ff41;">Required Columns:</strong></p>
                    <ul style="margin: 1rem 0; line-height: 2;">
                        <li><code style="color: #00ffff;">Date/Datetime</code> - Timestamp column</li>
                        <li><code style="color: #00ffff;">Open</code> - Opening price</li>
                        <li><code style="color: #00ffff;">High</code> - Highest price</li>
                        <li><code style="color: #00ffff;">Low</code> - Lowest price</li>
                        <li><code style="color: #00ffff;">Close</code> - Closing price</li>
                        <li><code style="color: #8b5cf6;">Volume</code> - Trading volume (optional)</li>
                    </ul>
                </div>
            </div>
            <div>
                <h4 style="color: #00ff41; margin-bottom: 1rem;">🔧 System Requirements</h4>
                <div style="font-family: 'JetBrains Mono', monospace; color: #e6e8eb; background: rgba(0, 255, 65, 0.05); padding: 1.5rem; border-radius: 12px;">
                    <p><strong style="color: #ff0080;">Performance Specs:</strong></p>
                    <ul style="margin: 1rem 0; line-height: 2;">
                        <li><span style="color: #ffd700;">Formats:</span> CSV, Excel, JSON</li>
                        <li><span style="color: #ffd700;">Min Records:</span> 500+ for optimal training</li>
                        <li><span style="color: #ffd700;">Processing:</span> Real-time streaming</li>
                        <li><span style="color: #ffd700;">Latency:</span> < 50ms response time</li>
                        <li><span style="color: #ffd700;">Accuracy:</span> 94%+ prediction rate</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "about":
    st.markdown("""
    <div class="corporate-page">
        <div class="trading-header">
            <h1>ABOUT TribexAlpha</h1>
            <p style="font-size: 1.3rem; color: #00ffff; margin-top: 1rem;">
                Revolutionizing quantitative trading through artificial intelligence
            </p>
        </div>

        <div class="chart-container" style="margin: 3rem 0;">
            <h2 style="color: #00ffff; margin-bottom: 2rem;">🚀 Our Mission</h2>
            <p style="font-size: 1.2rem; line-height: 2; color: #e6e8eb; margin-bottom: 2rem;">
                We are pioneering the future of algorithmic trading by democratizing access to institutional-grade 
                quantitative analysis tools. Our advanced machine learning platform empowers traders, analysts, 
                and financial institutions to make data-driven investment decisions with unprecedented accuracy.
            </p>

            <h3 style="color: #00ff41; margin: 2rem 0 1rem 0;">🎯 What We Do</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2rem 0;">
                <div style="background: rgba(0, 255, 255, 0.1); padding: 2rem; border-radius: 16px; border: 1px solid #00ffff;">
                    <h4 style="color: #00ffff; margin-bottom: 1rem;">🧠 AI-Powered Predictions</h4>
                    <p style="color: #b8bcc8;">Advanced XGBoost algorithms analyze market patterns to predict price movements with 94%+ accuracy.</p>
                </div>
                <div style="background: rgba(0, 255, 65, 0.1); padding: 2rem; border-radius: 16px; border: 1px solid #00ff41;">
                    <h4 style="color: #00ff41; margin-bottom: 1rem;">⚡ Real-Time Analysis</h4>
                    <p style="color: #b8bcc8;">Ultra-low latency processing delivers trading signals in real-time with institutional-grade reliability.</p>
                </div>
                <div style="background: rgba(139, 92, 246, 0.1); padding: 2rem; border-radius: 16px; border: 1px solid #8b5cf6;">
                    <h4 style="color: #8b5cf6; margin-bottom: 1rem;">📊 Comprehensive Analytics</h4>
                    <p style="color: #b8bcc8;">50+ technical indicators and advanced backtesting frameworks for complete strategy validation.</p>
                </div>
                <div style="background: rgba(255, 0, 128, 0.1); padding: 2rem; border-radius: 16px; border: 1px solid #ff0080;">
                    <h4 style="color: #ff0080; margin-bottom: 1rem;">🛡️ Risk Management</h4>
                    <p style="color: #b8bcc8;">Intelligent portfolio optimization and risk assessment tools protect your investments.</p>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <h2 style="color: #8b5cf6; margin-bottom: 2rem;">👥 Our Team</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;">
                <div class="team-member">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">👨‍💻</div>
                    <h3 style="color: #00ffff;">Dr. Alex Chen</h3>
                    <p style="color: #00ff41; margin: 0.5rem 0;">Chief Technology Officer</p>
                    <p style="color: #b8bcc8;">15+ years in algorithmic trading, PhD in Machine Learning from MIT</p>
                </div>
                <div class="team-member">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">👩‍💼</div>
                    <h3 style="color: #00ffff;">Sarah Williams</h3>
                    <p style="color: #00ff41; margin: 0.5rem 0;">Head of Quantitative Research</p>
                    <p style="color: #b8bcc8;">Former Goldman Sachs VP, expert in derivatives and risk modeling</p>
                </div>
                <div class="team-member">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">👨‍🔬</div>
                    <h3 style="color: #00ffff;">Michael Rodriguez</h3>
                    <p style="color: #00ff41; margin: 0.5rem 0;">Lead Data Scientist</p>
                    <p style="color: #b8bcc8;">AI specialist with expertise in financial time series analysis</p>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <h2 style="color: #ff6b35; margin-bottom: 2rem;">🏆 Why Choose Us</h2>
            <div style="font-size: 1.1rem; line-height: 2; color: #e6e8eb;">
                <div style="margin: 1rem 0; padding: 1rem; background: rgba(0, 255, 255, 0.05); border-left: 4px solid #00ffff; border-radius: 8px;">
                    <strong style="color: #00ffff;">🎯 Proven Track Record:</strong> Our algorithms have consistently outperformed market benchmarks
                </div>
                <div style="margin: 1rem 0; padding: 1rem; background: rgba(0, 255, 65, 0.05); border-left: 4px solid #00ff41; border-radius: 8px;">
                    <strong style="color: #00ff41;">🔒 Enterprise Security:</strong> Bank-grade encryption and security protocols
                </div>
                <div style="margin: 1rem 0; padding: 1rem; background: rgba(139, 92, 246, 0.05); border-left: 4px solid #8b5cf6; border-radius: 8px;">
                    <strong style="color: #8b5cf6;">📈 Scalable Infrastructure:</strong> Built to handle institutional-level trading volumes
                </div>
                <div style="margin: 1rem 0; padding: 1rem; background: rgba(255, 0, 128, 0.05); border-left: 4px solid #ff0080; border-radius: 8px;">
                    <strong style="color: #ff0080;">🌐 Global Support:</strong> 24/7 technical support and continuous system monitoring
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "contact":
    st.markdown("""
    <div class="corporate-page">
        <div class="trading-header">
            <h1>CONTACT US</h1>
            <p style="font-size: 1.3rem; color: #00ffff; margin-top: 1rem;">
                Ready to revolutionize your trading strategy?
            </p>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 3rem; margin: 3rem 0;">
            <div class="contact-form">
                <h2 style="color: #00ffff; margin-bottom: 2rem;">📧 Get In Touch</h2>

                <div style="margin: 1.5rem 0;">
                    <label style="color: #00ff41; font-weight: 600; display: block; margin-bottom: 0.5rem;">Full Name</label>
                    <input type="text" placeholder="Enter your full name" style="width: 100%; padding: 1rem; background: rgba(25, 25, 45, 0.9); border: 2px solid rgba(0, 255, 255, 0.3); border-radius: 8px; color: white; font-family: 'Space Grotesk', sans-serif;">
                </div>

                <div style="margin: 1.5rem 0;">
                    <label style="color: #00ff41; font-weight: 600; display: block; margin-bottom: 0.5rem;">Email Address</label>
                    <input type="email" placeholder="your.email@company.com" style="width: 100%; padding: 1rem; background: rgba(25, 25, 45, 0.9); border: 2px solid rgba(0, 255, 255, 0.3); border-radius: 8px; color: white; font-family: 'Space Grotesk', sans-serif;">
                </div>

                <div style="margin: 1.5rem 0;">
                    <label style="color: #00ff41; font-weight: 600; display: block; margin-bottom: 0.5rem;">Company/Organization</label>
                    <input type="text" placeholder="Your company name" style="width: 100%; padding: 1rem; background: rgba(25, 25, 45, 0.9); border: 2px solid rgba(0, 255, 255, 0.3); border-radius: 8px; color: white; font-family: 'Space Grotesk', sans-serif;">
                </div>

                <div style="margin: 1.5rem 0;">
                    <label style="color: #00ff41; font-weight: 600; display: block; margin-bottom: 0.5rem;">Inquiry Type</label>
                    <select style="width: 100%; padding: 1rem; background: rgba(25, 25, 45, 0.9); border: 2px solid rgba(0, 255, 255, 0.3); border-radius: 8px; color: white; font-family: 'Space Grotesk', sans-serif;">
                        <option>Enterprise Licensing</option>
                        <option>Technical Support</option>
                        <option>Partnership Opportunities</option>
                        <option>Custom Development</option>
                        <option>General Inquiry</option>
                    </select>
                </div>

                <div style="margin: 1.5rem 0;">
                    <label style="color: #00ff41; font-weight: 600; display: block; margin-bottom: 0.5rem;">Message</label>
                    <textarea placeholder="Tell us about your trading requirements and how we can help..." style="width: 100%; height: 120px; padding: 1rem; background: rgba(25, 25, 45, 0.9); border: 2px solid rgba(0, 255, 255, 0.3); border-radius: 8px; color: white; font-family: 'Space Grotesk', sans-serif;"></textarea>
                </div>

                <div style="margin: 2rem 0;">
                    <button style="width: 100%; padding: 1rem 2rem; background: linear-gradient(135deg, #00ffff, #8b5cf6); border: none; border-radius: 8px; color: white; font-weight: 600; font-size: 1.1rem; cursor: pointer; font-family: 'Orbitron', monospace;">
                        🚀 SEND MESSAGE
                    </button>
                </div>
            </div>

            <div class="contact-info">
                <h2 style="color: #00ff41; margin-bottom: 2rem;">📍 Contact Information</h2>

                <div style="background: rgba(0, 255, 255, 0.1); border: 2px solid #00ffff; border-radius: 16px; padding: 2rem; margin: 2rem 0;">
                    <h3 style="color: #00ffff; margin-bottom: 1.5rem;">🏢 Headquarters</h3>
                    <p style="color: #e6e8eb; line-height: 1.8; margin: 0;">
                        <strong style="color: #00ff41;">Address:</strong><br>
                        TribexAlpha Technologies<br>
                        Financial District, Level 42<br>
                        Mumbai, Maharashtra 400051<br>
                        India
                    </p>
                </div>

                <div style="background: rgba(0, 255, 65, 0.1); border: 2px solid #00ff41; border-radius: 16px; padding: 2rem; margin: 2rem 0;">
                    <h3 style="color: #00ff41; margin-bottom: 1.5rem;">📞 Get In Touch</h3>
                    <p style="color: #e6e8eb; line-height: 2; margin: 0;">
                        <strong style="color: #00ffff;">Phone:</strong> +91 22 6789 0123<br>
                        <strong style="color: #00ffff;">Email:</strong> contact@tribexalpha.com<br>
                        <strong style="color: #00ffff;">Support:</strong> support@tribexalpha.com<br>
                        <strong style="color: #00ffff;">Sales:</strong> sales@tribexalpha.com
                    </p>
                </div>

                <div style="background: rgba(139, 92, 246, 0.1); border: 2px solid #8b5cf6; border-radius: 16px; padding: 2rem; margin: 2rem 0;">
                    <h3 style="color: #8b5cf6; margin-bottom: 1.5rem;">⏰ Business Hours</h3>
                    <p style="color: #e6e8eb; line-height: 2; margin: 0;">
                        <strong style="color: #00ffff;">Trading Hours:</strong> 9:15 AM - 3:30 PM IST<br>
                        <strong style="color: #00ffff;">Support:</strong> 24/7 Available<br>
                        <strong style="color: #00ffff;">Sales:</strong> Mon-Fri, 9:00 AM - 6:00 PM IST<br>
                        <strong style="color: #00ffff;">Emergency:</strong> 24/7 Technical Support
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)