
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(page_title="Ultimate Predictions", page_icon="🎯", layout="wide")

# Initialize session state
if 'ultimate_manager' not in st.session_state:
    st.session_state.ultimate_manager = None
if 'unified_predictions' not in st.session_state:
    st.session_state.unified_predictions = None

st.title("🎯 Ultimate Prediction System")
st.markdown("**Unified predictions from all your AI models in one comprehensive view**")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("❌ No data available. Please upload data first.")
    st.stop()

# Initialize Ultimate Model Manager
if st.session_state.ultimate_manager is None:
    with st.spinner("Initializing Ultimate Model System..."):
        try:
            from models.ultimate_model_manager import UltimateModelManager
            st.session_state.ultimate_manager = UltimateModelManager()
            st.session_state.ultimate_manager.load_all_models()
            st.success("✅ Ultimate Model System initialized!")
        except Exception as e:
            st.error(f"❌ Failed to initialize Ultimate Model System: {str(e)}")
            st.stop()

ultimate_manager = st.session_state.ultimate_manager

# Model Status Dashboard
st.header("🤖 Model Status Dashboard")

model_summary = ultimate_manager.get_model_summary()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Volatility Model", model_summary.get('Volatility', 'Unknown'))
    st.metric("Direction Model", model_summary.get('Direction', 'Unknown'))

with col2:
    st.metric("Profit Probability", model_summary.get('Profit Prob', 'Unknown'))
    st.metric("Reversal Detection", model_summary.get('Reversal', 'Unknown'))

with col3:
    st.metric("Trend/Sideways", model_summary.get('Trend Sideways', 'Unknown'))
    st.metric("Ultimate Model", model_summary.get('Ultimate', 'Unknown'))

with col4:
    available_models = sum(1 for status in model_summary.values() if "✅" in status)
    st.metric("Total Models", f"{available_models}/6", f"{(available_models/6)*100:.0f}% Ready")

# Main Interface
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Unified Predictions", 
    "🔧 Train Ultimate Model", 
    "📊 Model Comparison", 
    "📈 Performance Analytics"
])

with tab1:
    st.header("🎯 Unified Prediction Dashboard")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_rows = st.selectbox(
            "Number of Recent Predictions",
            [50, 100, 200, 500, 1000],
            index=1
        )
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh predictions", value=False)
    
    with col3:
        if st.button("🔄 Generate Unified Predictions", type="primary"):
            st.session_state.unified_predictions = None  # Force refresh
    
    # Generate unified predictions
    if st.session_state.unified_predictions is None or auto_refresh:
        with st.spinner("Generating unified predictions from all models..."):
            try:
                df = st.session_state.data
                unified_df = ultimate_manager.create_unified_prediction_table(df, limit_rows=prediction_rows)
                st.session_state.unified_predictions = unified_df
                st.success(f"✅ Generated {len(unified_df)} unified predictions!")
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.session_state.unified_predictions = None
    
    # Display unified predictions
    if st.session_state.unified_predictions is not None:
        unified_df = st.session_state.unified_predictions
        
        # Summary metrics
        st.subheader("📊 Prediction Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Direction_Pred' in unified_df.columns:
                bullish_pct = (unified_df['Direction_Pred'] == 'Bullish').mean() * 100
                st.metric("Bullish Signals", f"{bullish_pct:.1f}%")
        
        with col2:
            if 'Vol_Regime' in unified_df.columns:
                high_vol_pct = unified_df['Vol_Regime'].isin(['High', 'Very High']).mean() * 100
                st.metric("High Volatility", f"{high_vol_pct:.1f}%")
        
        with col3:
            if 'Consensus_Score' in unified_df.columns:
                avg_consensus = unified_df['Consensus_Score'].mean() * 100
                st.metric("Avg Consensus", f"{avg_consensus:.1f}%")
        
        with col4:
            if 'Ultimate_Confidence' in unified_df.columns:
                avg_confidence = unified_df['Ultimate_Confidence'].mean() * 100
                st.metric("Ultimate Confidence", f"{avg_confidence:.1f}%")
        
        # Interactive Chart
        st.subheader("📈 Unified Prediction Chart")
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price & Signals', 'Volatility Regime', 'Consensus Score'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price candlestick with signals
        fig.add_trace(
            go.Candlestick(
                x=unified_df.index,
                open=unified_df['Open'],
                high=unified_df['High'],
                low=unified_df['Low'],
                close=unified_df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add direction signals
        if 'Direction_Pred' in unified_df.columns:
            bullish_points = unified_df[unified_df['Direction_Pred'] == 'Bullish']
            bearish_points = unified_df[unified_df['Direction_Pred'] == 'Bearish']
            
            fig.add_trace(
                go.Scatter(
                    x=bullish_points.index,
                    y=bullish_points['Close'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=8, color='green'),
                    name='Bullish Signal'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=bearish_points.index,
                    y=bearish_points['Close'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=8, color='red'),
                    name='Bearish Signal'
                ),
                row=1, col=1
            )
        
        # Volatility regime
        if 'Volatility_Pred' in unified_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=unified_df.index,
                    y=unified_df['Volatility_Pred'],
                    mode='lines',
                    name='Volatility',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
        
        # Consensus score
        if 'Consensus_Score' in unified_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=unified_df.index,
                    y=unified_df['Consensus_Score'],
                    mode='lines',
                    name='Consensus',
                    line=dict(color='purple'),
                    fill='tonexty'
                ),
                row=3, col=1
            )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data Table
        st.subheader("📋 Unified Prediction Data")
        
        # Format display columns
        display_columns = []
        if 'Timestamp' in unified_df.columns:
            display_columns.append('Timestamp')
        
        display_columns.extend(['Open', 'High', 'Low', 'Close'])
        
        # Add prediction columns
        pred_columns = [col for col in unified_df.columns if any(x in col for x in ['Pred', 'Signal', 'Regime', 'Confidence', 'Consensus'])]
        display_columns.extend(pred_columns)
        
        # Filter available columns
        available_columns = [col for col in display_columns if col in unified_df.columns]
        
        st.dataframe(
            unified_df[available_columns].tail(50),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = unified_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Unified Predictions CSV",
            data=csv,
            file_name=f"unified_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

with tab2:
    st.header("🔧 Train Ultimate Model")
    
    st.info("The Ultimate Model learns from all your individual model predictions to make even better predictions!")
    
    # Check if we have enough individual models
    trained_count = sum(1 for status in model_summary.values() if "✅" in status and "Ultimate" not in status)
    
    if trained_count < 2:
        st.warning(f"⚠️ You need at least 2 individual models trained. Currently have: {trained_count}")
        st.markdown("**Go train more individual models first:**")
        st.markdown("- Volatility Model")
        st.markdown("- Direction Model") 
        st.markdown("- Profit Probability Model")
        st.markdown("- Reversal Detection Model")
        st.markdown("- Trend/Sideways Model")
    else:
        st.success(f"✅ {trained_count} individual models available for ultimate training!")
        
        # Training configuration
        col1, col2 = st.columns(2)
        
        with col1:
            target_type = st.selectbox(
                "Ultimate Model Target",
                ['direction', 'return', 'volatility', 'profit_signal'],
                help="What should the ultimate model predict?"
            )
        
        with col2:
            train_split = st.slider("Training Split", 0.7, 0.9, 0.8, 0.05)
        
        # Train button
        if st.button("🚀 Train Ultimate Model", type="primary"):
            with st.spinner("Training Ultimate Model... This may take a few minutes..."):
                try:
                    df = st.session_state.data
                    result = ultimate_manager.train_ultimate_model(df, target_type)
                    
                    st.success("✅ Ultimate Model trained successfully!")
                    
                    # Display results
                    if 'metrics' in result:
                        metrics = result['metrics']
                        
                        col1, col2 = st.columns(2)
                        
                        if ultimate_manager.ultimate_model.task_type == 'classification':
                            with col1:
                                accuracy = metrics.get('accuracy', 0)
                                st.metric("Accuracy", f"{accuracy:.2%}")
                            
                            with col2:
                                if 'classification_report' in metrics:
                                    f1_score = metrics['classification_report']['macro avg']['f1-score']
                                    st.metric("F1 Score", f"{f1_score:.3f}")
                        else:
                            with col1:
                                rmse = metrics.get('rmse', 0)
                                st.metric("RMSE", f"{rmse:.6f}")
                            
                            with col2:
                                mae = metrics.get('mae', 0)
                                st.metric("MAE", f"{mae:.6f}")
                    
                    # Feature importance
                    if 'feature_importance' in result and result['feature_importance']:
                        st.subheader("🎯 Feature Importance")
                        
                        importance_df = pd.DataFrame(
                            list(result['feature_importance'].items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            importance_df.head(15),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 15 Most Important Features"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())

with tab3:
    st.header("📊 Model Comparison")
    st.info("Compare performance across all your individual models")
    
    # This would show comparative metrics across models
    st.markdown("**Coming Soon**: Detailed model comparison metrics")

with tab4:
    st.header("📈 Performance Analytics")
    st.info("Deep dive into Ultimate Model performance")
    
    # This would show detailed performance analytics
    st.markdown("**Coming Soon**: Advanced performance analytics and backtesting")
