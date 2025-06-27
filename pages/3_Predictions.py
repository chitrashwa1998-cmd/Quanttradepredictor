import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

st.title("🔮 Model Predictions")
st.markdown("Generate and analyze predictions using the trained models.")

# Check prerequisites
if st.session_state.data is None:
    st.error("❌ No data available. Please upload data first.")
    st.stop()

# Create tabs for different prediction types
volatility_tab, direction_tab = st.tabs(["📊 Volatility Predictions", "🎯 Direction Predictions"])

# Volatility Predictions Tab
with volatility_tab:
    st.header("📊 Volatility Predictions")
    
    # Check if volatility features and model are available
    if st.session_state.features is None:
        st.error("❌ No volatility features calculated. Please calculate technical indicators first.")
    elif not hasattr(st.session_state, 'trained_models') or not st.session_state.trained_models:
        st.error("❌ No trained models available. Please train the volatility model first.")
    elif 'volatility' not in st.session_state.trained_models or st.session_state.trained_models['volatility'] is None:
        st.error("❌ Volatility model not trained. Please train the model first.")
    else:
        # Volatility prediction controls
        st.subheader("🎯 Prediction Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vol_filter = st.selectbox(
                "📅 Time Period Filter",
                ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All data"],
                index=1,
                help="Select the time period for volatility predictions",
                key="vol_filter"
            )
        
        with col2:
            st.metric("Volatility Model Status", "✅ Ready", help="Volatility model is trained and ready")
        
        # Generate volatility predictions button
        if st.button("🚀 Generate Volatility Predictions", type="primary", key="vol_predict"):
            try:
                with st.spinner("Generating volatility predictions..."):
                    # Get model trainer
                    model_trainer = st.session_state.model_trainer
                    
                    # Use volatility features for prediction
                    data_for_prediction = st.session_state.features.copy()
                    
                    # Generate predictions
                    predictions, _ = model_trainer.predict('volatility', data_for_prediction)
                    
                    # Store predictions
                    st.session_state.volatility_predictions = predictions
                    
                    st.success("✅ Volatility predictions generated successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error generating volatility predictions: {str(e)}")
        
        # Display volatility predictions if available
        if hasattr(st.session_state, 'volatility_predictions') and st.session_state.volatility_predictions is not None:
            st.subheader("📈 Volatility Prediction Results")
            
            # Show prediction statistics
            predictions = st.session_state.volatility_predictions
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Volatility", f"{np.mean(predictions):.4f}")
            with col2:
                st.metric("Max Volatility", f"{np.max(predictions):.4f}")
            with col3:
                st.metric("Min Volatility", f"{np.min(predictions):.4f}")
            with col4:
                st.metric("Volatility Range", f"{np.max(predictions) - np.min(predictions):.4f}")
            
            # Create volatility prediction chart
            fig = go.Figure()
            
            # Add actual prices
            data_len = min(len(st.session_state.data), len(predictions))
            recent_data = st.session_state.data.tail(data_len)
            
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=1),
                yaxis='y1'
            ))
            
            # Add volatility predictions on secondary y-axis
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=predictions[-data_len:],
                mode='lines',
                name='Predicted Volatility',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
            
            # Update layout for dual y-axis
            fig.update_layout(
                title="Price vs Predicted Volatility",
                xaxis_title="Time",
                yaxis=dict(title="Price", side="left"),
                yaxis2=dict(title="Predicted Volatility", side="right", overlaying="y"),
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Direction Predictions Tab
with direction_tab:
    st.header("🎯 Direction Predictions")
    
    # Check if direction features and model are available
    if not hasattr(st.session_state, 'direction_features') or st.session_state.direction_features is None:
        st.error("❌ No direction features calculated. Please calculate direction indicators first.")
    elif not hasattr(st.session_state, 'direction_trained_models') or not st.session_state.direction_trained_models:
        st.error("❌ No trained direction models available. Please train the direction model first.")
    elif 'direction' not in st.session_state.direction_trained_models or st.session_state.direction_trained_models['direction'] is None:
        st.error("❌ Direction model not trained. Please train the direction model first.")
    else:
        # Direction prediction controls
        st.subheader("🎯 Prediction Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dir_filter = st.selectbox(
                "📅 Time Period Filter",
                ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All data"],
                index=1,
                help="Select the time period for direction predictions",
                key="dir_filter"
            )
        
        with col2:
            st.metric("Direction Model Status", "✅ Ready", help="Direction model is trained and ready")
        
        # Generate direction predictions button
        if st.button("🚀 Generate Direction Predictions", type="primary", key="dir_predict"):
            try:
                with st.spinner("Generating direction predictions..."):
                    # Get direction model
                    direction_model = st.session_state.direction_trained_models['direction']
                    
                    # Use direction features for prediction
                    direction_features = st.session_state.direction_features.copy()
                    
                    # Prepare features for prediction
                    features_prepared = direction_model.prepare_features(direction_features)
                    
                    # Generate predictions
                    predictions, probabilities = direction_model.predict(features_prepared)
                    
                    # Store predictions
                    st.session_state.direction_predictions = predictions
                    st.session_state.direction_probabilities = probabilities
                    
                    st.success("✅ Direction predictions generated successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error generating direction predictions: {str(e)}")
                import traceback
                st.error(f"Error details: {traceback.format_exc()}")
        
        # Display direction predictions if available
        if hasattr(st.session_state, 'direction_predictions') and st.session_state.direction_predictions is not None:
            st.subheader("📈 Direction Prediction Results")
            
            # Show prediction statistics
            predictions = st.session_state.direction_predictions
            probabilities = st.session_state.direction_probabilities
            
            col1, col2, col3, col4 = st.columns(4)
            
            bullish_count = np.sum(predictions == 1)
            bearish_count = np.sum(predictions == 0)
            bullish_pct = (bullish_count / len(predictions)) * 100
            avg_confidence = np.mean(np.max(probabilities, axis=1)) if probabilities is not None else 0
            
            with col1:
                st.metric("Bullish Signals", f"{bullish_count}")
            with col2:
                st.metric("Bearish Signals", f"{bearish_count}")
            with col3:
                st.metric("Bullish %", f"{bullish_pct:.1f}%")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Create direction prediction chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxis=True,
                vertical_spacing=0.1,
                subplot_titles=('Price Chart', 'Direction Predictions'),
                row_heights=[0.7, 0.3]
            )
            
            # Add price chart
            data_len = min(len(st.session_state.data), len(predictions))
            recent_data = st.session_state.data.tail(data_len)
            
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=1)
            ), row=1, col=1)
            
            # Add direction predictions
            bullish_mask = predictions[-data_len:] == 1
            bearish_mask = predictions[-data_len:] == 0
            
            # Bullish signals
            fig.add_trace(go.Scatter(
                x=recent_data.index[bullish_mask],
                y=[1] * np.sum(bullish_mask),
                mode='markers',
                name='Bullish',
                marker=dict(color='green', size=8, symbol='triangle-up'),
            ), row=2, col=1)
            
            # Bearish signals
            fig.add_trace(go.Scatter(
                x=recent_data.index[bearish_mask],
                y=[0] * np.sum(bearish_mask),
                mode='markers',
                name='Bearish',
                marker=dict(color='red', size=8, symbol='triangle-down'),
            ), row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title="Price vs Direction Predictions",
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Direction", row=2, col=1, range=[-0.1, 1.1])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recent predictions table
            st.subheader("📋 Recent Direction Predictions")
            
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'Timestamp': recent_data.index[-20:],  # Last 20 predictions
                'Price': recent_data['Close'].tail(20).values,
                'Direction': ['Bullish' if p == 1 else 'Bearish' for p in predictions[-20:]],
                'Confidence': [f"{np.max(prob):.1f}%" for prob in probabilities[-20:]] if probabilities is not None else ['N/A'] * 20
            })
            
            st.dataframe(predictions_df, use_container_width=True)