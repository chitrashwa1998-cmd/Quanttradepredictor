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
            
            # Enhanced statistics
            bullish_count = np.sum(predictions == 1)
            bearish_count = np.sum(predictions == 0)
            bullish_pct = (bullish_count / len(predictions)) * 100
            
            if probabilities is not None:
                avg_confidence = np.mean(np.max(probabilities, axis=1)) * 100
                high_confidence = np.sum(np.max(probabilities, axis=1) > 0.7)
                high_conf_pct = (high_confidence / len(predictions)) * 100
            else:
                avg_confidence = 0
                high_confidence = 0
                high_conf_pct = 0
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", f"{len(predictions)}")
            with col2:
                st.metric("Bullish Signals", f"{bullish_count} ({bullish_pct:.1f}%)")
            with col3:
                st.metric("Bearish Signals", f"{bearish_count} ({100-bullish_pct:.1f}%)")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Additional statistics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("High Confidence", f"{high_confidence} ({high_conf_pct:.1f}%)")
            with col6:
                recent_bullish = np.sum(predictions[-20:] == 1) if len(predictions) >= 20 else np.sum(predictions == 1)
                recent_pct = (recent_bullish / min(20, len(predictions))) * 100
                st.metric("Recent Bullish (Last 20)", f"{recent_bullish} ({recent_pct:.1f}%)")
            with col7:
                if probabilities is not None:
                    recent_conf = np.mean(np.max(probabilities[-20:], axis=1)) * 100 if len(probabilities) >= 20 else avg_confidence
                    st.metric("Recent Confidence", f"{recent_conf:.1f}%")
                else:
                    st.metric("Recent Confidence", "N/A")
            with col8:
                data_len = min(len(st.session_state.data), len(predictions))
                recent_data = st.session_state.data.tail(data_len)
                price_change = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]) * 100
                st.metric("Price Change", f"{price_change:.2f}%")
            
            # Create direction prediction chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price Chart', 'Direction Predictions'),
                row_heights=[0.7, 0.3]
            )
            
            # Add price chart with OHLC
            data_len = min(len(st.session_state.data), len(predictions))
            recent_data = st.session_state.data.tail(data_len)
            
            # Add candlestick chart for price
            fig.add_trace(go.Candlestick(
                x=recent_data.index,
                open=recent_data['Open'],
                high=recent_data['High'],
                low=recent_data['Low'],
                close=recent_data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ), row=1, col=1)
            
            # Add direction predictions with confidence coloring
            pred_data = predictions[-data_len:]
            prob_data = probabilities[-data_len:] if probabilities is not None else None
            
            # Create confidence-based colors
            if prob_data is not None:
                confidences = np.max(prob_data, axis=1)
                colors = ['rgba(0, 255, 0, ' + str(conf) + ')' for conf in confidences]
                sizes = [6 + 6 * conf for conf in confidences]  # Size based on confidence
            else:
                confidences = np.ones(len(pred_data)) * 0.5  # Default confidence for fallback
                colors = ['green'] * len(pred_data)
                sizes = [8] * len(pred_data)
            
            # Bullish signals
            bullish_mask = pred_data == 1
            if np.any(bullish_mask):
                bullish_colors = [colors[i] for i in range(len(colors)) if bullish_mask[i]]
                bullish_sizes = [sizes[i] for i in range(len(sizes)) if bullish_mask[i]]
                
                fig.add_trace(go.Scatter(
                    x=recent_data.index[bullish_mask],
                    y=[1] * np.sum(bullish_mask),
                    mode='markers',
                    name='Bullish',
                    marker=dict(color=bullish_colors if prob_data is not None else 'green', 
                               size=bullish_sizes if prob_data is not None else 10, 
                               symbol='triangle-up'),
                    text=[f'Confidence: {confidences[i]:.1%}' for i in range(len(confidences)) if bullish_mask[i]] if prob_data is not None else None,
                    hovertemplate='Bullish Signal<br>%{text}<extra></extra>' if prob_data is not None else 'Bullish Signal<extra></extra>'
                ), row=2, col=1)
            
            # Bearish signals
            bearish_mask = pred_data == 0
            if np.any(bearish_mask):
                bearish_colors = [colors[i] for i in range(len(colors)) if bearish_mask[i]]
                bearish_sizes = [sizes[i] for i in range(len(sizes)) if bearish_mask[i]]
                
                fig.add_trace(go.Scatter(
                    x=recent_data.index[bearish_mask],
                    y=[0] * np.sum(bearish_mask),
                    mode='markers',
                    name='Bearish',
                    marker=dict(color=bearish_colors if prob_data is not None else 'red', 
                               size=bearish_sizes if prob_data is not None else 10, 
                               symbol='triangle-down'),
                    text=[f'Confidence: {confidences[i]:.1%}' for i in range(len(confidences)) if bearish_mask[i]] if prob_data is not None else None,
                    hovertemplate='Bearish Signal<br>%{text}<extra></extra>' if prob_data is not None else 'Bearish Signal<extra></extra>'
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
            
            # Show detailed analysis section
            st.subheader("📊 Detailed Direction Analysis")
            
            # Create tabbed analysis
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📋 Recent Predictions", "📈 Performance Metrics", "🔍 Signal Quality"])
            
            with analysis_tab1:
                st.markdown("**Last 30 Direction Predictions**")
                
                # Create comprehensive predictions dataframe
                num_recent = min(30, len(predictions))
                recent_predictions = predictions[-num_recent:]
                recent_probs = probabilities[-num_recent:] if probabilities is not None else None
                recent_prices = recent_data.tail(num_recent)
                
                # Calculate price changes for validation
                price_changes = recent_prices['Close'].pct_change().shift(-1) * 100  # Next period change
                actual_direction = (price_changes > 0).astype(int)
                
                predictions_df = pd.DataFrame({
                    'Timestamp': recent_prices.index,
                    'Open': recent_prices['Open'].round(2),
                    'High': recent_prices['High'].round(2),
                    'Low': recent_prices['Low'].round(2),
                    'Close': recent_prices['Close'].round(2),
                    'Predicted Direction': ['🟢 Bullish' if p == 1 else '🔴 Bearish' for p in recent_predictions],
                    'Confidence': [f"{np.max(prob):.1f}%" for prob in recent_probs] if recent_probs is not None else ['N/A'] * num_recent,
                    'Next Change %': [f"{change:.2f}%" if not pd.isna(change) else 'N/A' for change in price_changes],
                    'Correct': ['✅' if pred == actual and not pd.isna(actual) else '❌' if not pd.isna(actual) else '⏳' 
                               for pred, actual in zip(recent_predictions, actual_direction)]
                })
                
                st.dataframe(predictions_df, use_container_width=True)
            
            with analysis_tab2:
                st.markdown("**Model Performance Analysis**")
                
                # Calculate accuracy where we have actual data
                valid_comparisons = ~pd.isna(actual_direction[:-1])  # Exclude last as it has no future data
                if valid_comparisons.sum() > 0:
                    accuracy = (recent_predictions[:-1][valid_comparisons] == actual_direction[:-1][valid_comparisons]).mean()
                    st.metric("Prediction Accuracy", f"{accuracy:.1%}")
                
                # Confidence distribution
                if probabilities is not None:
                    conf_dist = np.max(probabilities, axis=1)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Confidence histogram
                        fig_hist = go.Figure(data=[go.Histogram(x=conf_dist, nbinsx=20, name='Confidence Distribution')])
                        fig_hist.update_layout(title="Confidence Distribution", xaxis_title="Confidence Level", yaxis_title="Count")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Confidence vs accuracy
                        high_conf_mask = conf_dist > 0.7
                        med_conf_mask = (conf_dist >= 0.5) & (conf_dist <= 0.7)
                        low_conf_mask = conf_dist < 0.5
                        
                        st.metric("High Confidence (>70%)", f"{high_conf_mask.sum()} signals")
                        st.metric("Medium Confidence (50-70%)", f"{med_conf_mask.sum()} signals")
                        st.metric("Low Confidence (<50%)", f"{low_conf_mask.sum()} signals")
            
            with analysis_tab3:
                st.markdown("**Signal Quality Assessment**")
                
                # Signal strength analysis
                if probabilities is not None:
                    conf_scores = np.max(probabilities, axis=1)
                    
                    # Quality categories
                    very_high = (conf_scores > 0.8).sum()
                    high = ((conf_scores > 0.7) & (conf_scores <= 0.8)).sum()
                    medium = ((conf_scores > 0.6) & (conf_scores <= 0.7)).sum()
                    low = (conf_scores <= 0.6).sum()
                    
                    quality_df = pd.DataFrame({
                        'Quality Level': ['Very High (>80%)', 'High (70-80%)', 'Medium (60-70%)', 'Low (≤60%)'],
                        'Signal Count': [very_high, high, medium, low],
                        'Percentage': [f"{(very_high/len(conf_scores)*100):.1f}%",
                                     f"{(high/len(conf_scores)*100):.1f}%",
                                     f"{(medium/len(conf_scores)*100):.1f}%",
                                     f"{(low/len(conf_scores)*100):.1f}%"]
                    })
                    
                    st.dataframe(quality_df, use_container_width=True)
                    
                    # Quality over time
                    fig_quality = go.Figure()
                    fig_quality.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=conf_scores[-len(recent_data):],
                        mode='lines+markers',
                        name='Confidence Over Time',
                        line=dict(color='blue')
                    ))
                    fig_quality.update_layout(title="Signal Confidence Over Time", 
                                            xaxis_title="Time", 
                                            yaxis_title="Confidence Level",
                                            yaxis=dict(range=[0, 1]))
                    st.plotly_chart(fig_quality, use_container_width=True)