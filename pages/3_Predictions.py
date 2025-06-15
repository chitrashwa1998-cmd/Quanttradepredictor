import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

st.title("🎯 Model Predictions and Analysis")
st.markdown("---")

# Check if data and models are available
if st.session_state.data is None:
    st.warning("⚠️ No data loaded. Please go to the **Data Upload** page first.")
    st.stop()

if not st.session_state.models:
    st.warning("⚠️ No trained models found. Please go to the **Model Training** page first.")
    st.stop()

# Initialize model trainer if not available
if 'model_trainer' not in st.session_state or st.session_state.model_trainer is None:
    st.warning("⚠️ Model trainer not initialized. Please go to the **Model Training** page first.")
    st.stop()

df = st.session_state.data
models = st.session_state.models
model_trainer = st.session_state.model_trainer

# Available models
available_models = [name for name, info in models.items() if info is not None]

if not available_models:
    st.error("❌ No successfully trained models found.")
    st.info("Please go to the **Model Training** page to train models first.")
    
    # Show what's in session state for debugging
    with st.expander("🔍 Debug Information"):
        st.write("Models in session state:", list(models.keys()) if models else "None")
        st.write("Model values:", {k: "Loaded" if v is not None else "None" for k, v in models.items()} if models else "Empty")
    st.stop()

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">🎯 PREDICTION ENGINE</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Real-time Market Analysis & Forecasting
    </p>
</div>
""", unsafe_allow_html=True)

# Model selection with better error handling
try:
    selected_model = st.selectbox(
        "Select Model for Analysis",
        available_models,
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Select a trained model to generate predictions"
    )
except Exception as e:
    st.error(f"Error loading model options: {str(e)}")
    st.info("Try refreshing the page or retraining your models.")
    available_models = []
    selected_model = None

# Time range selection for predictions
st.subheader("Time Range Selection")

col1, col2 = st.columns(2)

with col1:
    date_range = st.selectbox(
        "Select Date Range",
        ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All data"]
    )

# Filter data based on selection
if date_range == "Last 30 days":
    start_date = df.index.max() - timedelta(days=30)
elif date_range == "Last 90 days":
    start_date = df.index.max() - timedelta(days=90)
elif date_range == "Last 6 months":
    start_date = df.index.max() - timedelta(days=180)
elif date_range == "Last year":
    start_date = df.index.max() - timedelta(days=365)
else:
    start_date = df.index.min()

df_filtered = df[df.index >= start_date]

# Get predictions for selected model
model_info = models[selected_model]

# Display model information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Type", model_info['task_type'].title())

with col2:
    if model_info['task_type'] == 'classification':
        st.metric("Accuracy", f"{model_info['metrics']['accuracy']:.3f}")
    else:
        st.metric("RMSE", f"{model_info['metrics']['rmse']:.4f}")

with col3:
    st.metric("Data Points", len(df_filtered))

# Generate predictions for the filtered data
if st.session_state.features is not None:
    
    st.header("Prediction Results")
    
    # Make predictions on the filtered data
    features_filtered = st.session_state.features[st.session_state.features.index >= start_date]
    
    try:
        predictions, probabilities = model_trainer.predict(selected_model, features_filtered)
        
        # Create prediction visualization based on model type
        if selected_model == 'direction':
            st.subheader("Direction Predictions")
            
            # Direction predictions (0=Down, 1=Up)
            pred_df = pd.DataFrame({
                'Date': features_filtered.index,
                'Actual_Price': df_filtered['Close'],
                'Prediction': predictions,
                'Direction': ['Up' if p == 1 else 'Down' for p in predictions]
            })
            
            if probabilities is not None:
                pred_df['Confidence'] = np.max(probabilities, axis=1)
            
            # Plot price with direction predictions
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Price Chart with Predictions', 'Prediction Confidence'),
                              vertical_spacing=0.1,
                              row_heights=[0.7, 0.3])
            
            # Price chart
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Actual_Price'], 
                          name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Add prediction markers
            up_predictions = pred_df[pred_df['Prediction'] == 1]
            down_predictions = pred_df[pred_df['Prediction'] == 0]
            
            fig.add_trace(
                go.Scatter(x=up_predictions['Date'], y=up_predictions['Actual_Price'],
                          mode='markers', marker=dict(symbol='triangle-up', color='green', size=8),
                          name='Predicted Up'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=down_predictions['Date'], y=down_predictions['Actual_Price'],
                          mode='markers', marker=dict(symbol='triangle-down', color='red', size=8),
                          name='Predicted Down'),
                row=1, col=1
            )
            
            # Confidence chart
            if 'Confidence' in pred_df.columns:
                fig.add_trace(
                    go.Scatter(x=pred_df['Date'], y=pred_df['Confidence'],
                              name='Confidence', line=dict(color='orange')),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, title="Direction Prediction Analysis")
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Confidence", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                up_pct = (predictions == 1).mean() * 100
                st.metric("Predicted Up %", f"{up_pct:.1f}%")
            
            with col2:
                down_pct = (predictions == 0).mean() * 100
                st.metric("Predicted Down %", f"{down_pct:.1f}%")
            
            with col3:
                if 'Confidence' in pred_df.columns:
                    avg_confidence = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        elif selected_model == 'magnitude':
            st.subheader("Magnitude Predictions")
            
            pred_df = pd.DataFrame({
                'Date': features_filtered.index,
                'Actual_Price': df_filtered['Close'],
                'Predicted_Magnitude': predictions
            })
            
            # Calculate actual magnitude for comparison
            actual_returns = df_filtered['Close'].pct_change().shift(-1)
            actual_magnitude = np.abs(actual_returns) * 100
            pred_df['Actual_Magnitude'] = actual_magnitude
            
            # Plot predicted vs actual magnitude
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pred_df['Date'], 
                y=pred_df['Predicted_Magnitude'],
                name='Predicted Magnitude',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_df['Date'], 
                y=pred_df['Actual_Magnitude'],
                name='Actual Magnitude',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title="Predicted vs Actual Magnitude of Price Moves",
                xaxis_title="Date",
                yaxis_title="Magnitude (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_pred = predictions.mean()
                st.metric("Avg Predicted Magnitude", f"{avg_pred:.2f}%")
            
            with col2:
                avg_actual = actual_magnitude.mean()
                st.metric("Avg Actual Magnitude", f"{avg_actual:.2f}%")
            
            with col3:
                correlation = np.corrcoef(predictions[~np.isnan(actual_magnitude)], 
                                       actual_magnitude.dropna())[0,1]
                st.metric("Correlation", f"{correlation:.3f}")
        
        elif selected_model == 'trading_signal':
            st.subheader("Trading Signals")
            
            # Signal mapping: 0=Sell, 1=Hold, 2=Buy
            signal_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
            signal_colors = {0: 'red', 1: 'gray', 2: 'green'}
            
            pred_df = pd.DataFrame({
                'Date': features_filtered.index,
                'Price': df_filtered['Close'],
                'Signal': predictions,
                'Signal_Name': [signal_map[s] for s in predictions]
            })
            
            # Plot price with trading signals
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pred_df['Date'], 
                y=pred_df['Price'],
                name='Price',
                line=dict(color='blue')
            ))
            
            # Add signal markers
            for signal_value, signal_name in signal_map.items():
                signal_data = pred_df[pred_df['Signal'] == signal_value]
                if len(signal_data) > 0:
                    marker_symbol = 'triangle-up' if signal_value == 2 else ('triangle-down' if signal_value == 0 else 'circle')
                    
                    fig.add_trace(go.Scatter(
                        x=signal_data['Date'],
                        y=signal_data['Price'],
                        mode='markers',
                        marker=dict(
                            symbol=marker_symbol,
                            color=signal_colors[signal_value],
                            size=10
                        ),
                        name=f'{signal_name} Signal'
                    ))
            
            fig.update_layout(
                title="Trading Signals",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal distribution
            signal_counts = pd.Series(predictions).value_counts()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_pct = (signal_counts.get(2, 0) / len(predictions)) * 100
                st.metric("Buy Signals", f"{buy_pct:.1f}%")
            
            with col2:
                hold_pct = (signal_counts.get(1, 0) / len(predictions)) * 100
                st.metric("Hold Signals", f"{hold_pct:.1f}%")
            
            with col3:
                sell_pct = (signal_counts.get(0, 0) / len(predictions)) * 100
                st.metric("Sell Signals", f"{sell_pct:.1f}%")
        
        elif selected_model == 'trend_sideways':
            st.subheader("Trend vs Sideways Classification")
            
            # Trend mapping: 0=Sideways, 1=Trending
            trend_map = {0: 'Sideways', 1: 'Trending'}
            trend_colors = {0: 'orange', 1: 'blue'}
            
            pred_df = pd.DataFrame({
                'Date': features_filtered.index,
                'Price': df_filtered['Close'],
                'Prediction': predictions,
                'Market_State': [trend_map[p] for p in predictions]
            })
            
            # Add confidence if available
            if probabilities is not None:
                pred_df['Confidence'] = np.max(probabilities, axis=1)
            
            # Create visualization
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=('Price with Market State', 'Market State Timeline'),
                              vertical_spacing=0.1,
                              row_heights=[0.7, 0.3])
            
            # Price chart with color-coded background for market state
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Price'], 
                          name='Price', line=dict(color='black', width=2)),
                row=1, col=1
            )
            
            # Add market state periods as background colors
            sideways_periods = pred_df[pred_df['Prediction'] == 0]
            trending_periods = pred_df[pred_df['Prediction'] == 1]
            
            # Add scatter points for market states
            if len(sideways_periods) > 0:
                fig.add_trace(
                    go.Scatter(x=sideways_periods['Date'], y=sideways_periods['Price'],
                              mode='markers', marker=dict(color='orange', size=6, symbol='square'),
                              name='Sideways Market'),
                    row=1, col=1
                )
            
            if len(trending_periods) > 0:
                fig.add_trace(
                    go.Scatter(x=trending_periods['Date'], y=trending_periods['Price'],
                              mode='markers', marker=dict(color='blue', size=6, symbol='diamond'),
                              name='Trending Market'),
                    row=1, col=1
                )
            
            # Market state timeline
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'],
                          mode='lines+markers', 
                          line=dict(color='purple', width=3),
                          marker=dict(size=4),
                          name='Market State',
                          hovertemplate='<b>Date:</b> %{x}<br><b>State:</b> %{customdata}<extra></extra>',
                          customdata=pred_df['Market_State']),
                row=2, col=1
            )
            
            fig.update_layout(height=700, title="Trend vs Sideways Market Analysis")
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Market State (0=Sideways, 1=Trending)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market state statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sideways_pct = (predictions == 0).mean() * 100
                st.metric("Sideways Market %", f"{sideways_pct:.1f}%")
            
            with col2:
                trending_pct = (predictions == 1).mean() * 100
                st.metric("Trending Market %", f"{trending_pct:.1f}%")
            
            with col3:
                if 'Confidence' in pred_df.columns:
                    avg_confidence = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            # Show periods analysis
            st.subheader("Market State Periods Analysis")
            
            # Calculate consecutive periods
            pred_df['State_Change'] = pred_df['Prediction'].diff().fillna(0) != 0
            pred_df['Period_ID'] = pred_df['State_Change'].cumsum()
            
            # Group by periods
            periods_summary = pred_df.groupby('Period_ID').agg({
                'Date': ['first', 'last'],
                'Prediction': 'first',
                'Market_State': 'first'
            }).round(2)
            
            periods_summary.columns = ['Start_Date', 'End_Date', 'State_Code', 'Market_State']
            periods_summary['Duration_Days'] = (periods_summary['End_Date'] - periods_summary['Start_Date']).dt.days + 1
            
            # Show recent periods
            st.write("**Recent Market State Periods:**")
            recent_periods = periods_summary.tail(10).copy()
            recent_periods['Start_Date'] = recent_periods['Start_Date'].dt.strftime('%Y-%m-%d')
            recent_periods['End_Date'] = recent_periods['End_Date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(recent_periods[['Start_Date', 'End_Date', 'Market_State', 'Duration_Days']], 
                        use_container_width=True)
        
        elif selected_model == 'volatility':
            st.subheader("Volatility Forecasting")
            
            pred_df = pd.DataFrame({
                'Date': features_filtered.index,
                'Price': df_filtered['Close'],
                'Predicted_Volatility': predictions
            })
            
            # Calculate actual volatility for comparison
            actual_vol = df_filtered['Close'].rolling(10).std()
            pred_df['Actual_Volatility'] = actual_vol
            
            # Plot volatility predictions
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=('Price', 'Volatility Forecast'),
                              vertical_spacing=0.1)
            
            # Price chart
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Price'], name='Price'),
                row=1, col=1
            )
            
            # Volatility chart
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Predicted_Volatility'], 
                          name='Predicted Volatility', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Actual_Volatility'], 
                          name='Actual Volatility', line=dict(color='blue')),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title="Volatility Forecasting")
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Generic visualization for other models
            st.subheader(f"{selected_model.replace('_', ' ').title()} Predictions")
            
            pred_df = pd.DataFrame({
                'Date': features_filtered.index,
                'Price': df_filtered['Close'],
                'Prediction': predictions
            })
            
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=('Price', 'Predictions'),
                              vertical_spacing=0.1)
            
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Price'], name='Price'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'], name='Prediction'),
                row=2, col=1
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction data table
        st.subheader("Recent Predictions")
        
        # Show last 20 predictions
        if 'pred_df' in locals():
            recent_predictions = pred_df.tail(20).copy()
            recent_predictions['Date'] = recent_predictions['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(recent_predictions, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")

# Model comparison section
st.header("Model Comparison")

if len(available_models) > 1:
    compare_models = st.multiselect(
        "Select models to compare",
        available_models,
        default=available_models[:2] if len(available_models) >= 2 else available_models,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if len(compare_models) >= 2:
        comparison_data = []
        
        for model_name in compare_models:
            model_info = models[model_name]
            metrics = model_info['metrics']
            
            if model_info['task_type'] == 'classification':
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Type': 'Classification',
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Primary Metric': f"{metrics['accuracy']:.3f}"
                })
            else:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Type': 'Regression',
                    'RMSE': f"{metrics['rmse']:.4f}",
                    'Primary Metric': f"{metrics['rmse']:.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

# Export predictions
st.header("Export Predictions")

if st.button("Download Predictions as CSV"):
    if 'pred_df' in locals():
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_model}_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No prediction data available to download")

# Next steps
st.markdown("---")
st.info("📋 **Next Steps:** Analyze your predictions and proceed to the **Backtesting** page to evaluate strategy performance.")
