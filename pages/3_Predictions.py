import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

# Load custom CSS if it exists
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass  # Continue without custom CSS

# Initialize session state variables if they don't exist
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

# Check if data and models are available
if st.session_state.data is None:
    st.warning("⚠️ No data loaded. Please go to the **Data Upload** page first.")
    st.stop()

if not st.session_state.models:
    st.warning("⚠️ No trained models found. Please go to the **Model Training** page first.")
    st.stop()

if st.session_state.model_trainer is None:
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
    st.stop()

# Header
st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">🎯 PREDICTION ENGINE</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Real-time Market Analysis & Forecasting
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Model and time range selection in columns for better layout
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_model = st.selectbox(
        "🤖 Select AI Model",
        available_models,
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Choose a trained model for predictions"
    )

with col2:
    date_range = st.selectbox(
        "📅 Time Period",
        ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All data"],
        index=1
    )

with col3:
    st.metric("Model Type", models[selected_model]['task_type'].title())

# Ensure Date is not both index and column
if 'Date' in df.columns:
    # Remove Date column if it exists as both index and column
    df = df.drop(columns=['Date'], errors='ignore')

if st.session_state.features is not None:
    if hasattr(st.session_state.features, 'columns') and 'Date' in st.session_state.features.columns:
        st.session_state.features = st.session_state.features.drop(columns=['Date'], errors='ignore')

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

df_filtered = df[df.index >= start_date].copy()

# Check if features exist, if not prepare them
if st.session_state.features is None:
    st.warning("Features not found. Preparing features from current data...")
    features_filtered = model_trainer.prepare_features(df_filtered)
    st.session_state.features = model_trainer.prepare_features(df)
else:
    features_filtered = st.session_state.features[st.session_state.features.index >= start_date].copy()

# Generate predictions
try:
    predictions, probabilities = model_trainer.predict(selected_model, features_filtered)

    # Create tabs for different views
    if selected_model == 'direction':
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "🔄 Reversals", "📊 Statistics", "📋 Data Table"])

        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'Price': df_filtered['Close'],
            'Prediction': predictions,
            'Direction': ['Up' if p == 1 else 'Down' for p in predictions]
        }, index=features_filtered.index)

        if probabilities is not None:
            pred_df['Confidence'] = np.max(probabilities, axis=1)

        # Calculate technical indicators for reversal detection
        pred_df['Direction_Change'] = pred_df['Prediction'].diff().fillna(0) != 0
        pred_df['SMA_10'] = pred_df['Price'].rolling(10).mean()
        pred_df['SMA_20'] = pred_df['Price'].rolling(20).mean()

        # Reversal detection
        bullish_reversal = (pred_df['Prediction'].shift(1) == 0) & (pred_df['Prediction'] == 1)
        bearish_reversal = (pred_df['Prediction'].shift(1) == 1) & (pred_df['Prediction'] == 0)
        price_support = pred_df['Price'] > pred_df['SMA_10']
        price_resistance = pred_df['Price'] < pred_df['SMA_10']

        pred_df['Confirmed_Bull_Reversal'] = bullish_reversal & price_support
        pred_df['Confirmed_Bear_Reversal'] = bearish_reversal & price_resistance

        with tab1:
            st.subheader("📈 Price Movement with Direction Predictions")

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index, 
                y=pred_df['Price'],
                name='Price',
                line=dict(color='#2E86C1', width=2),
                hovertemplate='<b>Price:</b> $%{y:.2f}<br><b>Date:</b> %{x}<extra></extra>'
            ))

            # Moving average
            fig.add_trace(go.Scatter(
                x=pred_df.index, 
                y=pred_df['SMA_10'],
                name='SMA 10',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.7
            ))

            # Prediction markers
            up_predictions = pred_df[pred_df['Prediction'] == 1]
            down_predictions = pred_df[pred_df['Prediction'] == 0]

            if len(up_predictions) > 0:
                fig.add_trace(go.Scatter(
                    x=up_predictions.index, 
                    y=up_predictions['Price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='#27AE60', size=8),
                    name='Predicted Up',
                    hovertemplate='<b>Prediction:</b> Up<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ))

            if len(down_predictions) > 0:
                fig.add_trace(go.Scatter(
                    x=down_predictions.index, 
                    y=down_predictions['Price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='#E74C3C', size=8),
                    name='Predicted Down',
                    hovertemplate='<b>Prediction:</b> Down<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ))

            fig.update_layout(
                height=500,
                title="Direction Predictions Overview",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                up_pct = (predictions == 1).mean() * 100
                st.metric("Predicted Up", f"{up_pct:.1f}%", delta=None)
            with col2:
                down_pct = (predictions == 0).mean() * 100
                st.metric("Predicted Down", f"{down_pct:.1f}%", delta=None)
            with col3:
                if 'Confidence' in pred_df.columns:
                    avg_conf = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                else:
                    st.metric("Data Points", len(pred_df))
            with col4:
                current_pred = "Up" if predictions[-1] == 1 else "Down"
                st.metric("Latest Signal", current_pred)

        with tab2:
            st.subheader("🔄 Reversal Detection Analysis")

            bull_reversals = pred_df[pred_df['Confirmed_Bull_Reversal']]
            bear_reversals = pred_df[pred_df['Confirmed_Bear_Reversal']]

            if len(bull_reversals) > 0 or len(bear_reversals) > 0:
                fig = go.Figure()

                # Price line
                fig.add_trace(go.Scatter(
                    x=pred_df.index, 
                    y=pred_df['Price'],
                    name='Price',
                    line=dict(color='#2E86C1', width=2)
                ))

                # Reversal points
                if len(bull_reversals) > 0:
                    fig.add_trace(go.Scatter(
                        x=bull_reversals.index, 
                        y=bull_reversals['Price'],
                        mode='markers',
                        marker=dict(symbol='star', color='#27AE60', size=15, line=dict(width=2, color='darkgreen')),
                        name='Bullish Reversal',
                        hovertemplate='<b>Bullish Reversal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ))

                if len(bear_reversals) > 0:
                    fig.add_trace(go.Scatter(
                        x=bear_reversals.index, 
                        y=bear_reversals['Price'],
                        mode='markers',
                        marker=dict(symbol='star', color='#E74C3C', size=15, line=dict(width=2, color='darkred')),
                        name='Bearish Reversal',
                        hovertemplate='<b>Bearish Reversal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ))

                fig.update_layout(
                    height=500,
                    title="Confirmed Reversal Points",
                    xaxis_title="Date",
                    yaxis_title="Price ($)"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Reversal summary
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Bullish Reversals", len(bull_reversals))
                with col2:
                    st.metric("Bearish Reversals", len(bear_reversals))

                # Recent reversals table
                if len(bull_reversals) > 0 or len(bear_reversals) > 0:
                    st.subheader("Recent Reversal Signals")

                    recent_reversals = []

                    for idx, row in bull_reversals.tail(5).iterrows():
                        try:
                            date_str = pd.to_datetime(idx).strftime('%Y-%m-%d %H:%M')
                        except:
                            date_str = str(idx)
                        recent_reversals.append({
                            'Date': date_str,
                            'Type': 'Bullish 🟢',
                            'Price': f"${row['Price']:.2f}",
                            'Confidence': f"{row.get('Confidence', 0):.3f}" if 'Confidence' in pred_df.columns else 'N/A'
                        })

                    for idx, row in bear_reversals.tail(5).iterrows():
                        try:
                            date_str = pd.to_datetime(idx).strftime('%Y-%m-%d %H:%M')
                        except:
                            date_str = str(idx)
                        recent_reversals.append({
                            'Date': date_str,
                            'Type': 'Bearish 🔴',
                            'Price': f"${row['Price']:.2f}",
                            'Confidence': f"{row.get('Confidence', 0):.3f}" if 'Confidence' in pred_df.columns else 'N/A'
                        })

                    if recent_reversals:
                        reversal_df = pd.DataFrame(recent_reversals)
                        reversal_df = reversal_df.sort_values('Date', ascending=False)
                        st.dataframe(reversal_df, use_container_width=True, hide_index=True)
            else:
                st.info("No confirmed reversal points detected in this time period.")

        with tab3:
            st.subheader("📊 Prediction Statistics")

            # Model performance metrics
            model_info = models[selected_model]

            col1, col2, col3 = st.columns(3)
            with col1:
                if 'accuracy' in model_info['metrics']:
                    st.metric("Model Accuracy", f"{model_info['metrics']['accuracy']:.3f}")
                if 'precision' in model_info['metrics']:
                    st.metric("Precision", f"{model_info['metrics']['precision']:.3f}")

            with col2:
                if 'recall' in model_info['metrics']:
                    st.metric("Recall", f"{model_info['metrics']['recall']:.3f}")
                if 'f1' in model_info['metrics']:
                    st.metric("F1 Score", f"{model_info['metrics']['f1']:.3f}")

            with col3:
                st.metric("Total Predictions", len(predictions))
                st.metric("Time Period", f"{(df_filtered.index.max() - df_filtered.index.min()).days} days")

            # Prediction distribution chart
            signal_counts = pd.Series(predictions).value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=['Down', 'Up'],
                values=[signal_counts.get(0, 0), signal_counts.get(1, 0)],
                hole=0.3,
                marker_colors=['#E74C3C', '#27AE60']
            )])

            fig.update_layout(
                title="Direction Prediction Distribution",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Confidence analysis if available
            if 'Confidence' in pred_df.columns:
                st.subheader("Confidence Analysis")

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=pred_df['Confidence'],
                    nbinsx=20,
                    name='Confidence Distribution',
                    marker_color='#3498DB'
                ))

                fig.update_layout(
                    title="Prediction Confidence Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Frequency",
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("📋 Detailed Predictions Data")

            # Format the dataframe for display
            display_df = pred_df.copy()
            display_df = display_df.reset_index()
            
            # Get the index column name (it could be 'Date' or the actual index name)
            index_col_name = display_df.columns[0]
            display_df['Date'] = pd.to_datetime(display_df[index_col_name]).dt.strftime('%Y-%m-%d %H:%M')
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")

            if 'Confidence' in display_df.columns:
                display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.3f}")

            # Remove the original index column and reorder
            display_df = display_df.drop(columns=[index_col_name])
            cols = ['Date'] + [col for col in display_df.columns if col != 'Date']
            display_df = display_df[cols]

            # Show recent predictions (last 50)
            st.dataframe(
                display_df.tail(50).sort_values('Date', ascending=False),
                use_container_width=True,
                hide_index=True
            )

            # Download button
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv,
                file_name=f"direction_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

    elif selected_model == 'trading_signal':
        tab1, tab2, tab3 = st.tabs(["📊 Trading Signals", "📈 Signal Analysis", "📋 Signal History"])

        signal_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
        signal_colors = {0: '#E74C3C', 1: '#F39C12', 2: '#27AE60'}

        pred_df = pd.DataFrame({
            'Price': df_filtered['Close'],
            'Signal': predictions,
            'Signal_Name': [signal_map[s] for s in predictions]
        }, index=features_filtered.index)

        with tab1:
            st.subheader("📊 Trading Signal Overview")

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index, 
                y=pred_df['Price'],
                name='Price',
                line=dict(color='#2E86C1', width=2)
            ))

            # Signal markers
            for signal_value, signal_name in signal_map.items():
                signal_data = pred_df[pred_df['Signal'] == signal_value]
                if len(signal_data) > 0:
                    marker_symbol = 'triangle-up' if signal_value == 2 else ('triangle-down' if signal_value == 0 else 'circle')

                    fig.add_trace(go.Scatter(
                        x=signal_data.index,
                        y=signal_data['Price'],
                        mode='markers',
                        marker=dict(
                            symbol=marker_symbol,
                            color=signal_colors[signal_value],
                            size=10
                        ),
                        name=f'{signal_name} Signal',
                        hovertemplate=f'<b>{signal_name} Signal</b><br>Price: $%{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
                    ))

            fig.update_layout(
                height=500,
                title="Trading Signals on Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Current signal
            current_signal = signal_map[predictions[-1]]
            signal_color = signal_colors[predictions[-1]]

            st.markdown(f"""
            <div style="background: rgba{signal_color[3:-1]}, 0.1); border: 2px solid {signal_color}; 
                 border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
                <h3 style="color: {signal_color}; margin: 0;">Current Signal: {current_signal}</h3>
                <p style="margin: 0.5rem 0 0 0;">Latest trading recommendation</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.subheader("📈 Signal Distribution Analysis")

            signal_counts = pd.Series(predictions).value_counts()

            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Sell', 'Hold', 'Buy'],
                values=[signal_counts.get(0, 0), signal_counts.get(1, 0), signal_counts.get(2, 0)],
                hole=0.3,
                marker_colors=['#E74C3C', '#F39C12', '#27AE60']
            )])

            fig.update_layout(
                title="Trading Signal Distribution",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Signal metrics
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

        with tab3:
            st.subheader("📋 Signal History")

            # Recent signals
            display_df = pred_df.copy()
            display_df = display_df.reset_index()
            
            # Get the index column name
            index_col_name = display_df.columns[0]
            display_df['Date'] = pd.to_datetime(display_df[index_col_name]).dt.strftime('%Y-%m-%d %H:%M')
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")

            # Remove the original index column and select required columns
            display_df = display_df.drop(columns=[index_col_name])
            
            st.dataframe(
                display_df[['Date', 'Price', 'Signal_Name']].tail(50).sort_values('Date', ascending=False),
                use_container_width=True,
                hide_index=True
            )

    else:
        # Generic handler for other models
        st.subheader(f"{selected_model.replace('_', ' ').title()} Predictions")

        pred_df = pd.DataFrame({
            'Price': df_filtered['Close'],
            'Prediction': predictions
        }, index=features_filtered.index)

        # Simple chart
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=('Price', 'Predictions'),
                          vertical_spacing=0.1)

        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Price'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Prediction'], name='Prediction'), row=2, col=1)

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        display_df = pred_df.copy()
        display_df = display_df.reset_index()
        
        # Get the index column name
        index_col_name = display_df.columns[0]
        display_df['Date'] = pd.to_datetime(display_df[index_col_name]).dt.strftime('%Y-%m-%d %H:%M')
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
        
        # Remove the original index column and reorder
        display_df = display_df.drop(columns=[index_col_name])
        cols = ['Date'] + [col for col in display_df.columns if col != 'Date']
        display_df = display_df[cols]

        st.subheader("Recent Predictions")
        st.dataframe(display_df.tail(20), use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error generating predictions: {str(e)}")
    st.info("Please try refreshing the page or check your model training.")

# Model comparison section
st.markdown("---")
st.header("🔍 Model Comparison")

if len(available_models) > 1:
    compare_models = st.multiselect(
        "Select models to compare performance",
        available_models,
        default=available_models[:3] if len(available_models) >= 3 else available_models,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    if len(compare_models) >= 2:
        comparison_data = []

        for model_name in compare_models:
            model_info = models[model_name]
            metrics = model_info['metrics']

            row_data = {
                'Model': model_name.replace('_', ' ').title(),
                'Type': model_info['task_type'].title()
            }

            if model_info['task_type'] == 'classification':
                row_data.update({
                    'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                    'Precision': f"{metrics.get('precision', 0):.3f}",
                    'Recall': f"{metrics.get('recall', 0):.3f}",
                    'F1 Score': f"{metrics.get('f1', 0):.3f}"
                })
            else:
                row_data.update({
                    'RMSE': f"{metrics.get('rmse', 0):.4f}",
                    'MAE': f"{metrics.get('mae', 0):.4f}",
                    'R²': f"{metrics.get('r2', 0):.3f}"
                })

            comparison_data.append(row_data)

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Next steps
st.markdown("---")
st.info("📋 **Next Steps:** Analyze your predictions and proceed to the **Backtesting** page to evaluate strategy performance.")