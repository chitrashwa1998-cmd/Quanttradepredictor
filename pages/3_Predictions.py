import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

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
st.title("🎯 Prediction Engine")
st.markdown("Real-time Market Analysis & Forecasting")
st.markdown("---")

# Model and time range selection
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

def safe_format_date(idx):
    """Safely format date index to string"""
    try:
        if hasattr(idx, 'strftime'):
            return idx.strftime('%Y-%m-%d %H:%M')
        else:
            return pd.to_datetime(idx).strftime('%Y-%m-%d %H:%M')
    except:
        return str(idx)

def create_display_dataframe(pred_df):
    """Create a safely formatted display dataframe"""
    display_df = pred_df.copy()
    display_df = display_df.reset_index()
    
    # Safely handle date column
    if len(display_df.columns) > 0:
        index_col = display_df.columns[0]
        display_df['Date'] = display_df[index_col].apply(safe_format_date)
        
        # Remove original index column if different from Date
        if index_col != 'Date':
            display_df = display_df.drop(columns=[index_col], errors='ignore')
    
    # Format numeric columns
    if 'Price' in display_df.columns:
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
    
    if 'Confidence' in display_df.columns:
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.3f}")
    
    # Reorder columns
    date_cols = [col for col in display_df.columns if 'Date' in col]
    other_cols = [col for col in display_df.columns if 'Date' not in col]
    ordered_cols = date_cols + other_cols
    
    return display_df[ordered_cols]

# Generate predictions
try:
    predictions, probabilities = model_trainer.predict(selected_model, features_filtered)
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'Price': df_filtered['Close'],
        'Prediction': predictions,
        'Direction': ['Up' if p == 1 else 'Down' for p in predictions]
    }, index=features_filtered.index)
    
    if probabilities is not None:
        pred_df['Confidence'] = np.max(probabilities, axis=1)
    
    # Create tabs for different views
    if selected_model == 'direction':
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Statistics", "📋 Data Table", "🔍 Analysis"])
        
        with tab1:
            st.subheader("📈 Direction Predictions")
            
            # Create price chart with predictions
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2),
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
                    marker=dict(symbol='triangle-up', color='green', size=8),
                    name='Predicted Up',
                    hovertemplate='<b>Prediction:</b> Up<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ))
            
            if len(down_predictions) > 0:
                fig.add_trace(go.Scatter(
                    x=down_predictions.index,
                    y=down_predictions['Price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=8),
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
        
        with tab2:
            st.subheader("📊 Prediction Statistics")
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                up_pct = (predictions == 1).mean() * 100
                st.metric("Predicted Up", f"{up_pct:.1f}%")
            with col2:
                down_pct = (predictions == 0).mean() * 100
                st.metric("Predicted Down", f"{down_pct:.1f}%")
            with col3:
                if 'Confidence' in pred_df.columns:
                    avg_conf = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                else:
                    st.metric("Data Points", len(pred_df))
            with col4:
                current_pred = "Up" if predictions[-1] == 1 else "Down"
                st.metric("Latest Signal", current_pred)
        
        with tab3:
            st.subheader("📋 Detailed Predictions Data")
            
            # Create display dataframe
            display_df = create_display_dataframe(pred_df)
            
            # Show recent predictions
            st.dataframe(
                display_df.tail(50),
                use_container_width=True,
                hide_index=True
            )
        
        with tab4:
            st.subheader("🔍 Analysis")
            st.info("Model analysis and insights would be displayed here.")
    
    else:
        # Handle other model types (trading_signal, profit_prob, etc.)
        st.subheader(f"📊 {selected_model.replace('_', ' ').title()} Predictions")
        
        # Create display dataframe
        display_df = create_display_dataframe(pred_df)
        
        # Show chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Price'],
            mode='lines+markers',
            name='Price with Predictions'
        ))
        
        fig.update_layout(
            height=400,
            title=f"{selected_model.replace('_', ' ').title()} Predictions",
            xaxis_title="Date",
            yaxis_title="Price ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("Recent Predictions")
        st.dataframe(display_df.tail(20), use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error generating predictions: {str(e)}")
    st.info("Please try refreshing the page or check your model training.")
    
    # Show debug information
    st.subheader("Debug Information")
    st.write("Available models:", available_models)
    st.write("Selected model:", selected_model)
    st.write("Features shape:", features_filtered.shape if features_filtered is not None else "None")
    st.write("Data shape:", df_filtered.shape)

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
    
    if compare_models:
        st.info(f"Comparing {len(compare_models)} models. Feature comparison would be displayed here.")
else:
    st.info("Train more models to enable comparison features.")