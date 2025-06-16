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
            st.subheader("🔍 Recent Predictions")
            
            # Create display dataframe for recent predictions
            display_df = create_display_dataframe(pred_df)
            
            # Show the most recent 20 predictions with better formatting
            recent_df = display_df.tail(20).copy()
            
            # Add signal interpretation
            if 'Direction' in recent_df.columns:
                recent_df['Signal'] = recent_df['Direction'].apply(
                    lambda x: "🟢 BUY" if x == "Up" else "🔴 SELL"
                )
            
            st.dataframe(
                recent_df[['Date', 'Price', 'Direction', 'Signal'] + 
                         ([col for col in recent_df.columns if 'Confidence' in col] if 'Confidence' in recent_df.columns else [])
                        ],
                use_container_width=True,
                hide_index=True
            )
    
    elif selected_model == 'magnitude':
        tab1, tab2, tab3 = st.tabs(["📊 Magnitude Analysis", "📈 Price Movement", "📋 Data Table"])
        
        # Add magnitude-specific fields
        pred_df['Magnitude'] = np.abs(predictions)
        pred_df['Magnitude_Category'] = pd.cut(pred_df['Magnitude'], 
                                             bins=[0, 0.01, 0.02, 0.05, float('inf')],
                                             labels=['Low', 'Medium', 'High', 'Extreme'])
        
        with tab1:
            st.subheader("📊 Price Movement Magnitude Analysis")
            
            # Magnitude distribution
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Magnitude Over Time', 'Magnitude Distribution'),
                              vertical_spacing=0.1)
            
            # Magnitude timeline
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Magnitude'],
                mode='lines+markers',
                name='Magnitude',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ), row=1, col=1)
            
            # Magnitude histogram
            fig.add_trace(go.Histogram(
                x=pred_df['Magnitude'],
                nbinsx=30,
                name='Distribution',
                marker_color='lightblue'
            ), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Magnitude", row=1, col=1)
            fig.update_xaxes(title_text="Magnitude Value", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Magnitude", f"{pred_df['Magnitude'].mean():.4f}")
            with col2:
                st.metric("Max Magnitude", f"{pred_df['Magnitude'].max():.4f}")
            with col3:
                high_magnitude = (pred_df['Magnitude_Category'].isin(['High', 'Extreme'])).mean() * 100
                st.metric("High Magnitude %", f"{high_magnitude:.1f}%")
            with col4:
                st.metric("Latest Magnitude", f"{pred_df['Magnitude'].iloc[-1]:.4f}")
        
        with tab2:
            st.subheader("📈 Price Chart with Magnitude Indicators")
            
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))
            
            # Color-coded magnitude markers
            colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Extreme': 'red'}
            for category in colors:
                category_data = pred_df[pred_df['Magnitude_Category'] == category]
                if len(category_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=category_data.index,
                        y=category_data['Price'],
                        mode='markers',
                        marker=dict(color=colors[category], size=8),
                        name=f'{category} Magnitude',
                        hovertemplate=f'<b>Magnitude:</b> {category}<br><b>Price:</b> $%{{y:.2f}}<extra></extra>'
                    ))
            
            fig.update_layout(
                height=500,
                title="Price Movement with Magnitude Categories",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📋 Magnitude Predictions Data")
            
            display_df = create_display_dataframe(pred_df)
            if 'Magnitude' in display_df.columns:
                display_df['Magnitude'] = display_df['Magnitude'].apply(lambda x: f"{float(x):.4f}" if isinstance(x, (int, float, str)) else x)
            
            st.dataframe(display_df.tail(50), use_container_width=True, hide_index=True)
    
    elif selected_model == 'profit_probability':
        tab1, tab2, tab3 = st.tabs(["🎯 Profit Probability", "📊 Risk Analysis", "📋 Data Table"])
        
        # Add profit probability specific fields
        pred_df['Profit_Prob'] = predictions
        pred_df['Risk_Level'] = pd.cut(pred_df['Profit_Prob'], 
                                     bins=[0, 0.3, 0.6, 0.8, 1.0],
                                     labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'])
        
        with tab1:
            st.subheader("🎯 Profit Probability Analysis")
            
            # Probability over time with risk zones
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Profit_Prob'],
                mode='lines+markers',
                name='Profit Probability',
                line=dict(color='green', width=2),
                fill='tonexty'
            ))
            
            # Add risk zone backgrounds
            fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                         annotation_text="Low Risk Zone")
            fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Risk Zone")
            fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                         annotation_text="High Risk Zone")
            
            fig.update_layout(
                height=500,
                title="Profit Probability Over Time",
                xaxis_title="Date",
                yaxis_title="Profit Probability",
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Probability", f"{pred_df['Profit_Prob'].mean():.3f}")
            with col2:
                high_prob = (pred_df['Profit_Prob'] >= 0.6).mean() * 100
                st.metric("High Confidence %", f"{high_prob:.1f}%")
            with col3:
                low_risk = (pred_df['Risk_Level'] == 'Low Risk').mean() * 100
                st.metric("Low Risk %", f"{low_risk:.1f}%")
            with col4:
                st.metric("Latest Probability", f"{pred_df['Profit_Prob'].iloc[-1]:.3f}")
        
        with tab2:
            st.subheader("📊 Risk Distribution Analysis")
            
            # Risk level pie chart
            risk_counts = pred_df['Risk_Level'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.3,
                marker_colors=['red', 'orange', 'lightgreen', 'darkgreen']
            )])
            
            fig.update_layout(
                title="Risk Level Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Price chart with risk indicators
            st.subheader("Price Movement with Risk Levels")
            
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))
            
            # Risk level markers
            risk_colors = {'High Risk': 'red', 'Medium Risk': 'orange', 
                          'Low Risk': 'lightgreen', 'Very Low Risk': 'darkgreen'}
            
            for risk_level in risk_colors:
                risk_data = pred_df[pred_df['Risk_Level'] == risk_level]
                if len(risk_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=risk_data.index,
                        y=risk_data['Price'],
                        mode='markers',
                        marker=dict(color=risk_colors[risk_level], size=8),
                        name=risk_level,
                        hovertemplate=f'<b>Risk:</b> {risk_level}<br><b>Price:</b> $%{{y:.2f}}<extra></extra>'
                    ))
            
            fig.update_layout(
                height=500,
                title="Price Movement with Risk Level Indicators",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📋 Profit Probability Data")
            
            display_df = create_display_dataframe(pred_df)
            if 'Profit_Prob' in display_df.columns:
                display_df['Profit_Prob'] = display_df['Profit_Prob'].apply(lambda x: f"{float(x):.3f}" if isinstance(x, (int, float, str)) else x)
            
            st.dataframe(display_df.tail(50), use_container_width=True, hide_index=True)
    
    else:
        # Handle other model types (trading_signal, etc.)
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
        
        # Show data table with enhanced formatting
        st.subheader("Recent Predictions")
        
        # Add signal interpretation for better readability
        recent_df = display_df.tail(20).copy()
        if 'Prediction' in recent_df.columns:
            recent_df['Signal'] = recent_df['Prediction'].apply(
                lambda x: "🟢 BUY" if x == 1 else "🔴 SELL" if x == 0 else "⚪ HOLD"
            )
        
        st.dataframe(recent_df, use_container_width=True, hide_index=True)

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