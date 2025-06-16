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

    elif selected_model in ['profit_probability', 'profit_prob', 'profit_prob_regression']:
        tab1, tab2, tab3 = st.tabs(["🎯 Profit Probability", "📊 Risk Analysis", "📋 Data Table"])

        # Add profit probability specific fields
        if selected_model == 'profit_prob_regression':
            # For regression model, predictions are actual probabilities (0-1)
            pred_df['Profit_Prob'] = np.clip(predictions, 0, 1)
            pred_df['Prediction_Type'] = 'Regression (Probability)'
        else:
            # For classification model
            if probabilities is not None and probabilities.shape[1] > 1:
                # Use the probability of the positive class (class 1)
                pred_df['Profit_Prob'] = probabilities[:, 1]
                pred_df['Prediction_Type'] = 'Classification (Confidence)'
            elif probabilities is not None:
                # Single probability array
                pred_df['Profit_Prob'] = probabilities[:, 0]
                pred_df['Prediction_Type'] = 'Classification (Confidence)'
            else:
                # Binary predictions only - convert to meaningful probabilities
                # High confidence for 1, low confidence for 0
                pred_df['Profit_Prob'] = np.where(predictions == 1, 0.75, 0.25)
                pred_df['Prediction_Type'] = 'Classification (Binary)'

        # Create risk levels based on probability thresholds
        pred_df['Risk_Level'] = pd.cut(pred_df['Profit_Prob'], 
                                     bins=[0, 0.25, 0.5, 0.75, 1.0],
                                     labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'],
                                     include_lowest=True)

        # Add signal interpretation
        pred_df['Signal'] = np.where(pred_df['Profit_Prob'] >= 0.6, '🟢 HIGH PROFIT', 
                                   np.where(pred_df['Profit_Prob'] >= 0.4, '🟡 MEDIUM PROFIT', '🔴 LOW PROFIT'))

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
            fig.add_hline(y=0.75, line_dash="dash", line_color="green", 
                         annotation_text="Low Risk Zone")
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Risk Zone")
            fig.add_hline(y=0.25, line_dash="dash", line_color="red", 
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
            if 'Signal' in display_df.columns:
                display_df['Signal'] = display_df['Signal'].astype(str)

            st.dataframe(display_df.tail(50), use_container_width=True, hide_index=True)

    elif selected_model == 'volatility':
        tab1, tab2, tab3 = st.tabs(["📊 Volatility Forecast", "📈 Volatility Trends", "📋 Data Table"])

        # Add volatility-specific fields
        pred_df['Volatility_Forecast'] = predictions
        pred_df['Volatility_Category'] = pd.cut(pred_df['Volatility_Forecast'], 
                                               bins=[0, 0.01, 0.02, 0.05, float('inf')],
                                               labels=['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol'])

        with tab1:
            st.subheader("📊 Volatility Forecasting Analysis")

            # Volatility forecast over time
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Predicted Volatility Over Time', 'Volatility Distribution'),
                              vertical_spacing=0.1)

            # Volatility timeline
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Volatility_Forecast'],
                mode='lines+markers',
                name='Volatility Forecast',
                line=dict(color='orange', width=2),
                marker=dict(size=4)
            ), row=1, col=1)

            # Volatility histogram
            fig.add_trace(go.Histogram(
                x=pred_df['Volatility_Forecast'],
                nbinsx=30,
                name='Distribution',
                marker_color='lightcoral'
            ), row=2, col=1)

            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", row=1, col=1)
            fig.update_xaxes(title_text="Volatility Value", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Volatility", f"{pred_df['Volatility_Forecast'].mean():.4f}")
            with col2:
                st.metric("Max Volatility", f"{pred_df['Volatility_Forecast'].max():.4f}")
            with col3:
                high_vol = (pred_df['Volatility_Category'].isin(['High Vol', 'Extreme Vol'])).mean() * 100
                st.metric("High Volatility %", f"{high_vol:.1f}%")
            with col4:
                st.metric("Latest Volatility", f"{pred_df['Volatility_Forecast'].iloc[-1]:.4f}")

        with tab2:
            st.subheader("📈 Price with Volatility Indicators")

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))

            # Volatility-based markers
            vol_colors = {'Low Vol': 'green', 'Medium Vol': 'yellow', 'High Vol': 'orange', 'Extreme Vol': 'red'}
            for vol_level in vol_colors:
                vol_data = pred_df[pred_df['Volatility_Category'] == vol_level]
                if len(vol_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=vol_data.index,
                        y=vol_data['Price'],
                        mode='markers',
                        marker=dict(color=vol_colors[vol_level], size=8),
                        name=vol_level,
                        hovertemplate=f'<b>Volatility:</b> {vol_level}<br><b>Price:</b> $%{{y:.2f}}<extra></extra>'
                    ))

            fig.update_layout(
                height=500,
                title="Price Movement with Volatility Level Indicators",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("📋 Volatility Predictions Data")

            # Create enhanced display dataframe for volatility
            display_df = create_display_dataframe(pred_df)
            
            # Format volatility values
            if 'Volatility_Forecast' in display_df.columns:
                display_df['Volatility_Forecast'] = display_df['Volatility_Forecast'].apply(
                    lambda x: f"{float(x):.4f}" if isinstance(x, (int, float, str)) else x
                )
            
            # Add volatility interpretation
            if 'Volatility_Category' in display_df.columns:
                display_df['Vol_Level'] = display_df['Volatility_Category'].apply(
                    lambda x: f"📊 {x}" if pd.notna(x) else "N/A"
                )
            
            # Show validation stats
            st.subheader("📊 Volatility Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                low_vol_count = (pred_df['Volatility_Category'] == 'Low Vol').sum() if 'Volatility_Category' in pred_df.columns else 0
                st.metric("Low Volatility", low_vol_count)
            
            with col2:
                medium_vol_count = (pred_df['Volatility_Category'] == 'Medium Vol').sum() if 'Volatility_Category' in pred_df.columns else 0
                st.metric("Medium Volatility", medium_vol_count)
            
            with col3:
                high_vol_count = (pred_df['Volatility_Category'] == 'High Vol').sum() if 'Volatility_Category' in pred_df.columns else 0
                st.metric("High Volatility", high_vol_count)
            
            with col4:
                extreme_vol_count = (pred_df['Volatility_Category'] == 'Extreme Vol').sum() if 'Volatility_Category' in pred_df.columns else 0
                st.metric("Extreme Volatility", extreme_vol_count)
            
            # Detailed data table
            columns_to_show = ['Date', 'Price', 'Volatility_Forecast']
            if 'Vol_Level' in display_df.columns:
                columns_to_show.append('Vol_Level')
            if 'Volatility_Category' in display_df.columns:
                columns_to_show.append('Volatility_Category')
            
            # Filter columns that exist
            available_columns = [col for col in columns_to_show if col in display_df.columns]
            
            # Add volatility statistics for recent data
            st.subheader("📈 Recent Volatility Data")
            recent_vol_data = pred_df.tail(20)
            
            # Show volatility statistics for recent data
            if len(recent_vol_data) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    recent_avg = recent_vol_data['Volatility_Forecast'].mean()
                    st.metric("Recent Avg Vol", f"{recent_avg:.4f}")
                with col2:
                    recent_max = recent_vol_data['Volatility_Forecast'].max()
                    st.metric("Recent Max Vol", f"{recent_max:.4f}")
                with col3:
                    latest_vol = recent_vol_data['Volatility_Forecast'].iloc[-1]
                    st.metric("Latest Vol", f"{latest_vol:.4f}")
            
            st.dataframe(
                display_df[available_columns].tail(50), 
                use_container_width=True, 
                hide_index=True
            )

    elif selected_model == 'trend_sideways':
        tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "📊 Market State", "📋 Data Table"])

        # Add trend analysis fields
        pred_df['Market_State'] = np.where(predictions == 1, 'Trending', 'Sideways')
        pred_df['Trend_Signal'] = predictions

        with tab1:
            st.subheader("📈 Trend vs Sideways Market Analysis")

            # Trend analysis chart
            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))

            # Trend/Sideways markers
            trending_data = pred_df[pred_df['Market_State'] == 'Trending']
            sideways_data = pred_df[pred_df['Market_State'] == 'Sideways']

            if len(trending_data) > 0:
                fig.add_trace(go.Scatter(
                    x=trending_data.index,
                    y=trending_data['Price'],
                    mode='markers',
                    marker=dict(color='green', size=8, symbol='triangle-up'),
                    name='Trending Market',
                    hovertemplate='<b>State:</b> Trending<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ))

            if len(sideways_data) > 0:
                fig.add_trace(go.Scatter(
                    x=sideways_data.index,
                    y=sideways_data['Price'],
                    mode='markers',
                    marker=dict(color='orange', size=8, symbol='circle'),
                    name='Sideways Market',
                    hovertemplate='<b>State:</b> Sideways<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ))

            fig.update_layout(
                height=500,
                title="Market State Classification",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📊 Market State Distribution")

            # Market state pie chart
            state_counts = pred_df['Market_State'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=state_counts.index,
                values=state_counts.values,
                hole=0.3,
                marker_colors=['green', 'orange']
            )])

            fig.update_layout(
                title="Trending vs Sideways Market Distribution",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                trending_pct = (predictions == 1).mean() * 100
                st.metric("Trending %", f"{trending_pct:.1f}%")
            with col2:
                sideways_pct = (predictions == 0).mean() * 100
                st.metric("Sideways %", f"{sideways_pct:.1f}%")
            with col3:
                if 'Confidence' in pred_df.columns:
                    avg_conf = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                else:
                    st.metric("Data Points", len(pred_df))
            with col4:
                current_state = "Trending" if predictions[-1] == 1 else "Sideways"
                st.metric("Current State", current_state)

        with tab3:
            st.subheader("📋 Trend Classification Data")

            display_df = create_display_dataframe(pred_df)
            st.dataframe(display_df.tail(50), use_container_width=True, hide_index=True)

    elif selected_model == 'reversal':
        tab1, tab2, tab3 = st.tabs(["🔄 Reversal Signals", "📊 Reversal Analysis", "📋 Data Table"])

        # Add reversal fields
        pred_df['Reversal_Signal'] = np.where(predictions == 1, 'Reversal Expected', 'No Reversal')
        pred_df['Signal_Type'] = predictions

        with tab1:
            st.subheader("🔄 Price Reversal Detection")

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))

            # Reversal markers
            reversal_data = pred_df[pred_df['Reversal_Signal'] == 'Reversal Expected']

            if len(reversal_data) > 0:
                fig.add_trace(go.Scatter(
                    x=reversal_data.index,
                    y=reversal_data['Price'],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='diamond'),
                    name='Reversal Signal',
                    hovertemplate='<b>Signal:</b> Reversal Expected<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ))

            fig.update_layout(
                height=500,
                title="Price Reversal Detection Points",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📊 Reversal Signal Analysis")

            # Reversal statistics
            reversal_counts = pred_df['Reversal_Signal'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=reversal_counts.index,
                values=reversal_counts.values,
                hole=0.3,
                marker_colors=['lightblue', 'red']
            )])

            fig.update_layout(
                title="Reversal Signal Distribution",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                reversal_pct = (predictions == 1).mean() * 100
                st.metric("Reversal Signals %", f"{reversal_pct:.1f}%")
            with col2:
                no_reversal_pct = (predictions == 0).mean() * 100
                st.metric("No Reversal %", f"{no_reversal_pct:.1f}%")
            with col3:
                if 'Confidence' in pred_df.columns:
                    avg_conf = pred_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                else:
                    st.metric("Total Signals", len(reversal_data))
            with col4:
                current_signal = "Reversal Expected" if predictions[-1] == 1 else "No Reversal"
                st.metric("Latest Signal", current_signal)

        with tab3:
            st.subheader("📋 Reversal Detection Data")

            display_df = create_display_dataframe(pred_df)
            st.dataframe(display_df.tail(50), use_container_width=True, hide_index=True)

    elif selected_model == 'trading_signal':
        tab1, tab2, tab3 = st.tabs(["📊 Trading Signals", "📈 Signal Analysis", "📋 Signal History"])

        signal_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
        signal_colors = {0: '#E74C3C', 1: '#F39C12', 2: '#27AE60'}

        pred_df['Signal'] = predictions
        pred_df['Signal_Name'] = [signal_map[p] for p in predictions]

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

            display_df = create_display_dataframe(pred_df)
            
            st.dataframe(
                display_df.tail(50),
                use_container_width=True,
                hide_index=True
            )

    else:
        # Handle any other model types
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