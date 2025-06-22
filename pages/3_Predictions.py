import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">🎯 AI PREDICTION ENGINE</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Real-time Market Analysis & Forecasting
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None
if 'features' not in st.session_state:
    st.session_state.features = None

# Check if data is loaded
if st.session_state.data is None:
    st.warning("⚠️ No data loaded. Please go to the **Data Upload** page first.")
    st.stop()

# Check if models are trained
if not st.session_state.models or not isinstance(st.session_state.models, dict):
    st.warning("⚠️ No trained models found. Please go to the **Model Training** page first.")
    st.stop()

# Check if model trainer is initialized
if st.session_state.model_trainer is None:
    st.warning("⚠️ Model trainer not initialized. Please go to the **Model Training** page first.")
    st.stop()

# Get available models
available_models = [name for name, info in st.session_state.models.items() 
                   if info is not None]

if not available_models:
    st.error("❌ No successfully trained models found.")
    st.info("Please go to the **Model Training** page to train models first.")
    st.stop()

# Model configuration - based on your actual trained models
MODEL_CONFIG = {
    'direction': {
        'name': 'Direction Prediction',
        'description': 'Predicts if price will go up (1) or down (0)',
        'task_type': 'classification',
        'classes': {0: 'Down ⬇️', 1: 'Up ⬆️'},
        'colors': {0: 'red', 1: 'green'}
    },
    'magnitude': {
        'name': 'Magnitude Prediction', 
        'description': 'Predicts the magnitude of price movements (continuous values)',
        'task_type': 'regression',
        'unit': 'percentage points'
    },
    'profit_prob': {
        'name': 'Profit Probability',
        'description': 'Predicts probability of profitable trades',
        'task_type': 'classification',
        'classes': {0: 'Low Profit 📉', 1: 'High Profit 📈'},
        'colors': {0: 'red', 1: 'green'}
    },
    'volatility': {
        'name': 'Volatility Forecast',
        'description': 'Forecasts market volatility levels (continuous values)',
        'task_type': 'regression',
        'unit': 'volatility ratio'
    },
    'trend_sideways': {
        'name': 'Trend vs Sideways',
        'description': 'Classifies market as trending (1) or sideways (0)',
        'task_type': 'classification',
        'classes': {0: 'Sideways 📊', 1: 'Trending 📈'},
        'colors': {0: 'orange', 1: 'blue'}
    },
    'reversal': {
        'name': 'Reversal Points',
        'description': 'Identifies potential trend reversals',
        'task_type': 'classification',
        'classes': {0: 'Continue ➡️', 1: 'Reversal 🔄'},
        'colors': {0: 'blue', 1: 'purple'}
    },
    'trading_signal': {
        'name': 'Trading Signals',
        'description': 'Generates trading recommendations',
        'task_type': 'classification',
        'classes': {0: 'SELL 🔴', 1: 'HOLD 🟡', 2: 'BUY 🟢'},
        'colors': {0: 'red', 1: 'yellow', 2: 'green'}
    }
}

# Header with model selection
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_model = st.selectbox(
        "🤖 Select AI Model",
        available_models,
        format_func=lambda x: MODEL_CONFIG.get(x, {}).get('name', x.replace('_', ' ').title()),
        help="Choose a trained model for predictions"
    )

with col2:
    date_range = st.selectbox(
        "📅 Time Period",
        ["Last 50 points", "Last 100 points", "Last 200 points", "Last 500 points", "All data"],
        index=1
    )

with col3:
    model_info = MODEL_CONFIG.get(selected_model, {})
    task_type = model_info.get('task_type', 'classification')
    st.metric("Model Type", task_type.title())

# Display model description
st.info(f"**{model_info.get('name', selected_model)}**: {model_info.get('description', 'No description available')}")

# Prepare data and features
df = st.session_state.data
model_trainer = st.session_state.model_trainer

# Filter data based on selection
try:
    if date_range == "Last 50 points":
        df_filtered = df.tail(50).copy()
    elif date_range == "Last 100 points":
        df_filtered = df.tail(100).copy()
    elif date_range == "Last 200 points":
        df_filtered = df.tail(200).copy()
    elif date_range == "Last 500 points":
        df_filtered = df.tail(500).copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        st.error("No data available for the selected time period.")
        st.stop()

except Exception as e:
    st.warning(f"Error filtering data: {str(e)}. Using all available data.")
    df_filtered = df.copy()

# Prepare features for prediction
try:
    # Always recalculate features from current data to ensure all indicators are present
    from features.technical_indicators import TechnicalIndicators
    
    with st.spinner("Calculating technical indicators..."):
        # Calculate all technical indicators on the full dataset
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
        
        # Filter to the requested time period after calculating indicators
        if date_range == "Last 50 points":
            df_filtered_indicators = df_with_indicators.tail(50).copy()
        elif date_range == "Last 100 points":
            df_filtered_indicators = df_with_indicators.tail(100).copy()
        elif date_range == "Last 200 points":
            df_filtered_indicators = df_with_indicators.tail(200).copy()
        elif date_range == "Last 500 points":
            df_filtered_indicators = df_with_indicators.tail(500).copy()
        else:
            df_filtered_indicators = df_with_indicators.copy()
        
        # Prepare features using model trainer (this will handle feature selection automatically)
        features_filtered = model_trainer.prepare_features(df_filtered_indicators)
        
        # Store full features in session state for consistency
        st.session_state.features = model_trainer.prepare_features(df_with_indicators)

    st.success(f"✅ Prepared {len(features_filtered)} data points with {features_filtered.shape[1]} features")
    
    # Show feature information
    if hasattr(model_trainer, 'feature_names') and model_trainer.feature_names:
        st.info(f"Using {len(model_trainer.feature_names)} features for {selected_model} model")
        
        # Debug: Show first few feature names
        with st.expander("🔍 View Features Used"):
            feature_preview = model_trainer.feature_names[:10]
            st.write(f"First 10 features: {feature_preview}")
            if len(model_trainer.feature_names) > 10:
                st.write(f"... and {len(model_trainer.feature_names) - 10} more features")

except Exception as e:
    st.error(f"❌ Error preparing features: {str(e)}")
    st.write("**Debug Info:**")
    st.write(f"- Data shape: {df.shape}")
    st.write(f"- Date range: {date_range}")
    st.write(f"- Selected model: {selected_model}")
    
    # Try to show what features are available
    try:
        from features.technical_indicators import TechnicalIndicators
        test_indicators = TechnicalIndicators.calculate_all_indicators(df.tail(100))
        st.write(f"- Available columns: {list(test_indicators.columns)}")
    except Exception as inner_e:
        st.write(f"- Could not calculate indicators: {str(inner_e)}")
    
    st.stop()

# Generate predictions
try:
    with st.spinner(f"Generating {model_info.get('name', selected_model)} predictions..."):
        # Ensure we have the right features for this model
        if hasattr(model_trainer, 'feature_names') and model_trainer.feature_names:
            # Get the exact features needed for this model
            model_features = model_trainer.feature_names
            available_features = list(features_filtered.columns)
            
            # Check if we have all required features
            missing_features = [f for f in model_features if f not in available_features]
            if missing_features:
                st.warning(f"Missing {len(missing_features)} features, using available features")
                # Use only available features that match the model
                matching_features = [f for f in model_features if f in available_features]
                if len(matching_features) >= 5:  # Need at least 5 features
                    features_for_prediction = features_filtered[matching_features]
                else:
                    features_for_prediction = features_filtered
            else:
                features_for_prediction = features_filtered
        else:
            features_for_prediction = features_filtered
        
        # Make predictions
        predictions, probabilities = model_trainer.predict(selected_model, features_for_prediction)
        
        st.success(f"✅ Generated {len(predictions)} predictions successfully!")

    # Align data with predictions
    pred_length = len(predictions)
    
    # Get the corresponding price data
    if date_range == "Last 50 points":
        df_aligned = df.tail(min(50, pred_length)).copy()
    elif date_range == "Last 100 points":
        df_aligned = df.tail(min(100, pred_length)).copy()
    elif date_range == "Last 200 points":
        df_aligned = df.tail(min(200, pred_length)).copy()
    elif date_range == "Last 500 points":
        df_aligned = df.tail(min(500, pred_length)).copy()
    else:
        df_aligned = df.tail(pred_length).copy()

    # Ensure we have the right number of rows
    if len(df_aligned) > pred_length:
        df_aligned = df_aligned.tail(pred_length)
    elif len(df_aligned) < pred_length:
        # If we don't have enough aligned data, use the last available data
        df_aligned = df.tail(pred_length).copy()

    # Create prediction results
    pred_df = pd.DataFrame({
        'Price': df_aligned['Close'].values[:pred_length],
        'Prediction': predictions
    }, index=df_aligned.index[:pred_length])

    # Add confidence scores
    if probabilities is not None:
        if task_type == 'classification':
            pred_df['Confidence'] = np.max(probabilities, axis=1)

            # Add class probabilities for multiclass models
            if selected_model == 'trading_signal' and probabilities.shape[1] == 3:
                pred_df['Sell_Prob'] = probabilities[:, 0]
                pred_df['Hold_Prob'] = probabilities[:, 1] 
                pred_df['Buy_Prob'] = probabilities[:, 2]
            elif probabilities.shape[1] == 2:
                pred_df['Class_0_Prob'] = probabilities[:, 0]
                pred_df['Class_1_Prob'] = probabilities[:, 1]
        else:
            # For regression, probabilities represent prediction confidence
            pred_df['Confidence'] = probabilities[:, 0] if probabilities.shape[1] > 0 else 0.5

    # Add prediction labels for classification models
    if task_type == 'classification':
        classes = model_info.get('classes', {})
        pred_df['Prediction_Label'] = pred_df['Prediction'].map(classes)
        if pred_df['Prediction_Label'].isna().any():
            pred_df['Prediction_Label'] = pred_df['Prediction'].astype(str)

    st.success(f"✅ Generated {len(predictions)} predictions using {selected_model} model")

except Exception as e:
    st.error(f"❌ Error generating predictions: {str(e)}")
    st.info("This might be due to:")
    st.write("- Model not properly trained")
    st.write("- Feature incompatibility")
    st.write("- Data format issues")

    with st.expander("🔍 Debug Information"):
        st.write(f"Selected model: {selected_model}")
        st.write(f"Available models: {list(st.session_state.models.keys())}")
        st.write(f"Features shape: {features_filtered.shape}")
        st.write(f"Error details: {str(e)}")
    st.stop()

# Display results based on model type
if task_type == 'classification':
    # Classification model visualization
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Statistics", "🎯 Analysis", "📋 Data Table"])

    with tab1:
        st.subheader(f"📈 {model_info.get('name')} - Price Chart")

        fig = go.Figure()

        # Price line
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Price'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2),
            opacity=0.8
        ))

        # Prediction markers
        classes = model_info.get('classes', {})
        colors = model_info.get('colors', {})

        for pred_class, label in classes.items():
            class_data = pred_df[pred_df['Prediction'] == pred_class]
            if len(class_data) > 0:
                marker_symbol = 'circle'
                if selected_model == 'direction':
                    marker_symbol = 'triangle-up' if pred_class == 1 else 'triangle-down'
                elif selected_model == 'trading_signal':
                    if pred_class == 2:  # BUY
                        marker_symbol = 'triangle-up'
                    elif pred_class == 0:  # SELL
                        marker_symbol = 'triangle-down'
                    else:  # HOLD
                        marker_symbol = 'circle'

                hover_template = f'<b>{label}</b><br>Price: $%{{y:.2f}}<br>Date: %{{x}}'
                if 'Confidence' in class_data.columns:
                    hover_template += '<br>Confidence: %{customdata:.3f}'
                    customdata = class_data['Confidence']
                else:
                    customdata = None
                hover_template += '<extra></extra>'

                fig.add_trace(go.Scatter(
                    x=class_data.index,
                    y=class_data['Price'],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        color=colors.get(pred_class, 'gray'),
                        size=10,
                        line=dict(width=1, color='white')
                    ),
                    name=label,
                    hovertemplate=hover_template,
                    customdata=customdata
                ))

        fig.update_layout(
            height=500,
            title=f"{model_info.get('name')} Predictions",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='closest',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📊 Prediction Statistics")

        # Prediction distribution
        pred_counts = pred_df['Prediction'].value_counts().sort_index()

        # Metrics row
        metric_cols = st.columns(len(classes) + 1)

        for i, (pred_class, label) in enumerate(classes.items()):
            with metric_cols[i]:
                count = pred_counts.get(pred_class, 0)
                percentage = (count / len(pred_df)) * 100
                st.metric(label, f"{percentage:.1f}%", f"{count} signals")

        with metric_cols[-1]:
            if 'Confidence' in pred_df.columns:
                avg_confidence = pred_df['Confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            else:
                st.metric("Total Predictions", len(pred_df))

        # Distribution chart
        fig = go.Figure(data=[go.Pie(
            labels=[classes.get(i, f'Class {i}') for i in pred_counts.index],
            values=pred_counts.values,
            hole=0.3,
            marker_colors=[colors.get(i, 'gray') for i in pred_counts.index]
        )])

        fig.update_layout(
            title="Prediction Distribution",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🎯 Recent Predictions Analysis")

        # Latest predictions table
        recent_df = pred_df.tail(20).copy()
        recent_df['Date'] = recent_df.index.strftime('%Y-%m-%d %H:%M')
        recent_df['Price_Formatted'] = recent_df['Price'].apply(lambda x: f"${x:.2f}")

        display_cols = ['Date', 'Price_Formatted', 'Prediction_Label']
        if 'Confidence' in recent_df.columns:
            recent_df['Confidence_Formatted'] = recent_df['Confidence'].apply(lambda x: f"{x:.3f}")
            display_cols.append('Confidence_Formatted')

        st.dataframe(
            recent_df[display_cols].rename(columns={
                'Price_Formatted': 'Price',
                'Prediction_Label': 'Signal',
                'Confidence_Formatted': 'Confidence'
            }),
            use_container_width=True,
            hide_index=True
        )

        # Confidence distribution
        if 'Confidence' in pred_df.columns:
            st.subheader("Confidence Distribution")
            fig = go.Figure(data=[go.Histogram(
                x=pred_df['Confidence'],
                nbinsx=20,
                marker_color='lightblue'
            )])
            fig.update_layout(
                title="Prediction Confidence Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("📋 Complete Prediction Data")

        # Format display dataframe
        display_df = pred_df.copy()
        display_df = display_df.reset_index()
        display_df['Date'] = display_df.iloc[:, 0].dt.strftime('%Y-%m-%d %H:%M')
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")

        if 'Confidence' in display_df.columns:
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.3f}")

        # Reorder columns
        cols = ['Date', 'Price', 'Prediction_Label']
        if 'Confidence' in display_df.columns:
            cols.append('Confidence')

        st.dataframe(
            display_df[cols].rename(columns={'Prediction_Label': 'Signal'}),
            use_container_width=True,
            hide_index=True
        )

else:
    # Regression model visualization
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Chart", "📈 Distribution", "📋 Data Table"])

    with tab1:
        st.subheader(f"📊 {model_info.get('name')} Forecast")

        # Create subplot for price and predictions
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Movement', f'{model_info.get("name")} Values'),
            vertical_spacing=0.1
        )

        # Price chart
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Price'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ), row=1, col=1)

        # Prediction chart
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Prediction'],
            mode='lines+markers',
            name=model_info.get('name', 'Predictions'),
            line=dict(color='orange', width=2),
            marker=dict(size=4)
        ), row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text=model_info.get('unit', 'Value'), row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Value", f"{pred_df['Prediction'].mean():.4f}")
        with col2:
            st.metric("Max Value", f"{pred_df['Prediction'].max():.4f}")
        with col3:
            st.metric("Min Value", f"{pred_df['Prediction'].min():.4f}")
        with col4:
            st.metric("Std Dev", f"{pred_df['Prediction'].std():.4f}")

    with tab2:
        st.subheader("📈 Value Distribution")

        # Distribution histogram
        fig = go.Figure(data=[go.Histogram(
            x=pred_df['Prediction'],
            nbinsx=30,
            marker_color='lightcoral'
        )])

        fig.update_layout(
            title=f"{model_info.get('name')} Distribution",
            xaxis_title=model_info.get('unit', 'Value'),
            yaxis_title="Frequency",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Box plot
        fig = go.Figure(data=[go.Box(
            y=pred_df['Prediction'],
            name=model_info.get('name', 'Predictions'),
            marker_color='lightgreen'
        )])

        fig.update_layout(
            title=f"{model_info.get('name')} Box Plot",
            yaxis_title=model_info.get('unit', 'Value'),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📋 Forecast Data")

        # Format display dataframe
        display_df = pred_df.copy()
        display_df = display_df.reset_index()
        display_df['Date'] = display_df.iloc[:, 0].dt.strftime('%Y-%m-%d %H:%M')
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
        display_df['Forecast'] = display_df['Prediction'].apply(lambda x: f"{x:.4f}")

        cols = ['Date', 'Price', 'Forecast']
        if 'Confidence' in display_df.columns:
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.3f}")
            cols.append('Confidence')

        st.dataframe(
            display_df[cols],
            use_container_width=True,
            hide_index=True
        )

# Model information sidebar
with st.sidebar:
    st.markdown("### 🤖 Model Information")
    st.write(f"**Selected Model:** {model_info.get('name', selected_model)}")
    st.write(f"**Task Type:** {task_type.title()}")
    st.write(f"**Data Points:** {len(pred_df)}")

    if hasattr(model_trainer, 'feature_names') and model_trainer.feature_names:
        st.write(f"**Features Used:** {len(model_trainer.feature_names)}")

        with st.expander("View Features"):
            for i, feature in enumerate(model_trainer.feature_names, 1):
                st.write(f"{i}. {feature}")

    # Model performance info
    if selected_model in st.session_state.models:
        model_data = st.session_state.models[selected_model]
        if isinstance(model_data, dict) and 'metrics' in model_data:
            metrics = model_data['metrics']
            st.markdown("### 📊 Model Performance")

            if task_type == 'classification' and 'accuracy' in metrics:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            elif task_type == 'regression':
                if 'rmse' in metrics:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                if 'mae' in metrics:
                    st.metric("MAE", f"{metrics['mae']:.4f}")

# Footer
st.markdown("---")
st.markdown("### 💡 Model Guide")

model_descriptions = {
    'direction': "🎯 **Direction**: Predicts if price will move up or down next",
    'magnitude': "📏 **Magnitude**: Forecasts the size of price movements", 
    'profit_prob': "💰 **Profit Probability**: Estimates likelihood of profitable trades",
    'volatility': "🌊 **Volatility**: Forecasts market volatility levels",
    'trend_sideways': "📈 **Trend vs Sideways**: Classifies market conditions",
    'reversal': "🔄 **Reversal**: Identifies potential trend reversal points",
    'trading_signal': "🎯 **Trading Signal**: Generates BUY/HOLD/SELL recommendations"
}

for model, desc in model_descriptions.items():
    if model in available_models:
        st.info(desc)

st.markdown("**📊 Model Performance**: Check the Model Training page for detailed accuracy metrics")