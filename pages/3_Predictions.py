import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import all model classes and utilities
from models.volatility_model import VolatilityModel
from models.direction_model import DirectionModel
from models.profit_probability_model import ProfitProbabilityModel
from models.reversal_model import ReversalModel
from utils.database_adapter import DatabaseAdapter

def show_predictions_page():
    """Main predictions page with all 4 models - NO FALLBACK LOGIC"""
    
    st.title("🔮 Real-Time Predictions")
    st.markdown("### Advanced ML Model Predictions - Authentic Data Only")
    
    # Initialize database
    db = DatabaseAdapter()
    
    # Check if data exists
    if 'features' not in st.session_state or st.session_state.features is None:
        st.error("⚠️ No data available. Please upload data first in the Data Upload page.")
        st.stop()
    
    # Create tabs for all 4 models
    vol_tab, dir_tab, profit_tab, reversal_tab = st.tabs([
        "📊 Volatility Predictions", 
        "📈 Direction Predictions", 
        "💰 Profit Probability", 
        "🔄 Reversal Detection"
    ])
    
    # Volatility Predictions Tab
    with vol_tab:
        show_volatility_predictions(db)
    
    # Direction Predictions Tab
    with dir_tab:
        show_direction_predictions(db)
    
    # Profit Probability Tab  
    with profit_tab:
        show_profit_predictions(db)
    
    # Reversal Detection Tab
    with reversal_tab:
        show_reversal_predictions(db)

def show_volatility_predictions(db):
    """Volatility predictions with authentic data only"""
    
    st.header("📊 Volatility Forecasting")
    
    # Initialize model manager and check for trained models
    from models.model_manager import ModelManager
    model_manager = ModelManager()
    
    # Check if volatility model exists in session state or database
    if not model_manager.is_model_trained('volatility'):
        st.warning("⚠️ Volatility model not trained. Please train the model first.")
        return
    
    # Get authentic data
    features = st.session_state.features
    if features is None or len(features) == 0:
        st.error("No feature data available")
        return
    
    # Make predictions using trained model
    try:
        predictions, _ = model_manager.predict('volatility', features)
        
        if predictions is None or len(predictions) == 0:
            st.error("Model prediction failed")
            return
        
        # Use authentic datetime index from original data
        datetime_index = features.index
        
        # Create DataFrame with authentic timestamps
        pred_df = pd.DataFrame({
            'DateTime': datetime_index,
            'Predicted_Volatility': predictions,
            'Date': datetime_index.strftime('%Y-%m-%d'),
            'Time': datetime_index.strftime('%H:%M:%S')
        })
        
        # Display recent predictions (last 100 rows)
        recent_predictions = pred_df.tail(100)
        
        st.subheader("Recent Volatility Predictions")
        st.dataframe(recent_predictions, use_container_width=True)
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_predictions['DateTime'],
            y=recent_predictions['Predicted_Volatility'],
            mode='lines+markers',
            name='Predicted Volatility',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Volatility Predictions - Last 100 Data Points",
            xaxis_title="DateTime",
            yaxis_title="Predicted Volatility",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating volatility predictions: {str(e)}")

def show_direction_predictions(db):
    """Direction predictions with authentic data only"""
    
    st.header("📈 Direction Predictions")
    
    # Check if direction model exists in trained models
    if 'direction_trained_models' not in st.session_state or 'direction' not in st.session_state.direction_trained_models:
        st.warning("⚠️ Direction model not trained. Please train the model first.")
        return
    
    # Get authentic data
    if 'direction_features' not in st.session_state or st.session_state.direction_features is None:
        st.error("No direction features available")
        return
    
    features = st.session_state.direction_features
    
    try:
        model = st.session_state.direction_trained_models['direction']
        predictions, probabilities = model.predict(features)
        
        if predictions is None or len(predictions) == 0:
            st.error("Model prediction failed")
            return
        
        # Use authentic datetime index
        datetime_index = features.index
        
        # Create DataFrame with authentic data only
        pred_df = pd.DataFrame({
            'DateTime': datetime_index,
            'Direction': ['Bullish' if p == 1 else 'Bearish' for p in predictions],
            'Confidence': [np.max(prob) for prob in probabilities] if probabilities is not None else None,
            'Date': datetime_index.strftime('%Y-%m-%d'),
            'Time': datetime_index.strftime('%H:%M:%S')
        })
        
        # Display recent predictions
        recent_predictions = pred_df.tail(100)
        
        st.subheader("Recent Direction Predictions")
        
        # Only show data if we have valid predictions
        if len(recent_predictions) > 0:
            st.dataframe(recent_predictions, use_container_width=True)
            
            # Create chart
            fig = go.Figure()
            
            # Add bullish signals
            bullish_data = recent_predictions[recent_predictions['Direction'] == 'Bullish']
            if len(bullish_data) > 0:
                fig.add_trace(go.Scatter(
                    x=bullish_data['DateTime'],
                    y=[1] * len(bullish_data),
                    mode='markers',
                    name='Bullish',
                    marker=dict(color='green', size=10)
                ))
            
            # Add bearish signals
            bearish_data = recent_predictions[recent_predictions['Direction'] == 'Bearish']
            if len(bearish_data) > 0:
                fig.add_trace(go.Scatter(
                    x=bearish_data['DateTime'],
                    y=[0] * len(bearish_data),
                    mode='markers',
                    name='Bearish',
                    marker=dict(color='red', size=10)
                ))
            
            fig.update_layout(
                title="Direction Predictions - Last 100 Data Points",
                xaxis_title="DateTime",
                yaxis_title="Direction (1=Bullish, 0=Bearish)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating direction predictions: {str(e)}")

def show_profit_predictions(db):
    """Profit probability predictions with authentic data only"""
    
    st.header("💰 Profit Probability Predictions")
    
    # Check if profit model exists in trained models
    if 'profit_prob_trained_models' not in st.session_state or 'profit_probability' not in st.session_state.profit_prob_trained_models:
        st.warning("⚠️ Profit probability model not trained. Please train the model first.")
        return
    
    # Get authentic data
    if 'profit_prob_features' not in st.session_state or st.session_state.profit_prob_features is None:
        st.error("No profit probability features available")
        return
    
    features = st.session_state.profit_prob_features
    
    try:
        model = st.session_state.profit_prob_trained_models['profit_probability']
        predictions, probabilities = model.predict(features)
        
        if predictions is None or len(predictions) == 0:
            st.error("Model prediction failed")
            return
        
        # Use authentic datetime index
        datetime_index = features.index
        
        # Create DataFrame with authentic data only
        pred_df = pd.DataFrame({
            'DateTime': datetime_index,
            'Profit_Probability': ['High Profit' if p == 1 else 'Low Profit' for p in predictions],
            'Confidence': [np.max(prob) for prob in probabilities] if probabilities is not None else None,
            'Date': datetime_index.strftime('%Y-%m-%d'),
            'Time': datetime_index.strftime('%H:%M:%S')
        })
        
        # Display recent predictions
        recent_predictions = pred_df.tail(100)
        
        st.subheader("Recent Profit Probability Predictions")
        
        if len(recent_predictions) > 0:
            st.dataframe(recent_predictions, use_container_width=True)
            
            # Create chart
            fig = go.Figure()
            
            # Add high profit signals
            high_profit = recent_predictions[recent_predictions['Profit_Probability'] == 'High Profit']
            if len(high_profit) > 0:
                fig.add_trace(go.Scatter(
                    x=high_profit['DateTime'],
                    y=[1] * len(high_profit),
                    mode='markers',
                    name='High Profit',
                    marker=dict(color='green', size=10)
                ))
            
            # Add low profit signals
            low_profit = recent_predictions[recent_predictions['Profit_Probability'] == 'Low Profit']
            if len(low_profit) > 0:
                fig.add_trace(go.Scatter(
                    x=low_profit['DateTime'],
                    y=[0] * len(low_profit),
                    mode='markers',
                    name='Low Profit',
                    marker=dict(color='red', size=10)
                ))
            
            fig.update_layout(
                title="Profit Probability Predictions - Last 100 Data Points",
                xaxis_title="DateTime",
                yaxis_title="Profit Probability (1=High, 0=Low)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating profit probability predictions: {str(e)}")

def show_reversal_predictions(db):
    """Reversal detection predictions with authentic data only"""
    
    st.header("🔄 Reversal Detection")
    
    # Check if reversal model exists in trained models
    if 'reversal_trained_models' not in st.session_state or 'reversal' not in st.session_state.reversal_trained_models:
        st.warning("⚠️ Reversal model not trained. Please train the model first.")
        return
    
    # Get authentic data
    if 'reversal_features' not in st.session_state or st.session_state.reversal_features is None:
        st.error("No reversal features available")
        return
    
    features = st.session_state.reversal_features
    
    try:
        model = st.session_state.reversal_trained_models['reversal']
        predictions, probabilities = model.predict(features)
        
        if predictions is None or len(predictions) == 0:
            st.error("Model prediction failed")
            return
        
        # Use authentic datetime index
        datetime_index = features.index
        
        # Create DataFrame with authentic data only
        pred_df = pd.DataFrame({
            'DateTime': datetime_index,
            'Reversal_Signal': ['Reversal' if p == 1 else 'No Reversal' for p in predictions],
            'Confidence': [np.max(prob) for prob in probabilities] if probabilities is not None else None,
            'Date': datetime_index.strftime('%Y-%m-%d'),
            'Time': datetime_index.strftime('%H:%M:%S')
        })
        
        # Display recent predictions
        recent_predictions = pred_df.tail(100)
        
        st.subheader("Recent Reversal Predictions")
        
        if len(recent_predictions) > 0:
            st.dataframe(recent_predictions, use_container_width=True)
            
            # Create chart
            fig = go.Figure()
            
            # Add reversal signals
            reversals = recent_predictions[recent_predictions['Reversal_Signal'] == 'Reversal']
            if len(reversals) > 0:
                fig.add_trace(go.Scatter(
                    x=reversals['DateTime'],
                    y=[1] * len(reversals),
                    mode='markers',
                    name='Reversal',
                    marker=dict(color='orange', size=12)
                ))
            
            # Add no reversal signals
            no_reversals = recent_predictions[recent_predictions['Reversal_Signal'] == 'No Reversal']
            if len(no_reversals) > 0:
                fig.add_trace(go.Scatter(
                    x=no_reversals['DateTime'],
                    y=[0] * len(no_reversals),
                    mode='markers',
                    name='No Reversal',
                    marker=dict(color='blue', size=8)
                ))
            
            fig.update_layout(
                title="Reversal Detection - Last 100 Data Points",
                xaxis_title="DateTime",
                yaxis_title="Reversal Signal (1=Reversal, 0=No Reversal)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating reversal predictions: {str(e)}")

if __name__ == "__main__":
    show_predictions_page()