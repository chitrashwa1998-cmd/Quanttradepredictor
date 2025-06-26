import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

st.title("🔮 Volatility Predictions")
st.markdown("Generate and analyze volatility forecasts using the trained model.")

# Check prerequisites
if st.session_state.data is None:
    st.error("❌ No data available. Please upload data first.")
    st.stop()

if st.session_state.features is None:
    st.error("❌ No features calculated. Please calculate technical indicators first.")
    st.stop()

# Check if models are trained
if not hasattr(st.session_state, 'trained_models') or not st.session_state.trained_models:
    st.error("❌ No trained models available. Please train the volatility model first.")
    st.stop()

if 'volatility' not in st.session_state.trained_models or st.session_state.trained_models['volatility'] is None:
    st.error("❌ Volatility model not trained. Please train the model first.")
    st.stop()

# Prediction controls
st.header("🎯 Prediction Controls")

col1, col2 = st.columns(2)

with col1:
    prediction_start = st.date_input(
        "Prediction Start Date",
        value=datetime.now().date() - timedelta(days=30),
        help="Start date for predictions"
    )

with col2:
    prediction_end = st.date_input(
        "Prediction End Date", 
        value=datetime.now().date(),
        help="End date for predictions"
    )

# Generate predictions button
if st.button("🚀 Generate Volatility Predictions", type="primary"):
    try:
        with st.spinner("Generating volatility predictions..."):
            # Get model trainer
            model_trainer = st.session_state.model_trainer

            # Filter data for prediction period
            data_for_prediction = st.session_state.features.copy()

            # Convert dates for filtering
            if hasattr(data_for_prediction.index, 'date'):
                mask = (data_for_prediction.index.date >= prediction_start) & (data_for_prediction.index.date <= prediction_end)
            else:
                # Try to convert index to datetime if needed
                try:
                    data_for_prediction.index = pd.to_datetime(data_for_prediction.index)
                    mask = (data_for_prediction.index.date >= prediction_start) & (data_for_prediction.index.date <= prediction_end)
                except:
                    st.warning("Using all available data for predictions (date filtering failed)")
                    mask = data_for_prediction.index.notna()

            filtered_data = data_for_prediction[mask]

            if filtered_data.empty:
                st.error("❌ No data available for the selected date range")
                st.stop()

            st.info(f"📊 Making predictions on {len(filtered_data)} data points")

            # Make predictions
            predictions, probabilities = model_trainer.predict('volatility', filtered_data)

            # Create results dataframe
            pred_df = pd.DataFrame({
                'Volatility_Forecast': predictions
            }, index=filtered_data.index)

            # Add actual volatility for comparison if possible
            try:
                # Calculate actual volatility
                if 'Close' in st.session_state.data.columns:
                    returns = st.session_state.data['Close'].pct_change()
                    actual_volatility = returns.rolling(10).std()

                    # Align with prediction index
                    common_index = pred_df.index.intersection(actual_volatility.index)
                    pred_df.loc[common_index, 'Actual_Volatility'] = actual_volatility.loc[common_index]
            except Exception as e:
                st.warning(f"Could not calculate actual volatility for comparison: {str(e)}")

            # Add price data for context
            try:
                if 'Close' in st.session_state.data.columns:
                    common_index = pred_df.index.intersection(st.session_state.data.index)
                    pred_df.loc[common_index, 'Close_Price'] = st.session_state.data.loc[common_index, 'Close']
            except Exception as e:
                st.warning(f"Could not add price data: {str(e)}")

            # Store predictions in session state
            st.session_state.predictions = pred_df

            st.success(f"✅ Generated {len(predictions)} volatility predictions!")

# Display predictions if available
if hasattr(st.session_state, 'predictions') and st.session_state.predictions is not None:
    pred_df = st.session_state.predictions

    st.header("📊 Volatility Forecast Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_volatility = pred_df['Volatility_Forecast'].mean()
        st.metric("Average Predicted Volatility", f"{avg_volatility:.4f}")

    with col2:
        max_volatility = pred_df['Volatility_Forecast'].max()
        st.metric("Maximum Volatility", f"{max_volatility:.4f}")

    with col3:
        min_volatility = pred_df['Volatility_Forecast'].min()
        st.metric("Minimum Volatility", f"{min_volatility:.4f}")

    with col4:
        volatility_std = pred_df['Volatility_Forecast'].std()
        st.metric("Volatility Std Dev", f"{volatility_std:.4f}")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Volatility Chart", "📊 Analysis", "📋 Data Table"])

    with tab1:
        st.subheader("📈 Volatility Forecast Over Time")

        fig = go.Figure()

        # Add predicted volatility
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Volatility_Forecast'],
            mode='lines',
            name='Predicted Volatility',
            line=dict(color='blue', width=2)
        ))

        # Add actual volatility if available
        if 'Actual_Volatility' in pred_df.columns:
            actual_data = pred_df['Actual_Volatility'].dropna()
            if not actual_data.empty:
                fig.add_trace(go.Scatter(
                    x=actual_data.index,
                    y=actual_data.values,
                    mode='lines',
                    name='Actual Volatility',
                    line=dict(color='red', width=2, dash='dash')
                ))

        fig.update_layout(
            height=500,
            title="Volatility Forecast vs Actual",
            xaxis_title="Date",
            yaxis_title="Volatility",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Price chart with volatility overlay if price data available
        if 'Close_Price' in pred_df.columns:
            st.subheader("💰 Price Movement with Volatility Forecast")

            # Create subplot with secondary y-axis
            from plotly.subplots import make_subplots

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])

            # Add price
            price_data = pred_df['Close_Price'].dropna()
            fig2.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data.values,
                    name="Price",
                    line=dict(color='green')
                ),
                secondary_y=False,
            )

            # Add volatility
            fig2.add_trace(
                go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Volatility_Forecast'],
                    name="Predicted Volatility",
                    line=dict(color='orange', dash='dot')
                ),
                secondary_y=True,
            )

            # Set y-axes titles
            fig2.update_yaxes(title_text="Price ($)", secondary_y=False)
            fig2.update_yaxes(title_text="Volatility", secondary_y=True)

            fig2.update_layout(
                height=500,
                title="Price and Volatility Forecast",
                xaxis_title="Date"
            )

            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("📊 Volatility Analysis")

        # Volatility distribution
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Volatility Distribution**")
            fig_hist = px.histogram(
                pred_df, 
                x='Volatility_Forecast',
                bins=30,
                title="Distribution of Predicted Volatility"
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.write("**Volatility Statistics**")
            volatility_stats = pred_df['Volatility_Forecast'].describe()
            st.dataframe(volatility_stats.to_frame().T, use_container_width=True)

        # Accuracy metrics if actual volatility is available
        if 'Actual_Volatility' in pred_df.columns:
            actual_clean = pred_df['Actual_Volatility'].dropna()
            pred_clean = pred_df.loc[actual_clean.index, 'Volatility_Forecast']

            if len(actual_clean) > 0:
                st.subheader("🎯 Prediction Accuracy")

                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                mse = mean_squared_error(actual_clean, pred_clean)
                mae = mean_absolute_error(actual_clean, pred_clean)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual_clean, pred_clean)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("MSE", f"{mse:.6f}")
                with col2:
                    st.metric("MAE", f"{mae:.6f}")
                with col3:
                    st.metric("RMSE", f"{rmse:.6f}")
                with col4:
                    st.metric("R² Score", f"{r2:.4f}")

                # Scatter plot of predicted vs actual
                fig_scatter = px.scatter(
                    x=actual_clean,
                    y=pred_clean,
                    title="Predicted vs Actual Volatility",
                    labels={'x': 'Actual Volatility', 'y': 'Predicted Volatility'}
                )

                # Add perfect prediction line
                min_val = min(actual_clean.min(), pred_clean.min())
                max_val = max(actual_clean.max(), pred_clean.max())
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))

                st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.subheader("📋 Volatility Predictions Data")

        # Helper function to safely format dates
        def safe_format_date(timestamp):
            try:
                if pd.isna(timestamp):
                    return ""
                if hasattr(timestamp, 'strftime'):
                    return timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    return str(timestamp)
            except:
                return str(timestamp)

        # Format dataframe for display
        display_df = pred_df.copy()

        # Reset index to show timestamps as a column
        display_df = display_df.reset_index()

        if len(display_df.columns) > 0:
            # Format timestamp column
            timestamp_col = display_df.columns[0]
            display_df['Date'] = display_df[timestamp_col].apply(safe_format_date)

            # Remove original timestamp column if it's not 'Date'
            if timestamp_col != 'Date':
                display_df = display_df.drop(columns=[timestamp_col])

        # Round numerical columns
        for col in display_df.columns:
            if col != 'Date' and display_df[col].dtype in ['float64', 'float32']:
                display_df[col] = display_df[col].round(6)

        # Calculate prediction error if both columns exist
        if 'Volatility_Forecast' in display_df.columns and 'Actual_Volatility' in display_df.columns:
            pred_clean = pd.to_numeric(display_df['Volatility_Forecast'], errors='coerce')
            actual_clean = pd.to_numeric(display_df['Actual_Volatility'], errors='coerce')

            valid_mask = pd.notna(pred_clean) & pd.notna(actual_clean)
            display_df['Volatility_Error'] = np.nan
            display_df.loc[valid_mask, 'Volatility_Error'] = np.abs(pred_clean[valid_mask] - actual_clean[valid_mask])
            display_df['Volatility_Error'] = display_df['Volatility_Error'].round(6)

        # Show the dataframe
        st.dataframe(display_df, use_container_width=True, height=400)

        # Download button
        if len(display_df) > 0:
            csv_data = display_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Volatility Predictions CSV",
                data=csv_data,
                file_name=f"volatility_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download all {len(display_df)} prediction records"
            )

            st.info(f"📊 Ready to download: {len(display_df)} rows of volatility prediction data")

else:
    st.info("👆 Click the button above to generate volatility predictions")

# Model performance section
if hasattr(st.session_state, 'trained_models') and 'volatility' in st.session_state.trained_models:
    st.header("🎯 Model Performance")

    model_info = st.session_state.trained_models['volatility']

    if 'metrics' in model_info:
        metrics = model_info['metrics']

        col1, col2, col3 = st.columns(3)

        with col1:
            rmse = metrics.get('rmse', 0)
            st.metric("Training RMSE", f"{rmse:.4f}")

        with col2:
            mae = metrics.get('mae', 0)
            st.metric("Training MAE", f"{mae:.4f}")

        with col3:
            mse = metrics.get('mse', 0)
            st.metric("Training MSE", f"{mse:.4f}")

    # Feature importance
    if 'feature_importance' in model_info and model_info['feature_importance']:
        st.subheader("🎯 Feature Importance")

        importance_data = model_info['feature_importance']
        importance_df = pd.DataFrame(
            list(importance_data.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)

        # Show feature importance chart
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Volatility Model Feature Importance"
        )
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)