
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

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
    filter_option = st.selectbox(
        "📅 Time Period Filter",
        ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All data"],
        index=1,
        help="Select the time period for predictions"
    )

with col2:
    st.metric("Model Status", "✅ Ready", help="Volatility model is trained and ready")

# Generate predictions button
if st.button("🚀 Generate Volatility Predictions", type="primary"):
    try:
        with st.spinner("Generating volatility predictions..."):
            # Get model trainer
            model_trainer = st.session_state.model_trainer

            # Filter data based on selection
            data_for_prediction = st.session_state.features.copy()
            
            # Convert index to datetime if needed
            if not isinstance(data_for_prediction.index, pd.DatetimeIndex):
                try:
                    data_for_prediction.index = pd.to_datetime(data_for_prediction.index)
                except:
                    st.error("❌ Could not convert data index to datetime")
                    st.stop()

            # Apply time filter
            if filter_option == "Last 30 days":
                start_date = data_for_prediction.index.max() - timedelta(days=30)
            elif filter_option == "Last 90 days":
                start_date = data_for_prediction.index.max() - timedelta(days=90)
            elif filter_option == "Last 6 months":
                start_date = data_for_prediction.index.max() - timedelta(days=180)
            elif filter_option == "Last year":
                start_date = data_for_prediction.index.max() - timedelta(days=365)
            else:  # All data
                start_date = data_for_prediction.index.min()

            filtered_data = data_for_prediction[data_for_prediction.index >= start_date]

            if filtered_data.empty:
                st.error("❌ No data available for the selected time period")
                st.stop()

            st.info(f"📊 Making predictions on {len(filtered_data)} data points from {start_date.strftime('%Y-%m-%d')} to {filtered_data.index.max().strftime('%Y-%m-%d')}")

            # Make predictions
            predictions, probabilities = model_trainer.predict('volatility', filtered_data)

            # Create results dataframe
            pred_df = pd.DataFrame({
                'Volatility_Forecast': predictions
            }, index=filtered_data.index)

            # Add actual volatility for comparison if possible
            try:
                # Calculate actual volatility from original data
                if 'Close' in st.session_state.data.columns:
                    returns = st.session_state.data['Close'].pct_change()
                    actual_volatility = returns.rolling(10).std()

                    # Align with prediction index
                    common_index = pred_df.index.intersection(actual_volatility.index)
                    pred_df.loc[common_index, 'Actual_Volatility'] = actual_volatility.loc[common_index]
            except Exception as e:
                st.warning(f"Could not calculate actual volatility: {str(e)}")

            # Add price data for context
            try:
                if 'Close' in st.session_state.data.columns:
                    common_index = pred_df.index.intersection(st.session_state.data.index)
                    pred_df.loc[common_index, 'Close_Price'] = st.session_state.data.loc[common_index, 'Close']
                    pred_df.loc[common_index, 'Open_Price'] = st.session_state.data.loc[common_index, 'Open']
                    pred_df.loc[common_index, 'High_Price'] = st.session_state.data.loc[common_index, 'High']
                    pred_df.loc[common_index, 'Low_Price'] = st.session_state.data.loc[common_index, 'Low']
            except Exception as e:
                st.warning(f"Could not add price data: {str(e)}")

            # Store predictions in session state
            st.session_state.predictions = pred_df
            st.session_state.prediction_filter = filter_option

            st.success(f"✅ Generated {len(predictions)} volatility predictions for {filter_option.lower()}!")

    except Exception as e:
        st.error(f"❌ Failed to generate predictions: {str(e)}")
        st.info("Please check your model training and data processing.")

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
        st.metric("Volatility Range", f"{volatility_std:.4f}")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Volatility Chart", "💰 Price & Volatility", "📊 Analysis", "📋 Data Table"])

    with tab1:
        st.subheader("📈 Volatility Forecast Over Time")

        fig = go.Figure()

        # Add predicted volatility
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Volatility_Forecast'],
            mode='lines+markers',
            name='Predicted Volatility',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))

        # Add actual volatility if available
        if 'Actual_Volatility' in pred_df.columns:
            actual_data = pred_df['Actual_Volatility'].dropna()
            if not actual_data.empty:
                fig.add_trace(go.Scatter(
                    x=actual_data.index,
                    y=actual_data.values,
                    mode='lines+markers',
                    name='Actual Volatility',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=4)
                ))

        fig.update_layout(
            height=500,
            title="Volatility Forecast vs Actual",
            xaxis_title="Date",
            yaxis_title="Volatility",
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Volatility distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                pred_df, 
                x='Volatility_Forecast',
                nbins=30,
                title="Predicted Volatility Distribution",
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot for volatility
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=pred_df['Volatility_Forecast'],
                name='Predicted Volatility',
                marker_color='#1f77b4'
            ))
            
            if 'Actual_Volatility' in pred_df.columns:
                actual_clean = pred_df['Actual_Volatility'].dropna()
                if not actual_clean.empty:
                    fig_box.add_trace(go.Box(
                        y=actual_clean.values,
                        name='Actual Volatility',
                        marker_color='#ff7f0e'
                    ))
            
            fig_box.update_layout(
                height=350,
                title="Volatility Distribution Comparison",
                yaxis_title="Volatility"
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        st.subheader("💰 Price Movement with Volatility Forecast")

        if 'Close_Price' in pred_df.columns:
            # Create subplot with secondary y-axis
            fig2 = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=('Price Chart', 'Volatility Forecast'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Add candlestick chart if OHLC data available
            if all(col in pred_df.columns for col in ['Open_Price', 'High_Price', 'Low_Price', 'Close_Price']):
                price_data = pred_df[['Open_Price', 'High_Price', 'Low_Price', 'Close_Price']].dropna()
                if not price_data.empty:
                    fig2.add_trace(
                        go.Candlestick(
                            x=price_data.index,
                            open=price_data['Open_Price'],
                            high=price_data['High_Price'],
                            low=price_data['Low_Price'],
                            close=price_data['Close_Price'],
                            name="Price",
                            showlegend=False
                        ),
                        row=1, col=1
                    )
            else:
                # Add price line if only close price available
                price_data = pred_df['Close_Price'].dropna()
                if not price_data.empty:
                    fig2.add_trace(
                        go.Scatter(
                            x=price_data.index,
                            y=price_data.values,
                            name="Price",
                            line=dict(color='#2ca02c', width=2),
                            showlegend=False
                        ),
                        row=1, col=1
                    )

            # Add volatility in second subplot
            fig2.add_trace(
                go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Volatility_Forecast'],
                    name="Predicted Volatility",
                    line=dict(color='#1f77b4', width=2),
                    fill='tonexty',
                    showlegend=False
                ),
                row=2, col=1
            )

            # Add actual volatility if available
            if 'Actual_Volatility' in pred_df.columns:
                actual_data = pred_df['Actual_Volatility'].dropna()
                if not actual_data.empty:
                    fig2.add_trace(
                        go.Scatter(
                            x=actual_data.index,
                            y=actual_data.values,
                            name="Actual Volatility",
                            line=dict(color='#ff7f0e', width=2, dash='dash'),
                            showlegend=False
                        ),
                        row=2, col=1
                    )

            fig2.update_layout(
                height=600,
                title="Price Movement and Volatility Analysis",
                xaxis_title="Date",
                showlegend=True
            )

            fig2.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig2.update_yaxes(title_text="Volatility", row=2, col=1)

            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("Price data not available. Please ensure your data includes price columns.")

    with tab3:
        st.subheader("📊 Volatility Analysis & Accuracy")

        # Volatility statistics
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Volatility Statistics**")
            volatility_stats = pred_df['Volatility_Forecast'].describe()
            stats_df = pd.DataFrame({
                'Statistic': volatility_stats.index,
                'Value': [f"{val:.6f}" for val in volatility_stats.values]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with col2:
            if 'Actual_Volatility' in pred_df.columns:
                actual_clean = pred_df['Actual_Volatility'].dropna()
                pred_clean = pred_df.loc[actual_clean.index, 'Volatility_Forecast']

                if len(actual_clean) > 10:
                    st.write("**Prediction Accuracy Metrics**")
                    
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                    mse = mean_squared_error(actual_clean, pred_clean)
                    mae = mean_absolute_error(actual_clean, pred_clean)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(actual_clean, pred_clean)

                    accuracy_df = pd.DataFrame({
                        'Metric': ['MSE', 'MAE', 'RMSE', 'R² Score'],
                        'Value': [f"{mse:.6f}", f"{mae:.6f}", f"{rmse:.6f}", f"{r2:.4f}"]
                    })
                    st.dataframe(accuracy_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Not enough data points for accuracy calculation")
            else:
                st.info("Actual volatility data not available for comparison")

        # Accuracy visualization if available
        if 'Actual_Volatility' in pred_df.columns:
            actual_clean = pred_df['Actual_Volatility'].dropna()
            pred_clean = pred_df.loc[actual_clean.index, 'Volatility_Forecast']

            if len(actual_clean) > 10:
                st.subheader("🎯 Prediction Accuracy Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
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

                    fig_scatter.update_layout(height=400)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Residuals plot
                    residuals = pred_clean - actual_clean
                    fig_residuals = px.scatter(
                        x=actual_clean,
                        y=residuals,
                        title="Prediction Residuals",
                        labels={'x': 'Actual Volatility', 'y': 'Residuals (Predicted - Actual)'}
                    )
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_residuals.update_layout(height=400)
                    st.plotly_chart(fig_residuals, use_container_width=True)

    with tab4:
        st.subheader("📋 Volatility Predictions Data Table")

        # Format dataframe for display
        display_df = pred_df.copy()
        display_df = display_df.reset_index()

        # Format datetime column
        date_col_created = False
        if len(display_df.columns) > 0:
            timestamp_col = display_df.columns[0]
            try:
                display_df['Date'] = pd.to_datetime(display_df[timestamp_col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df = display_df.drop(columns=[timestamp_col])
                date_col_created = True
            except Exception as e:
                # If datetime conversion fails, keep the original column name
                display_df = display_df.rename(columns={timestamp_col: 'Date'})
                date_col_created = True

        # Round numerical columns
        numeric_columns = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Date':
                display_df[col] = display_df[col].round(6)

        # Calculate prediction error if both columns exist
        if 'Volatility_Forecast' in display_df.columns and 'Actual_Volatility' in display_df.columns:
            actual_mask = pd.notna(display_df['Actual_Volatility'])
            display_df['Prediction_Error'] = np.nan
            display_df.loc[actual_mask, 'Prediction_Error'] = np.abs(
                display_df.loc[actual_mask, 'Volatility_Forecast'] - 
                display_df.loc[actual_mask, 'Actual_Volatility']
            ).round(6)

        # Reorder columns with Date first if it exists
        if date_col_created and 'Date' in display_df.columns:
            cols = ['Date'] + [col for col in display_df.columns if col != 'Date']
            display_df = display_df[cols]

        # Show data with pagination
        st.dataframe(
            display_df, 
            use_container_width=True, 
            height=400,
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn("Date", width="medium"),
                "Volatility_Forecast": st.column_config.NumberColumn("Predicted Volatility", format="%.6f"),
                "Actual_Volatility": st.column_config.NumberColumn("Actual Volatility", format="%.6f"),
                "Prediction_Error": st.column_config.NumberColumn("Prediction Error", format="%.6f"),
                "Close_Price": st.column_config.NumberColumn("Close Price", format="$%.2f")
            }
        )

        # Summary information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(display_df))
        with col2:
            if 'Actual_Volatility' in display_df.columns:
                actual_count = display_df['Actual_Volatility'].notna().sum()
                st.metric("Records with Actual Data", actual_count)
            else:
                st.metric("Forecast Records", len(display_df))
        with col3:
            time_span = (pd.to_datetime(display_df['Date']).max() - pd.to_datetime(display_df['Date']).min()).days
            st.metric("Time Span (Days)", time_span)

        # Download button
        if len(display_df) > 0:
            csv_data = display_df.to_csv(index=False)
            
            filter_name = st.session_state.get('prediction_filter', 'predictions').lower().replace(' ', '_')
            filename = f"volatility_{filter_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                help=f"Download all {len(display_df)} prediction records"
            )

else:
    st.info("👆 Select a time period and click the button above to generate volatility predictions")

# Model performance section
if hasattr(st.session_state, 'trained_models') and 'volatility' in st.session_state.trained_models:
    st.header("🎯 Model Performance Summary")

    model_info = st.session_state.trained_models['volatility']

    col1, col2 = st.columns(2)
    
    with col1:
        if 'metrics' in model_info:
            metrics = model_info['metrics']
            
            st.subheader("📊 Training Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'MSE'],
                'Value': [
                    f"{metrics.get('rmse', 0):.6f}",
                    f"{metrics.get('mae', 0):.6f}",
                    f"{metrics.get('mse', 0):.6f}"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col2:
        # Feature importance
        if 'feature_importance' in model_info and model_info['feature_importance']:
            st.subheader("🎯 Top Features")
            
            importance_data = model_info['feature_importance']
            importance_df = pd.DataFrame(
                list(importance_data.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False).head(10)

            # Show top 10 features
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features"
            )
            fig_importance.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=400
            )
            st.plotly_chart(fig_importance, use_container_width=True)
