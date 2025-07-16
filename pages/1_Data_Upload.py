import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_processing import DataProcessor
from features.technical_indicators import TechnicalIndicators
from utils.database_adapter import DatabaseAdapter

# Initialize database
trading_db = DatabaseAdapter()

st.set_page_config(page_title="Data Upload", page_icon="📊", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">📊 DATA UPLOAD CENTER</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Load and Process Market Data
    </p>
</div>
""", unsafe_allow_html=True)

# File upload section
st.header("Upload OHLC Data")
st.markdown("""
Upload your historical price data in CSV format. The file should contain columns for Date, Open, High, Low, Close, and optionally Volume.

**Supported formats:**
- Date formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY
- Column names: Date/Datetime, Open, High, Low, Close, Volume (case-insensitive)
""")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file with OHLC data"
)

# Data storage and purpose options
st.subheader("📊 Data Storage Options")
col1, col2 = st.columns(2)

with col1:
    preserve_full_data = st.checkbox(
        "Preserve Full Dataset", 

# Show current dataset configuration
st.sidebar.subheader("📋 Current Datasets")
try:
    training_datasets = trading_db.get_datasets_by_purpose('training')
    pre_seed_datasets = trading_db.get_datasets_by_purpose('pre_seed')
    
    if training_datasets:
        st.sidebar.success(f"🎯 Training: {training_datasets[0]['name']} ({training_datasets[0]['rows']} rows)")
    else:
        st.sidebar.info("🎯 No training dataset")
        
    if pre_seed_datasets:
        st.sidebar.success(f"🌱 Pre-seed: {pre_seed_datasets[0]['name']} ({pre_seed_datasets[0]['rows']} rows)")
    else:
        st.sidebar.info("🌱 No pre-seed dataset")
except:
    st.sidebar.info("📊 Upload datasets to see configuration")

with col1:
    preserve_full_data = st.checkbox(
        "Preserve Full Dataset",
        value=True,
        help="Keep all data points without sampling. Recommended for most datasets."
    )

with col2:
    if preserve_full_data:
        st.info("✅ Full dataset will be preserved")
    else:
        st.info("⚠️ Large datasets will be intelligently sampled (50k rows max)")

# Dataset purpose selection
st.subheader("🎯 Dataset Purpose")
col1, col2, col3 = st.columns(3)

with col1:
    dataset_purpose = st.selectbox(
        "Dataset Purpose",
        ["training", "pre_seed", "validation", "testing"],
        index=0,
        help="Select the purpose of this dataset"
    )

with col2:
    if dataset_purpose == "training":
        st.info("🎯 Used for model training")
    elif dataset_purpose == "pre_seed":
        st.info("🌱 Used for live data seeding")
    elif dataset_purpose == "validation":
        st.info("✅ Used for model validation")
    else:
        st.info("🧪 Used for model testing")

with col3:
    # Auto-generate dataset name based on purpose
    auto_dataset_name = st.text_input(
        "Dataset Name",
        value=f"{dataset_purpose}_dataset",
        help="Name for this dataset"
    )

if uploaded_file is not None:
    try:
        # Display file info
        st.info(f"**File Info**: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

        # Validate file size (limit to 50MB to prevent memory issues)
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("❌ File too large. Please upload a file smaller than 50MB.")
            st.error("💡 **Tip**: For large datasets, try splitting your data into smaller chunks or use data sampling.")
            st.stop()

        # Validate file type
        if not uploaded_file.name.lower().endswith('.csv'):
            st.error("❌ Please upload a CSV file.")
            st.stop()

        # Additional validation for empty files
        if uploaded_file.size == 0:
            st.error("❌ The uploaded file is empty.")
            st.stop()

        # Check if file is too small (likely corrupted)
        if uploaded_file.size < 100:
            st.error("❌ File is too small. Please check if the file is corrupted.")
            st.stop()

        with st.spinner("Loading and processing data..."):
            # Reset file pointer before processing
            uploaded_file.seek(0)
            
            # Add try-catch specifically for file processing
            try:
                df, message = DataProcessor.load_and_process_data(uploaded_file)
            except Exception as processing_error:
                st.error(f"❌ Error processing file: {str(processing_error)}")
                st.error("💡 **Common fixes:**")
                st.error("• Ensure your CSV uses comma separators")
                st.error("• Check that your file encoding is UTF-8")
                st.error("• Verify column names include Date, Open, High, Low, Close")
                st.error("• Remove any special characters from the file")
                st.stop()

        if df is not None:
            # Validate dataframe before storing
            if len(df) == 0:
                st.error("❌ Uploaded file contains no valid data rows.")
                st.stop()

            # Validate datetime index
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                st.error("❌ Data must have a valid datetime index. Please ensure your CSV has a proper Date/DateTime column.")
                st.stop()

            # Check for synthetic datetime patterns
            sample_datetime_str = str(df.index[0])
            is_synthetic = (
                any(pattern in sample_datetime_str for pattern in ['Data_', 'Point_']) or
                (sample_datetime_str == '09:15:00')  # Time only without date
            )
            
            if is_synthetic:
                st.error("❌ Invalid datetime values detected. Please upload data with proper datetime format (YYYY-MM-DD HH:MM:SS).")
                st.stop()

            # Clear existing session data properly
            st.session_state.data = df
            st.session_state.features = None
            st.session_state.models = {}
            st.session_state.predictions = None
            st.session_state.volatility_predictions = None
            st.session_state.direction_predictions = None
            st.session_state.direction_probabilities = None

            # Automatically save to database with error handling and retry logic
            try:
                from utils.database_adapter import DatabaseAdapter
                import time
                
                # Retry logic for database save (handles table creation timing issues)
                max_retries = 3
                save_success = False
                
                for attempt in range(max_retries):
                    try:
                        trading_db = DatabaseAdapter()
                        
                        # Test database connection first
                        if not trading_db._test_connection():
                            if attempt == max_retries - 1:
                                st.error("❌ Database connection failed. Please check your PostgreSQL setup.")
                                st.stop()
                            time.sleep(1)
                            continue
                        
                        # Always preserve full data for datasets under 100k rows
                        preserve_setting = preserve_full_data or len(df) < 100000
                        save_success = trading_db.save_ohlc_data(df, auto_dataset_name, preserve_setting, dataset_purpose)
                        if save_success:
                            break
                        elif attempt < max_retries - 1:
                            st.info(f"Database save attempt {attempt + 1} failed, retrying...")
                            time.sleep(1)
                    except Exception as retry_error:
                        if "does not exist" in str(retry_error) and attempt < max_retries - 1:
                            st.info(f"Database initializing, retrying save attempt {attempt + 1}...")
                            time.sleep(2)
                            continue
                        elif attempt == max_retries - 1:
                            raise retry_error
                
                if save_success:
                    # Verify data was actually saved by trying to load it back
                    verification_data = trading_db.load_ohlc_data(auto_dataset_name)
                    if verification_data is not None and len(verification_data) > 0:
                        actual_rows = len(verification_data)
                        original_rows = len(df)
                        
                        if actual_rows == original_rows:
                            st.success(f"✅ {message} & Full dataset '{auto_dataset_name}' saved for {dataset_purpose}! ({actual_rows} rows)")
                        else:
                            st.warning(f"⚠️ {message} & Dataset '{auto_dataset_name}' saved for {dataset_purpose} but may have been processed: {actual_rows} rows saved from {original_rows} original rows")
                        
                        # Show detailed database info
                        db_info = trading_db.get_database_info()
                        st.info(f"📊 Database now contains {db_info['total_datasets']} dataset(s) with {db_info['total_records']} total records")
                    else:
                        st.error("❌ Data save verification failed. Data was not properly stored.")
                        st.stop()
                else:
                    st.error("❌ Failed to save data to database. Please try again.")
                    st.stop()
                    
            except Exception as db_error:
                st.error(f"❌ Database error: {str(db_error)}")
                st.error("💡 **Try these solutions:**")
                st.error("• Check if PostgreSQL database is properly configured")
                st.error("• Verify DATABASE_URL environment variable is set")
                st.error("• Try refreshing the page and uploading again")
                st.stop()
            
            st.rerun()
    except Exception as upload_error:
        st.error(f"❌ Upload failed: {str(upload_error)}")
        st.error("Please check your file format and try again.")

        # Manual save to database
        st.subheader("💾 Save to Database")
        col1, col2 = st.columns([2, 1])

        with col1:
            dataset_name = st.text_input("Dataset name", value="main_dataset", key="dataset_name")

        with col2:
            if st.button("💾 Save to Database", type="primary"):
                with st.spinner("Saving to database..."):
                    if trading_db.save_ohlc_data(df, dataset_name, preserve_full_data):
                        if preserve_full_data:
                            st.success(f"✅ Full dataset saved to database as '{dataset_name}'")
                        else:
                            st.success(f"✅ Data saved to database as '{dataset_name}' (optimized)")
                    else:
                        st.error("❌ Failed to save data to database. Try with a smaller dataset or different name.")

        # Display data summary
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Summary")
            summary = DataProcessor.get_data_summary(df)

            st.metric("Total Records", summary['total_rows'])
            st.metric("Date Range", f"{summary['date_range']['days']} days")
            st.metric("Data Frequency", DataProcessor.detect_data_frequency(df))

            # Price statistics
            st.markdown("**Price Statistics:**")
            st.write(f"- Close Price Range: ${summary['price_summary']['min_close']:.2f} - ${summary['price_summary']['max_close']:.2f}")
            st.write(f"- Average Close: ${summary['price_summary']['mean_close']:.2f}")
            st.write(f"- Daily Volatility: {summary['returns']['volatility']:.2%}")
            st.write(f"- Sharpe Ratio: {summary['returns']['sharpe_ratio']:.2f}")

        with col2:
            st.subheader("Data Quality")

            # Missing values check
            missing_values = summary['missing_values']
            total_missing = sum(missing_values.values())

            if total_missing == 0:
                st.success("✅ No missing values detected")
            else:
                st.warning(f"⚠️ {total_missing} missing values found")
                for col, count in missing_values.items():
                    if count > 0:
                        st.write(f"- {col}: {count} missing")

            # Data validation
            is_valid, validation_message = DataProcessor.validate_ohlc_data(df)
            if is_valid:
                st.success("✅ Data validation passed")
            else:
                st.error(f"❌ {validation_message}")

        # Display raw data sample
        st.subheader("Data Preview")
        col1, col2 = st.columns([3, 1])

        with col2:
            show_rows = st.selectbox("Rows to display", [10, 25, 50, 100], index=0)

        with col1:
            st.dataframe(df.head(show_rows), use_container_width=True)

        # Price chart
        st.subheader("Price Chart")

        # Chart options
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"])
        with col2:
            time_range = st.selectbox("Time Range", ["All", "Last 30 days", "Last 90 days", "Last 365 days"])
        with col3:
            show_volume = st.checkbox("Show Volume", value=True if 'Volume' in df.columns else False)

        # Filter data based on time range
        if time_range != "All":
            days = int(time_range.split()[1])
            df_chart = df.tail(days) if len(df) > days else df
        else:
            df_chart = df

        # Create chart
        if chart_type == "Candlestick":
            fig = go.Figure(data=go.Candlestick(
                x=df_chart.index,
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name="Price"
            ))
        elif chart_type == "OHLC":
            fig = go.Figure(data=go.Ohlc(
                x=df_chart.index,
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name="Price"
            ))
        else:  # Line chart
            fig = go.Figure(data=go.Scatter(
                x=df_chart.index,
                y=df_chart['Close'],
                mode='lines',
                name='Close Price'
            ))

        # Add volume if requested and available
        if show_volume and 'Volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=('Price', 'Volume')
            )

            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=df_chart.index,
                    open=df_chart['Open'],
                    high=df_chart['High'],
                    low=df_chart['Low'],
                    close=df_chart['Close'],
                    name="Price"
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=df_chart.index,
                    y=df_chart['Close'],
                    mode='lines',
                    name='Close Price'
                ), row=1, col=1)

            fig.add_trace(go.Bar(
                x=df_chart.index,
                y=df_chart['Volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.6)'
            ), row=2, col=1)

        fig.update_layout(
            title=f"Price Chart - {time_range}",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Technical indicators preprocessing
        st.subheader("Technical Indicators Preprocessing")

        if st.button("Generate Technical Indicators", type="primary"):
            with st.spinner("Calculating technical indicators..."):
                df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)

                # Clean the data
                df_clean = DataProcessor.clean_data(df_with_indicators)

                # Update session state
                st.session_state.features = df_clean

            st.success("✅ Technical indicators calculated successfully!")

            # Show feature summary
            feature_cols = [col for col in df_clean.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

            st.info(f"Generated {len(feature_cols)} technical indicators")

            # Display feature columns
            with st.expander("View Generated Features"):
                col1, col2, col3 = st.columns(3)

                for i, feature in enumerate(feature_cols):
                    col = [col1, col2, col3][i % 3]
                    col.write(f"• {feature}")

            # Show correlation heatmap of some key features
            key_features = ['Close', 'rsi', 'macd', 'bb_position', 'volatility_10', 'price_momentum_5']
            available_features = [f for f in key_features if f in df_clean.columns]

            if len(available_features) > 2:
                st.subheader("Feature Correlation Matrix")
                corr_matrix = df_clean[available_features].corr()

                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0
                ))

                fig.update_layout(
                    title="Correlation Matrix of Key Features",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

        # Data cleaning options
        st.subheader("Data Cleaning Options")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clean Data", help="Remove outliers and handle missing values"):
                with st.spinner("Cleaning data..."):
                    df_clean = DataProcessor.clean_data(df)
                    st.session_state.data = df_clean

                st.success("✅ Data cleaned successfully!")
                st.rerun()

        with col2:
            if st.button("Reset to Original", help="Reset to originally uploaded data"):
                # Reload original data
                df_original, _ = DataProcessor.load_and_process_data(uploaded_file)
                st.session_state.data = df_original
                st.success("✅ Data reset to original")
                st.rerun()

        # Next steps
        st.markdown("---")
        st.info("📋 **Next Steps:** Once your data is loaded and processed, go to the **Model Training** page to train the XGBoost models.")

    else:
        st.error(f"❌ Error loading data: {message}")

        # Show troubleshooting section
        with st.expander("🔧 Troubleshooting Guide", expanded=True):
            st.markdown("""
            **Common issues and solutions:**

            1. **Column Names**: Ensure your CSV has columns named Date/Datetime, Open, High, Low, Close
               - Variations like 'O', 'H', 'L', 'C' are automatically detected
               - Column names are case-insensitive

            2. **File Format**: 
               - Use standard CSV format with comma separators
               - Try different separators (semicolon `;` or tab) if needed
               - Ensure file encoding is UTF-8

            3. **Date Format**: Supported formats include:
               - YYYY-MM-DD HH:MM:SS (e.g., 2023-01-01 09:30:00)
               - YYYY-MM-DD (e.g., 2023-01-01)
               - MM/DD/YYYY, DD/MM/YYYY

            4. **Data Quality**:
               - All price values must be positive numbers
               - High ≥ Low, High ≥ Open, High ≥ Close
               - Low ≤ Open, Low ≤ Close
               - Need at least 100 data rows

            5. **File Size**: Large files (>500MB) may take longer to process
            """)

            # Show first few lines of uploaded file for debugging
            if uploaded_file is not None:
                st.markdown("**File Preview (first 5 lines):**")
                try:
                    uploaded_file.seek(0)
                    preview_lines = []
                    for i, line in enumerate(uploaded_file):
                        if i >= 5:
                            break
                        preview_lines.append(line.decode('utf-8', errors='ignore').strip())
                    st.code('\n'.join(preview_lines))
                except Exception as e:
                    st.warning(f"Could not preview file: {e}")
                finally:
                    uploaded_file.seek(0)

else:
    st.info("👆 Please upload a CSV file with OHLC data to get started.")

    # Show sample data format
    st.subheader("Expected Data Format")

    sample_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Open': [100.0, 101.0, 102.0],
        'High': [105.0, 106.0, 107.0],
        'Low': [99.0, 100.0, 101.0],
        'Close': [104.0, 105.0, 106.0],
        'Volume': [1000000, 1100000, 1200000]
    })

    st.dataframe(sample_data, use_container_width=True)

    st.markdown("""
    **Column Requirements:**
    - **Date/Datetime**: Any standard date format
    - **Open**: Opening price (numeric)
    - **High**: Highest price (numeric)
    - **Low**: Lowest price (numeric)
    - **Close**: Closing price (numeric)
    - **Volume**: Trading volume (optional, numeric)
    """)