import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import streamlit as st

class DataProcessor:
    """Utility class for data processing and validation."""

    @staticmethod
    def validate_ohlc_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate OHLC data format and quality."""
        required_columns = ['Open', 'High', 'Low', 'Close']

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            available_cols = list(df.columns)
            return False, f"Missing required columns: {missing_columns}. Available columns: {available_cols}"

        # Check for non-numeric data
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_count = df[col].apply(lambda x: not pd.api.types.is_number(x)).sum()
                return False, f"Column {col} must be numeric. Found {non_numeric_count} non-numeric values."

        # Check for NaN values
        for col in required_columns:
            nan_count = df[col].isna().sum()
            if nan_count > len(df) * 0.1:  # More than 10% NaN values
                return False, f"Column {col} has too many missing values: {nan_count}/{len(df)} ({nan_count/len(df)*100:.1f}%)"

        # Check for negative or zero prices
        for col in required_columns:
            invalid_values = (df[col] <= 0).sum()
            if invalid_values > 0:
                return False, f"Column {col} contains {invalid_values} non-positive values. All prices must be > 0."

        # Check OHLC logic
        invalid_high_low = (df['High'] < df['Low']).sum()
        if invalid_high_low > 0:
            return False, f"Found {invalid_high_low} rows where High < Low"

        # Check OHLC logic
        invalid_high_low = (df['High'] < df['Low']).sum()
        if invalid_high_low > 0:
            return False, f"Found {invalid_high_low} rows where High < Low"

        invalid_high_open = (df['High'] < df['Open']).sum()
        if invalid_high_open > 0:
            return False, f"Found {invalid_high_open} rows where High < Open"

        invalid_high_close = (df['High'] < df['Close']).sum()
        if invalid_high_close > 0:
            return False, f"Found {invalid_high_close} rows where High < Close"

        invalid_low_open = (df['Low'] > df['Open']).sum()
        if invalid_low_open > 0:
            return False, f"Found {invalid_low_open} rows where Low > Open"

        invalid_low_close = (df['Low'] > df['Close']).sum()
        if invalid_low_close > 0:
            return False, f"Found {invalid_low_close} rows where Low > Close"

        # Check for sufficient data
        if len(df) < 10:
            return False, f"Insufficient data: {len(df)} rows. Need at least 10 rows for basic validation."

        # Check for reasonable price ranges (detect potential data issues)
        for col in required_columns:
            price_range = df[col].max() / df[col].min()
            if price_range > 1000:  # Prices vary by more than 1000x
                return False, f"Suspicious price range in {col}: min={df[col].min():.2f}, max={df[col].max():.2f}. Please verify data quality."

        return True, f"Data validation passed. {len(df)} rows of valid OHLC data."

    @staticmethod
    def load_and_process_data(uploaded_file) -> Tuple[pd.DataFrame, str]:
        """Load and process uploaded OHLC data."""
        try:
            # Validate file object
            if uploaded_file is None:
                return None, "No file provided"

            # Check if file has content
            if uploaded_file.size == 0:
                return None, "File is empty"

            # Check file size limit (25MB for processing)
            if uploaded_file.size > 25 * 1024 * 1024:
                return None, "File too large for processing. Please use a smaller file or contact support."

            # Reset file pointer to beginning
            uploaded_file.seek(0)

            # Validate file content by reading first few bytes
            try:
                first_bytes = uploaded_file.read(1024)
                uploaded_file.seek(0)

                # Check if file contains readable content
                if not first_bytes:
                    return None, "File appears to be empty or corrupted"

                # Basic CSV validation - should contain commas or semicolons
                first_line = first_bytes.decode('utf-8', errors='ignore').split('\n')[0]
                if ',' not in first_line and ';' not in first_line and '\t' not in first_line:
                    return None, "File doesn't appear to be a valid CSV format"

            except Exception as validation_error:
                return None, f"File validation failed: {str(validation_error)}"

            # Try different encodings and separators
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            separators = [',', ';', '\t']

            df = None
            successful_config = None

            for encoding in encodings:
                for sep in separators:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep, low_memory=False)
                        if len(df.columns) >= 4 and len(df) > 0:  # At least Date, Open, High, Low, Close and has data
                            successful_config = f"encoding={encoding}, separator='{sep}'"
                            break
                    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
                        continue
                    except Exception as e:
                        # Log other exceptions for debugging
                        print(f"Error reading with {encoding}, {sep}: {str(e)}")
                        continue
                if df is not None and len(df.columns) >= 4 and len(df) > 0:
                    break

            if df is None:
                return None, "Could not read CSV file. Please check file format, encoding, and separator. Try saving your CSV file with UTF-8 encoding and comma separators."

            # Clean column names (remove spaces, make case-insensitive)
            df.columns = df.columns.str.strip().str.lower()

            # Map common column name variations
            column_mapping = {
                'datetime': 'date',
                'timestamp': 'date',
                'time': 'date',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vol': 'volume',
                'adj close': 'close',
                'adj_close': 'close'
            }

            # Apply column mapping
            df.columns = [column_mapping.get(col, col) for col in df.columns]

            # Ensure we have required columns
            required_base_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_base_cols if col not in df.columns]

            if missing_cols:
                available_cols = list(df.columns)
                return None, f"Missing required columns: {missing_cols}. Available columns: {available_cols}"

            # Capitalize column names for consistency
            df.columns = [col.capitalize() for col in df.columns]

            # Handle date column
            date_col = None
            for col in df.columns:
                if col.lower() in ['date', 'datetime', 'timestamp', 'time']:
                    date_col = col
                    break

            if date_col:
                # Convert to datetime with multiple format attempts
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)

                # Remove rows with invalid dates
                initial_rows = len(df)
                df = df.dropna(subset=[date_col])

                if len(df) == 0:
                    return None, "All date values are invalid. Please check your date format."

                if len(df) < initial_rows:
                    print(f"Warning: Removed {initial_rows - len(df)} rows with invalid dates")

                # Validate that we have actual datetime values, not just times
                sample_date = df[date_col].iloc[0]
                if sample_date.date() == pd.Timestamp('1900-01-01').date():
                    return None, "Date column contains only time values. Please provide full datetime (YYYY-MM-DD HH:MM:SS) or date (YYYY-MM-DD) format."

                # Set as index
                df.set_index(date_col, inplace=True)
                df.index.name = 'DateTime'

                # Sort by datetime to ensure proper chronological order
                df = df.sort_index()

            else:
                return None, "No date/datetime column found. Please ensure your CSV has a column named 'Date', 'DateTime', 'Timestamp', or 'Time' with proper datetime values."

            # Remove any duplicate dates
            df = df[~df.index.duplicated(keep='first')]

            # Sort by date
            df.sort_index(inplace=True)

            # Convert price columns to numeric
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Handle Volume column if present
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

            # Remove rows with all NaN values in price columns
            df = df.dropna(subset=price_cols, how='all')

            if len(df) == 0:
                return None, "No valid data rows found after processing"

            # Validate data
            is_valid, message = DataProcessor.validate_ohlc_data(df)

            if not is_valid:
                return None, f"Data validation failed: {message}"

            success_msg = f"Data loaded successfully using {successful_config}. Processed {len(df)} rows."
            return df, success_msg

        except Exception as e:
            return None, f"Error loading data: {str(e)}. Please check that your file is a valid CSV with OHLC columns."

    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        summary = {
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'days': (df.index.max() - df.index.min()).days
            },
            'price_summary': {
                'min_close': df['Close'].min(),
                'max_close': df['Close'].max(),
                'mean_close': df['Close'].mean(),
                'std_close': df['Close'].std()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'columns': list(df.columns)
        }

        # Calculate returns
        returns = df['Close'].pct_change().dropna()
        summary['returns'] = {
            'mean_daily_return': returns.mean(),
            'volatility': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_daily_return': returns.max(),
            'min_daily_return': returns.min()
        }

        return summary

    @staticmethod
    def detect_data_frequency(df: pd.DataFrame) -> str:
        """Detect the frequency of the data (daily, hourly, etc.)."""
        if len(df) < 2:
            return "Unknown"

        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()

        if median_diff <= pd.Timedelta(minutes=5):
            return "5-minute or less"
        elif median_diff <= pd.Timedelta(minutes=15):
            return "15-minute"
        elif median_diff <= pd.Timedelta(hours=1):
            return "Hourly"
        elif median_diff <= pd.Timedelta(hours=4):
            return "4-hour"
        elif median_diff <= pd.Timedelta(days=1):
            return "Daily"
        elif median_diff <= pd.Timedelta(days=7):
            return "Weekly"
        else:
            return "Monthly or longer"

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers."""
        df_clean = df.copy()

        # Forward fill missing values
        df_clean = df_clean.ffill()

        # Remove any remaining NaN values
        df_clean = df_clean.dropna()

        # Remove extreme outliers (beyond 3 standard deviations)
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_clean.columns:
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                outlier_mask = np.abs(df_clean[col] - mean_val) > 3 * std_val

                if outlier_mask.sum() > 0:
                    st.warning(f"Removed {outlier_mask.sum()} outliers from {col}")
                    # Replace outliers with median
                    df_clean.loc[outlier_mask, col] = df_clean[col].median()

        return df_clean