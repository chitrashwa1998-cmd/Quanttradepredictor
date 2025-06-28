
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.database_adapter import DatabaseAdapter
from datetime import datetime

st.set_page_config(page_title="Database Manager", page_icon="💾", layout="wide")

# Initialize session state variables
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

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">💾 DATA CONTROL CENTER</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Database Management & Storage
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize database
trading_db = DatabaseAdapter()

# Database overview
st.header("📊 Database Overview")

db_info = trading_db.get_database_info()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Datasets", db_info.get('total_datasets', 0))

with col2:
    st.metric("Total Keys", db_info.get('total_keys', 0))

with col3:
    if st.button("🔄 Refresh", help="Refresh database information"):
        st.rerun()

# Datasets management
st.header("📈 Saved Datasets")

datasets = db_info.get('datasets', [])

if len(datasets) > 0:
    st.success(f"Found {len(datasets)} dataset(s) in database")
    for i, dataset in enumerate(datasets):
        with st.expander(f"📊 {dataset['name']} ({dataset['rows']} rows)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Rows:** {dataset['rows']}")
                # Handle date range display safely
                if dataset.get('start_date') and dataset.get('end_date'):
                    st.write(f"**Date Range:** {dataset['start_date']} to {dataset['end_date']}")
                else:
                    st.write(f"**Date Range:** Not available")
                
                # Handle saved/created timestamp
                if dataset.get('updated_at'):
                    st.write(f"**Updated:** {dataset['updated_at']}")
                elif dataset.get('created_at'):
                    st.write(f"**Created:** {dataset['created_at']}")
                else:
                    st.write(f"**Saved:** Not available")
            
            with col2:
                if st.button(f"Load Dataset", key=f"load_{i}"):
                    loaded_data = trading_db.load_ohlc_data(dataset['name'])
                    if loaded_data is not None:
                        st.session_state.data = loaded_data
                        st.success(f"✅ Loaded dataset: {dataset['name']}")
                        st.rerun()
                    else:
                        st.error("Failed to load dataset")
                
                if st.button(f"Preview Data", key=f"preview_{i}"):
                    preview_data = trading_db.load_ohlc_data(dataset['name'])
                    if preview_data is not None:
                        st.subheader(f"Preview of {dataset['name']}")
                        st.dataframe(preview_data.head(10), use_container_width=True)
                    else:
                        st.error("Failed to load preview")
                
                # Export button for each dataset
                export_data = trading_db.load_ohlc_data(dataset['name'])
                if export_data is not None:
                    csv_data = export_data.to_csv()
                    st.download_button(
                        label="📥 Export CSV",
                        data=csv_data,
                        file_name=f"{dataset['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"export_{i}"
                    )
            
            with col3:
                # Initialize confirmation state
                confirm_key = f"confirm_delete_{dataset['name']}"
                if confirm_key not in st.session_state:
                    st.session_state[confirm_key] = False
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_{i}", type="secondary"):
                    st.session_state[confirm_key] = True
                
                # Show confirmation if delete was clicked
                if st.session_state[confirm_key]:
                    st.warning(f"⚠️ Delete '{dataset['name']}'?")
                    col3a, col3b = st.columns(2)
                    
                    with col3a:
                        if st.button("✅ Yes", key=f"confirm_yes_{i}", type="primary"):
                            if trading_db.delete_dataset(dataset['name']):
                                st.success(f"✅ Deleted dataset: {dataset['name']}")
                                # Reset confirmation state
                                st.session_state[confirm_key] = False
                                st.rerun()
                            else:
                                st.error("Failed to delete dataset")
                                st.session_state[confirm_key] = False
                    
                    with col3b:
                        if st.button("❌ No", key=f"confirm_no_{i}"):
                            st.session_state[confirm_key] = False
                            st.rerun()
else:
    st.info("No datasets found in database. Upload data first!")

# Model results management
st.header("🤖 Model Results")

model_keys = [key for key in db_info.get('available_keys', []) if key.startswith('model_results_')]

if model_keys:
    for key in model_keys:
        model_name = key.replace('model_results_', '')
        results = trading_db.load_model_results(model_name)
        
        if results:
            with st.expander(f"🎯 {model_name} Model"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'accuracy' in results:
                        st.metric("Accuracy", f"{results['accuracy']:.4f}")
                    if 'precision' in results:
                        st.metric("Precision", f"{results['precision']:.4f}")
                
                with col2:
                    if 'recall' in results:
                        st.metric("Recall", f"{results['recall']:.4f}")
                    if 'f1' in results:
                        st.metric("F1 Score", f"{results['f1']:.4f}")
                
                if st.button(f"Delete {model_name} Results", key=f"delete_model_{model_name}"):
                    try:
                        # For PostgreSQL, we need to implement delete methods
                        success = trading_db.db.delete_model_results(model_name)
                        
                        if success:
                            st.success(f"✅ Deleted {model_name} model results")
                            st.rerun()
                        else:
                            st.error("Failed to delete model results")
                    except Exception as e:
                        st.error(f"Failed to delete model results: {str(e)}")
else:
    st.info("No model results found. Train models first!")

# Predictions management
st.header("🔮 Saved Predictions")

pred_keys = [key for key in db_info.get('available_keys', []) if key.startswith('predictions_')]

if pred_keys:
    for key in pred_keys:
        model_name = key.replace('predictions_', '')
        
        with st.expander(f"📈 {model_name} Predictions"):
            predictions = trading_db.load_predictions(model_name)
            
            if predictions is not None:
                st.write(f"**Shape:** {predictions.shape}")
                st.write(f"**Columns:** {', '.join(predictions.columns)}")
                
                if st.button(f"View {model_name} Predictions", key=f"view_pred_{model_name}"):
                    st.dataframe(predictions.head(20), use_container_width=True)
                
                if st.button(f"Delete {model_name} Predictions", key=f"delete_pred_{model_name}"):
                    try:
                        success = trading_db.db.delete_predictions(model_name)
                        if success:
                            st.success(f"✅ Deleted {model_name} predictions")
                            st.rerun()
                        else:
                            st.error("Failed to delete predictions")
                    except Exception as e:
                        st.error(f"Failed to delete predictions: {str(e)}")
else:
    st.info("No predictions found. Generate predictions first!")

# Database maintenance
st.header("🔧 Database Maintenance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Export Data")
    
    # Export current session data
    if st.session_state.data is not None:
        csv_data = st.session_state.data.to_csv()
        st.download_button(
            label="📥 Download Current Dataset as CSV",
            data=csv_data,
            file_name=f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Export all datasets from database
    if db_info.get('total_datasets', 0) > 0:
        st.write("**Export from Database:**")
        # Get datasets with proper error handling
        datasets_for_export = db_info.get('datasets', [])
        if datasets_for_export:
            selected_datasets = st.multiselect(
                "Select datasets to export",
                [dataset['name'] for dataset in datasets_for_export],
                key="bulk_export_selection"
            )
        else:
            selected_datasets = []
            st.info("No datasets available for export")
        
        if selected_datasets:
            if st.button("📥 Export Selected Datasets", key="bulk_export"):
                for dataset_name in selected_datasets:
                    export_data = trading_db.load_ohlc_data(dataset_name)
                    if export_data is not None:
                        csv_data = export_data.to_csv()
                        st.download_button(
                            label=f"📥 Download {dataset_name}",
                            data=csv_data,
                            file_name=f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key=f"bulk_download_{dataset_name}"
                        )

with col2:
    st.subheader("⚠️ Danger Zone")
    
    # Session data clearing
    if st.button("🧹 Clear Session Data", type="secondary", help="Clear all session state variables"):
        # Clear all session state thoroughly
        keys_to_clear = [
            'data', 'features', 'models', 'predictions', 'model_trainer',
            'trained_models', 'direction_features', 'direction_trained_models',
            'direction_predictions', 'direction_probabilities', 'volatility_predictions',
            'auto_recovery_done', 'backtest_results', 'backtest_signals'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Also clear any other prediction-related keys
        for key in list(st.session_state.keys()):
            if any(term in key.lower() for term in ['prediction', 'model', 'feature', 'train']):
                del st.session_state[key]
        
        st.success("✅ Session data cleared successfully!")
        st.rerun()
    
    st.markdown("---")
    
    # Initialize confirmation state
    if 'confirm_clear_all' not in st.session_state:
        st.session_state.confirm_clear_all = False
    
    if st.button("🗑️ Clear All Database", type="secondary"):
        st.session_state.confirm_clear_all = True
    
    if st.session_state.confirm_clear_all:
        st.warning("⚠️ This will permanently delete ALL data from the database!")
        st.write("This includes:")
        st.write("- All datasets")
        st.write("- All trained models")
        st.write("- All model results")
        st.write("- All predictions")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            if st.button("✅ Yes, Delete Everything", type="primary", key="confirm_clear_yes"):
                with st.spinner("Clearing database..."):
                    success = trading_db.clear_all_data()
                    
                if success:
                    st.success("✅ All database data cleared successfully")
                    
                    # Clear all session state thoroughly
                    keys_to_clear = [
                        'data', 'features', 'models', 'predictions', 'model_trainer',
                        'trained_models', 'direction_features', 'direction_trained_models',
                        'direction_predictions', 'direction_probabilities', 'volatility_predictions',
                        'auto_recovery_done', 'backtest_results', 'backtest_signals',
                        'confirm_clear_all'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Also clear any other prediction-related keys
                    for key in list(st.session_state.keys()):
                        if any(term in key.lower() for term in ['prediction', 'model', 'feature', 'train']):
                            del st.session_state[key]
                    
                    # Reset models dictionary
                    st.session_state.models = {}
                    
                    # Reset confirmation state
                    st.session_state.confirm_clear_all = False
                    
                    # Force page refresh
                    st.rerun()
                else:
                    st.error("❌ Failed to clear database. Please check console for details.")
                    st.session_state.confirm_clear_all = False
        
        with col2b:
            if st.button("❌ Cancel", key="confirm_clear_no"):
                st.session_state.confirm_clear_all = False
                st.rerun()

# Data Recovery Section
st.header("🔄 Data Recovery")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Check Session Data")
    if 'data' in st.session_state and st.session_state.data is not None:
        st.success(f"✅ Session data exists: {len(st.session_state.data)} rows")
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            if st.button("💾 Save Session Data to Database"):
                if trading_db.save_ohlc_data(st.session_state.data, "recovered_data"):
                    st.success("✅ Session data saved to database as 'recovered_data'")
                    st.rerun()
                else:
                    st.error("Failed to save session data")
        
        with col1b:
            if st.button("🗑️ Clear Session Data", type="secondary"):
                # Clear all session state data
                st.session_state.data = None
                st.session_state.features = None
                st.session_state.models = {}
                st.session_state.predictions = None
                st.session_state.model_trainer = None
                
                st.success("✅ Session data cleared")
                st.rerun()
    else:
        st.warning("⚠️ No data in current session")

with col2:
    st.subheader("Auto-save Settings")
    auto_save = st.checkbox("Auto-save uploaded data", value=True)
    if auto_save:
        st.info("New uploads will be automatically saved to database")

# Raw database view (for debugging)
with st.expander("🔍 Raw Database View (Debug)"):
    st.write("**All Keys:**")
    try:
        # Get keys through the database info method
        db_info = trading_db.get_database_info()
        all_keys = db_info.get('available_keys', [])
        st.write(all_keys)
        
        # Show database size
        st.write(f"**Total Keys:** {len(all_keys)}")
        st.write(f"**Database Type:** {db_info.get('adapter_type', 'Unknown')}")
        
        if all_keys:
            selected_key = st.selectbox("Select key to inspect:", all_keys)
            if st.button("View Key Content"):
                try:
                    # Access through the underlying database object
                    if hasattr(trading_db.db, 'db'):
                        # For TradingDatabase (Key-Value store)
                        content = trading_db.db.db[selected_key]
                    else:
                        # For PostgreSQL or other databases
                        st.warning("Key inspection not available for this database type")
                        content = None
                    
                    if content is not None:
                        if isinstance(content, dict) and 'data' in content:
                            st.write(f"Data type: {type(content)}")
                            st.write(f"Keys: {list(content.keys())}")
                            if 'metadata' in content:
                                st.write("Metadata:")
                                st.json(content['metadata'])
                        else:
                            st.json(content)
                except Exception as e:
                    st.error(f"Error viewing key: {str(e)}")
    except Exception as e:
        st.error(f"Error accessing database keys: {str(e)}")
        st.write("Database connection may be unavailable or using incompatible format")
