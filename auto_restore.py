"""
Automatic model restoration system for TribexAlpha trading app
This script ensures trained models persist between app restarts
"""

import streamlit as st
from datetime import datetime

def auto_restore_system():
    """Automatically restore data and models on app startup"""
    
    # Only run once per session
    if getattr(st.session_state, 'auto_restore_complete', False):
        return
    
    try:
        from utils.database_adapter import DatabaseAdapter
        from models.xgboost_models import QuantTradingModels
        from features.technical_indicators import TechnicalIndicators
        
        db = TradingDatabase()
        restoration_status = []
        
        # 1. Restore OHLC data
        if not hasattr(st.session_state, 'data') or st.session_state.data is None:
            data = db.load_ohlc_data("main_dataset")
            if data is not None and len(data) > 1000:
                st.session_state.data = data
                restoration_status.append(f"Data: {len(data)} rows")
        
        # 2. Restore features
        if not hasattr(st.session_state, 'features') or st.session_state.features is None:
            if hasattr(st.session_state, 'data') and st.session_state.data is not None:
                try:
                    features = TechnicalIndicators.calculate_all_indicators(st.session_state.data)
                    st.session_state.features = features.dropna()
                    restoration_status.append(f"Features: {len(st.session_state.features)} rows")
                except Exception:
                    pass
        
        # 3. Restore model trainer
        if not hasattr(st.session_state, 'model_trainer') or st.session_state.model_trainer is None:
            st.session_state.model_trainer = QuantTradingModels()
        
        # 4. Restore trained models
        if not hasattr(st.session_state, 'models') or not st.session_state.models:
            st.session_state.models = {}
            
            # Load trained model objects
            try:
                trained_models = db.load_trained_models()
                if trained_models and st.session_state.model_trainer:
                    st.session_state.model_trainer.models = trained_models
                    
                    # Load model results for display
                    model_names = ['direction', 'magnitude', 'profit_prob', 'volatility', 'trend_sideways', 'reversal', 'trading_signal']
                    restored_count = 0
                    
                    for model_name in model_names:
                        model_results = db.load_model_results(model_name)
                        if model_results and 'metrics' in model_results:
                            st.session_state.models[model_name] = model_results['metrics']
                            restored_count += 1
                    
                    if restored_count > 0:
                        restoration_status.append(f"Models: {restored_count} trained models")
                        
            except Exception as e:
                print(f"Model restoration error: {str(e)}")
        
        # Mark restoration as complete
        st.session_state.auto_restore_complete = True
        
        # Show restoration status if anything was restored
        if restoration_status:
            st.success(f"System automatically restored: {', '.join(restoration_status)}")
            
    except Exception as e:
        print(f"Auto-restore error: {str(e)}")
        st.session_state.auto_restore_complete = True

if __name__ == "__main__":
    auto_restore_system()