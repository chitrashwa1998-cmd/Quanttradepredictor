
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

st.set_page_config(page_title="Model Validation", page_icon="🔍", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="trading-header">
    <h1 style="margin:0;">🔍 MODEL VALIDATION</h1>
    <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
        Comprehensive Model Quality Assessment
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("❌ No data available. Please upload data first in the Data Upload page.")
    st.stop()

# Validation Controls
st.header("🔧 Validation Controls")

col1, col2, col3 = st.columns(3)

with col1:
    validation_type = st.selectbox(
        "Validation Type",
        ["Quick Validation", "Comprehensive Validation", "Performance Analysis"]
    )

with col2:
    data_subset = st.selectbox(
        "Data Subset",
        ["Last 1000 rows", "Last 5000 rows", "All data"]
    )

with col3:
    if st.button("🚀 Run Validation", type="primary"):
        # Prepare data subset
        if data_subset == "Last 1000 rows":
            validation_data = st.session_state.data.tail(1000)
        elif data_subset == "Last 5000 rows":
            validation_data = st.session_state.data.tail(5000)
        else:
            validation_data = st.session_state.data
        
        st.info(f"Running {validation_type} on {len(validation_data)} rows...")
        
        # Initialize validator
        from model_validation import ModelValidator
        validator = ModelValidator()
        
        # Run validation
        with st.spinner("Validating models..."):
            validation_results = validator.validate_all_models(validation_data)
            
            # Store results in session state
            st.session_state.validation_results = validation_results
            st.session_state.validation_timestamp = datetime.now()
        
        st.success("✅ Validation completed!")
        st.rerun()

# Display validation results
if 'validation_results' in st.session_state:
    results = st.session_state.validation_results
    
    # Overall Assessment
    if "overall_assessment" in results:
        assessment = results["overall_assessment"]
        
        st.header("📊 Overall Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = {
                "excellent": "🟢",
                "good": "🟡", 
                "needs_improvement": "🔴"
            }.get(assessment['status'], "⚪")
            st.metric("Status", f"{status_color} {assessment['status'].title()}")
        
        with col2:
            st.metric("Overall Score", f"{assessment['overall_score']:.1%}")
        
        with col3:
            st.metric("Total Models", assessment['total_models'])
        
        with col4:
            good_models = assessment['good_models']
            total_models = assessment['total_models']
            st.metric("Good Models", f"{good_models}/{total_models}")
        
        # Message and recommendations
        st.info(f"**Assessment:** {assessment['message']}")
        
        if assessment['recommendations']:
            st.subheader("🎯 Recommendations")
            for rec in assessment['recommendations']:
                st.write(f"• {rec}")
    
    # Individual Model Results
    st.header("📈 Individual Model Results")
    
    model_results = {k: v for k, v in results.items() 
                    if isinstance(v, dict) and 'overall_score' in v}
    
    if model_results:
        # Create model comparison chart
        models = list(model_results.keys())
        scores = [model_results[model]['overall_score'] for model in models]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=scores,
                text=[f"{score:.1%}" for score in scores],
                textposition='auto',
                marker_color=['green' if score >= 0.8 else 'orange' if score >= 0.6 else 'red' for score in scores]
            )
        ])
        
        fig.update_layout(
            title="Model Performance Scores",
            xaxis_title="Models",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed model information
        for model_name, result in model_results.items():
            with st.expander(f"🤖 {model_name.title()} Model Details"):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Score", f"{result['overall_score']:.1%}")
                    st.metric("Tests Passed", result['tests_passed'])
                    st.metric("Tests Failed", result['tests_failed'])
                
                with col2:
                    st.metric("Features", result['feature_count'])
                    st.metric("Data Points", result['data_points'])
                    if 'performance_quality' in result:
                        quality_color = {
                            "Good": "🟢",
                            "Fair": "🟡",
                            "Poor": "🔴"
                        }.get(result['performance_quality'], "⚪")
                        st.metric("Quality", f"{quality_color} {result['performance_quality']}")
                
                with col3:
                    if 'feature_alignment' in result:
                        st.metric("Feature Alignment", f"{result['feature_alignment']:.1%}")
                    if 'prediction_count' in result:
                        st.metric("Predictions", result['prediction_count'])
                    if 'unique_predictions' in result:
                        st.metric("Unique Predictions", result['unique_predictions'])
                
                # Performance metrics
                if 'performance_metrics' in result:
                    st.subheader("📊 Performance Metrics")
                    metrics = result['performance_metrics']
                    
                    if model_name == 'volatility':
                        if 'rmse' in metrics:
                            st.write(f"**RMSE:** {metrics['rmse']:.6f}")
                        if 'mae' in metrics:
                            st.write(f"**MAE:** {metrics['mae']:.6f}")
                        if 'mse' in metrics:
                            st.write(f"**MSE:** {metrics['mse']:.6f}")
                    else:
                        if 'accuracy' in metrics:
                            st.write(f"**Accuracy:** {metrics['accuracy']:.2%}")
                        if 'precision' in metrics:
                            st.write(f"**Precision:** {metrics['precision']:.2%}")
                        if 'recall' in metrics:
                            st.write(f"**Recall:** {metrics['recall']:.2%}")
                
                # Feature importance
                if 'top_features' in result:
                    st.subheader("🔝 Top Features")
                    for feature, importance in result['top_features'].items():
                        st.write(f"**{feature}:** {importance:.4f}")
                
                # Warnings
                if result.get('warnings'):
                    st.subheader("⚠️ Warnings")
                    for warning in result['warnings']:
                        st.warning(warning)
                
                # Errors
                if result.get('errors'):
                    st.subheader("❌ Errors")
                    for error in result['errors']:
                        st.error(error)
    
    # Model Comparison Analysis
    st.header("📊 Model Comparison Analysis")
    
    if len(model_results) > 1:
        comparison_data = []
        for model_name, result in model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Overall Score': result['overall_score'],
                'Tests Passed': result['tests_passed'],
                'Tests Failed': result['tests_failed'],
                'Feature Count': result['feature_count'],
                'Performance Quality': result.get('performance_quality', 'Unknown')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Model feature distribution
        feature_counts = [result['feature_count'] for result in model_results.values()]
        model_names = list(model_results.keys())
        
        fig_features = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=feature_counts,
                name='Feature Count',
                text=feature_counts,
                textposition='auto'
            )
        ])
        
        fig_features.update_layout(
            title="Feature Count by Model",
            xaxis_title="Models",
            yaxis_title="Number of Features",
            height=400
        )
        
        st.plotly_chart(fig_features, use_container_width=True)
    
    # Export Results
    st.header("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download Validation Report"):
            from model_validation import ModelValidator
            validator = ModelValidator()
            report = validator.generate_validation_report(results)
            
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"model_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📋 Download JSON Results"):
            json_data = json.dumps(results, indent=2, default=str)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"model_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Model Improvement Suggestions
    st.header("🔧 Model Improvement Suggestions")
    
    poor_models = [name for name, result in model_results.items() 
                  if result['overall_score'] < 0.6]
    
    if poor_models:
        st.warning(f"Models needing improvement: {', '.join(poor_models)}")
        
        improvement_suggestions = {
            'volatility': [
                "Increase training data size",
                "Add more volatility-specific features",
                "Tune hyperparameters",
                "Use ensemble methods"
            ],
            'direction': [
                "Balance class distribution",
                "Add momentum indicators",
                "Use cross-validation",
                "Feature selection optimization"
            ],
            'profit_probability': [
                "Optimize probability thresholds",
                "Add risk-adjusted features",
                "Use stratified sampling",
                "Ensemble multiple models"
            ],
            'reversal': [
                "Add pattern recognition features",
                "Increase lookback periods",
                "Use support/resistance levels",
                "Optimize for rare events"
            ]
        }
        
        for model in poor_models:
            if model in improvement_suggestions:
                st.subheader(f"📈 {model.title()} Model Improvements")
                for suggestion in improvement_suggestions[model]:
                    st.write(f"• {suggestion}")
    
    else:
        st.success("🎉 All models are performing well!")
    
    # Validation timestamp
    if 'validation_timestamp' in st.session_state:
        st.caption(f"Last validation: {st.session_state.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.info("👆 Click 'Run Validation' to start comprehensive model assessment")
    
    # Quick model status check
    st.header("📋 Quick Model Status")
    
    from models.model_manager import ModelManager
    model_manager = ModelManager()
    
    trained_models = model_manager.get_trained_models()
    
    if trained_models:
        st.success(f"✅ Found {len(trained_models)} trained models: {', '.join(trained_models)}")
        
        for model_name in trained_models:
            model_info = model_manager.get_model_info(model_name)
            if model_info:
                with st.expander(f"🤖 {model_name.title()} Model Info"):
                    st.write(f"**Features:** {len(model_info.get('feature_names', []))}")
                    st.write(f"**Task Type:** {model_info.get('task_type', 'Unknown')}")
                    
                    metrics = model_info.get('metrics', {})
                    if metrics:
                        st.write("**Metrics:**")
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                st.write(f"  - {key}: {value:.4f}")
    else:
        st.warning("⚠️ No trained models found. Please train models first.")

# Tips for model improvement
st.markdown("---")
st.subheader("💡 Model Quality Tips")

st.markdown("""
**What makes a good trading model:**
- **Accuracy > 60%** for classification models
- **RMSE < 0.01** for volatility models
- **Feature diversity** - not overly dependent on few features
- **Consistent predictions** - not predicting only one class
- **Proper feature alignment** - model expects the right features

**Red flags to watch for:**
- Models predicting only one class
- Very high feature concentration
- Missing feature importance
- Poor feature alignment
- Low accuracy/high RMSE

**Improvement strategies:**
- More diverse training data
- Better feature engineering
- Hyperparameter tuning
- Ensemble methods
- Cross-validation
""")
