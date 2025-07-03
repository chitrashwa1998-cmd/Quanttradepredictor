
#!/usr/bin/env python3
"""
Comprehensive Model Validation System
Tests model quality, performance, and reliability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelValidator:
    """Comprehensive model validation and testing"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_all_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive validation on all trained models"""
        
        print("🔍 Starting comprehensive model validation...")
        
        # Load model manager
        from models.model_manager import ModelManager
        model_manager = ModelManager()
        
        validation_results = {}
        
        # Get trained models
        trained_models = model_manager.get_trained_models()
        print(f"Found {len(trained_models)} trained models: {trained_models}")
        
        if not trained_models:
            print("❌ No trained models found!")
            return {"error": "No trained models available"}
        
        # Validate each model
        for model_name in trained_models:
            print(f"\n📊 Validating {model_name} model...")
            
            try:
                # Get model info
                model_info = model_manager.get_model_info(model_name)
                
                # Prepare features for this model
                features = self._prepare_model_features(data, model_name)
                
                if features is None or len(features) == 0:
                    print(f"❌ Could not prepare features for {model_name}")
                    continue
                
                # Run validation tests
                validation_result = self._validate_single_model(
                    model_manager, model_name, features, data
                )
                
                validation_results[model_name] = validation_result
                
            except Exception as e:
                print(f"❌ Error validating {model_name}: {str(e)}")
                validation_results[model_name] = {"error": str(e)}
        
        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(validation_results)
        validation_results["overall_assessment"] = overall_assessment
        
        return validation_results
    
    def _prepare_model_features(self, data: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Prepare features specific to each model"""
        
        try:
            if model_name == 'volatility':
                from features.technical_indicators import TechnicalIndicators
                features = TechnicalIndicators.calculate_all_indicators(data)
                return features
                
            elif model_name == 'direction':
                from features.direction_technical_indicators import DirectionTechnicalIndicators
                features = DirectionTechnicalIndicators.calculate_all_direction_indicators(data)
                return features
                
            elif model_name == 'profit_probability':
                from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
                features = ProfitProbabilityTechnicalIndicators.calculate_all_profit_probability_indicators(data)
                return features
                
            elif model_name == 'reversal':
                from models.reversal_model import ReversalModel
                reversal_model = ReversalModel()
                features = reversal_model.prepare_features(data)
                return features
                
            else:
                print(f"Unknown model type: {model_name}")
                return None
                
        except Exception as e:
            print(f"Error preparing features for {model_name}: {str(e)}")
            return None
    
    def _validate_single_model(self, model_manager, model_name: str, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate a single model comprehensively"""
        
        validation_result = {
            "model_name": model_name,
            "feature_count": len(features.columns),
            "data_points": len(features),
            "tests_passed": 0,
            "tests_failed": 0,
            "warnings": [],
            "errors": []
        }
        
        # Test 1: Model Loading
        print(f"  ✓ Test 1: Model Loading...")
        try:
            model_info = model_manager.get_model_info(model_name)
            if model_info and ('model' in model_info or 'ensemble' in model_info):
                validation_result["tests_passed"] += 1
                validation_result["model_loaded"] = True
            else:
                validation_result["tests_failed"] += 1
                validation_result["errors"].append("Model not properly loaded")
        except Exception as e:
            validation_result["tests_failed"] += 1
            validation_result["errors"].append(f"Model loading error: {str(e)}")
        
        # Test 2: Feature Alignment
        print(f"  ✓ Test 2: Feature Alignment...")
        try:
            model_info = model_manager.get_model_info(model_name)
            expected_features = model_info.get('feature_names', [])
            
            if expected_features:
                available_features = [f for f in expected_features if f in features.columns]
                feature_alignment = len(available_features) / len(expected_features)
                
                validation_result["feature_alignment"] = feature_alignment
                validation_result["expected_features"] = len(expected_features)
                validation_result["available_features"] = len(available_features)
                
                if feature_alignment >= 0.8:
                    validation_result["tests_passed"] += 1
                else:
                    validation_result["tests_failed"] += 1
                    validation_result["errors"].append(f"Poor feature alignment: {feature_alignment:.2%}")
            else:
                validation_result["warnings"].append("No expected features found")
                
        except Exception as e:
            validation_result["tests_failed"] += 1
            validation_result["errors"].append(f"Feature alignment error: {str(e)}")
        
        # Test 3: Prediction Generation
        print(f"  ✓ Test 3: Prediction Generation...")
        try:
            # Use last 100 rows for testing
            test_features = features.tail(100)
            predictions, probabilities = model_manager.predict(model_name, test_features)
            
            if predictions is not None and len(predictions) > 0:
                validation_result["tests_passed"] += 1
                validation_result["predictions_generated"] = True
                validation_result["prediction_count"] = len(predictions)
                
                # Check prediction distribution
                unique_predictions = np.unique(predictions)
                validation_result["unique_predictions"] = len(unique_predictions)
                validation_result["prediction_distribution"] = {
                    str(val): int(count) for val, count in zip(*np.unique(predictions, return_counts=True))
                }
                
                # Check for prediction diversity
                if len(unique_predictions) == 1:
                    validation_result["warnings"].append("Model only predicts one class")
                
            else:
                validation_result["tests_failed"] += 1
                validation_result["errors"].append("Could not generate predictions")
                
        except Exception as e:
            validation_result["tests_failed"] += 1
            validation_result["errors"].append(f"Prediction generation error: {str(e)}")
        
        # Test 4: Performance Metrics
        print(f"  ✓ Test 4: Performance Metrics...")
        try:
            model_info = model_manager.get_model_info(model_name)
            metrics = model_info.get('metrics', {})
            
            if metrics:
                validation_result["performance_metrics"] = metrics
                validation_result["tests_passed"] += 1
                
                # Check metric quality
                if model_name == 'volatility':
                    rmse = metrics.get('rmse', float('inf'))
                    if rmse < 0.01:  # Good volatility prediction
                        validation_result["performance_quality"] = "Good"
                    elif rmse < 0.02:
                        validation_result["performance_quality"] = "Fair"
                    else:
                        validation_result["performance_quality"] = "Poor"
                        validation_result["warnings"].append(f"High RMSE: {rmse:.4f}")
                
                else:  # Classification models
                    accuracy = metrics.get('accuracy', 0)
                    if accuracy > 0.7:
                        validation_result["performance_quality"] = "Good"
                    elif accuracy > 0.6:
                        validation_result["performance_quality"] = "Fair"
                    else:
                        validation_result["performance_quality"] = "Poor"
                        validation_result["warnings"].append(f"Low accuracy: {accuracy:.2%}")
            else:
                validation_result["tests_failed"] += 1
                validation_result["errors"].append("No performance metrics available")
                
        except Exception as e:
            validation_result["tests_failed"] += 1
            validation_result["errors"].append(f"Performance metrics error: {str(e)}")
        
        # Test 5: Feature Importance
        print(f"  ✓ Test 5: Feature Importance...")
        try:
            feature_importance = model_manager.get_feature_importance(model_name)
            
            if feature_importance:
                validation_result["feature_importance_available"] = True
                validation_result["top_features"] = dict(list(feature_importance.items())[:5])
                validation_result["tests_passed"] += 1
                
                # Check for feature concentration
                top_5_importance = sum(list(feature_importance.values())[:5])
                if top_5_importance > 0.8:
                    validation_result["warnings"].append("High feature concentration in top 5 features")
                    
            else:
                validation_result["tests_failed"] += 1
                validation_result["errors"].append("No feature importance available")
                
        except Exception as e:
            validation_result["tests_failed"] += 1
            validation_result["errors"].append(f"Feature importance error: {str(e)}")
        
        # Calculate overall score
        total_tests = validation_result["tests_passed"] + validation_result["tests_failed"]
        if total_tests > 0:
            validation_result["overall_score"] = validation_result["tests_passed"] / total_tests
        else:
            validation_result["overall_score"] = 0
        
        return validation_result
    
    def _generate_overall_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of all models"""
        
        model_results = {k: v for k, v in validation_results.items() if isinstance(v, dict) and 'overall_score' in v}
        
        if not model_results:
            return {"status": "failed", "message": "No valid model results"}
        
        # Calculate overall statistics
        scores = [result['overall_score'] for result in model_results.values()]
        avg_score = np.mean(scores)
        
        # Count models by quality
        good_models = sum(1 for score in scores if score >= 0.8)
        fair_models = sum(1 for score in scores if 0.6 <= score < 0.8)
        poor_models = sum(1 for score in scores if score < 0.6)
        
        # Generate recommendations
        recommendations = []
        
        if avg_score >= 0.8:
            status = "excellent"
            message = "Your models are performing excellently!"
        elif avg_score >= 0.6:
            status = "good"
            message = "Your models are performing well with some room for improvement."
            recommendations.append("Consider retraining models with poor performance")
        else:
            status = "needs_improvement"
            message = "Your models need significant improvement."
            recommendations.append("Retrain models with more data and better features")
            recommendations.append("Check for data quality issues")
        
        # Specific recommendations
        for model_name, result in model_results.items():
            if result['overall_score'] < 0.6:
                recommendations.append(f"Retrain {model_name} model - current score: {result['overall_score']:.2%}")
            
            if result.get('warnings'):
                recommendations.append(f"Address warnings in {model_name} model")
        
        return {
            "status": status,
            "message": message,
            "overall_score": avg_score,
            "total_models": len(model_results),
            "good_models": good_models,
            "fair_models": fair_models,
            "poor_models": poor_models,
            "recommendations": recommendations
        }
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report"""
        
        report = []
        report.append("=" * 60)
        report.append("🔍 MODEL VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall Assessment
        if "overall_assessment" in validation_results:
            assessment = validation_results["overall_assessment"]
            report.append(f"\n📊 OVERALL ASSESSMENT: {assessment['status'].upper()}")
            report.append(f"Message: {assessment['message']}")
            report.append(f"Overall Score: {assessment['overall_score']:.2%}")
            report.append(f"Total Models: {assessment['total_models']}")
            report.append(f"Good Models: {assessment['good_models']}")
            report.append(f"Fair Models: {assessment['fair_models']}")
            report.append(f"Poor Models: {assessment['poor_models']}")
            
            if assessment['recommendations']:
                report.append("\n🎯 RECOMMENDATIONS:")
                for rec in assessment['recommendations']:
                    report.append(f"  • {rec}")
        
        # Individual Model Results
        report.append("\n" + "=" * 60)
        report.append("📈 INDIVIDUAL MODEL RESULTS")
        report.append("=" * 60)
        
        for model_name, result in validation_results.items():
            if isinstance(result, dict) and 'overall_score' in result:
                report.append(f"\n🤖 {model_name.upper()} MODEL")
                report.append(f"Overall Score: {result['overall_score']:.2%}")
                report.append(f"Tests Passed: {result['tests_passed']}")
                report.append(f"Tests Failed: {result['tests_failed']}")
                report.append(f"Features: {result['feature_count']}")
                report.append(f"Data Points: {result['data_points']}")
                
                if result.get('performance_quality'):
                    report.append(f"Performance Quality: {result['performance_quality']}")
                
                if result.get('warnings'):
                    report.append("⚠️  Warnings:")
                    for warning in result['warnings']:
                        report.append(f"    • {warning}")
                
                if result.get('errors'):
                    report.append("❌ Errors:")
                    for error in result['errors']:
                        report.append(f"    • {error}")
        
        return "\n".join(report)


def main():
    """Run comprehensive model validation"""
    
    print("🚀 Starting Model Validation System...")
    
    # Load data
    from utils.database_adapter import get_trading_database
    db = get_trading_database()
    
    data = db.load_ohlc_data("main_dataset")
    
    if data is None or len(data) == 0:
        print("❌ No data available for validation")
        return
    
    print(f"✅ Loaded {len(data)} rows of data")
    
    # Initialize validator
    validator = ModelValidator()
    
    # Run validation
    validation_results = validator.validate_all_models(data)
    
    # Generate and print report
    report = validator.generate_validation_report(validation_results)
    print("\n" + report)
    
    # Save results
    import json
    with open('model_validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n💾 Validation results saved to 'model_validation_results.json'")


if __name__ == "__main__":
    main()
