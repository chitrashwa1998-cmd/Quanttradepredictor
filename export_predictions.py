
#!/usr/bin/env python3
"""
Direct prediction export script - bypasses Streamlit UI
"""

import pandas as pd
import numpy as np
from utils.database_adapter import DatabaseAdapter
from models.model_manager import ModelManager

def export_all_predictions():
    """Export predictions for all trained models"""
    
    print("🔄 Starting direct prediction export...")
    
    # Initialize components
    db = DatabaseAdapter()
    model_manager = ModelManager()
    
    # Load data
    print("📊 Loading OHLC data...")
    data = db.load_ohlc_data()
    if data is None:
        print("❌ No data found in database")
        return
    
    print(f"✅ Loaded {len(data)} records")
    
    # Export predictions for each model
    models = ['volatility', 'direction', 'profit_probability', 'reversal']
    
    for model_name in models:
        try:
            if model_manager.is_model_trained(model_name):
                print(f"\n🔮 Generating {model_name} predictions...")
                
                if model_name == 'volatility':
                    from features.technical_indicators import TechnicalIndicators
                    ti = TechnicalIndicators()
                    features = ti.calculate_all_indicators(data)
                elif model_name == 'direction':
                    from features.direction_technical_indicators import DirectionTechnicalIndicators
                    dti = DirectionTechnicalIndicators()
                    features = dti.calculate_all_direction_indicators(data)
                elif model_name == 'profit_probability':
                    from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
                    features = ProfitProbabilityTechnicalIndicators.calculate_all_profit_probability_indicators(data)
                elif model_name == 'reversal':
                    from models.reversal_model import ReversalModel
                    reversal_model = ReversalModel()
                    features = reversal_model.prepare_features(data)
                
                # Make predictions
                predictions, probabilities = model_manager.predict(model_name, features)
                
                # Create output DataFrame
                result_df = pd.DataFrame({
                    'DateTime': features.index,
                    'Prediction': predictions
                })
                
                # Add probabilities if available
                if probabilities is not None:
                    if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                        result_df['Confidence'] = np.max(probabilities, axis=1)
                        if model_name in ['direction', 'profit_probability', 'reversal']:
                            result_df['Probability_Class_0'] = probabilities[:, 0]
                            result_df['Probability_Class_1'] = probabilities[:, 1]
                    else:
                        result_df['Confidence'] = probabilities.flatten()
                
                # Add human-readable labels
                if model_name == 'direction':
                    result_df['Direction_Label'] = result_df['Prediction'].map({0: 'Bearish', 1: 'Bullish'})
                elif model_name == 'profit_probability':
                    result_df['Profit_Label'] = result_df['Prediction'].map({0: 'Low Profit', 1: 'High Profit'})
                elif model_name == 'reversal':
                    result_df['Reversal_Label'] = result_df['Prediction'].map({0: 'No Reversal', 1: 'Reversal'})
                
                # Export to CSV
                filename = f"{model_name}_predictions.csv"
                result_df.to_csv(filename, index=False)
                
                print(f"✅ Exported {len(result_df)} {model_name} predictions to {filename}")
                
                # Show sample
                print(f"📋 Sample predictions:")
                print(result_df.head(3).to_string(index=False))
                
            else:
                print(f"⚠️  {model_name} model not trained - skipping")
                
        except Exception as e:
            print(f"❌ Error exporting {model_name}: {str(e)}")
    
    print("\n🎉 Export complete! Files saved in current directory.")

if __name__ == "__main__":
    export_all_predictions()
