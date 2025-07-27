
#!/usr/bin/env python3
"""
Database Export Script for TribexAlpha
Exports all PostgreSQL data to local folder structure
"""

import os
import json
import pickle
import pandas as pd
from datetime import datetime
from utils.database_adapter import DatabaseAdapter

def create_export_folder():
    """Create export folder structure"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_dir = f"database_export_{timestamp}"
    
    # Create directory structure
    folders = [
        export_dir,
        f"{export_dir}/datasets",
        f"{export_dir}/models", 
        f"{export_dir}/metadata",
        f"{export_dir}/backup"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    return export_dir

def export_datasets(db, export_dir):
    """Export all datasets as CSV files"""
    print("📊 Exporting datasets...")
    
    try:
        # Get all datasets
        db_info = db.get_database_info()
        datasets = db_info.get('datasets', [])
        
        exported_datasets = []
        
        for dataset in datasets:
            dataset_name = dataset['name']
            print(f"  📥 Exporting {dataset_name}...")
            
            # Load dataset
            data = db.load_ohlc_data(dataset_name)
            
            if data is not None and len(data) > 0:
                # Save as CSV
                csv_file = f"{export_dir}/datasets/{dataset_name}.csv"
                data.to_csv(csv_file)
                
                # Create dataset info
                dataset_info = {
                    'name': dataset_name,
                    'rows': len(data),
                    'columns': list(data.columns),
                    'start_date': data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'file_path': csv_file
                }
                
                exported_datasets.append(dataset_info)
                print(f"    ✅ Exported {len(data)} rows to {csv_file}")
            else:
                print(f"    ⚠️ No data found for {dataset_name}")
        
        # Save datasets metadata
        datasets_meta_file = f"{export_dir}/metadata/datasets_info.json"
        with open(datasets_meta_file, 'w') as f:
            json.dump(exported_datasets, f, indent=2)
        
        print(f"✅ Exported {len(exported_datasets)} datasets")
        return exported_datasets
        
    except Exception as e:
        print(f"❌ Error exporting datasets: {str(e)}")
        return []

def export_trained_models(db, export_dir):
    """Export trained models"""
    print("🤖 Exporting trained models...")
    
    try:
        # Load trained models
        trained_models = db.load_trained_models()
        
        if trained_models:
            # Save models as pickle file
            models_file = f"{export_dir}/models/trained_models.pkl"
            with open(models_file, 'wb') as f:
                pickle.dump(trained_models, f)
            
            # Create models info
            models_info = {
                'total_models': len(trained_models),
                'model_names': list(trained_models.keys()),
                'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'file_path': models_file
            }
            
            # Save individual model details
            for model_name, model_obj in trained_models.items():
                try:
                    # Extract model information
                    model_details = {
                        'name': model_name,
                        'type': str(type(model_obj)),
                        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Try to get model-specific info
                    if hasattr(model_obj, 'feature_names_in_'):
                        model_details['features'] = list(model_obj.feature_names_in_)
                    
                    # Save individual model
                    individual_model_file = f"{export_dir}/models/{model_name}_model.pkl"
                    with open(individual_model_file, 'wb') as f:
                        pickle.dump(model_obj, f)
                    
                    model_details['individual_file'] = individual_model_file
                    
                    # Save model details
                    model_info_file = f"{export_dir}/models/{model_name}_info.json"
                    with open(model_info_file, 'w') as f:
                        json.dump(model_details, f, indent=2)
                    
                    print(f"    ✅ Exported {model_name} model")
                    
                except Exception as e:
                    print(f"    ⚠️ Error exporting {model_name}: {str(e)}")
            
            # Save overall models metadata
            models_meta_file = f"{export_dir}/metadata/models_info.json"
            with open(models_meta_file, 'w') as f:
                json.dump(models_info, f, indent=2)
            
            print(f"✅ Exported {len(trained_models)} trained models")
            return models_info
        else:
            print("⚠️ No trained models found")
            return None
            
    except Exception as e:
        print(f"❌ Error exporting trained models: {str(e)}")
        return None

def export_model_results(db, export_dir):
    """Export model training results"""
    print("📈 Exporting model results...")
    
    try:
        # Try to get model results for known models
        model_names = ['volatility', 'direction', 'profit_probability', 'reversal']
        exported_results = []
        
        for model_name in model_names:
            try:
                results = db.load_model_results(model_name)
                if results:
                    # Save results as JSON
                    results_file = f"{export_dir}/models/{model_name}_results.json"
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    exported_results.append({
                        'model_name': model_name,
                        'file_path': results_file,
                        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    print(f"    ✅ Exported {model_name} results")
            except Exception as e:
                print(f"    ⚠️ No results for {model_name}: {str(e)}")
        
        if exported_results:
            # Save results metadata
            results_meta_file = f"{export_dir}/metadata/model_results_info.json"
            with open(results_meta_file, 'w') as f:
                json.dump(exported_results, f, indent=2)
        
        print(f"✅ Exported {len(exported_results)} model results")
        return exported_results
        
    except Exception as e:
        print(f"❌ Error exporting model results: {str(e)}")
        return []

def export_predictions(db, export_dir):
    """Export saved predictions"""
    print("🔮 Exporting predictions...")
    
    try:
        # Try to get predictions for known models
        model_names = ['volatility', 'direction', 'profit_probability', 'reversal']
        exported_predictions = []
        
        for model_name in model_names:
            try:
                predictions = db.load_predictions(model_name)
                if predictions is not None and len(predictions) > 0:
                    # Save predictions as CSV
                    pred_file = f"{export_dir}/models/{model_name}_predictions.csv"
                    predictions.to_csv(pred_file)
                    
                    exported_predictions.append({
                        'model_name': model_name,
                        'rows': len(predictions),
                        'columns': list(predictions.columns),
                        'file_path': pred_file,
                        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    print(f"    ✅ Exported {model_name} predictions ({len(predictions)} rows)")
            except Exception as e:
                print(f"    ⚠️ No predictions for {model_name}: {str(e)}")
        
        if exported_predictions:
            # Save predictions metadata
            pred_meta_file = f"{export_dir}/metadata/predictions_info.json"
            with open(pred_meta_file, 'w') as f:
                json.dump(exported_predictions, f, indent=2)
        
        print(f"✅ Exported {len(exported_predictions)} prediction sets")
        return exported_predictions
        
    except Exception as e:
        print(f"❌ Error exporting predictions: {str(e)}")
        return []

def create_database_dump(db, export_dir):
    """Create PostgreSQL database dump"""
    print("💾 Creating database dump...")
    
    try:
        # Get database info
        db_info = db.get_database_info()
        
        # Create comprehensive database info
        dump_info = {
            'export_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'database_type': db_info.get('database_type'),
            'backend': db_info.get('backend'),
            'total_datasets': db_info.get('total_datasets'),
            'total_records': db_info.get('total_records'),
            'total_models': db_info.get('total_models'),
            'total_trained_models': db_info.get('total_trained_models'),
            'total_predictions': db_info.get('total_predictions'),
            'export_directory': export_dir,
            'database_url_host': os.getenv('DATABASE_URL', '').split('@')[1].split('/')[0] if '@' in os.getenv('DATABASE_URL', '') else 'Hidden'
        }
        
        # Save database dump info
        dump_file = f"{export_dir}/backup/database_dump_info.json"
        with open(dump_file, 'w') as f:
            json.dump(dump_info, f, indent=2)
        
        print(f"✅ Database dump info saved to {dump_file}")
        return dump_info
        
    except Exception as e:
        print(f"❌ Error creating database dump: {str(e)}")
        return None

def create_restore_script(export_dir, datasets, models_info, results, predictions):
    """Create a script to restore the database"""
    restore_script = f"""#!/usr/bin/env python3
'''
Database Restore Script for TribexAlpha
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''

import os
import json
import pickle
import pandas as pd
from utils.database_adapter import DatabaseAdapter

def restore_database():
    '''Restore database from exported files'''
    print("🔄 Starting database restoration...")
    
    try:
        # Initialize database
        db = DatabaseAdapter()
        
        # Restore datasets
        print("📊 Restoring datasets...")
"""

    # Add dataset restoration code
    for dataset in datasets:
        dataset_name = dataset['name']
        restore_script += f"""
        print("  📥 Restoring {dataset_name}...")
        data = pd.read_csv("datasets/{dataset_name}.csv", index_col=0, parse_dates=True)
        if db.save_ohlc_data(data, "{dataset_name}"):
            print("    ✅ Restored {dataset_name}")
        else:
            print("    ❌ Failed to restore {dataset_name}")
"""

    # Add models restoration code
    if models_info:
        restore_script += f"""
        # Restore trained models
        print("🤖 Restoring trained models...")
        try:
            with open("models/trained_models.pkl", "rb") as f:
                trained_models = pickle.load(f)
            
            if db.save_trained_models(trained_models):
                print(f"    ✅ Restored {{len(trained_models)}} models")
            else:
                print("    ❌ Failed to restore trained models")
        except Exception as e:
            print(f"    ❌ Error restoring models: {{str(e)}}")
"""

    restore_script += """
        print("✅ Database restoration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Database restoration failed: {str(e)}")
        return False

if __name__ == "__main__":
    restore_database()
"""

    # Save restore script
    restore_file = f"{export_dir}/restore_database.py"
    with open(restore_file, 'w') as f:
        f.write(restore_script)
    
    # Make it executable
    os.chmod(restore_file, 0o755)
    
    print(f"✅ Restore script created: {restore_file}")

def main():
    """Main export function"""
    print("🚀 TribexAlpha Database Export")
    print("=" * 50)
    
    try:
        # Initialize database
        print("🔄 Initializing database connection...")
        db = DatabaseAdapter()
        
        # Create export folder
        export_dir = create_export_folder()
        print(f"📁 Created export directory: {export_dir}")
        
        # Export datasets
        datasets = export_datasets(db, export_dir)
        
        # Export trained models
        models_info = export_trained_models(db, export_dir)
        
        # Export model results
        results = export_model_results(db, export_dir)
        
        # Export predictions
        predictions = export_predictions(db, export_dir)
        
        # Create database dump
        dump_info = create_database_dump(db, export_dir)
        
        # Create restore script
        create_restore_script(export_dir, datasets, models_info, results, predictions)
        
        # Create summary
        summary = {
            'export_completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'export_directory': export_dir,
            'exported_datasets': len(datasets),
            'exported_models': len(models_info['model_names']) if models_info else 0,
            'exported_results': len(results),
            'exported_predictions': len(predictions),
            'total_files_exported': len(datasets) + (len(models_info['model_names']) if models_info else 0) + len(results) + len(predictions)
        }
        
        summary_file = f"{export_dir}/export_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 50)
        print("✅ DATABASE EXPORT COMPLETED!")
        print("=" * 50)
        print(f"📁 Export Location: {export_dir}")
        print(f"📊 Datasets Exported: {summary['exported_datasets']}")
        print(f"🤖 Models Exported: {summary['exported_models']}")
        print(f"📈 Results Exported: {summary['exported_results']}")
        print(f"🔮 Predictions Exported: {summary['exported_predictions']}")
        print(f"📄 Total Files: {summary['total_files_exported']}")
        print("\n📋 Export Contents:")
        print(f"  • {export_dir}/datasets/        - CSV files of your OHLC data")
        print(f"  • {export_dir}/models/          - Trained models and results")
        print(f"  • {export_dir}/metadata/        - Database metadata")
        print(f"  • {export_dir}/backup/          - Database dump info")
        print(f"  • {export_dir}/restore_database.py - Restoration script")
        print(f"  • {export_dir}/export_summary.json - Export summary")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        return False

if __name__ == "__main__":
    main()
