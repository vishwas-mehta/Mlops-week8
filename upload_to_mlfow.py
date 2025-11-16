"""Quick MLflow Upload - Uses existing experiment"""
import mlflow
import mlflow.sklearn
import json
import time

# Load results
print("Loading results...")
with open('results/experiment_results.json', 'r') as f:
    results = json.load(f)

# Connect to MLflow
mlflow.set_tracking_uri("http://localhost:8100")

# Get existing experiment (don't create new one)
try:
    experiment = mlflow.get_experiment_by_name("iris_data_contamination_study")
    if experiment:
        mlflow.set_experiment("iris_data_contamination_study")
        print(f"✓ Using existing experiment (ID: {experiment.experiment_id})")
    else:
        mlflow.set_experiment("iris_data_contamination_study")
        print("✓ Created new experiment")
except:
    mlflow.set_experiment("iris_data_contamination_study")
    print("✓ Set experiment")

print("\nUploading 8 runs to MLflow...")

success = 0
failed = 0

for result in results:
    run_name = f"{result['model']}_corrupt_{result['corruption_pct']}pct"
    
    print(f"\n  Uploading: {run_name}")
    
    try:
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("classifier_name", result['model'])
            mlflow.log_param("corruption_percentage", result['corruption_pct'])
            mlflow.log_param("corrupted_sample_count", result['corrupted_samples'])
            mlflow.log_param("seed_value", result['seed'])
            
            # Log metrics
            mlflow.log_metric("training_accuracy", result['train_acc'])
            mlflow.log_metric("testing_accuracy", result['test_acc'])
            mlflow.log_metric("testing_precision", result['test_precision'])
            mlflow.log_metric("testing_recall", result['test_recall'])
            mlflow.log_metric("testing_f1_score", result['test_f1'])
            mlflow.log_metric("overfitting_metric", result['overfit_gap'])
            
            # Log artifacts
            mlflow.log_artifact(result['plot_file'])
            
            print(f"    ✓ Success")
            success += 1
            
    except Exception as e:
        print(f"    ✗ Failed: {str(e)[:100]}")
        failed += 1
    
    time.sleep(0.5)  # Small delay between uploads

print(f"\n{'='*80}")
print(f"UPLOAD COMPLETE")
print(f"  Success: {success}/8")
print(f"  Failed: {failed}/8")
print(f"{'='*80}")
print(f"\nView results at: http://34.44.83.201:8100")