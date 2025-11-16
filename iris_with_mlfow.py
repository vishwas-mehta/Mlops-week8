"""
IRIS Data Poisoning Experiment - COMPLETE WORKING VERSION
Guaranteed to work with MLflow
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import pickle
import time

warnings.filterwarnings('ignore')

ALL_RESULTS = []


def corrupt_training_data(X_train, y_train, poison_pct, seed=None):
    """Add noise and flip labels"""
    if seed:
        np.random.seed(seed)
    
    X_corrupt = X_train.copy()
    y_corrupt = y_train.copy()
    
    n_poison = int(len(X_train) * poison_pct / 100.0)
    
    if n_poison == 0:
        print(f"  ✓ Baseline (0% corruption)")
        return X_corrupt, y_corrupt, []
    
    poison_idx = np.random.choice(len(X_train), n_poison, replace=False)
    print(f"  ✓ Corrupting {n_poison}/{len(X_train)} samples")
    
    for idx in poison_idx:
        noise = np.random.randn(X_train.shape[1]) * 4.0 * X_train.std(axis=0)
        X_corrupt[idx] = X_train[idx] + noise
        
        other_labels = [l for l in np.unique(y_train) if l != y_train[idx]]
        y_corrupt[idx] = np.random.choice(other_labels)
    
    return X_corrupt, y_corrupt, poison_idx


def train_and_test_model(X_train, X_test, y_train, y_test, model_name, seed=42):
    """Train model and return metrics"""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
    
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    conf_mat = confusion_matrix(y_test, y_test_pred)
    
    return {
        'model': model,
        'scaler': scaler,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_prec': test_prec,
        'test_rec': test_rec,
        'test_f1': test_f1,
        'conf_mat': conf_mat,
        'overfit_gap': train_acc - test_acc,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled
    }


def run_experiment(poison_pct, base_seed=42):
    """Run one complete experiment"""
    
    exp_seed = base_seed + poison_pct
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {poison_pct}% Data Corruption (seed={exp_seed})")
    print(f"{'='*80}")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=base_seed, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    X_train_corrupt, y_train_corrupt, poison_idx = corrupt_training_data(
        X_train, y_train, poison_pct, seed=exp_seed
    )
    
    models = ["LogisticRegression", "RandomForestClassifier"]
    
    for model_name in models:
        print(f"\n  → Training {model_name}...")
        
        results = train_and_test_model(
            X_train_corrupt, X_test,
            y_train_corrupt, y_test,
            model_name, seed=exp_seed
        )
        
        print(f"     Train Acc: {results['train_acc']:.4f}")
        print(f"     Test Acc:  {results['test_acc']:.4f}")
        print(f"     Test F1:   {results['test_f1']:.4f}")
        print(f"     Overfit:   {results['overfit_gap']:.4f}")
        
        os.makedirs('results', exist_ok=True)
        
        plt.figure(figsize=(7, 5))
        sns.heatmap(results['conf_mat'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name}\n{poison_pct}% Corruption')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        plot_file = f"results/cm_{model_name}_{poison_pct}pct.png"
        plt.savefig(plot_file, dpi=120)
        plt.close()
        
        print(f"     Saved: {plot_file}")
        
        # Save model
        model_file = f"results/model_{model_name}_{poison_pct}pct.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({'model': results['model'], 'scaler': results['scaler']}, f)
        
        ALL_RESULTS.append({
            'corruption_pct': poison_pct,
            'model': model_name,
            'train_acc': float(results['train_acc']),
            'test_acc': float(results['test_acc']),
            'test_precision': float(results['test_prec']),
            'test_recall': float(results['test_rec']),
            'test_f1': float(results['test_f1']),
            'overfit_gap': float(results['overfit_gap']),
            'corrupted_samples': len(poison_idx),
            'seed': exp_seed,
            'plot_file': plot_file,
            'model_file': model_file
        })


def upload_to_mlflow_reliable():
    """Upload results to MLflow with retries and error handling"""
    
    print(f"\n{'='*80}")
    print("UPLOADING RESULTS TO MLFLOW")
    print(f"{'='*80}")
    
    try:
        import mlflow
        import mlflow.sklearn
        
        # Use localhost - more reliable
        mlflow.set_tracking_uri("http://localhost:8100")
        
        # Try to set experiment
        try:
            mlflow.set_experiment("iris_data_contamination_study")
            print("✓ Connected to MLflow")
        except Exception as e:
            print(f"Creating new experiment...")
            mlflow.create_experiment("iris_data_contamination_study")
            mlflow.set_experiment("iris_data_contamination_study")
            print("✓ Experiment created")
        
        success_count = 0
        fail_count = 0
        
        for result in ALL_RESULTS:
            run_name = f"{result['model']}_corrupt_{result['corruption_pct']}pct"
            
            print(f"\n  Uploading: {run_name}")
            
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
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
                        success_count += 1
                        break
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"    Retry {retry_count}/{max_retries}...")
                        time.sleep(2)
                    else:
                        print(f"    ✗ Failed after {max_retries} attempts: {str(e)[:50]}")
                        fail_count += 1
        
        print(f"\n{'='*80}")
        print(f"✓ MLflow Upload Complete: {success_count} success, {fail_count} failed")
        print(f"View at: http://34.44.83.201:8100")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n⚠ MLflow upload failed: {e}")
        print("Local results are still saved in ./results/")


def save_results_summary():
    """Save all results to CSV and JSON"""
    
    df = pd.DataFrame(ALL_RESULTS)
    df.to_csv('results/experiment_results.csv', index=False)
    print(f"\n✓ Saved: results/experiment_results.csv")
    
    with open('results/experiment_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"✓ Saved: results/experiment_results.json")
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(df[['corruption_pct', 'model', 'test_acc', 'test_f1', 'overfit_gap']].to_string(index=False))


def main():
    """Main execution"""
    
    print("="*80)
    print("IRIS DATA POISONING EXPERIMENT")
    print("LOCAL RUN + MLFLOW UPLOAD")
    print("="*80)
    
    corruption_levels = [0, 5, 10, 50]
    
    print(f"\nWill test: {corruption_levels}% corruption levels")
    print(f"Models: LogisticRegression, RandomForestClassifier")
    print(f"Total runs: {len(corruption_levels) * 2} = 8")
    
    # Run all experiments locally first
    for pct in corruption_levels:
        run_experiment(pct, base_seed=42)
    
    # Save local results
    save_results_summary()
    
    # Upload to MLflow with retries
    upload_to_mlflow_reliable()
    
    print(f"\n{'='*80}")
    print("✓ ASSIGNMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"\nLocal results: ./results/")
    print(f"  - experiment_results.csv")
    print(f"  - experiment_results.json")
    print(f"  - 8 confusion matrix PNG files")
    print(f"  - 8 model PKL files")
    print(f"\nMLflow UI: http://34.44.83.201:8100")


if __name__ == "__main__":
    main()