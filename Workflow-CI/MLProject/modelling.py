# MLProject/modelling.py - Script yang disesuaikan untuk MLflow Project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
import mlflow
import mlflow.sklearn
import argparse
import os
import json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a Diabetes classifier')
    parser.add_argument('--data-path', type=str, default='diabetes_preprocessed_full.csv',
                       help='Path to the input data CSV file')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    # 1. Parse arguments from MLflow run
    args = parse_args()
    
    # 2. Setup MLflow tracking - SANGAT PENTING untuk CI/CD
    #    MLFLOW_TRACKING_URI akan diset oleh GitHub Actions secret
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"📡 MLflow Tracking URI set to: {tracking_uri}")
    
    # Gunakan experiment name yang konsisten untuk CI
    mlflow.set_experiment("Diabetes_CI_Pipeline")
    
    # 3. Start MLflow run
    with mlflow.start_run(run_name="ci_training_run"):
        print("="*70)
        print("MLFLOW PROJECT TRAINING - CI/CD PIPELINE")
        print("="*70)
        
        # Nonaktifkan autolog untuk manual control
        mlflow.autolog(disable=True)
        
        # 4. Load and prepare data
        print(f"\n📂 Loading data from: {args.data_path}")
        df = pd.read_csv(args.data_path)
        
        # Fix one-hot encoding redundancy (sama seperti sebelumnya)
        cols_to_drop = []
        if 'age_category_Young' in df.columns:
            cols_to_drop.append('age_category_Young')
        if 'bmi_category_Underweight' in df.columns:
            cols_to_drop.append('bmi_category_Underweight')
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Diabetes positive: {df['diabetes'].sum()} ({df['diabetes'].sum()/len(df)*100:.2f}%)")
        
        X = df.drop(columns=['diabetes'])
        y = df['diabetes']
        
        # 5. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
        )
        
        print(f"\nTrain set: {X_train.shape}, Positive: {y_train.sum():,}")
        print(f"Test set:  {X_test.shape}, Positive: {y_test.sum():,}")
        
        # 6. Hyperparameter tuning (GridSearchCV)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [10, 15, None],
            'min_samples_split': [5, 10],
            'class_weight': ['balanced', {0: 1, 1: 3}]
        }
        
        print("\n🔍 Performing GridSearchCV...")
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=args.random_state, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 7. Best model evaluation
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'best_cv_score': grid_search.best_score_
        }
        
        # 8. Manual Logging (Kunci Kriteria 2 & 3)
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param('test_size', args.test_size)
        mlflow.log_param('random_state', args.random_state)
        mlflow.log_param('data_path', args.data_path)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log dataset info
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("train_positive", y_train.sum())
        mlflow.log_metric("test_positive", y_test.sum())
        
        # 9. Log model - METODE YANG PALING KOMPATIBEL
        # Menggunakan log_artifact untuk kompatibilitas maksimal dengan DagsHub[citation:6]
        import pickle
        model_path = 'best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact(model_path, artifact_path="model")
        
        # Alternatif: Log dengan log_model jika registry didukung
        # mlflow.sklearn.log_model(best_model, "model")
        
        # 10. Log essential artifacts untuk CI (opsional, sederhana)
        # Confusion matrix as JSON
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cm_data = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        with open('confusion_matrix.json', 'w') as f:
            json.dump(cm_data, f)
        mlflow.log_artifact('confusion_matrix.json')
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        with open('classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact('classification_report.json')
        
        # 11. Print results
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\n📊 Best Parameters: {grid_search.best_params_}")
        print(f"📈 Test Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print(f"\n✅ Model and artifacts logged to MLflow.")
        if tracking_uri:
            print(f"🔗 Tracking Server: {tracking_uri}")
        
        # Return success
        run_id = mlflow.active_run().info.run_id
        print(f"🏷️  Run ID for Docker build: {run_id}")

        # Simpan run_id ke file untuk step berikutnya
        with open('run_id.txt', 'w') as f:
            f.write(run_id)

            return 0

if __name__ == "__main__":
    main()
    
    