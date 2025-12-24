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
    
    # 2. SANGAT PENTING: JANGAN pakai set_experiment() sama sekali!
    # Biarkan MLflow run command yang handle experiment
    
    # 3. Start MLflow run - Biarkan kosong, MLflow akan handle
    with mlflow.start_run():
        print("="*70)
        print("MLFLOW PROJECT TRAINING - CI/CD PIPELINE")
        print("="*70)
        
        # Nonaktifkan autolog untuk manual control
        mlflow.autolog(disable=True)
        
        # 4. Load and prepare data
        print(f"\n📂 Loading data from: {args.data_path}")
        df = pd.read_csv(args.data_path)
        
        # Fix one-hot encoding redundancy
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
        print("\n🔍 Performing GridSearchCV...")
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [10, 15, None],
            'min_samples_split': [5, 10],
            'class_weight': ['balanced', {0: 1, 1: 3}]
        }
        
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
        
        # 8. Manual Logging
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
        
        # 9. Log model - Cara paling sederhana
        import pickle
        model_path = 'best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact(model_path, artifact_path="model")
        
        # Juga log dengan sklearn untuk compatibility
        mlflow.sklearn.log_model(best_model, "sklearn_model")
        
        # 10. Save run_id untuk Docker build
        run_id = mlflow.active_run().info.run_id
        print(f"\n🏷️  Run ID: {run_id}")
        
        # Save run_id to file
        with open('run_id.txt', 'w') as f:
            f.write(run_id)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\n📊 Best Parameters: {grid_search.best_params_}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        return 0

if __name__ == "__main__":
    main()