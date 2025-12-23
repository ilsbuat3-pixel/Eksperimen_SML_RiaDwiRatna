# modelling.py - BASIC MODEL (untuk kriteria basic saja)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def load_data():
    """Load preprocessed dataset"""
    df = pd.read_csv('diabetes_preprocessed_full.csv')
    
    # Simple fix for one-hot encoding
    cols_to_drop = []
    if 'age_category_Young' in df.columns:
        cols_to_drop.append('age_category_Young')
    if 'bmi_category_Underweight' in df.columns:
        cols_to_drop.append('bmi_category_Underweight')
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    print(f"Dataset shape: {df.shape}")
    
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']
    return X, y

def train_basic_model():
    """Basic model with MLflow AUTOLOG"""
    X, y = load_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # ENABLE AUTOLOG (kriteria basic)
    mlflow.autolog()
    
    with mlflow.start_run(run_name="basic_rf_autolog"):
        # Simple model, no tuning
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        print("Training basic RandomForest with autolog...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("BASIC MODEL RESULTS (Autolog)")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Model akan otomatis di-log oleh autolog
        
    return model

if __name__ == "__main__":
    # Local MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Diabetes_Basic_Autolog")
    
    print("="*60)
    print("RUNNING BASIC MODEL FOR CRITERIA 2 BASIC")
    print("="*60)
    
    model = train_basic_model()
    
    print("\n✅ Basic model training completed!")
    print("Check MLflow UI at http://127.0.0.1:5000")
    print("Take screenshot of autolog results")