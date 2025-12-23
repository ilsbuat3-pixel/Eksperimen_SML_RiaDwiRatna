# automate_RiaDwiRatna.py
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
import argparse
import os

def load_data(filepath):
    """Load dataset dari file CSV"""
    return pd.read_csv(filepath)

def remove_duplicates(df):
    """Hapus data duplikat"""
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"Removed {removed} duplicate rows")
    return df

def handle_outliers(df):
    """Handle outliers dengan winsorizing"""
    outlier_cols = ['blood_glucose_level', 'bmi', 'HbA1c_level']
    
    for col in outlier_cols:
        # Apply winsorizing (5% di kedua sisi)
        df[col] = winsorize(df[col], limits=[0.05, 0.05])
    
    print(f"Winsorized {len(outlier_cols)} columns")
    return df

def create_features(df):
    """Feature engineering"""
    # 1. Age categories
    def categorize_age(age):
        if age < 30:
            return 'Young'
        elif age < 50:
            return 'Middle'
        else:
            return 'Senior'
    
    df['age_category'] = df['age'].apply(categorize_age)
    
    # 2. BMI categories
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    df['bmi_category'] = df['bmi'].apply(categorize_bmi)
    
    # 3. Risk Score
    df['risk_score'] = (
        (df['age'] > 45).astype(int) +
        (df['bmi'] > 25).astype(int) +
        (df['HbA1c_level'] > 5.7).astype(int) +
        (df['blood_glucose_level'] > 140).astype(int) +
        df['hypertension'] +
        df['heart_disease']
    )
    
    # 4. Interaction features
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['glucose_hba1c_ratio'] = df['blood_glucose_level'] / df['HbA1c_level']
    
    print(f"Created 7 new features")
    return df

def encode_categorical(df):
    """Encode categorical variables"""
    # Encode gender
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    df['gender_encoded'] = df['gender'].map(gender_map)
    
    # Encode smoking history
    smoking_map = {
        'never': 0,
        'No Info': 1,
        'not current': 2,
        'former': 3,
        'current': 4,
        'ever': 4
    }
    df['smoking_encoded'] = df['smoking_history'].map(smoking_map)
    
    # One-hot encoding untuk age_category dan bmi_category
    df = pd.get_dummies(df, columns=['age_category', 'bmi_category'], dtype=int)
    
    print("Encoded categorical variables")
    return df

def scale_features(df):
    """Scale numerical features dengan RobustScaler"""
    scale_cols = [
        'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
        'age_bmi_interaction', 'glucose_hba1c_ratio'
    ]
    
    scaler = RobustScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    print(f"Scaled {len(scale_cols)} numerical features")
    return df, scaler

def final_preparation(df):
    """Persiapan final dataset"""
    # Drop original text columns yang sudah diencode
    columns_to_drop = ['gender', 'smoking_history']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Pastikan diabetes adalah kolom terakhir
    if 'diabetes' in df.columns:
        diabetes_col = df['diabetes']
        df = df.drop(columns=['diabetes'])
        df['diabetes'] = diabetes_col
    
    print(f"Final dataset shape: {df.shape}")
    return df

def preprocess_data(input_path, output_path):
    """
    Main preprocessing function
    """
    print("="*50)
    print("STARTING DATA PREPROCESSING")
    print("="*50)
    
    # 1. Load data
    print("\n1. Loading data...")
    df = load_data(input_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 2. Remove duplicates
    print("\n2. Removing duplicates...")
    df = remove_duplicates(df)
    
    # 3. Handle outliers
    print("\n3. Handling outliers...")
    df = handle_outliers(df)
    
    # 4. Feature engineering
    print("\n4. Feature engineering...")
    df = create_features(df)
    
    # 5. Encode categorical
    print("\n5. Encoding categorical variables...")
    df = encode_categorical(df)
    
    # 6. Scale features
    print("\n6. Scaling numerical features...")
    df, scaler = scale_features(df)
    
    # 7. Final preparation
    print("\n7. Final preparation...")
    df = final_preparation(df)
    
    # 8. Save processed data
    print("\n8. Saving processed data...")
    df.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    
    # 9. Generate report
    report = {
        'input_file': input_path,
        'output_file': output_path,
        'original_rows': 100000,  # Dataset original
        'processed_rows': len(df),
        'original_cols': 9,
        'processed_cols': len(df.columns),
        'diabetes_positive': df['diabetes'].sum(),
        'diabetes_percentage': (df['diabetes'].sum() / len(df) * 100),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"\nSummary Report:")
    print(f"- Input: {report['input_file']}")
    print(f"- Output: {report['output_file']}")
    print(f"- Rows: {report['original_rows']} → {report['processed_rows']}")
    print(f"- Columns: {report['original_cols']} → {report['processed_cols']}")
    print(f"- Diabetes positive: {report['diabetes_positive']} ({report['diabetes_percentage']:.2f}%)")
    print(f"- Memory usage: {report['memory_usage_mb']:.2f} MB")
    
    return df, report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess diabetes dataset')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Cek file input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    # Jalankan preprocessing
    try:
        df_processed, report = preprocess_data(args.input, args.output)
        
        # Cek output
        if os.path.exists(args.output):
            print(f"\n✓ Verification: Output file created successfully")
            print(f"  File size: {os.path.getsize(args.output) / 1024**2:.2f} MB")
        else:
            print(f"\n✗ Error: Output file not created")
            
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)