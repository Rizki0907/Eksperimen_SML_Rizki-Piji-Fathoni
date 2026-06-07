import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import kaggle
import os
import warnings
warnings.filterwarnings('ignore')

def download_dataset():
    """Download dataset dari Kaggle API"""
    print("="*50)
    print("DOWNLOADING DATASET")
    print("="*50)
    
    os.makedirs('heart_raw', exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'johnsmith88/heart-disease-dataset',
        path='heart_raw',
        unzip=True
    )
    print("Dataset berhasil didownload!")
    return 'heart_raw/heart.csv'

def load_dataset(filepath):
    """Load dataset dari filepath"""
    print("="*50)
    print("LOADING DATASET")
    print("="*50)
    
    df = pd.read_csv(filepath)
    print(f"Shape dataset: {df.shape}")
    print(f"Kolom: {list(df.columns)}")
    return df

def remove_duplicates(df):
    """Menghapus data duplikat"""
    print("="*50)
    print("MENGHAPUS DUPLIKAT")
    print("="*50)
    
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Shape sebelum: {before}")
    print(f"Shape sesudah: {after}")
    print(f"Duplikat dihapus: {before - after}")
    return df

def remove_outliers_iqr(df, columns):
    """Menghapus outlier menggunakan metode IQR"""
    print("="*50)
    print("MENANGANI OUTLIER (IQR METHOD)")
    print("="*50)
    
    df_out = df.copy()
    total_removed = 0
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = df_out.shape[0]
        df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
        removed = before - df_out.shape[0]
        total_removed += removed
        print(f"  {col}: lower={lower:.2f}, upper={upper:.2f}, removed={removed} rows")
    
    print(f"\nShape setelah remove outlier: {df_out.shape}")
    print(f"Total baris dihapus: {total_removed}")
    return df_out

def encode_features(df):
    """Encoding fitur kategorikal"""
    print("="*50)
    print("ENCODING FITUR KATEGORIKAL")
    print("="*50)
    
    binary_cols = ['sex', 'fbs', 'exang']
    onehot_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
        print(f"Label Encoding: {col} -> {df[col].unique()}")
    
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    print(f"\nShape setelah encoding: {df.shape}")
    return df

def normalize_features(df):
    """Normalisasi fitur numerik"""
    print("="*50)
    print("NORMALISASI FITUR NUMERIK")
    print("="*50)
    
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print("Statistik setelah normalisasi:")
    print(X[numerical_cols].describe().round(3))
    
    return X, y

def split_and_save(X, y, output_dir='heart_preprocessed'):
    """Split data dan simpan ke CSV"""
    print("="*50)
    print("SPLIT DATA & SIMPAN")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(f'{output_dir}/heart_train.csv', index=False)
    test_df.to_csv(f'{output_dir}/heart_test.csv', index=False)
    
    print(f"Total data  : {X.shape[0]}")
    print(f"Train set   : {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"Test set    : {X_test.shape[0]} ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"\nTrain disimpan: {output_dir}/heart_train.csv")
    print(f"Test disimpan : {output_dir}/heart_test.csv")
    print("\nPreprocessing selesai!")

def main():
    # 1. Download dataset
    filepath = download_dataset()
    
    # 2. Load dataset
    df = load_dataset(filepath)
    
    # 3. Hapus duplikat
    df = remove_duplicates(df)
    
    # 4. Hapus outlier
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df = remove_outliers_iqr(df, numerical_cols)
    
    # 5. Encoding
    df = encode_features(df)
    
    # 6. Normalisasi
    X, y = normalize_features(df)
    
    # 7. Split dan simpan
    split_and_save(X, y)

if __name__ == "__main__":
    main()