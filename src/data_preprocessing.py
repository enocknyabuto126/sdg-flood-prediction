"""
Data Preprocessing Module for Flash Flood Risk Prediction
Addresses SDG 13 (Climate Action) and SDG 11 (Sustainable Cities and Communities)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class SDGDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def load_and_merge(self, filepaths):
        # Load and concatenate data from both Excel files
        dfs = [pd.read_excel(fp) for fp in filepaths]
        data = pd.concat(dfs, ignore_index=True)
        return data

    def filter_and_prepare(self, data):
        # Filter for the relevant SeriesDescription
        filtered = data[data['SeriesDescription'] == 'Number of people affected by disaster (number)'].copy()
        # Drop rows with missing target
        filtered = filtered.dropna(subset=['Value'])
        # Fill missing values in features
        filtered = filtered.fillna('Unknown')
        # Select features
        features = ['GeoAreaName', 'TimePeriod', 'Source', 'Nature', 'Reporting Type']
        X = filtered[features].copy()
        y = filtered['Value'].astype(float)
        # Encode categorical features
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        self.feature_names = list(X.columns)
        return X, y

    def scale_features(self, X):
        return self.scaler.fit_transform(X)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def preprocess_pipeline(self, filepaths):
        data = self.load_and_merge(filepaths)
        X, y = self.filter_and_prepare(data)
        X_scaled = self.scale_features(X)
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y)
        return X_train, X_test, y_train, y_test, self.feature_names

    def explore_files(self, filepaths, nrows=5):
        for fp in filepaths:
            print(f"\nExploring {fp}:")
            df = pd.read_excel(fp)
            print("Columns:", list(df.columns))
            print(df.head(nrows))

if __name__ == '__main__':
    preprocessor = SDGDataPreprocessor()
    filepaths = [
        'data/Goal11.xlsx',
        'data/Goal13.xlsx'
    ]
    preprocessor.explore_files(filepaths)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(filepaths)
    print('Features:', feature_names)
    print('Train shape:', X_train.shape)
    print('Test shape:', X_test.shape) 