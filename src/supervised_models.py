"""
Supervised Learning Models for Flash Flood Risk Prediction
Addresses SDG 13 (Climate Action) and SDG 11 (Sustainable Cities and Communities)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt

class SDGFloodRegressor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_performance = {}
        self.feature_importance = None

    def train_random_forest(self, X_train, y_train, X_test, y_test, feature_names):
        print("🌳 Training Random Forest Regressor...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        performance = self._calculate_metrics(y_test, y_pred, 'Random Forest')
        self.models['Random Forest'] = best_rf
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"✅ Random Forest best params: {grid_search.best_params_}")
        return performance

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test, feature_names):
        print("🚀 Training Gradient Boosting Regressor...")
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5],
        }
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_gb = grid_search.best_estimator_
        y_pred = best_gb.predict(X_test)
        performance = self._calculate_metrics(y_test, y_pred, 'Gradient Boosting')
        self.models['Gradient Boosting'] = best_gb
        print(f"✅ Gradient Boosting best params: {grid_search.best_params_}")
        return performance

    def train_neural_network(self, X_train, y_train, X_test, y_test, feature_names):
        print("🧠 Training Neural Network Regressor...")
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,)],
            'alpha': [0.001, 0.01],
        }
        nn = MLPRegressor(random_state=42, max_iter=500)
        grid_search = GridSearchCV(nn, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_nn = grid_search.best_estimator_
        y_pred = best_nn.predict(X_test)
        performance = self._calculate_metrics(y_test, y_pred, 'Neural Network')
        self.models['Neural Network'] = best_nn
        print(f"✅ Neural Network best params: {grid_search.best_params_}")
        return performance

    def train_linear_regression(self, X_train, y_train, X_test, y_test, feature_names):
        print("📊 Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        performance = self._calculate_metrics(y_test, y_pred, 'Linear Regression')
        self.models['Linear Regression'] = lr
        return performance

    def _calculate_metrics(self, y_true, y_pred, model_name):
        metrics = {
            'model_name': model_name,
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
        self.model_performance[model_name] = metrics
        return metrics

    def train_all_models(self, X_train, y_train, X_test, y_test, feature_names):
        print("🎯 Training all regression models...")
        self.train_random_forest(X_train, y_train, X_test, y_test, feature_names)
        self.train_gradient_boosting(X_train, y_train, X_test, y_test, feature_names)
        self.train_neural_network(X_train, y_train, X_test, y_test, feature_names)
        self.train_linear_regression(X_train, y_train, X_test, y_test, feature_names)
        performance_df = pd.DataFrame(self.model_performance).T
        best_model_name = performance_df['MAE'].idxmin()
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        print(f"🏆 Best model: {best_model_name} (MAE: {performance_df.loc[best_model_name, 'MAE']:.3f})")
        return performance_df

    def predict(self, X, model_name=None):
        if model_name is None:
            model_name = self.best_model_name
        model = self.models[model_name]
        return model.predict(X)

    def get_feature_importance(self, top_n=10):
        if self.feature_importance is not None:
            return self.feature_importance.head(top_n)
        return None

    def save_models(self, filepath_prefix='models/sdg_regressor'):
        for model_name, model in self.models.items():
            filename = f"{filepath_prefix}_{model_name.replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            print(f"✅ Saved {model_name} to {filename}")

if __name__ == '__main__':
    from data_preprocessing import SDGDataPreprocessor
    preprocessor = SDGDataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline([
        'data/Goal11.xlsx',
        'data/Goal13.xlsx'
    ])
    reg = SDGFloodRegressor()
    performance_df = reg.train_all_models(X_train, y_train, X_test, y_test, feature_names)
    print(performance_df)
    print('Top features:', reg.get_feature_importance()) 