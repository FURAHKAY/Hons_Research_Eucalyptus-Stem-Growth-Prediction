import os
import pandas as pd
import numpy as np
import joblib
import yaml
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load configuration
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define directories for models and predictions
model_dirs = {
    'random_forest': '../Models/random_forest',
    'xgboost': '../Models/xgboost',
    'lstm': '../Models/lstm',
    'linear_regression': '../Models/linear_regression'
}
predictions_dir = '../Outputs/predictions'
split_data_dir = config['data']['splits_dir']
os.makedirs(predictions_dir, exist_ok=True)

# Conditional function to load data based on LSTM or non-LSTM model
def load_data_splits(group_name, data_split, is_lstm=False):
    if is_lstm:
        # LSTM-specific path loading
        X = pd.read_csv(f"{model_dirs['lstm']}/lstm_{data_split}_data_X_{data_split}_{group_name}.csv", index_col='Datetime')
        y = pd.read_csv(f"{model_dirs['lstm']}/lstm_{data_split}_data_y_{data_split}_{group_name}.csv", index_col='Datetime').values.ravel()
    else:
        # Standard split data path loading
        X = pd.read_csv(f"{split_data_dir}/X_{data_split}_{group_name}.csv", index_col=0)
        y = pd.read_csv(f"{split_data_dir}/y_{data_split}_{group_name}.csv", index_col=0, header=0).values.ravel()
    return X, y

# Save predictions with features based on model type
def save_predictions_with_features(X, y_true, y_pred, model_name, group_name, data_split, is_lstm=False):
    if is_lstm:
        test_results_df = pd.DataFrame({
            'DateTime': X.index[:len(y_pred)],
            'Actual': y_true[:len(y_pred)],
            'Predicted': y_pred.flatten(),
            'Air Temp': X.iloc[:len(y_pred), 0],
            'Air Hum': X.iloc[:len(y_pred), 1],
            'Soil Temp': X.iloc[:len(y_pred), 2],
            'Soil Moisture': X.iloc[:len(y_pred), 3],
            'VPD': X.iloc[:len(y_pred), 4],
            'Device_ID': X.iloc[:len(y_pred), 5]
        })
    else:
        test_results_df = pd.DataFrame({
            'DateTime': X.index,
            'Device_ID': X['Device_ID'],
            'Actual': y_true,
            'Predicted': y_pred,
            'Air Temp': X['Air Temp'],
            'Air Hum': X['Air Hum'],
            'Soil Temp': X['Soil Temp'],
            'Soil Moisture': X['Soil Moisture'],
            'VPD': X['VPD']
        })
        
    file_path = os.path.join(predictions_dir, f"{model_name.replace(' ', '_')}_{group_name}_{data_split}_predictions.csv")
    test_results_df.to_csv(file_path, index=False)
    print(f"Predictions with features saved to {file_path}")

# LSTM prediction function
def predict_lstm_model(group_name, data_split):
    X, y = load_data_splits(group_name, data_split, is_lstm=True)
    model = load_model(f"{model_dirs['lstm']}/lstm_model_{group_name}.h5", custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load(f"{model_dirs['lstm']}/lstm_scaler_{group_name}.pkl")
    y_scaler = joblib.load(f"{model_dirs['lstm']}/lstm_y_scaler_{group_name}.pkl")

    n_timesteps = 10
    n_features = 6

    X_reshaped = X.values.reshape(-1, n_timesteps, n_features)
    X_scaled = scaler.transform(X_reshaped.reshape(-1, n_features)).reshape(X_reshaped.shape)

    y_pred_scaled = model.predict(X_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    save_predictions_with_features(X, y, y_pred, "LSTM", group_name, data_split, is_lstm=True)

# Non-LSTM prediction function
def predict_sklearn_model(model_path, scaler_path, X, y, model_name, group_name, data_split):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    save_predictions_with_features(X, y, y_pred, model_name, group_name, data_split, is_lstm=False)

# Main function
if __name__ == "__main__":
    device_groups = {'full': list(range(4, 11)), 'stellenbosch': [4, 5, 6], 'portugal': [7, 8, 9, 10]}
    
    for group_name in device_groups.keys():
        for data_split in ['train', 'test']:
            print(f"\nGenerating {data_split} predictions for {group_name} group")

            # Load data for sklearn models
            X, y = load_data_splits(group_name, data_split, is_lstm=False)

            # Predict with Random Forest
            predict_sklearn_model(
                model_path=f"{model_dirs['random_forest']}/random_forest_model_{group_name}.pkl",
                scaler_path=f"{model_dirs['random_forest']}/scaler_X_{group_name}.pkl",
                X=X,
                y=y,
                model_name="Random Forest",
                group_name=group_name,
                data_split=data_split
            )

            # Predict with XGBoost
            predict_sklearn_model(
                model_path=f"{model_dirs['xgboost']}/xgboost_model_{group_name}.pkl",
                scaler_path=f"{model_dirs['xgboost']}/scaler_X_{group_name}.pkl",
                X=X,
                y=y,
                model_name="XGBoost",
                group_name=group_name,
                data_split=data_split
            )

            # Predict with Linear Regression
            predict_sklearn_model(
                model_path=f"{model_dirs['linear_regression']}/linear_regression_model_{group_name}.pkl",
                scaler_path=f"{model_dirs['linear_regression']}/scaler_X_{group_name}.pkl",
                X=X,
                y=y,
                model_name="Linear Regression",
                group_name=group_name,
                data_split=data_split
            )

            # Predict with LSTM
            predict_lstm_model(
                group_name=group_name,
                data_split=data_split
            )

    print("\nAll predictions generated and saved for both train and test splits.")
