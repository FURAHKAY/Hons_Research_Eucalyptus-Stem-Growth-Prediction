import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load configuration
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define model directories
model_dirs = {
    'random_forest': '../Models/random_forest',
    'xgboost': '../Models/xgboost',
    'lstm': '../Models/lstm',
    'linear_regression': '../Models/linear_regression'
}
evaluation_dir = '../Outputs/evaluation'
os.makedirs(evaluation_dir, exist_ok=True)

# Helper function to load test data
def load_test_data(group_name):
    X_test = pd.read_csv(f"{config['data']['splits_dir']}/X_test_{group_name}.csv", index_col=0)
    y_test = pd.read_csv(f"{config['data']['splits_dir']}/y_test_{group_name}.csv", index_col=0).values.ravel()
    return X_test, y_test

# Function to calculate Model Efficiency Index (MEI)
def calculate_mei(y_true, y_pred):
    mean_observed = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - mean_observed) ** 2)
    mei = 1 - (numerator / denominator)
    return mei

# Function to evaluate a model's performance
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    accuracy_percentage = r2 * 100
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mbe = np.mean(y_true - y_pred)  # Mean Bias Error
    mei = calculate_mei(y_true, y_pred)
    print(f"\n{model_name} Evaluation Results:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R² Score: {r2}")
    # Save to file
    results_file = os.path.join(evaluation_dir, f"{model_name.replace(' ', '_')}_{group_name}_evaluation.txt")
    with open(results_file, "w") as f:
        f.write(f"{model_name} Evaluation Results for {group_name}:\n")
        f.write(f"Mean Squared Error (MSE): {mse}\n")
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"R² Score: {r2}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
        f.write(f'Accuracy: {accuracy_percentage:.2f}%\n')
        f.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2%}\n')
        f.write(f'Mean Bias Error (MBE): {mbe:.2f}\n')
        f.write(f"Model Efficiency Index (MEI): {mei}\n")

    print(f"Results saved to {results_file}")


# Evaluation for Random Forest and XGBoost models
def evaluate_sklearn_model(model_path, scaler_path, X_test, y_test, model_name):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred, model_name)

# Helper function to create sequences for LSTM evaluation
def create_lstm_sequences(data, n_timesteps):
    sequences = []
    for i in range(len(data) - n_timesteps + 1):
        sequence = data[i:i + n_timesteps]
        sequences.append(sequence)
    return np.array(sequences)

# Evaluation for LSTM model
def evaluate_lstm_model(group_name, model_name):
    # Load the test data
    X_test = pd.read_csv(f"{model_dirs['lstm']}/lstm_test_data_X_test_{group_name}.csv", index_col='Datetime')
    y_test = pd.read_csv(f"{model_dirs['lstm']}/lstm_test_data_y_test_{group_name}.csv", index_col='Datetime')
    

    model = load_model(f'{model_dirs['lstm']}/lstm_model_{group_name}.h5', custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load(f'{model_dirs['lstm']}/lstm_scaler_{group_name}.pkl')
    y_scaler = joblib.load(f'{model_dirs['lstm']}/lstm_y_scaler_{group_name}.pkl')

    # Define number of timesteps and features
    n_timesteps = 10
    n_features = 6

    # Reshape the data back to 3D for LSTM input
    X_test_reshaped = X_test.values.reshape(-1, n_timesteps, n_features)

    # Apply the scaler to the reshaped data (but flatten the data first for scaling)
    X_test_scaled = scaler.transform(X_test_reshaped.reshape(-1, n_features)).reshape(X_test_reshaped.shape)

    # Make predictions for both training and test sets
    y_test_pred = model.predict(X_test_scaled)

    # Inverse transform predictions back to the original scale
    y_test_pred = y_scaler.inverse_transform(y_test_pred)

    evaluate_model(y_test, y_test_pred, model_name)


# Main evaluation process for each model
if __name__ == "__main__":
    device_groups = {'full': list(range(4, 11)), 'stellenbosch': [4, 5, 6], 'portugal': [7, 8, 9, 10]}
    
    for group_name in device_groups.keys():
        print(f"\nEvaluating models for {group_name} group")
        X_test, y_test = load_test_data(group_name)

        # Evaluate Random Forest model
        print(f"\nEvaluating Random Forest model for {group_name} group")
        evaluate_sklearn_model(
            model_path=f"{model_dirs['random_forest']}/random_forest_model_{group_name}.pkl",
            scaler_path=f"{model_dirs['random_forest']}/scaler_X_{group_name}.pkl",
            X_test=X_test,
            y_test=y_test,
            model_name=f"Random Forest"
        )

        # Evaluate XGBoost model
        print(f"\nEvaluating XGBoost model for {group_name} group")
        evaluate_sklearn_model(
            model_path=f"{model_dirs['xgboost']}/xgboost_model_{group_name}.pkl",
            scaler_path=f"{model_dirs['xgboost']}/scaler_X_{group_name}.pkl",
            X_test=X_test,
            y_test=y_test,
            model_name=f"XGBoost"
        )

        # Evaluate Linear Regression model
        print(f"\nEvaluating Linear Regression model for {group_name} group")
        evaluate_sklearn_model(
            model_path=f"{model_dirs['linear_regression']}/linear_regression_model_{group_name}.pkl",
            scaler_path=f"{model_dirs['linear_regression']}/scaler_X_{group_name}.pkl",
            X_test=X_test,
            y_test=y_test,
            model_name=f"Linear Regression"
        )
        
        # Define number of timesteps for the LSTM sliding window
        n_timesteps = config['preprocessing']['n_timesteps']

        # Evaluate LSTM model
        print(f"\nEvaluating LSTM model for {group_name} group")
        evaluate_lstm_model(
            group_name=group_name,
            model_name=f"LSTM"
        )
    
    print("\nAll models' evaluation completed.")
