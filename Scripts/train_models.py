import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load configuration
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = config['data']['processed_data_path']
data = pd.read_csv(data_path)
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

# Define paths
split_data_dir = config['data']['splits_dir']
processed_data_dir = config['data']['processed_dir']

# Define model directories
model_dirs = {
    'random_forest': '../Models/random_forest',
    'xgboost': '../Models/xgboost',
    'lstm': '../Models/lstm',
    'linear_regression': '../Models/linear_regression'
}
for path in model_dirs.values():
    os.makedirs(path, exist_ok=True)

def load_data_splits(group_name):
    X_train = pd.read_csv(f"{split_data_dir}/X_train_{group_name}.csv", index_col=0)  # Adjust index_col to keep index
    X_test = pd.read_csv(f"{split_data_dir}/X_test_{group_name}.csv", index_col=0)
    y_train = pd.read_csv(f"{split_data_dir}/y_train_{group_name}.csv", index_col=0, header=0).values.ravel()
    y_test = pd.read_csv(f"{split_data_dir}/y_test_{group_name}.csv", index_col=0, header=0).values.ravel()

    # Print shapes to verify consistency
    print(f"Loaded sizes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test

# Model training functions
def train_random_forest(X_train, y_train, group_name):
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    joblib.dump(scaler_X, f"{model_dirs['random_forest']}/scaler_X_{group_name}.pkl")

    model = RandomForestRegressor(n_estimators=config['random_forest']['n_estimators'], random_state=config['random_forest']['random_state'])
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"{model_dirs['random_forest']}/random_forest_model_{group_name}.pkl")
    print(f"Random Forest model saved for {group_name}")

def train_xgboost(X_train, y_train, group_name):
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    joblib.dump(scaler_X, f"{model_dirs['xgboost']}/scaler_X_{group_name}.pkl")

    model = XGBRegressor(n_estimators=config['xgboost']['n_estimators'], random_state=config['xgboost']['random_state'])
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"{model_dirs['xgboost']}/xgboost_model_{group_name}.pkl")
    print(f"XGBoost model saved for {group_name}")

def train_linear_regression(X_train, y_train, group_name):
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    joblib.dump(scaler_X, f"{model_dirs['linear_regression']}/scaler_X_{group_name}.pkl")

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"{model_dirs['linear_regression']}/linear_regression_model_{group_name}.pkl")
    print(f"Linear Regression model saved for {group_name}")


# Define number of timesteps for the LSTM sliding window
n_timesteps = config['preprocessing']['n_timesteps']

# Create sequences using a sliding window approach
def create_sequences(data, target_column, n_timesteps):
    sequences = []
    labels = []
    for i in range(len(data) - n_timesteps):
        sequence = data.iloc[i:i + n_timesteps].drop(columns=[target_column]).values  # Exclude the target column
        label = data.iloc[i + n_timesteps][target_column]  # Target value (Dendro)
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Build the LSTM model
def build_lstm_model(input_shape, n_units=100, dropout_rate=0.2, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(n_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(n_units))  # Second LSTM layer
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Function to train and evaluate LSTM model for specific Device IDs
def train_lstm_for_device_group(data, device_ids, group_name):
    print(f"Training LSTM model for Device IDs: {device_ids}")

    # Filter data for the specified device group
    group_data = data[data['Device_ID'].isin(device_ids)]
    
    # Define features and target
    features = ['Device_ID', 'Air Temp', 'Air Hum', 'Soil Temp', 'Soil Moisture', 'VPD']
    target = 'Dendro'

    # Create sequences for features and target
    X_sequences, y_sequences = create_sequences(group_data, target_column=target, n_timesteps=n_timesteps)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    joblib.dump(scaler, f"{model_dirs['lstm']}/lstm_scaler_{group_name}.pkl")

    # Normalize the target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    joblib.dump(y_scaler, f"{model_dirs['lstm']}/lstm_y_scaler_{group_name}.pkl")

    # Build and compile the model
    model = build_lstm_model(input_shape=(n_timesteps, X_train_scaled.shape[2]), optimizer=Adam(learning_rate=0.0001))

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train_scaled, y_train_scaled, validation_split=0.2, epochs=config['lstm']['epochs'], batch_size=config['lstm']['batch_size'], callbacks=[early_stopping])

    # Save the trained model
    model.save(f"{model_dirs['lstm']}/lstm_model_{group_name}.h5")

    # Save the training and test data with proper indexes
    X_train_df = pd.DataFrame(X_train.reshape(X_train.shape[0], -1), index=group_data.index[:X_train.shape[0]])
    X_train_df.to_csv(f'{model_dirs['lstm']}/lstm_train_data_X_train_{group_name}.csv', index_label='Datetime')

    X_test_df = pd.DataFrame(X_test.reshape(X_test.shape[0], -1), index=group_data.index[X_train.shape[0]:X_train.shape[0] + X_test.shape[0]])
    X_test_df.to_csv(f'{model_dirs['lstm']}/lstm_test_data_X_test_{group_name}.csv', index_label='Datetime')

    y_train_df = pd.DataFrame(y_train, index=group_data.index[:X_train.shape[0]], columns=[target])
    y_train_df.to_csv(f'{model_dirs['lstm']}/lstm_train_data_y_train_{group_name}.csv', index_label='Datetime')

    y_test_df = pd.DataFrame(y_test, index=group_data.index[X_train.shape[0]:X_train.shape[0] + X_test.shape[0]], columns=[target])
    y_test_df.to_csv(f'{model_dirs['lstm']}/lstm_test_data_y_test_{group_name}.csv', index_label='Datetime')

    print(f"LSTM model training completed for Device IDs: {device_ids}\n")



# Main function to train models for each group
if __name__ == "__main__":
    device_groups = {'full': list(range(4, 11)), 'stellenbosch': [4, 5, 6], 'portugal': [7, 8, 9, 10]}
    
    # Iterate over each device group to train models
    for group_name, device_ids in device_groups.items():
        print(f"\nTraining models for {group_name} group")
        
        # Load preprocessed splits
        X_train, X_test, y_train, y_test = load_data_splits(group_name)
        
        # Train Random Forest
        print(f"Training Random Forest model for {group_name} group")
        train_random_forest(X_train, y_train, group_name)
        
        # Train LSTM (requires sequence reshaping)
        print(f"Training LSTM model for {group_name} group")
        train_lstm_for_device_group(data, device_ids, group_name)
        
        # Train XGBoost
        print(f"Training XGBoost model for {group_name} group")
        train_xgboost(X_train, y_train, group_name)
        
        # Train Linear Regression
        print(f"Training Linear Regression model for {group_name} group")
        train_linear_regression(X_train, y_train, group_name)

    print("\nAll models training completed.")
