import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load configuration
with open('../Config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define paths and ensure directories exist
raw_data_path = config['data']['raw_data_path']
processed_data_dir = config['data']['processed_dir']
split_data_dir = config['data']['splits_dir']
os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(split_data_dir, exist_ok=True)

def load_data(path):
    """Load raw data from the specified path."""
    data = pd.read_csv(path)
    print("Data loaded successfully.")
    return data

def preprocess_data(data):
    # Convert DateTime to proper datetime format
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    
    # Set the DateTime column as the index
    data.set_index('DateTime', inplace=True)
    
    # Drop duplicates
    data.drop_duplicates(inplace=True)
    
    # Check for missing values
    # print("Missing values before processing:")
    # print(data.isna().sum())
    
    # Handle missing values (forward fill)
    data.ffill(inplace=True)

    # Rename 'Dendro_Corrected' to 'Dendro'
    data.rename(columns={'Dendro_Corrected': 'Dendro'}, inplace=True)

    # Define the numeric columns
    numeric_columns = ['Dendro', 'Air Temp', 'Air Hum', 'Soil Moisture', 'Soil Temp', 'VPD']
    
    # Handle outliers by capping values outside 3 standard deviations
    for column in numeric_columns:
        upper_limit = data[column].mean() + 3 * data[column].std()
        lower_limit = data[column].mean() - 3 * data[column].std()
        data[column] = np.where(data[column] > upper_limit, upper_limit, data[column])
        data[column] = np.where(data[column] < lower_limit, lower_limit, data[column])

    # # Check missing values after forward fill
    # print("Missing values after forward fill:")
    # print(data.isna().sum())

    # Feature Scaling for numeric features
    features = ['Air Temp', 'Air Hum', 'Soil Moisture', 'Soil Temp', 'VPD']
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    return data

def save_splits(data, group_name):
    """Scale features, split data, and save train/test sets for each device group."""
    X = data[config['preprocessing']['features']]
    y = data[config['preprocessing']['target']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X_train)
    # print(y_test)
        # Print to check consistent sizes
    print(f"Size of X_train: {X_train.shape}, Size of y_train: {y_train.shape}")
    print(f"Size of X_test: {X_test.shape}, Size of y_test: {y_test.shape}")

    # Save splits with pandas to avoid scientific notation and retain column headers
    X_train.to_csv(os.path.join(split_data_dir, f'X_train_{group_name}.csv'), index=True)
    X_test.to_csv(os.path.join(split_data_dir, f'X_test_{group_name}.csv'), index=True)
    y_train.to_csv(os.path.join(split_data_dir, f'y_train_{group_name}.csv'), index=True)
    y_test.to_csv(os.path.join(split_data_dir, f'y_test_{group_name}.csv'), index=True)

    print(f"Data split and saved for group: {group_name}")

if __name__ == "__main__":
    # Load and preprocess the raw data
    data = load_data(raw_data_path)
    data = preprocess_data(data)
    
    # Save the fully preprocessed dataset
    preprocessed_data_path = config['data']['processed_data_path']
    data.to_csv(preprocessed_data_path)
    print(f"Preprocessed data saved to {preprocessed_data_path}")

    # Create train/test splits for each device group
    device_groups = {'full': list(range(4, 11)), 'stellenbosch': [4, 5, 6], 'portugal': [7, 8, 9, 10]}
    for group_name, device_ids in device_groups.items():
        group_data = data[data['Device_ID'].isin(device_ids)]
        save_splits(group_data, group_name)
    
    print("Preprocessing and splitting completed successfully.")
