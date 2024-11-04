import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
output_base_dir = '../Outputs/visuals'
models = ['XGBoost', 'Random_Forest', 'LSTM']
groups = ['full', 'stellenbosch', 'portugal']
numeric_columns = ['Actual', 'Predicted', 'Air Temp', 'Air Hum', 'Soil Temp', 'Soil Moisture', 'VPD']

# Ensure directories
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load and process data for a specific model and group
def load_and_resample_data(model, group_name):
    predictions_dir = f'../Outputs/predictions'
    output_dir = f'{output_base_dir}/{model}_{group_name}/model_results'
    ensure_dir(output_dir)
    
    # Load the prediction results
    train_file_path = f'{predictions_dir}/{model}_{group_name}_train_predictions.csv'
    test_file_path = f'{predictions_dir}/{model}_{group_name}_test_predictions.csv'
    
    # Check if files exist before loading
    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Files for {model} in group {group_name} not found.")

    train_results = pd.read_csv(train_file_path)
    test_results = pd.read_csv(test_file_path)
    # Combine train and test data for plotting
    train_results['Dataset'] = 'Train'
    test_results['Dataset'] = 'Test'
    data = pd.concat([train_results, test_results])

    # Convert 'DateTime' to datetime format and set as index
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.set_index('DateTime', inplace=True)
    data = data.sort_values('DateTime')

    # Split and resample
    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    resampled_train = train_data[numeric_columns].resample('3H').mean()
    resampled_test = test_data[numeric_columns].resample('3H').mean()

    # Calculate error
    train_data['Error'] = train_data['Actual'] - train_data['Predicted']
    test_data['Error'] = test_data['Actual'] - test_data['Predicted']

    return output_dir, train_data, test_data, resampled_train, resampled_test

# Plot combined Actual vs Predicted time series
def plot_combined_time_series(model, output_dir, resampled_train, resampled_test):
    plt.figure(figsize=(10, 6))
    plt.plot(resampled_train.index, resampled_train['Actual'], label=f'{model} Actual (Training)', color='blue', linewidth=0.7, alpha=0.7)
    plt.plot(resampled_train.index, resampled_train['Predicted'], label=f'{model} Predicted (Training)', color='green', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.plot(resampled_test.index, resampled_test['Actual'], label=f'{model} Actual (Testing)', color='blue', linewidth=1.2, alpha=0.7)
    plt.plot(resampled_test.index, resampled_test['Predicted'], label=f'{model} Predicted (Testing)', color='red', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.title(f'{model.capitalize()} Time Series of Actual and Predicted Dendrometer Values')
    plt.xlabel('DateTime')
    plt.ylabel('Values (μm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model}_combined_time_series.png')



# Plot Actual and Predicted scatter plots for specified features
def plot_feature_vs_growth(model, output_dir, feature, resampled_train, resampled_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(resampled_train[feature], resampled_train['Actual'], label=f'{model} Actual (Training)', color='blue', alpha=0.7)
    plt.scatter(resampled_train[feature], resampled_train['Predicted'], label=f'{model} Predicted (Training)', color='green', alpha=0.7)
    plt.scatter(resampled_test[feature], resampled_test['Actual'], label=f'{model} Actual (Testing)', color='orange', alpha=0.7)
    plt.scatter(resampled_test[feature], resampled_test['Predicted'], label=f'{model} Predicted (Testing)', color='red', alpha=0.7)
    plt.title(f'{model.capitalize()} {feature} vs. Tree Growth')
    plt.xlabel(feature)
    plt.ylabel('Tree Growth (μm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model}_{feature.lower()}_vs_tree_growth.png')


# Plot Error Distribution
def plot_error_distribution(model, output_dir, train_data, test_data):
    plt.figure(figsize=(10, 6))
    plt.hist(train_data['Error'], bins=50, color='blue', alpha=0.7, label=f'{model} Training Error')
    plt.hist(test_data['Error'], bins=50, color='red', alpha=0.7, label=f'{model} Testing Error')
    plt.title(f'{model} Distribution of Errors (Actual - Predicted)')
    plt.xlabel('Error (μm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model}_error_distribution.png')



# Plot per-device scatter of Actual vs Predicted with Ideal Fit Line and Linear Regression Line
def plot_actual_vs_predicted_per_device(model, output_dir, train_data, test_data):
    device_ids = pd.concat([train_data['Device_ID'], test_data['Device_ID']]).unique()
    
    for device_id in device_ids:
        plt.figure(figsize=(10, 6))
        
        # Filter data for the current device
        train_device_data = train_data[train_data['Device_ID'] == device_id]
        test_device_data = test_data[test_data['Device_ID'] == device_id]

        # Scatter plot of Actual vs. Predicted Values (Training)
        plt.scatter(train_device_data['Actual'], train_device_data['Predicted'], 
                    label=f'{model} - Train Device {device_id}', color='purple', alpha=0.7)

        # Scatter plot of Actual vs. Predicted Values (Testing)
        plt.scatter(test_device_data['Actual'], test_device_data['Predicted'], 
                    label=f'{model} - Test Device {device_id}', color='green', alpha=0.7)

        # Ideal fit line (Predicted = Actual)
        max_value = max(train_device_data['Actual'].max(), test_device_data['Actual'].max())
        plt.plot([0, max_value], [0, max_value], color='black', linestyle='--', label='Ideal Fit (y = x)')
        
        # Linear regression line for training data
        if not train_device_data.empty:
            slope_train, intercept_train = np.polyfit(train_device_data['Actual'], train_device_data['Predicted'], 1)
            plt.plot(train_device_data['Actual'], slope_train * train_device_data['Actual'] + intercept_train,
                     color='blue', linestyle='-', label='Train Linear Fit')

            # Display the equation for training data
            equation_text_train = f"Train: ŷ = {slope_train:.2f}x + {intercept_train:.2f}"
            plt.text(train_device_data['Actual'].min(), train_device_data['Predicted'].max(), 
                     equation_text_train, fontsize=12, color='blue')

        # Linear regression line for testing data
        if not test_device_data.empty:
            slope_test, intercept_test = np.polyfit(test_device_data['Actual'], test_device_data['Predicted'], 1)
            plt.plot(test_device_data['Actual'], slope_test * test_device_data['Actual'] + intercept_test,
                     color='orange', linestyle='-', label='Test Linear Fit')

            # Display the equation for testing data
            equation_text_test = f"Test: ŷ = {slope_test:.2f}x + {intercept_test:.2f}"
            plt.text(test_device_data['Actual'].min(), test_device_data['Predicted'].max(), 
                     equation_text_test, fontsize=12, color='orange')
        
        # Plot settings
        plt.title(f'{model} - Actual vs Predicted for Device {device_id} (Train and Test)')
        plt.xlabel('Actual Tree Growth (μm)')
        plt.ylabel('Predicted Tree Growth (μm)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.xlim(train_device_data['Actual'].min(), max_value)
        plt.ylim(train_device_data['Predicted'].min(), max_value)
        plt.savefig(f'{output_dir}/{model}_device_{device_id}_actual_vs_predicted.png')


# Plot overall Actual vs Predicted scatter plot with Ideal Fit Line and Linear Regression Line for the full dataset
def plot_actual_vs_predicted_full_dataset(model, output_dir, full_train_data, full_test_data):
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of Actual vs. Predicted Values (Training)
    plt.scatter(full_train_data['Actual'], full_train_data['Predicted'], label=f'{model} - Train', color='purple', alpha=0.6)

    # Scatter plot of Actual vs. Predicted Values (Testing)
    plt.scatter(full_test_data['Actual'], full_test_data['Predicted'], label=f'{model} - Test', color='green', alpha=0.6)

    # Ideal fit line (Predicted = Actual)
    max_value = max(full_train_data['Actual'].max(), full_test_data['Actual'].max())
    min_value = min(full_train_data['Actual'].min(), full_test_data['Actual'].min())
    plt.plot([min_value, max_value], [min_value, max_value], color='black', linestyle='--', label='Ideal Fit (y = x)')
    
    # Linear regression line for training data
    if not full_train_data.empty:
        slope_train, intercept_train = np.polyfit(full_train_data['Actual'], full_train_data['Predicted'], 1)
        plt.plot(full_train_data['Actual'], slope_train * full_train_data['Actual'] + intercept_train,
                 color='blue', linestyle='-', label='Train Linear Fit')

        # Display the equation for training data
        equation_text_train = f"Train: ŷ = {slope_train:.2f}x + {intercept_train:.2f}"
        plt.text(min_value, max_value * 0.95, equation_text_train, fontsize=12, color='blue')

    # Linear regression line for testing data
    if not full_test_data.empty:
        slope_test, intercept_test = np.polyfit(full_test_data['Actual'], full_test_data['Predicted'], 1)
        plt.plot(full_test_data['Actual'], slope_test * full_test_data['Actual'] + intercept_test,
                 color='orange', linestyle='-', label='Test Linear Fit')

        # Display the equation for testing data
        equation_text_test = f"Test: ŷ = {slope_test:.2f}x + {intercept_test:.2f}"
        plt.text(min_value, max_value * 0.85, equation_text_test, fontsize=12, color='orange')
    
    # Plot settings
    plt.title(f'{model} - Overall Actual vs Predicted (Train and Test)')
    plt.xlabel('Actual Tree Growth (μm)')
    plt.ylabel('Predicted Tree Growth (μm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.savefig(f'{output_dir}/{model}_overall_actual_vs_predicted.png')



# Generate all plots for a model and group
def generate_model_visuals(model, group_name):
    output_dir, train_data, test_data, resampled_train, resampled_test = load_and_resample_data(model, group_name)
    plot_combined_time_series(model, output_dir, resampled_train, resampled_test)
    for feature in ['Air Temp', 'Soil Moisture', 'VPD']:
        plot_feature_vs_growth(model, output_dir, feature, resampled_train, resampled_test)
    plot_error_distribution(model, output_dir, train_data, test_data)
    plot_actual_vs_predicted_per_device(model, output_dir, train_data, test_data)
    plot_actual_vs_predicted_full_dataset(model, output_dir, train_data, test_data)




# Run visualization for all models and groups
for model in models:
    for group_name in groups:
        try:
            generate_model_visuals(model, group_name)
        except FileNotFoundError as e:
            print(e)
