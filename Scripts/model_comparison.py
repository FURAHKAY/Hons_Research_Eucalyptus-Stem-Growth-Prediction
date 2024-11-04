import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Load configuration
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Directory Configuration
output_dir = '../Outputs/visuals/combine_comparison'
metrics_dir = config['outputs']['evaluation_dir']
predictions_dir = config['outputs']['predictions_dir']
os.makedirs(output_dir, exist_ok=True)

# Load Prediction Data
models = ['random_forest', 'lstm', 'xgboost', 'linear_regression']
train_files = {
    'random_forest': '../Outputs/predictions/Random_Forest_full_train_predictions.csv',
    'lstm': '../Outputs/predictions/LSTM_full_train_predictions.csv',
    'xgboost': '../Outputs/predictions/XGBoost_full_train_predictions.csv',
    'linear_regression': '../Outputs/predictions/Linear_Regression_full_train_predictions.csv'
}
test_files = {
    'random_forest': '../Outputs/predictions/Random_Forest_full_test_predictions.csv',
    'lstm': '../Outputs/predictions/LSTM_full_test_predictions.csv',
    'xgboost': '../Outputs/predictions/XGBoost_full_test_predictions.csv',
    'linear_regression': '../Outputs/predictions/Linear_Regression_full_test_predictions.csv'
}

# Load and Prepare Data for Each Model
model_data = {}
model_data_no_resampling = {}
for model in models:
    # Load train and test data
    train_df = pd.read_csv(train_files[model])
    test_df = pd.read_csv(test_files[model])

    # Set Dataset column and combine train and test
    train_df['Dataset'] = 'Train'
    test_df['Dataset'] = 'Test'
    data = pd.concat([train_df, test_df])

    # Convert 'DateTime' to datetime format and set as index
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.set_index('DateTime', inplace=True)

    # Save unresampled data for residual histogram
    model_data_no_resampling[model] = data

    # Resample data to 3-hour intervals, calculating the mean for numeric columns
    numeric_columns = ['Actual', 'Predicted']
    resampled_data = data[numeric_columns].resample('3h').mean()  # Use 'h' instead of 'H'
    model_data[model] = resampled_data.sort_index()


# 1. Combined Time Series of Actual vs Predicted Values (Resampled)
plt.figure(figsize=(10, 6))
for model, data in model_data.items():
    plt.plot(data.index, data['Actual'], label=f'{model.capitalize()} Actual', linestyle='-', alpha=0.7)
    plt.plot(data.index, data['Predicted'], label=f'{model.capitalize()} Predicted', linestyle='--', alpha=0.7)

plt.title('Combined Time Series of Actual vs Predicted Values (3-Hour Resampled)')
plt.xlabel('DateTime')
plt.ylabel('Dendrometer Values (μm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'combined_time_series.png'))
plt.show()

# 2. Scatter Plot: Actual vs Predicted Values for Each Model (Resampled)
plt.figure(figsize=(10, 6))
max_value = max([data['Actual'].max() for data in model_data.values()])

plt.plot([0, max_value], [0, max_value], color='black', linestyle='--', label='Ideal Fit (Predicted = Actual)')
for model, data in model_data.items():
    plt.scatter(data['Actual'], data['Predicted'], label=f'{model.capitalize()} Predicted', alpha=0.6)

plt.title('Actual vs. Predicted Values for Each Model')
plt.xlabel('Actual Values (μm)')
plt.ylabel('Predicted Values (μm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'combined_actual_vs_predicted.png'))
plt.show()

plt.figure(figsize=(10, 6))
for model, data in model_data_no_resampling.items():
    residuals = data['Actual'] - data['Predicted']
    plt.hist(residuals, bins=50, alpha=0.6, label=f'{model.capitalize()} Residuals')

plt.title('Distribution of Residuals (Actual - Predicted)')
plt.xlabel('Residuals (μm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residuals_distribution.png'))
plt.show()
