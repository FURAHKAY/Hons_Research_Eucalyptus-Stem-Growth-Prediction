import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import xgboost as xgb
import yaml

# Load configuration
with open('../Config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define paths and ensure directories exist
raw_data_path = config['data']['raw_data_path']
output_dir = config['outputs']['experimentation_dir']
os.makedirs(output_dir, exist_ok=True)

# Load data and set DateTime index
data = pd.read_csv(raw_data_path)
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

# Check for NaN values and handle them (e.g., drop or impute)
if data.isna().sum().sum() > 0:
    print("Data contains NaN values. Handling them by dropping.")
    data = data.dropna()

# Define features and target variable
features = ['Air Temp', 'Air Hum', 'Soil Temp', 'Soil Moisture', 'VPD']
target = 'Dendro_Corrected'
X = data[features]
y = data[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  


# --- RANDOM FOREST MODEL ---
def random_forest_experiment():
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, f'{output_dir}/random_forest_best_model.pkl')

    y_pred = best_model.predict(X_test_scaled)
    mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)
    save_metrics('Random Forest', mae, mse, rmse, r2, grid_search.best_params_)
    plot_feature_importance(best_model, 'Random Forest', features)

# --- LSTM MODEL ---
def lstm_experiment():
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=2)

    model.save(f'{output_dir}/lstm_best_model.keras')  # Use new Keras format

    y_pred = model.predict(X_test_lstm).flatten()
    mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)
    save_metrics('LSTM', mae, mse, rmse, r2)
    plot_loss_history(history, 'LSTM')

# --- XGBOOST MODEL ---
def xgboost_experiment():
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, f'{output_dir}/xgboost_best_model.pkl')

    y_pred = best_model.predict(X_test_scaled)
    mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)
    save_metrics('XGBoost', mae, mse, rmse, r2, grid_search.best_params_)

# --- Evaluation and Helper Functions ---
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")
    return mae, mse, rmse, r2

def save_metrics(model_name, mae, mse, rmse, r2, best_params=None):
    with open(f'{output_dir}/{model_name.lower()}_evaluation.txt', 'w') as f:
        f.write(f"{model_name} Model Performance:\n")
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"Mean Squared Error (MSE): {mse}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
        f.write(f"R² Score: {r2}\n")
        if best_params:
            f.write(f"Best Parameters: {best_params}\n")

def plot_feature_importance(model, model_name, feature_names):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance for {model_name} Model')
    plt.savefig(f'{output_dir}/{model_name.lower()}_feature_importance.png')
    plt.show()

def plot_loss_history(history, model_name):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'{model_name} Loss History')
    plt.savefig(f'{output_dir}/{model_name.lower()}_loss_history.png')
    plt.show()

# Run experiments for all models
random_forest_experiment()
lstm_experiment()
xgboost_experiment()

print("All experiments completed. Models and metrics saved.")
