
## Project Structure

- **Config/**: Contains configuration files (e.g., `config.yaml`) for paths and hyperparameters.
- **Data/**: Directory to store raw and processed data files.
- **Models/**:directory is used to store the trained model files (e.g., .pkl files for Random Forest, .h5 for LSTM).
  **Note: If the files are too large, they can be regenerated by running the training scripts.**
- **Outputs/**: Stores outputs such as evaluation metrics, predictions, and generated visuals.
- **Scripts/**: Directory containing the main scripts for each stage of the project:
  - **preprocessing.py**: Prepares and preprocesses the data for model training.
  - **train_models.py**: Trains various machine learning models based on configuration.
  - **evaluate.py**: Evaluates trained models and saves evaluation metrics.
  - **predictions.py**: Generates predictions using the trained models.
  - **visuals.py**: Generates visualizations of model results.
  - **model_comparison.py**: Compares the performance of different models based on evaluation metrics and visualizations.
  - **experimentation.py**: Conducts hyperparameter tuning and experimentation for model selection.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost tensorflow keras pyyaml

   ```

3. Ensure that your configuration file `config.yaml` in the `Config/` folder is set up with the correct paths and parameters.

## Running the Project

### Step-by-Step Execution

1. **Preprocess the Data**:
   ```bash
   python Scripts/preprocessing.py
   ```
   This script loads the raw data, performs cleaning and feature engineering, and saves the processed data in the `Data/` directory.

2. **Train Models**:
   ```bash
   python Scripts/train_models.py
   ```
   This script trains the defined models (e.g., Random Forest, LSTM, XGBoost) and saves the trained models to the `Outputs/` directory.

3. **Evaluate Models**:
   ```bash
   python Scripts/evaluate.py
   ```
   This script loads the trained models, evaluates their performance on test data, and stores metrics (e.g., MAE, RMSE) in the `Outputs/evaluation` folder.

4. **Generate Predictions**:
   ```bash
   python Scripts/predictions.py
   ```
   This script uses the trained models to generate predictions and saves the output to the `Outputs/predictions` folder.

5. **Generate Visuals**:
   ```bash
   python Scripts/visuals.py
   ```
   This script creates visualizations of model performance, such as time series and residual plots, saving them to the `Outputs/visuals` folder.

6. **Compare Models**:
   ```bash
   python Scripts/model_comparison.py
   ```
   This script compares model performance based on evaluation metrics and visualizations, generating comparison plots in the `Outputs/visuals/combine_comparison` folder.

7. **Experimentation and Hyperparameter Tuning**:
   ```bash
   python Scripts/experimentation.py
   ```
   This script performs hyperparameter tuning for selected models, saving the best configurations and results to the `Outputs/experimentation` folder.

## Notes

- Make sure that the `Config/config.yaml` file is configured correctly with all necessary paths and parameters.
- You can modify the scripts to use different models or add additional preprocessing steps as needed.

---
