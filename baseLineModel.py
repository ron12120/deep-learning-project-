import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
file_path = './Fifa 23 Players Data.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

# Define target variable
y = df['Value(in Euro)']  # Replace with your actual target column name

# Split dataset into train and test
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

# Ensure they are NumPy arrays
y_train = y_train.to_numpy().astype(float).flatten()
y_test = y_test.to_numpy().astype(float).flatten()

# Baseline model
baseline_prediction = np.mean(y_train)

y_train_baseline = np.full(y_train.shape, baseline_prediction)
y_test_baseline = np.full(y_test.shape, baseline_prediction)

# Evaluate baseline performance
baseline_train_mse = mean_squared_error(y_train, y_train_baseline)
baseline_train_mae = mean_absolute_error(y_train, y_train_baseline)
baseline_test_mse = mean_squared_error(y_test, y_test_baseline)
baseline_test_mae = mean_absolute_error(y_test, y_test_baseline)

# Print results
print("Baseline Model Performance:")
print(f"Training MSE: {baseline_train_mse:.2f}, MAE: {baseline_train_mae:.2f}")
print(f"Testing MSE: {baseline_test_mse:.2f}, MAE: {baseline_test_mae:.2f}")
