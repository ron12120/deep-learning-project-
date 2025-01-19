import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
file_path = './Fifa_23_Players_Data_with_Wikipedia.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False).head(1701)

# Define target variable
y = df['Value(in Euro)']  # Replace with your actual target column name

# Convert target variable to NumPy array
y = y.to_numpy().astype(float).flatten()

# Baseline model
baseline_prediction = np.mean(y)

# Create baseline predictions
y_baseline = np.full(y.shape, baseline_prediction)

# Evaluate baseline performance
baseline_mse = mean_squared_error(y, y_baseline)
baseline_mae = mean_absolute_error(y, y_baseline)

# Calculate the mean squared value of y
mean_squared_value_y = np.mean(y ** 2)

# Calculate percentage metrics correctly
mse_percentage = (baseline_mse / mean_squared_value_y) * 100

# Print corrected results
print("Baseline Model Performance:")
print(f"MSE: {mse_percentage:.2f}%")
print(f"MAE: {baseline_mae:.2f}")


# Plot the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y)), y, color='blue', alpha=0.5, label='Actual Values')
plt.axhline(y=baseline_prediction, color='red', linestyle='--', label='Baseline Prediction')
plt.title('Actual Values vs Baseline Prediction')
plt.xlabel('Index')
plt.ylabel('Value (in Euro)')
plt.legend()
plt.show()
