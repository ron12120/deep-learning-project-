import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "./Fifa_23_Players_Data_with_Wikipedia.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False).head(1701)

# Shuffle the dataset to ensure randomness
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Define numerical features and target
numerical_features = [
    'Overall', 'Age', 'Height(in cm)', 'Weight(in kg)', 'TotalStats',
    'Weak Foot Rating', 'Skill Moves', 'Shooting Total', 'Pace Total',
    'Passing Total', 'Dribbling Total', 'Defending Total', 'Physicality Total',
    'Finishing', 'Sprint Speed', 'Agility', 'Reactions', 'Stamina',
    'Strength', 'Vision', 'Penalties']
X = data[numerical_features]
y = data['Value(in Euro)']

# Log-transform the target variable to reduce scale skewness
y_log = np.log1p(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42, shuffle=True)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add bias (intercept) term to the scaled features
X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

# Implement Linear Regression using the Normal Equation
# theta = (X.T * X)^-1 * X.T * y
theta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ y_train

# Predictions
y_train_pred_log = X_train_scaled @ theta
y_test_pred_log = X_test_scaled @ theta

# Inverse the log transformation to compare with original values
y_train_actual = np.expm1(y_train)  # Original training values
y_test_actual = np.expm1(y_test)    # Original testing values
y_train_pred_actual = np.expm1(y_train_pred_log)
y_test_pred_actual = np.expm1(y_test_pred_log)

# Calculate MSE and MAE
train_mse = mean_squared_error(y_train_actual, y_train_pred_actual)
train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
test_mse = mean_squared_error(y_test_actual, y_test_pred_actual)
test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)

# Calculate Training and Test Loss (Log Scale)
train_loss = mean_squared_error(y_train, y_train_pred_log)
test_loss = mean_squared_error(y_test, y_test_pred_log)

# Convert losses to percentages
train_loss_percentage = (train_loss / np.mean(y_train)) * 100
test_loss_percentage = (test_loss / np.mean(y_test)) * 100

# Print Performance Metrics
print(f"Training Set Performance:")
print(f"Mean Squared Error (MSE): {train_mse:.2f}")
print(f"Mean Absolute Error (MAE): {train_mae:.2f}")
print(f"Training Loss (Log Scale): {train_loss_percentage:.2f}%\n")

print(f"Test Set Performance:")
print(f"Mean Squared Error (MSE): {test_mse:.2f}")
print(f"Mean Absolute Error (MAE): {test_mae:.2f}")
print(f"Test Loss (Log Scale): {test_loss_percentage:.2f}%")

# Plot Training Set: Predicted vs. Actual
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train_actual, y_train_pred_actual, alpha=0.5, color='b', label='Predicted vs Actual')
plt.plot([min(y_train_actual), max(y_train_actual)],
         [min(y_train_actual), max(y_train_actual)],
         color='red', linestyle='--', label='Ideal Prediction')
plt.title('Training Set: Predicted vs Actual')
plt.xlabel('Actual Values (€)')
plt.ylabel('Predicted Values (€)')
plt.legend()

# Plot Testing Set: Predicted vs. Actual
plt.subplot(1, 2, 2)
plt.scatter(y_test_actual, y_test_pred_actual, alpha=0.5, color='g', label='Predicted vs Actual')
plt.plot([min(y_test_actual), max(y_test_actual)],
         [min(y_test_actual), max(y_test_actual)],
         color='red', linestyle='--', label='Ideal Prediction')
plt.title('Test Set: Predicted vs Actual')
plt.xlabel('Actual Values (€)')
plt.ylabel('Predicted Values (€)')
plt.legend()

# Show Plots
plt.tight_layout()
plt.show()
