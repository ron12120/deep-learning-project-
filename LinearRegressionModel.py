import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt

# Load the data
file_path = "Fifa_23_Players_Data_with_Wikipedia.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

# Select features and target
X = df[['Overall', 'Age', 'Height(in cm)', 'Weight(in kg)', 'TotalStats',
        'Weak Foot Rating', 'Skill Moves', 'Shooting Total', 'Pace Total',
        'Passing Total', 'Dribbling Total', 'Defending Total', 'Physicality Total',
        'Finishing', 'Sprint Speed', 'Agility', 'Reactions', 'Stamina', 'Strength', 'Vision', 'Penalties']]
y = df['Value(in Euro)']

# Splitting the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Scaling features and target
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

value_scaler = StandardScaler()
y_train_scaled = value_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = value_scaler.transform(y_test.values.reshape(-1, 1))

# Converting to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = LinearRegressionModel(X_train_tensor.shape[1])

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 20
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation loss
    with torch.no_grad():
        model.eval()

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}')

# Testing the model
with torch.no_grad():
    model.eval()
    y_pred_test_scaled = model(X_test_tensor)
    test_loss = criterion(y_pred_test_scaled, y_test_tensor).item()
    y_pred_test = value_scaler.inverse_transform(y_pred_test_scaled.numpy())
    y_test_original = value_scaler.inverse_transform(y_test_tensor.numpy())

# Calculate MSE and MAE for testing data
test_mse = mean_squared_error(y_test_original, y_pred_test)
test_mae = mean_absolute_error(y_test_original, y_pred_test)

# Evaluate on training data
with torch.no_grad():
    y_pred_train_scaled = model(X_train_tensor)
    train_loss = criterion(y_pred_train_scaled, y_train_tensor).item()
    y_pred_train = value_scaler.inverse_transform(y_pred_train_scaled.numpy())
    y_train_original = value_scaler.inverse_transform(y_train_tensor.numpy())

# Calculate MSE and MAE for training data
train_mse = mean_squared_error(y_train_original, y_pred_train)
train_mae = mean_absolute_error(y_train_original, y_pred_train)

# Print performance metrics
print(f"Train Loss (MSE): {train_loss * 100:.4f} %")
print(f"Train MSE: {train_mse:.2f}, Train MAE: {train_mae:.2f}")

print(f"Test Loss (MSE): {test_loss * 100:.4f} %")
print(f"Test MSE: {test_mse:.2f}, Test MAE: {test_mae:.2f}")

# Plotting the comparison of actual vs predicted values for both training and testing sets
plt.figure(figsize=(14, 6))

# Plot for training set
plt.subplot(1, 2, 1)
plt.scatter(y_train_original, y_pred_train, alpha=0.5, color='blue', label='Train Data')
plt.plot([min(y_train_original), max(y_train_original)],
         [min(y_train_original), max(y_train_original)], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Training Set: Predicted vs Actual')
plt.xlabel('Actual Value (€)')
plt.ylabel('Predicted Value (€)')
plt.legend()

# Plot for testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test_original, y_pred_test, alpha=0.5, color='green', label='Test Data')
plt.plot([min(y_test_original), max(y_test_original)],
         [min(y_test_original), max(y_test_original)], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Testing Set: Predicted vs Actual')
plt.xlabel('Actual Value (€)')
plt.ylabel('Predicted Value (€)')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
