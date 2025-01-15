import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
file_path = "Fifa_23_Players_Data_with_Wikipedia.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False).head(1701)

# Extracting features and target
X = data[[
    'Overall', 'Age', 'Height(in cm)', 'Weight(in kg)', 'TotalStats',
    'Weak Foot Rating', 'Skill Moves', 'Shooting Total', 'Pace Total', 'Passing Total',
    'Dribbling Total', 'Defending Total', 'Physicality Total', 'Finishing', 'Sprint Speed',
    'Agility', 'Reactions', 'Stamina', 'Strength', 'Vision', 'Penalties'
]]
y = data['Value(in Euro)']

# Splitting the data into training, validation, and testing sets with randomized shuffling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

# Scaling the features and target
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

value_scaler = StandardScaler()
y_train_scaled = value_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = value_scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = value_scaler.transform(y_test.values.reshape(-1, 1))

# Converting data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Defining a simple neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Initializing the model
input_size = X_train_tensor.shape[1]
output_size = 1
model = NeuralNetwork(input_size, output_size)

# Defining loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 20  # Number of epochs to wait for improvement in validation loss
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training the model with early stopping
num_epochs = 200
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation loss
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0  # Reset counter
    else:
        epochs_without_improvement += 1

    # Stop training if no improvement for `patience` epochs
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

# Testing the model on the test set
with torch.no_grad():
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
