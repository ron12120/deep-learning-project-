import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
file_path = "Fifa 23 Players Data.csv"
data = pd.read_csv(file_path).head(1000)

# Extracting features and target
X = data[[
    'Overall', 'Age', 'Height(in cm)', 'Weight(in kg)', 'TotalStats',
    'Weak Foot Rating', 'Skill Moves', 'Shooting Total', 'Pace Total', 'Passing Total',
    'Dribbling Total', 'Defending Total', 'Physicality Total', 'Finishing', 'Sprint Speed',
    'Agility', 'Reactions', 'Stamina', 'Strength', 'Vision', 'Penalties'
]]
y = data['Value(in Euro)']

# Splitting the data into training and testing sets with randomized shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Scaling the features and target
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

value_scaler = StandardScaler()
y_train_scaled = value_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = value_scaler.transform(y_test.values.reshape(-1, 1))

# Converting data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

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

# Training the model with randomized batches
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}')

# Testing the model on the test set
with torch.no_grad():
    y_pred_test_scaled = model(X_test_tensor)
    test_loss = criterion(y_pred_test_scaled, y_test_tensor).item()
    y_pred_test = value_scaler.inverse_transform(y_pred_test_scaled.numpy())
    y_test_original = value_scaler.inverse_transform(y_test_tensor.numpy())

print(f"Test Loss: {test_loss:.4f}")

# Display results for a few test samples
for i in range(5):
    print(f"Predicted: €{y_pred_test[i][0]:,.2f}, Actual: €{y_test_original[i][0]:,.2f}")

# Evaluate on training data
with torch.no_grad():
    y_pred_train_scaled = model(X_train_tensor)
    y_pred_train = value_scaler.inverse_transform(y_pred_train_scaled.numpy())
    y_train_original = value_scaler.inverse_transform(y_train_tensor.numpy())

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
