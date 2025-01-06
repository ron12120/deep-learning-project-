import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess the data
file_path = "Fifa 23 Players Data.csv"
data = pd.read_csv(file_path)

X = data[['Overall', 'Age', 'Height(in cm)', 'Weight(in kg)', 'TotalStats',
    'Weak Foot Rating', 'Skill Moves', 'Shooting Total', 'Pace Total', 'Passing Total',
    'Dribbling Total', 'Defending Total', 'Physicality Total', 'Finishing', 'Sprint Speed',
          'Agility', 'Reactions', 'Stamina', 'Strength', 'Vision', 'Penalties']]  # Features
y = data['Value(in Euro)']  # Target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalize target variable
value_scaler = StandardScaler()
y_scaled = value_scaler.fit_transform(y.values.reshape(-1, 1))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define a more complex neural network model
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkModel, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.hidden2 = nn.Linear(128, 64)         # Second hidden layer
        self.output = nn.Linear(64, output_dim)   # Output layer
        self.dropout = nn.Dropout(0.2)            # Dropout to prevent overfitting

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.dropout(x)
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# Model instantiation
input_dim = X_train_tensor.shape[1]
output_dim = 1
model = NeuralNetworkModel(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on training and testing data
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    y_train_pred_scaled = model(X_train_tensor).detach().numpy()
    y_train_pred = value_scaler.inverse_transform(y_train_pred_scaled)
    y_train_original = value_scaler.inverse_transform(y_train_tensor.numpy())

    y_test_pred_scaled = model(X_test_tensor).detach().numpy()
    y_test_pred = value_scaler.inverse_transform(y_test_pred_scaled)
    y_test_original = value_scaler.inverse_transform(y_test_tensor.numpy())

# Calculate Evaluation Metrics
train_mse = mean_squared_error(y_train_original, y_train_pred)
train_mae = mean_absolute_error(y_train_original, y_train_pred)
train_r2 = r2_score(y_train_original, y_train_pred)

test_mse = mean_squared_error(y_test_original, y_test_pred)
test_mae = mean_absolute_error(y_test_original, y_test_pred)
test_r2 = r2_score(y_test_original, y_test_pred)

# Print Training and Testing Loss
print(f"Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
print(f"Testing MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# Print some samples from the test set
print("\nTest Samples - Real vs Predicted:")
for i in range(5):  # Print the first 5 samples
    print(f"Real Value: €{y_test_original[i][0]:,.2f}, Predicted Value: €{y_test_pred[i][0]:,.2f}")

# Plotting the results
plt.figure(figsize=(12, 6))

# Training Set: Predicted vs Actual
plt.subplot(1, 2, 1)
plt.scatter(y_train_original, y_train_pred, alpha=0.5, color='blue')
plt.plot([min(y_train_original), max(y_train_original)],
         [min(y_train_original), max(y_train_original)],
         color='red', linestyle='--')
plt.title('Training Set: Predicted vs Actual')
plt.xlabel('Actual Value (€)')
plt.ylabel('Predicted Value (€)')

# Testing Set: Predicted vs Actual
plt.subplot(1, 2, 2)
plt.scatter(y_test_original, y_test_pred, alpha=0.5, color='green')
plt.plot([min(y_test_original), max(y_test_original)],
         [min(y_test_original), max(y_test_original)],
         color='red', linestyle='--')
plt.title('Testing Set: Predicted vs Actual')
plt.xlabel('Actual Value (€)')
plt.ylabel('Predicted Value (€)')

# Show Plots
plt.tight_layout()
plt.show()

# Test prediction on a new sample
sample_data = pd.DataFrame({
    'Overall': [88],
    'Age': [24],
    'Height(in cm)': [180],
    'Weight(in kg)': [75],
    'TotalStats': [2200],
    'Weak Foot Rating': [4],
    'Skill Moves': [4],
    'Shooting Total': [85],
    'Pace Total': [88],
    'Passing Total': [82],
    'Dribbling Total': [87],
    'Defending Total': [60],
    'Physicality Total': [78],
    'Finishing': [83],
    'Sprint Speed': [90],
    'Agility': [86],
    'Reactions': [85],
    'Stamina': [84],
    'Strength': [80],
    'Vision': [83],
    'Penalties': [75]
})

# Normalize using the same scaler used during training
sample_data_scaled = scaler.transform(sample_data)

# Convert to tensor
sample_tensor = torch.tensor(sample_data_scaled, dtype=torch.float32)

# Make a prediction
predicted_value = model(sample_tensor).detach().numpy()
predicted_original = value_scaler.inverse_transform(predicted_value)

print(f"Predicted Value (in Euro): €{predicted_original[0][0]:,.2f}")
