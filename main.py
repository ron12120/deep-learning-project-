import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
file_path = "fifa_players.csv"
data = pd.read_csv(file_path)

X = data[['crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys',
    'dribbling', 'curve', 'freekick_accuracy', 'long_passing', 'ball_control',
    'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 
    'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'aggression',
    'interceptions', 'positioning', 'vision', 'penalties', 'composure', 
    'marking', 'standing_tackle', 'sliding_tackle']]  # Features
y = data['overall_rating']  # Target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Model instantiation
input_dim = X_tensor.shape[1]
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Lower learning rate

# Training the model
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
predicted = model(X_tensor).detach().numpy()
print("Training completed. Here are some predicted ratings:")
print(predicted[:29])

# Test prediction on a new sample
sample_data = torch.tensor([[86, 95, 70, 92, 86, 97, 93, 94, 89, 96, 91, 86, 93, 95, 95, 85, 
    68, 72, 66, 94, 48, 22, 94, 94, 75, 96, 33, 28, 26]], dtype=torch.float32)  # Example input
sample_data_scaled = torch.tensor(scaler.transform(sample_data), dtype=torch.float32)
predicted_rating = model(sample_data_scaled).item()
print(f"Predicted Rating for sample data {sample_data}: {predicted_rating:.2f}")
