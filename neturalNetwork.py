import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the data
file_path = "Fifa 23 Players Data.csv"
data = pd.read_csv(file_path)

# Extracting features and target
X = data[[
    'Overall', 'Age', 'Height(in cm)', 'Weight(in kg)', 'TotalStats',
    'Weak Foot Rating', 'Skill Moves', 'Shooting Total', 'Pace Total', 'Passing Total',
    'Dribbling Total', 'Defending Total', 'Physicality Total', 'Finishing', 'Sprint Speed',
    'Agility', 'Reactions', 'Stamina', 'Strength', 'Vision', 'Penalties'
]]
y = data['Value(in Euro)']

# Scaling the features and target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
value_scaler = StandardScaler()
y_scaled = value_scaler.fit_transform(y.values.reshape(-1, 1))

# Converting data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

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
input_size = X_tensor.shape[1]
output_size = 1
model = NeuralNetwork(input_size, output_size)

# Defining loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Making a prediction for a sample player
sample_player = data.sample(1)
sample_features = sample_player[[
    'Overall', 'Age', 'Height(in cm)', 'Weight(in kg)', 'TotalStats',
    'Weak Foot Rating', 'Skill Moves', 'Shooting Total', 'Pace Total', 'Passing Total',
    'Dribbling Total', 'Defending Total', 'Physicality Total', 'Finishing', 'Sprint Speed',
    'Agility', 'Reactions', 'Stamina', 'Strength', 'Vision', 'Penalties'
]]

sample_scaled = scaler.transform(sample_features)
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

# Predicting value
predicted_value_scaled = model(sample_tensor).detach().numpy()
predicted_value_original = value_scaler.inverse_transform(predicted_value_scaled)

# Display results
print(f"Predicted Value: €{predicted_value_original[0][0]:,.2f}")
print(f"True Value: €{sample_player['Value(in Euro)'].values[0]:,.2f}")
