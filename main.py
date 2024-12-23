import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

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

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)


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
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10000
for epoch in range(num_epochs):
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
predicted = model(X_tensor).detach().numpy()
predicted_original = value_scaler.inverse_transform(predicted)

print("Training completed. Here are some predicted player values:")
print(predicted_original[:21])

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

print(f"Predicted Value (in Euro): {predicted_original[0][0]:.2f}")


def get_player_value(data, **kwargs):
    """
    Retrieve the Value(in Euro) of a player based on provided features.

    Parameters:
        data (pd.DataFrame): The dataset containing player information.
        **kwargs: Key-value pairs of features to match (e.g., Overall=88, Age=24).

    Returns:
        str: Player's name and their value if found, else a not found message.
    """
    query = " & ".join([f"({key} == {repr(value)})" for key, value in kwargs.items()])

    filtered_players = data.query(query)

    if not filtered_players.empty:
        player = filtered_players.iloc[0]
        return f"Player: {player['Known As']} ({player['Full Name']}), Value: €{player['Value(in Euro)']:,}"
    else:
        return "No player found with the given features."


print(player_value)