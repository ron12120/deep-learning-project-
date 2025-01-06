import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt

# Load the data
file_path = "./Fifa_23_Players_with_Wikipedia.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False).head(1000)

# Shuffle the dataset manually to ensure randomness
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Extracting numerical features and target
numerical_features = [
    'Overall', 'Age', 'Height(in cm)', 'Weight(in kg)', 'TotalStats',
    'Weak Foot Rating', 'Skill Moves', 'Shooting Total', 'Pace Total',
    'Passing Total', 'Dribbling Total', 'Defending Total', 'Physicality Total',
    'Finishing', 'Sprint Speed', 'Agility', 'Reactions', 'Stamina',
    'Strength', 'Vision', 'Penalties'
]

X_num = data[numerical_features]
y = data['Value(in Euro)']
X_text = data['Wikipedia_Intro']  # Text column

print(data.columns)


# Splitting the data into training and testing sets after shuffling
X_num_train, X_num_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_num, X_text, y, test_size=0.2, random_state=None, shuffle=True
)

# Scaling numerical features
scaler = StandardScaler()
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled = scaler.transform(X_num_test)

value_scaler = StandardScaler()
y_train_scaled = value_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = value_scaler.transform(y_test.values.reshape(-1, 1))

# Tokenize text data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_text(text_list):
    # Ensure all values are strings and filter out any invalid entries
    text_list = text_list.fillna('')  # Replace NaN values with empty strings
    text_list = text_list.astype(str)  # Convert all entries to strings
    return tokenizer(
        text_list.tolist(),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=128
    )

# Tokenizing the text columns
X_text_train_tokens = tokenize_text(X_text_train)
X_text_test_tokens = tokenize_text(X_text_test)

# PyTorch tensors
X_num_train_tensor = torch.tensor(X_num_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_num_test_tensor = torch.tensor(X_num_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


# Define the hybrid RNN model
class HybridRNNModel(nn.Module):
    def __init__(self, num_input_size, output_size, vocab_size, embedding_dim, rnn_hidden_size):
        super(HybridRNNModel, self).__init__()

        # Text branch using RNN (LSTM)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, rnn_hidden_size, batch_first=True)
        self.text_fc = nn.Linear(rnn_hidden_size, 128)

        # Numerical branch
        self.num_fc = nn.Sequential(
            nn.Linear(num_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Combined layers
        self.fc_combined = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, num_inputs, text_inputs):
        # Text processing
        text_embeddings = self.embedding(text_inputs['input_ids'])
        _, (hidden, _) = self.rnn(text_embeddings)
        text_features = self.text_fc(hidden[-1])  # Take the last layer's hidden state

        # Numerical processing
        num_features = self.num_fc(num_inputs)

        # Combine both features
        combined = torch.cat((text_features, num_features), dim=1)
        output = self.fc_combined(combined)
        return output


# Initialize the model
vocab_size = tokenizer.vocab_size
embedding_dim = 128
rnn_hidden_size = 64
input_size = X_num_train_tensor.shape[1]

model = HybridRNNModel(
    num_input_size=input_size,
    output_size=1,
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_hidden_size=rnn_hidden_size
)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_num_train_tensor, X_text_train_tokens)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_test_scaled = model(X_num_test_tensor, X_text_test_tokens)
    test_loss = criterion(y_pred_test_scaled, y_test_tensor).item()
    y_pred_test = value_scaler.inverse_transform(y_pred_test_scaled.numpy())
    y_test_original = value_scaler.inverse_transform(y_test_tensor.numpy())

print(f"Test Loss: {test_loss:.4f}")

# Display results
for i in range(5):
    print(f"Predicted: €{y_pred_test[i][0]:,.2f}, Actual: €{y_test_original[i][0]:,.2f}")

# Evaluate on training data
model.eval()
with torch.no_grad():
    y_pred_train_scaled = model(X_num_train_tensor, X_text_train_tokens)
    train_loss = criterion(y_pred_train_scaled, y_train_tensor).item()
    y_pred_train = value_scaler.inverse_transform(y_pred_train_scaled.numpy())
    y_train_original = value_scaler.inverse_transform(y_train_tensor.numpy())

print(f"Train Loss: {train_loss:.4f}")

# Plot Training Set: Predicted vs. Actual
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train_original, y_pred_train, alpha=0.5, color='b')
plt.plot([min(y_train_original), max(y_train_original)],
         [min(y_train_original), max(y_train_original)],
         color='red', linestyle='--')
plt.title('Training Set: Predicted vs Actual')
plt.xlabel('Actual Value (€)')
plt.ylabel('Predicted Value (€)')

# Plot Testing Set: Predicted vs. Actual
plt.subplot(1, 2, 2)
plt.scatter(y_test_original, y_pred_test, alpha=0.5, color='g')
plt.plot([min(y_test_original), max(y_test_original)],
         [min(y_test_original), max(y_test_original)],
         color='red', linestyle='--')
plt.title('Testing Set: Predicted vs Actual')
plt.xlabel('Actual Value (€)')
plt.ylabel('Predicted Value (€)')

# Show Plots
plt.tight_layout()
plt.show()
