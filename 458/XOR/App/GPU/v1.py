import torch
import torch.nn as nn
import torch.optim as optim

print(torch.cuda.is_available())
print(torch.cuda.is_available())
print(torch.cuda.is_available())

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

# Define the neural network model
class XORNeuralNetwork(nn.Module):
    def __init__(self):
        super(XORNeuralNetwork, self).__init__()
        self.hidden = nn.Linear(2, 2)  # 2 input nodes -> 2 hidden nodes
        self.output = nn.Linear(2, 1)  # 2 hidden nodes -> 1 output node
        self.activation = nn.Sigmoid()  # Activation function

    def forward(self, x):
        x = self.activation(self.hidden(x))  # Apply activation to hidden layer
        x = self.activation(self.output(x))  # Apply activation to output layer
        return x

# Initialize model and move to GPU
model = XORNeuralNetwork().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 1000000
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, Y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Test the trained model
with torch.no_grad():
    predictions = model(X)
    print("\nPredictions after training:")
    print(predictions.cpu().numpy())  # Move to CPU for printing
