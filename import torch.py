import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Step 1: Data Loading
# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1]
])

# Download and load the training and test dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Define the Autoencoder Model
class LinearAutoencoder(nn.Module):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 4)  # Compress to 4 latent variables
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Tanh()  # Output layer with Tanh to match input normalization
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Step 3: Initialize the Model, Loss Function, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LinearAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0008)

# Step 4: Train the Autoencoder
def train(model, train_loader, criterion, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.view(inputs.size(0), -1).to(device)  # Flatten the images
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")

# Step 5: Evaluate the Autoencoder
def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            #inputs += 0.6 * torch.randn_like(inputs)  # Add noise to the inputs
            for input in inputs:
                input[ :, :, :input.size(2)//2] = 0.0
            inputs = inputs.view(inputs.size(0), -1).to(device)
            outputs = model(inputs)
            return inputs, outputs

# Step 6: Visualize the Results
def visualize(inputs, outputs):
    inputs = inputs.view(-1, 28, 28).cpu().numpy()
    outputs = outputs.view(-1, 28, 28).cpu().numpy()
    
    num_images = 10
    fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
    for i in range(num_images):
        axes[0, i].imshow(inputs[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(outputs[i], cmap='gray')
        axes[1, i].axis('off')
    plt.show()

# Main function to run the steps
def main():
    train(model, train_loader, criterion, optimizer, device, epochs=10)
    inputs, outputs = evaluate(model, test_loader, device)
    visualize(inputs, outputs)

if __name__ == "__main__":
    main()
