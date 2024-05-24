import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

class ResidualEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_size=3, stride=2):
        super(ResidualEncoderBlock, self).__init__()
        layers = []
        for i in range(num_layers-1):
            stride_i = stride if i == 0 else 1
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride=stride_i, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)
        
        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        
    def forward(self, x):
        identity = x
        
        out = self.block(x)
        
        if self.projection is not None:
            identity = self.projection(identity)
        
        out += identity
        out = nn.ReLU(inplace=True)(out)
        
        return out

class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_size=3, stride=2):
        super(ResidualDecoderBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            stride_i = stride if i == 0 else 1
            output_padding = 1 if i == 0 and stride > 1 else 0
            layers.append(nn.ConvTranspose2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride=stride_i, padding=1, output_padding=output_padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        
        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding)
        
    def forward(self, x):
        identity = x
        
        out = self.block(x)
        
        if self.projection is not None:
            identity = self.projection(identity)
        
        out += identity
        out = nn.ReLU(inplace=True)(out)
        
        return out
    
class ResidualEncoder(nn.Module):
    def __init__(self, num_blocks=3, block_depth=2, bottleneck_dim=128, input_size=128):
        super(ResidualEncoder, self).__init__()

        encoder_layers = []
        in_channels = 3
        out_channels = 64
        for i in range(num_blocks):
            encoder_layers.append(ResidualEncoderBlock(in_channels, out_channels, num_layers=block_depth))
            in_channels = out_channels
            out_channels *= 2
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate the size of the feature map after the last encoder block
        self.feature_map_size = input_size // (2 ** num_blocks)
        flattened_dim = in_channels * self.feature_map_size * self.feature_map_size  # Flattened dimension after downsampling
        
        self.bottleneck_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, bottleneck_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck_encoder(x)
        return x
    

class ResidualDecoder(nn.Module):
    def __init__(self, num_blocks=3, block_depth=2, bottleneck_dim=128, input_size=128):
        super(ResidualDecoder, self).__init__()


        # Calculate the size of the feature map after the last encoder block
        self.feature_map_size = input_size // (2 ** num_blocks)
        flattened_dim = in_channels * self.feature_map_size * self.feature_map_size  # Flattened dimension after downsampling
        
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, flattened_dim),
            nn.ReLU(inplace=True)
            nn.Unflatten(1, (in_channels, self.feature_map_size, self.feature_map_size))
        )
        
        decoder_layers = []
        out_channels = in_channels // 2
        for i in range(num_blocks):
            decoder_layers.append(ResidualDecoderBlock(in_channels, out_channels, num_layers=block_depth))
            in_channels = out_channels
            out_channels //= 2
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        

    def forward(self, x):
        x = self.bottleneck_decoder(x)
        x = self.decoder(x)
        return x

class ResidualAutoencoder(nn.Module):
    def __init__(self, num_blocks=3, block_depth=2, bottleneck_dim=128, input_size=128):
        super(ResidualAutoencoder, self).__init__()
        
        self.ResidualEncoder = ResidualEncoder(num_blocks=num_blocks, block_depth=block_depth, bottleneck_dim=bottleneck_dim, input_size=input_size)
        
        self.ResidualDecoder = ResidualDecoder(num_blocks=num_blocks, block_depth=block_depth, bottleneck_dim=bottleneck_dim, input_size=input_size)i      
        
    def forward(self, x):
        x = self.ResidualEncoder(x)
        x = self.ResidualDecoder(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_autoencoder(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Data transformation and loading
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        # Initialize model, loss function, and optimizer
        model = ResidualAutoencoder(
            num_blocks=config.num_blocks,
            num_blocks=config.num_blocks,
            block_depth=config.block_depth,
            block_depth=config.block_depth,
            bottleneck_dim=config.bottleneck_dim,
            input_size=128  # Assuming CIFAR-10 images are resized to 128x128
        ).to(config.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        num_params = count_parameters(model)
        
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.to(config.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            # Calculate average loss
            train_loss /= len(train_loader.dataset)
            
            # Log metrics to wandb
            wandb.log({"epoch": epoch + 1, "loss": train_loss, "num_params": num_params, "bottleneck_dim": config.bottleneck_dim})
            print(f"Epoch [{epoch + 1}/{config.epochs}], Loss: {train_loss:.4f}, Num Params: {num_params}")

# Define hyperparameter sweep configuration
sweep_config = {
    'method': 'random',  # or 'grid'
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'epochs': {
            'values': [10]  # Fixed number of epochs for training
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01
        },
        'num_blocks': {
            'values': [2, 3, 4]
        },
        'num_blocks': {
            'values': [2, 3, 4]
        },
        'block_depth': {
            'values': [1, 2, 3]
        },
        'block_depth': {
            'values': [1, 2, 3]
        },
        'bottleneck_dim': {
            'values': [64, 128, 256]
        },
        'device': {
            'values': ['cuda' if torch.cuda.is_available() else 'cpu']
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="residual-autoencoder")

# Run sweep
wandb.agent(sweep_id, train_autoencoder)
