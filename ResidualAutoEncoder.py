import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchmetrics as metrics
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

class ResidualEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_size=3, stride=2):
        super(ResidualEncoderBlock, self).__init__()
        layers = []
        for i in range(num_layers-1):
            stride_i = stride if i == 0 else 1
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride=stride_i, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels if num_layers == 1 else out_channels, out_channels, kernel_size, padding=1))
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
        for i in range(num_layers-1):
            stride_i = stride if i == 0 else 1
            output_padding = 1 if i == 0 and stride > 1 else 0
            layers.append(nn.ConvTranspose2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride=stride_i, padding=1, output_padding=output_padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(in_channels if num_layers == 1 else out_channels, out_channels, kernel_size, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)
        
        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=1)
        
    def forward(self, x):
        identity = x
        
        out = self.block(x)
        
        if self.projection is not None:
            identity = self.projection(identity)
        
        out += identity
        out = nn.ReLU(inplace=True)(out)
        
        return out
    
class ResidualEncoder(nn.Module):
    def __init__(self, num_blocks=3, block_depth=2, bottleneck_dim=128, input_size=128, channels=3):
        super(ResidualEncoder, self).__init__()

        encoder_layers = []
        in_channels = channels
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
    def __init__(self, num_blocks=3, block_depth=2, bottleneck_dim=128, input_size=128, channels=3):
        super(ResidualDecoder, self).__init__()

        in_channels = 64 * (2 ** (num_blocks - 1))

        # Calculate the size of the feature map after the last encoder block
        self.feature_map_size = input_size // (2 ** num_blocks)
        flattened_dim = in_channels * self.feature_map_size * self.feature_map_size  # Flattened dimension after downsampling
        
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, flattened_dim),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (in_channels, self.feature_map_size, self.feature_map_size))
        )
        
        decoder_layers = []
        out_channels = in_channels // 2
        for i in range(num_blocks):
            decoder_layers.append(ResidualDecoderBlock(in_channels, out_channels, num_layers=block_depth))
            in_channels = out_channels
            out_channels //= 2
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.final_conv = nn.Conv2d(in_channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.bottleneck_decoder(x)
        x = self.decoder(x)
        x = F.tanh(self.final_conv(x))
        return x

class ResidualAutoencoder(nn.Module):
    def __init__(self, num_blocks=3, block_depth=2, bottleneck_dim=128, input_size=128, channels=3, asym_block=1, asym_depth=1, device=torch.device('cpu')):
        super(ResidualAutoencoder, self).__init__()
        
        self.ResidualEncoder = ResidualEncoder(num_blocks=num_blocks, block_depth=block_depth, bottleneck_dim=bottleneck_dim, input_size=input_size, channels=channels)
        self.ResidualDecoder = ResidualDecoder(num_blocks=num_blocks * asym_block, block_depth=block_depth * asym_depth, bottleneck_dim=bottleneck_dim, input_size=input_size, channels=channels)
        
        self.device = device
        self.to(self.device)
    def forward(self, x):
        h = self.ResidualEncoder(x)
        x = self.ResidualDecoder(h)
        return x, h
    
    def encode(self, x):
        with torch.no_grad():
            return self.ResidualEncoder(x)
    
    def decode(self, x):
        with torch.no_grad():
            return self.ResidualDecoder(x)
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
        
    
    def train_harness(self, model, train_loader, criterion, optimizer, epochs=10):
        device = model.device
        model.train()
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        for epoch in range(epochs):
            train_loss = 0.0
            for data in train_loader:
                inputs, _ = data

                inputs = inputs.to(device)
                optimizer.zero_grad()
                with autocast():
                    # Forward pass
                    outputs, h = model(inputs)
                    loss, recon_loss = criterion(outputs, inputs, h)
            
                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                recon_loss = recon_loss.item() * inputs.size(0)
                train_loss += loss.item() * inputs.size(0)
           
            # Calculate average loss
            train_loss /= len(train_loader.dataset)
            recon_loss /= len(train_loader.dataset)
            scheduler.step(train_loss)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, recon_loss: {recon_loss:.4f} learning rate: {scheduler.get_last_lr()[0]}")
            
            torch.cuda.empty_cache()

    def evaluate_harness(self,model, test_loader, device):
        model.eval()
        with torch.no_grad():
            data_iter = iter(test_loader)
            inputs, _ = next(data_iter)
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            return inputs.cpu(), outputs.cpu()
    
    def l1_reg(self, l1_lambda_bottleneck=0.01):
        L1_term = torch.tensor(0., device=self.device)  # Ensure tensor is on the same device as the model

        for name, weights in self.named_parameters():
            if 'bias' not in name and 'bottleneck' in name:
                weights_sum = torch.sum(torch.abs(weights))
                L1_term += weights_sum * l1_lambda_bottleneck                   

        # Regularize loss using L1 regularization
        return L1_term

    def sparse_loss(self, h, p=0.1):
        q = h.mean(dim=0)
        return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).sum()

class Criterion():
    def __init__(self, model, l1_lambda_bottleneck=0.01, lambda_sparse=0.1, lambda_tv=1e-6, lambda_perceptual=0.1, lambda_mse=1.0, lambda_ssim=1.0, lambda_psnr=1.0):
        self.model = model
        self.l1_lambda_bottleneck = l1_lambda_bottleneck
        self.lambda_sparse = lambda_sparse
        self.lambda_tv = lambda_tv
        self.lambda_perceptual = lambda_perceptual
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_psnr = lambda_psnr
        
        self.reconstruction_loss = nn.MSELoss().to(self.model.device)
        self.psnr_loss = metrics.image.PeakSignalNoiseRatio().to(self.model.device)
        
        self.ssim_metric = metrics.image.StructuralSimilarityIndexMeasure().to(self.model.device)
        self.tvloss = metrics.image.TotalVariation().to(self.model.device)
        self.l1loss = self.model.l1_reg
        self.kl_divergence = self.model.sparse_loss
        self.lpips = metrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.model.device)
        
    def __call__(self, outputs,targets, h):
        recon_loss = self.reconstruction_loss(outputs, targets)
        loss = F.softplus(recon_loss) * self.lambda_mse
        loss += F.softplus(self.l1loss(l1_lambda_bottleneck=self.l1_lambda_bottleneck))
        loss += F.softplus(self.kl_divergence(h)) * self.lambda_sparse
        
        if torch.rand(1).item() < 0.5:  # 50% of the time
            loss += F.softplus(self.tvloss(outputs)) * self.lambda_tv
            
        if torch.rand(1).item() < 0.2:  # 20% of the time
            loss += F.softplus(self.lpips(outputs, targets)) * self.lambda_perceptual
        
        if torch.rand(1).item() < 0.5:  # 50% of the time
            loss += F.softplus(self.psnr_loss(outputs, targets)) * self.lambda_psnr
            
        if torch.rand(1).item() < 0.5:  # 50% of the time
            loss += F.softplus(self.ssim_metric(outputs, targets)) * self.lambda_ssim
        return loss, recon_loss
        
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
            block_depth=config.block_depth,
            bottleneck_dim=config.bottleneck_dim,
            input_size=32  # Assuming CIFAR-10 images are 32x32
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
#sweep_id = wandb.sweep(sweep_config, project="residual-autoencoder")

# Run sweep
#wandb.agent(sweep_id, train_autoencoder)





def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 30
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transformation and loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), ( 0.5)),
        transforms.Resize((128, 128))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, loss function, and optimizer
    model = ResidualAutoencoder(num_blocks=3, block_depth=2, bottleneck_dim=128, channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    model.train_harness(model, train_loader, criterion, optimizer, device, epochs)
    
    # Evaluate the model
    inputs, outputs = model.evaluate_harness(model, test_loader, device)
    outputs = unnormalize(outputs)
    # Visualize the results
    num_images = 10
    fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
    for i in range(num_images):
        axes[0, i].imshow(inputs[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        axes[1, i].imshow(outputs[i].permute(1, 2, 0))
        axes[1, i].axis('off')
    
    plt.show()
    
def unnormalize(tensor):
    return tensor * 0.5 + 0.5

if __name__ == "__main__":
    main()
