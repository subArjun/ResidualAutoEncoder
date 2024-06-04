import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchmetrics as metrics
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.quantization import fuse_modules, quantize_dynamic
import torch.onnx
import wandb
import numpy as np

# Residual Encoder Block
class ResidualEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_size=3, stride=2, dropout=0.2):
        super(ResidualEncoderBlock, self).__init__()
        layers = []
        # Creating the layers for the block
        for i in range(num_layers-1):
            stride_i = stride if i == 0 else 1
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride=stride_i, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Conv2d(in_channels if num_layers == 1 else out_channels, out_channels, kernel_size, padding=1, stride=1 if num_layers > 1 else 2))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)
        
        # Projection layer if dimensions change
        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x  # Save the input for residual connection
        out = self.block(x)  # Pass through the block
        if self.projection is not None:
            identity = self.projection(identity)  # Adjust dimensions if necessary
        out += identity  # Add the residual
        out = self.relu(out)  # Apply activation
        return out

# Residual Decoder Block
class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_size=3, stride=2, dropout=0.2):
        super(ResidualDecoderBlock, self).__init__()
        layers = []
        # Creating the layers for the block
        for i in range(num_layers-1):
            stride_i = stride if i == 0 else 1
            output_padding = 1 if i == 0 and stride > 1 else 0
            layers.append(nn.ConvTranspose2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride=stride_i, padding=1, output_padding=output_padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ConvTranspose2d(in_channels if num_layers == 1 else out_channels, out_channels, kernel_size, stride=1 if num_layers > 1 else 2, padding=1, output_padding=1 if num_layers == 1 else 0))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)
        
        # Projection layer if dimensions change
        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=1)
            
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x  # Save the input for residual connection
        out = self.block(x)  # Pass through the block
        if self.projection is not None:
            identity = self.projection(identity)  # Adjust dimensions if necessary
        out += identity  # Add the residual
        out = self.relu(out)  # Apply activation
        return out
    
# Residual Encoder
class ResidualEncoder(nn.Module):
    def __init__(self, num_blocks=3, block_depth=2, bottleneck_dim=128, input_size=128, channels=3, dropout=0.2):
        super(ResidualEncoder, self).__init__()

        encoder_layers = []
        in_channels = channels
        out_channels = 64
        for i in range(num_blocks):
            encoder_layers.append(ResidualEncoderBlock(in_channels, out_channels, num_layers=block_depth, dropout=dropout))
            in_channels = out_channels
            out_channels *= 2
        
        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate the size of the feature map after the last encoder block
        self.feature_map_size = input_size // (2 ** num_blocks)
        flattened_dim = in_channels * self.feature_map_size * self.feature_map_size  # Flattened dimension after downsampling
        
        # Bottleneck layer
        self.bottleneck_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, bottleneck_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)  # Pass through encoder blocks
        x = self.bottleneck_encoder(x)  # Pass through bottleneck layer
        return x
    
# Residual Decoder
class ResidualDecoder(nn.Module):
    def __init__(self, num_blocks=3, block_depth=2, bottleneck_dim=128, input_size=128, channels=3, dropout=0.2):
        super(ResidualDecoder, self).__init__()

        in_channels = 64 * (2 ** (num_blocks - 1))

        # Calculate the size of the feature map after the last encoder block
        self.feature_map_size = input_size // (2 ** num_blocks)
        flattened_dim = in_channels * self.feature_map_size * self.feature_map_size  # Flattened dimension after downsampling
        
        # Bottleneck layer
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, flattened_dim),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (in_channels, self.feature_map_size, self.feature_map_size))
        )
        
        decoder_layers = []
        out_channels = in_channels // 2
        for i in range(num_blocks):
            decoder_layers.append(ResidualDecoderBlock(in_channels, out_channels, num_layers=block_depth, dropout=dropout))
            in_channels = out_channels
            out_channels //= 2
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.final_conv = nn.Conv2d(in_channels, channels, kernel_size=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bottleneck_decoder(x)  # Pass through bottleneck layer
        x = self.decoder(x)  # Pass through decoder blocks
        x = self.final_conv(x)  # Apply final convolution and tanh activation
        x = self.relu(x)
        return x

# Residual Autoencoder
class ResidualAutoencoder(nn.Module):
    def __init__(self, num_blocks=3, block_depth=2, bottleneck_dim=128, input_size=128, channels=3, asym_block=1, asym_depth=1, dropout=0.2, device=torch.device('cpu')):
        super(ResidualAutoencoder, self).__init__()
        self.num_blocks = num_blocks
        self.block_depth = block_depth
        self.bottleneck_dim = bottleneck_dim
        self.asym_block = asym_block
        self.asym_depth = asym_depth
        self.input_size = input_size
        self.channels = channels
        self.ResidualEncoder = ResidualEncoder(num_blocks=num_blocks, block_depth=block_depth, bottleneck_dim=bottleneck_dim, input_size=input_size, channels=channels, dropout=dropout)
        self.ResidualDecoder = ResidualDecoder(num_blocks=num_blocks * asym_block, block_depth=block_depth * asym_depth, bottleneck_dim=bottleneck_dim, input_size=input_size, channels=channels, dropout=dropout)
        
        self.device = device
        self.to(self.device)
        self.apply(self.weights_init)
        
    def forward(self, x):
        h = self.ResidualEncoder(x)  # Encode input
        x = self.ResidualDecoder(h)  # Decode latent representation
        return x, h
    
    def encode(self, x):
        with torch.no_grad():
            return self.ResidualEncoder(x)  # Encode input without gradients
    
    def decode(self, x):
        with torch.no_grad():
            return self.ResidualDecoder(x)  # Decode latent representation without gradients
        
    def save(self, path):
        torch.save(self.state_dict(), path)  # Save model state
        
    def load(self, path):
        self.load_state_dict(torch.load(path))  # Load model state
        
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)  # Initialize weights with Kaiming normal
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases with zero

    def prune_model(self, amount=0.4):
        model = self.cpu()
        for module in model.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        return model

    def fuse_model(self, model):
        for module_name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                for block_name, block in module.named_children():
                    if isinstance(block, ResidualEncoderBlock) or isinstance(block, ResidualDecoderBlock):
                        if self.block_depth == 2:
                            fuse_modules(block.block, [['0', '1', '2'], ['4', '5']], inplace=True)
                        else:
                            fuse_modules(block.block, [['0', '1', '2'], ['3', '4'], ['6', '7']], inplace=True)
        return model    

    def quantize_model(self, model):
        model.eval()
        quantized_model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8,)
        return quantized_model

    def script_model(self, model):
        model.eval()
        return torch.jit.script(model)
    
    def convert_to_onnx(self, model, path='model.onnx'):
        dummy_input = torch.randn(1, 3, 128, 128)
        torch.onnx.export(model, dummy_input, path, verbose=True, input_names=['input'], output_names=['output'])

    def optimize_for_inference(self, path='models/model.pt', type='lite', pruning_amount=0.4, save=False):
        if type == 'lite':
            pruned_model = self.prune_model(pruning_amount)
            fused_model = self.fuse_model(pruned_model)
            quantized_model = self.quantize_model(fused_model)
            script_model = self.script_model(quantized_model)
            frozen_model = torch.jit.optimize_for_inference(script_model)
            if save:
                torch.jit.save(frozen_model, path)
            return frozen_model
        elif type == 'micro':
            pruned_model = self.prune_model(pruning_amount)
            quantized_model = self.quantize_model(pruned_model)
            if save:
                self.convert_to_onnx(quantized_model, path)
            return quantized_model
        
    def load_for_optimized_inference(self, path):
        model = torch.jit.load(path)
        model.eval()
        return model

    def train_harness(self, model, train_loader, test_loader, criterion, optimizer, epochs=10, wandb_log=False):
        device = model.device
        model.train()
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        for epoch in range(epochs):
            train_loss = 0.0
            reconstruction_loss = 0.0
            for data in train_loader:
                inputs, _ = data

                inputs = inputs.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs, h = model(inputs)  # Forward pass
                    loss, recon_loss = criterion(outputs, inputs, h)  # Calculate loss
            
                scaler.scale(loss).backward()  # Backward pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
                scaler.step(optimizer)  # Optimizer step
                scaler.update()
                
                reconstruction_loss += recon_loss.item() * inputs.size(0)  # Accumulate reconstruction loss
                train_loss += loss.item() * inputs.size(0)  # Accumulate total loss
           
            train_loss /= len(train_loader.dataset)  # Average total loss
            reconstruction_loss /= len(train_loader.dataset)  # Average reconstruction loss
            _, _, val_loss = model.evaluate_harness(model, test_loader, device)  # Evaluate on validation set
            scheduler.step(val_loss)  # Step scheduler
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, recon_loss: {recon_loss:.4f}, val_loss: {val_loss:.4f} learning rate: {scheduler.get_last_lr()[0]}")
            torch.cuda.empty_cache()
            if wandb_log:
                wandb.log({'Epoch': epoch+1, 'Loss': train_loss, 'Reconstruction Loss': reconstruction_loss, 'val_loss': val_loss, 'Learning Rate': scheduler.get_last_lr()[0], 'parameters': count_parameters(model)})
            
    def evaluate_harness(self, model, test_loader, device):
        model.to(device)
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs, _ = model(inputs)
                loss = F.mse_loss(outputs, inputs, reduction='mean')
                total_loss += loss.item()
        average_loss = total_loss / len(test_loader)  # Average loss
        return inputs.cpu(), outputs.cpu(), average_loss
    
    def l1_reg(self, l1_lambda_bottleneck=0.01):
        L1_term = torch.tensor(0., device=self.device)  # Initialize L1 term
        for name, weights in self.named_parameters():
            if 'bias' not in name and 'bottleneck' in name:
                weights_sum = torch.sum(torch.abs(weights))  # Calculate L1 norm
                L1_term += weights_sum * l1_lambda_bottleneck  # Add to L1 term
        return L1_term
    
def wandb_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Data transformation and loading
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
        model = ResidualAutoencoder(num_blocks=config.num_blocks, 
                                    block_depth=config.block_depth, 
                                    bottleneck_dim=config.bottleneck_dim, 
                                    channels=3, 
                                    input_size=128, 
                                    asym_block=config.asym_block, 
                                    asym_depth=config.asym_depth,
                                    dropout=config.dropout, 
                                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                                                        
        criterion = Criterion(model, 
                            lambda_kl=config.lambda_kl, 
                            lambda_perceptual=config.lambda_perceptual, 
                            lambda_mse=config.lambda_mse, 
                            lambda_ssim=config.lambda_ssim
                            )
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        model.train_harness(model, train_loader, test_loader, criterion, optimizer, epochs=25, wandb_log=True)

#Custom autoencoder criterion class        
class Criterion():
    def __init__(self, model, lambda_kl=1e-6,lambda_perceptual=1e-5, lambda_mse=30.0, lambda_ssim=1e-6, greedy=False):
        # Initialize the criterion with the model and hyperparameters
        self.model = model
        
        #Greedy optimization for Reconstruction loss will preserve the sparsity of the latent representation
        self.greedy = greedy
        
        #Sparsity constraint
        self.lambda_kl = lambda_kl
        
        #Perceptual loss
        self.lambda_perceptual = lambda_perceptual
        
        #Reconstruction loss
        self.lambda_mse = lambda_mse
        
        #Structural similarity loss
        self.lambda_ssim = lambda_ssim
        
        self.reconstruction_loss = nn.MSELoss()
        self.ssim_metric = metrics.image.StructuralSimilarityIndexMeasure().to(self.model.device)
        self.kl_divergence = nn.KLDivLoss(reduction='batchmean')
        self.lpips = metrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.model.device)
        self.lpips_flag = False
        
    def __call__(self, outputs, targets, h):
        if self.greedy:
            recon_loss = self.reconstruction_loss(outputs, targets)
            loss = F.softplus(recon_loss) * self.lambda_mse
            
            sparse_loss = F.softplus(self.kl_divergence(h, torch.zeros_like(h))) * self.lambda_kl 
            loss += sparse_loss
            
            return loss, recon_loss
        
        else:    
            recon_loss = self.reconstruction_loss(outputs, targets)
            loss = F.softplus(recon_loss) * self.lambda_mse
            
            sparse_loss = F.softplus(self.kl_divergence(h, torch.zeros_like(h))) * self.lambda_kl
            loss += sparse_loss

            if torch.rand(1).item() < 0.5:
                lpips_loss = F.softplus(self.lpips(outputs, targets)) * self.lambda_perceptual
                loss += lpips_loss
                self.lpips_flag = True
                
            if not self.lpips_flag:
                ssim_loss = F.softplus((1 - self.ssim_metric(outputs, targets))) * self.lambda_ssim
                loss += ssim_loss
            self.lpips_flag = False
                
            return loss, recon_loss




#Helper functions        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# De-normalize function for visualization
def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_images(tensor, title="Images"):
    tensor = denormalize(tensor)
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (0, 2, 3, 1))
    fig, axes = plt.subplots(1, len(tensor), figsize=(15, 8))
    for i, img in enumerate(tensor):
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.suptitle(title)
    plt.show()