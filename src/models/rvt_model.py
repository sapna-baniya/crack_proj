"""
Residual Vision Transformer (RvT) Model for Crack Detection
Combines CNN feature extraction with Vision Transformer
"""

import torch
import torch.nn as nn
import timm
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualViTBlock(nn.Module):
    """Residual connection around ViT block"""
    
    def __init__(self, vit_block, hidden_dim):
        super(ResidualViTBlock, self).__init__()
        self.vit_block = vit_block
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """Forward with residual connection"""
        residual = x
        x = self.vit_block(x)
        x = x + self.residual_proj(residual)
        x = self.layer_norm(x)
        return x


class CNNFeatureExtractor(nn.Module):
    """CNN backbone for initial feature extraction"""
    
    def __init__(self, output_channels=256):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """Extract CNN features"""
        return self.conv_blocks(x)


class CrackDetectionRvT(nn.Module):
    """Residual Vision Transformer for binary crack detection"""
    
    def __init__(self, config_path='config/config.yaml', pretrained=True):
        """
        Initialize RvT model
        
        Args:
            config_path: Path to configuration file
            pretrained: Whether to use pretrained ViT weights
        """
        super(CrackDetectionRvT, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['rvt']
        self.num_classes = self.config['models']['cnn']['num_classes']
        
        logger.info("Building Residual Vision Transformer model...")
        
        # CNN feature extractor
        self.cnn_features = CNNFeatureExtractor(output_channels=256)
        
        # Load pre-trained ViT and modify
        base_vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        
        # Patch embedding adjustment for CNN features
        # After CNN: 224x224 -> 56x56 (after 2 maxpools of stride 2)
        # We need to convert 56x56x256 to patches
        self.patch_size = 7  # 56/7 = 8 patches per dimension -> 64 patches total
        self.num_patches = (56 // self.patch_size) ** 2
        
        # Project CNN features to ViT embedding dimension
        self.feature_projection = nn.Conv2d(
            256, 768, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # ViT components
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, 768))
        
        # Use ViT blocks with residual connections
        self.blocks = nn.ModuleList([
            ResidualViTBlock(base_vit.blocks[i], 768)
            for i in range(len(base_vit.blocks))
        ])
        
        self.norm = base_vit.norm
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(self.model_config['dropout']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.model_config['dropout']),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"RvT model initialized")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, 224, 224)
            
        Returns:
            Output predictions (B, 1)
        """
        B = x.shape[0]
        
        # Extract CNN features
        cnn_features = self.cnn_features(x)  # (B, 256, 56, 56)
        
        # Project to patches
        patches = self.feature_projection(cnn_features)  # (B, 768, 8, 8)
        patches = patches.flatten(2).transpose(1, 2)  # (B, 64, 768)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, 65, 768)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks with residual connections
        for block in self.blocks:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Classification head (use CLS token)
        cls_output = x[:, 0]
        output = self.head(cls_output)
        
        return output
    
    def freeze_cnn_backbone(self):
        """Freeze CNN feature extractor"""
        logger.info("Freezing CNN backbone...")
        for param in self.cnn_features.parameters():
            param.requires_grad = False
    
    def freeze_vit_backbone(self):
        """Freeze ViT blocks"""
        logger.info("Freezing ViT blocks...")
        for param in self.blocks.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all layers"""
        logger.info("Unfreezing all layers...")
        for param in self.parameters():
            param.requires_grad = True


class RvTWrapper:
    """Wrapper class for RvT model compatible with training pipeline"""
    
    def __init__(self, config_path='config/config.yaml', device='cuda'):
        """
        Initialize wrapper
        
        Args:
            config_path: Path to configuration file
            device: Device to use (cuda/cpu)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Build model
        self.model = CrackDetectionRvT(config_path=config_path).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        
        training_config = self.config['training']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_config['learning_rate']
        )
        
        # Learning rate scheduler
        if training_config['lr_scheduler']['enabled']:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=training_config['lr_scheduler']['factor'],
                patience=training_config['lr_scheduler']['patience'],
                min_lr=training_config['lr_scheduler']['min_lr']
            )
    
    def train_step(self, images, labels):
        """Single training step"""
        self.model.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs.squeeze(), labels.float())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, images, labels):
        """Single evaluation step"""
        self.model.eval()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            loss = self.criterion(outputs.squeeze(), labels.float())
        
        return loss.item(), outputs.cpu().numpy()
    
    def predict(self, images):
        """Make predictions"""
        self.model.eval()
        
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
        
        return outputs.cpu().numpy()
    
    def save(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")


def main():
    """Test RvT model"""
    # Build model
    rvt_wrapper = RvTWrapper()
    
    print("\n" + "="*60)
    print("RESIDUAL VISION TRANSFORMER MODEL")
    print("="*60)
    print(f"Device: {rvt_wrapper.device}")
    print(f"Total parameters: {sum(p.numel() for p in rvt_wrapper.model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(rvt_wrapper.device)
    output = rvt_wrapper.model(dummy_input)
    print(f"\nTest input shape: {dummy_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Test predictions: {output.squeeze().detach().cpu().numpy()}")


if __name__ == "__main__":
    main()