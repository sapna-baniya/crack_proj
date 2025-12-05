"""
Vision Transformer (ViT) Model for Crack Detection
Using pre-trained ViT from timm library with fine-tuning
"""

import torch
import torch.nn as nn
import timm
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrackDetectionViT(nn.Module):
    """Vision Transformer for binary crack detection"""
    
    def __init__(self, config_path='config/config.yaml', pretrained=True):
        """
        Initialize ViT model
        
        Args:
            config_path: Path to configuration file
            pretrained: Whether to use pretrained weights
        """
        super(CrackDetectionViT, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['vit']
        self.num_classes = self.config['models']['cnn']['num_classes']  # Binary classification
        
        # Load pre-trained ViT
        logger.info("Loading Vision Transformer model...")
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=1  # Binary classification (sigmoid output)
        )
        
        # Modify the head for binary classification
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(self.model_config['dropout']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.model_config['dropout']),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"ViT model initialized")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output predictions (B, 1)
        """
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the head"""
        logger.info("Freezing backbone layers...")
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        
        logger.info(f"Trainable parameters after freezing: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        logger.info("Unfreezing all layers...")
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def unfreeze_last_n_blocks(self, n=2):
        """
        Unfreeze the last n transformer blocks
        
        Args:
            n: Number of blocks to unfreeze
        """
        logger.info(f"Unfreezing last {n} transformer blocks...")
        
        # First freeze all
        self.freeze_backbone()
        
        # Unfreeze last n blocks
        total_blocks = len(self.model.blocks)
        for i in range(total_blocks - n, total_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True
        
        # Unfreeze norm and head
        for param in self.model.norm.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")


class ViTWrapper:
    """Wrapper class for ViT model compatible with training pipeline"""
    
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
        self.model = CrackDetectionViT(config_path=config_path).to(self.device)
        
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
        """
        Single training step
        
        Args:
            images: Batch of images
            labels: Batch of labels
            
        Returns:
            loss: Training loss
        """
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
        """
        Single evaluation step
        
        Args:
            images: Batch of images
            labels: Batch of labels
            
        Returns:
            loss: Validation loss
            predictions: Model predictions
        """
        self.model.eval()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            loss = self.criterion(outputs.squeeze(), labels.float())
        
        return loss.item(), outputs.cpu().numpy()
    
    def predict(self, images):
        """
        Make predictions
        
        Args:
            images: Batch of images
            
        Returns:
            predictions: Model predictions
        """
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
    """Test ViT model"""
    import numpy as np
    
    # Build model
    vit_wrapper = ViTWrapper()
    
    print("\n" + "="*60)
    print("VISION TRANSFORMER MODEL")
    print("="*60)
    print(f"Device: {vit_wrapper.device}")
    print(f"Total parameters: {sum(p.numel() for p in vit_wrapper.model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(vit_wrapper.device)
    output = vit_wrapper.model(dummy_input)
    print(f"\nTest input shape: {dummy_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Test predictions: {output.squeeze().detach().cpu().numpy()}")


if __name__ == "__main__":
    main()