"""
Data preprocessing for crack detection
- Load images from S3 or local
- Resize and normalize
- Data augmentation
- Train/val/test split
- Save processed data
"""

import os
import numpy as np
import cv2
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
import json
from sklearn.model_selection import train_test_split
import boto3
from io import BytesIO
from PIL import Image
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess crack detection images"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize preprocessor
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_dir = Path(self.config['data']['local']['raw_data_dir'])
        self.processed_data_dir = Path(self.config['data']['local']['processed_data_dir'])
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_size = tuple(self.config['data']['image']['target_size'])
        self.image_format = self.config['data']['image']['format']
        
        # Train/val/test split ratios
        self.train_ratio = self.config['data']['split']['train_ratio']
        self.val_ratio = self.config['data']['split']['val_ratio']
        self.test_ratio = self.config['data']['split']['test_ratio']
        self.random_seed = self.config['data']['split']['random_seed']
        
        # S3 client (if needed)
        try:
            self.s3_client = boto3.client('s3', region_name=self.config['aws']['region'])
            self.use_s3 = True
        except:
            self.use_s3 = False
            logger.warning("S3 not configured, will use local data only")
    
    def load_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            
            if img is None:
                logger.warning(f"Could not read image: {image_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def load_dataset_from_directory(self, data_dir):
        """
        Load dataset from directory structure
        
        Args:
            data_dir: Directory containing positive/ and negative/ folders
            
        Returns:
            images: numpy array of images
            labels: numpy array of labels
            paths: list of image paths
        """
        images = []
        labels = []
        paths = []
        
        # Load positive samples (label = 1)
        positive_dir = data_dir / 'positive'
        if positive_dir.exists():
            logger.info(f"Loading positive samples from {positive_dir}")
            pos_files = list(positive_dir.rglob('*.jpg')) + list(positive_dir.rglob('*.png'))
            
            for img_path in tqdm(pos_files, desc="Loading positive"):
                img = self.load_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(1)
                    paths.append(str(img_path))
        
        # Load negative samples (label = 0)
        negative_dir = data_dir / 'negative'
        if negative_dir.exists():
            logger.info(f"Loading negative samples from {negative_dir}")
            neg_files = list(negative_dir.rglob('*.jpg')) + list(negative_dir.rglob('*.png'))
            
            for img_path in tqdm(neg_files, desc="Loading negative"):
                img = self.load_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(0)
                    paths.append(str(img_path))
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images")
        logger.info(f"  - Positive (cracks): {np.sum(labels == 1)}")
        logger.info(f"  - Negative (no cracks): {np.sum(labels == 0)}")
        
        return images, labels, paths
    
    def split_data(self, images, labels, paths):
        """
        Split data into train/val/test sets
        
        Args:
            images: numpy array of images
            labels: numpy array of labels
            paths: list of image paths
            
        Returns:
            Dictionary containing train/val/test splits
        """
        logger.info("Splitting data into train/val/test sets...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test, paths_temp, paths_test = train_test_split(
            images, labels, paths,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
            X_temp, y_temp, paths_temp,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            stratify=y_temp
        )
        
        logger.info(f"Train set: {len(X_train)} images")
        logger.info(f"  - Positive: {np.sum(y_train == 1)}, Negative: {np.sum(y_train == 0)}")
        logger.info(f"Validation set: {len(X_val)} images")
        logger.info(f"  - Positive: {np.sum(y_val == 1)}, Negative: {np.sum(y_val == 0)}")
        logger.info(f"Test set: {len(X_test)} images")
        logger.info(f"  - Positive: {np.sum(y_test == 1)}, Negative: {np.sum(y_test == 0)}")
        
        return {
            'train': {'images': X_train, 'labels': y_train, 'paths': paths_train},
            'val': {'images': X_val, 'labels': y_val, 'paths': paths_val},
            'test': {'images': X_test, 'labels': y_test, 'paths': paths_test}
        }
    
    def save_processed_data(self, data_splits):
        """
        Save processed data to disk
        
        Args:
            data_splits: Dictionary containing train/val/test splits
        """
        logger.info("Saving processed data...")
        
        for split_name, split_data in data_splits.items():
            split_dir = self.processed_data_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Save images and labels as numpy arrays
            np.save(split_dir / 'images.npy', split_data['images'])
            np.save(split_dir / 'labels.npy', split_data['labels'])
            
            # Save paths as JSON
            with open(split_dir / 'paths.json', 'w') as f:
                json.dump(split_data['paths'], f, indent=2)
            
            logger.info(f"Saved {split_name} set to {split_dir}")
        
        # Save dataset statistics
        stats = {
            'total_images': sum(len(split['images']) for split in data_splits.values()),
            'target_size': self.target_size,
            'splits': {
                name: {
                    'num_images': len(split['images']),
                    'num_positive': int(np.sum(split['labels'] == 1)),
                    'num_negative': int(np.sum(split['labels'] == 0))
                }
                for name, split in data_splits.items()
            }
        }
        
        with open(self.processed_data_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved dataset statistics to {self.processed_data_dir / 'dataset_stats.json'}")
    
    def preprocess(self, source='local'):
        """
        Run complete preprocessing pipeline
        
        Args:
            source: 'local' or 's3'
        """
        logger.info("=" * 60)
        logger.info("DATA PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Load data
        if source == 'local':
            images, labels, paths = self.load_dataset_from_directory(self.raw_data_dir)
        else:
            raise NotImplementedError("S3 loading not yet implemented")
        
        # Split data
        data_splits = self.split_data(images, labels, paths)
        
        # Save processed data
        self.save_processed_data(data_splits)
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"Processed data saved to: {self.processed_data_dir}")
        logger.info("=" * 60)
        
        return data_splits


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess crack detection dataset')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--source', type=str, default='local', choices=['local', 's3'], 
                       help='Data source: local or s3')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config_path=args.config)
    
    # Run preprocessing
    preprocessor.preprocess(source=args.source)


if __name__ == "__main__":
    main()