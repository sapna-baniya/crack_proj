"""
Download crack detection dataset from Google Drive
Supports both folder downloads and ZIP file downloads
"""

import os
import argparse
import gdown
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
import zipfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoogleDriveDownloader:
    """Download datasets from Google Drive"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize the downloader
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.raw_data_dir = Path(self.config['data']['local']['raw_data_dir'])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_zip(self, zip_path, extract_to):
        """
        Extract ZIP file
        
        Args:
            zip_path: Path to ZIP file
            extract_to: Directory to extract to
        """
        try:
            logger.info(f"Extracting {zip_path.name}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files
                file_list = zip_ref.namelist()
                
                # Extract with progress bar
                for file in tqdm(file_list, desc="Extracting"):
                    zip_ref.extract(file, extract_to)
            
            logger.info(f"Extracted to {extract_to}")
            
            # Count extracted files
            num_files = len(list(extract_to.rglob('*')))
            logger.info(f"Extracted {num_files} items")
            
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {str(e)}")
            raise
    
    def download_folder(self, folder_id, output_dir, folder_name):
        """
        Download a Google Drive folder
        
        Args:
            folder_id: Google Drive folder ID
            output_dir: Local output directory
            folder_name: Name for the folder
        """
        try:
            logger.info(f"Downloading {folder_name} from Google Drive...")
            
            folder_path = output_dir / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Download folder using gdown
            url = f"https://drive.google.com/drive/folders/{folder_id}"
            gdown.download_folder(url, output=str(folder_path), quiet=False, use_cookies=False)
            
            logger.info(f"Successfully downloaded {folder_name}")
            
            # Count downloaded files
            num_files = len(list(folder_path.rglob('*')))
            logger.info(f"Downloaded {num_files} files to {folder_path}")
            
        except Exception as e:
            logger.error(f"Error downloading {folder_name}: {str(e)}")
            raise
    
    def download_file(self, file_id, output_path, extract=True):
        """
        Download a single file from Google Drive
        
        Args:
            file_id: Google Drive file ID
            output_path: Local output path
            extract: If True and file is ZIP, extract it
        """
        try:
            logger.info(f"Downloading file to {output_path}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=False)
            logger.info(f"Successfully downloaded file")
            
            # If it's a ZIP file and extract=True, extract it
            if extract and str(output_path).endswith('.zip'):
                extract_dir = output_path.parent
                self.extract_zip(output_path, extract_dir)
                
                # Optionally remove the ZIP file after extraction
                logger.info(f"Removing ZIP file {output_path}")
                output_path.unlink()
                
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise
    
    def download_all(self, positive_folder_id=None, negative_folder_id=None):
        """
        Download all datasets
        
        Args:
            positive_folder_id: Google Drive folder/file ID for positive samples (cracks)
            negative_folder_id: Google Drive folder/file ID for negative samples (no cracks)
        """
        # Use provided IDs or fall back to config
        pos_id = positive_folder_id or self.config['data']['google_drive'].get('positive_folder_id')
        neg_id = negative_folder_id or self.config['data']['google_drive'].get('negative_folder_id')
        
        if not pos_id or pos_id == "YOUR_POSITIVE_FOLDER_ID":
            logger.error("Please provide positive folder/file ID")
            return
            
        if not neg_id or neg_id == "YOUR_NEGATIVE_FOLDER_ID":
            logger.error("Please provide negative folder/file ID")
            return
        
        # Download positive samples (with cracks)
        logger.info("=" * 60)
        logger.info("Downloading POSITIVE samples (with cracks)")
        logger.info("=" * 60)
        self.download_dataset(pos_id, 'positive')
        
        # Download negative samples (without cracks)
        logger.info("=" * 60)
        logger.info("Downloading NEGATIVE samples (without cracks)")
        logger.info("=" * 60)
        self.download_dataset(neg_id, 'negative')
        
        # Summary
        self.print_summary()
    
    def download_dataset(self, file_or_folder_id, dataset_name):
        """
        Download dataset - automatically detects if it's a file or folder
        
        Args:
            file_or_folder_id: Google Drive file or folder ID
            dataset_name: Name for the dataset (e.g., 'positive', 'negative')
        """
        try:
            # First, try downloading as a file (assumes it's a ZIP)
            zip_path = self.raw_data_dir / f"{dataset_name}.zip"
            
            logger.info(f"Attempting to download as ZIP file...")
            url = f"https://drive.google.com/uc?id={file_or_folder_id}"
            
            try:
                gdown.download(url, str(zip_path), quiet=False)
                
                # If successful, it's a file - extract it
                logger.info(f"Downloaded ZIP file successfully")
                extract_dir = self.raw_data_dir / dataset_name
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                self.extract_zip(zip_path, extract_dir)
                
                # Remove ZIP file after extraction
                logger.info(f"Removing ZIP file")
                zip_path.unlink()
                
            except Exception as file_error:
                # If file download fails, try as folder
                logger.info(f"Not a direct file, trying as folder...")
                if zip_path.exists():
                    zip_path.unlink()
                
                self.download_folder(file_or_folder_id, self.raw_data_dir, dataset_name)
                
        except Exception as e:
            logger.error(f"Error downloading {dataset_name} dataset: {str(e)}")
            raise
    
    def print_summary(self):
        """Print download summary"""
        logger.info("=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        
        positive_dir = self.raw_data_dir / 'positive'
        negative_dir = self.raw_data_dir / 'negative'
        
        if positive_dir.exists():
            num_positive = len(list(positive_dir.rglob('*.jpg'))) + len(list(positive_dir.rglob('*.png')))
            logger.info(f"Positive samples (cracks): {num_positive}")
        
        if negative_dir.exists():
            num_negative = len(list(negative_dir.rglob('*.jpg'))) + len(list(negative_dir.rglob('*.png')))
            logger.info(f"Negative samples (no cracks): {num_negative}")
            
        logger.info(f"Data saved to: {self.raw_data_dir}")
        logger.info("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download crack detection dataset from Google Drive')
    parser.add_argument('--positive-folder-id', type=str, help='Google Drive folder ID for positive samples')
    parser.add_argument('--negative-folder-id', type=str, help='Google Drive folder ID for negative samples')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = GoogleDriveDownloader(config_path=args.config)
    
    # Download datasets
    downloader.download_all(
        positive_folder_id=args.positive_folder_id,
        negative_folder_id=args.negative_folder_id
    )


if __name__ == "__main__":
    main()