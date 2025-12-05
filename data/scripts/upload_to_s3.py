"""
Upload crack detection dataset to AWS S3
"""

import os
import boto3
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
from botocore.exceptions import ClientError
import mimetypes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class S3Uploader:
    """Upload datasets to AWS S3"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize the uploader
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=self.config['aws']['region'])
        self.s3_resource = boto3.resource('s3', region_name=self.config['aws']['region'])
        
        self.raw_bucket = self.config['aws']['s3']['raw_data_bucket']
        self.processed_bucket = self.config['aws']['s3']['processed_data_bucket']
        
        self.raw_data_dir = Path(self.config['data']['local']['raw_data_dir'])
        
    def create_bucket_if_not_exists(self, bucket_name):
        """
        Create S3 bucket if it doesn't exist
        
        Args:
            bucket_name: Name of the bucket
        """
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} already exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if self.config['aws']['region'] == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.config['aws']['region']}
                        )
                    logger.info(f"Created bucket: {bucket_name}")
                    
                    # Enable versioning
                    versioning = self.s3_resource.BucketVersioning(bucket_name)
                    versioning.enable()
                    logger.info(f"Enabled versioning for bucket: {bucket_name}")
                    
                except ClientError as e:
                    logger.error(f"Error creating bucket {bucket_name}: {str(e)}")
                    raise
            else:
                logger.error(f"Error accessing bucket {bucket_name}: {str(e)}")
                raise
    
    def upload_file(self, file_path, bucket_name, s3_key):
        """
        Upload a single file to S3
        
        Args:
            file_path: Local file path
            bucket_name: S3 bucket name
            s3_key: S3 object key
        """
        try:
            # Determine content type
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                content_type = 'application/octet-stream'
            
            # Upload file
            extra_args = {'ContentType': content_type}
            self.s3_client.upload_file(
                str(file_path),
                bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
        except ClientError as e:
            logger.error(f"Error uploading {file_path} to {s3_key}: {str(e)}")
            raise
    
    def upload_directory(self, local_dir, bucket_name, s3_prefix=''):
        """
        Upload a directory to S3
        
        Args:
            local_dir: Local directory path
            bucket_name: S3 bucket name
            s3_prefix: S3 prefix (folder path)
        """
        local_dir = Path(local_dir)
        
        if not local_dir.exists():
            logger.error(f"Directory {local_dir} does not exist")
            return
        
        # Get all files
        files = list(local_dir.rglob('*'))
        files = [f for f in files if f.is_file()]
        
        logger.info(f"Uploading {len(files)} files from {local_dir} to s3://{bucket_name}/{s3_prefix}")
        
        # Upload files with progress bar
        for file_path in tqdm(files, desc=f"Uploading to {s3_prefix}"):
            # Calculate relative path
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
            
            # Upload file
            self.upload_file(file_path, bucket_name, s3_key)
        
        logger.info(f"Successfully uploaded {len(files)} files to s3://{bucket_name}/{s3_prefix}")
    
    def upload_raw_data(self):
        """Upload raw data to S3"""
        logger.info("=" * 60)
        logger.info("UPLOADING RAW DATA TO S3")
        logger.info("=" * 60)
        
        # Create bucket if needed
        self.create_bucket_if_not_exists(self.raw_bucket)
        
        # Upload positive samples
        positive_dir = self.raw_data_dir / 'positive'
        if positive_dir.exists():
            logger.info("Uploading positive samples (with cracks)...")
            self.upload_directory(positive_dir, self.raw_bucket, 'raw/positive')
        else:
            logger.warning(f"Positive directory not found: {positive_dir}")
        
        # Upload negative samples
        negative_dir = self.raw_data_dir / 'negative'
        if negative_dir.exists():
            logger.info("Uploading negative samples (without cracks)...")
            self.upload_directory(negative_dir, self.raw_bucket, 'raw/negative')
        else:
            logger.warning(f"Negative directory not found: {negative_dir}")
        
        logger.info("=" * 60)
        logger.info("UPLOAD COMPLETE")
        logger.info(f"Data uploaded to: s3://{self.raw_bucket}/raw/")
        logger.info("=" * 60)
    
    def list_bucket_contents(self, bucket_name, prefix=''):
        """
        List contents of an S3 bucket
        
        Args:
            bucket_name: S3 bucket name
            prefix: S3 prefix to filter
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                logger.info(f"\nContents of s3://{bucket_name}/{prefix}:")
                for obj in response['Contents']:
                    size_mb = obj['Size'] / (1024 * 1024)
                    logger.info(f"  - {obj['Key']} ({size_mb:.2f} MB)")
                
                total_size_gb = sum(obj['Size'] for obj in response['Contents']) / (1024 * 1024 * 1024)
                logger.info(f"\nTotal: {len(response['Contents'])} objects, {total_size_gb:.2f} GB")
            else:
                logger.info(f"No objects found in s3://{bucket_name}/{prefix}")
                
        except ClientError as e:
            logger.error(f"Error listing bucket contents: {str(e)}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload crack detection dataset to AWS S3')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--list', action='store_true', help='List bucket contents after upload')
    
    args = parser.parse_args()
    
    # Initialize uploader
    uploader = S3Uploader(config_path=args.config)
    
    # Upload data
    uploader.upload_raw_data()
    
    # List contents if requested
    if args.list:
        uploader.list_bucket_contents(uploader.raw_bucket, 'raw/')


if __name__ == "__main__":
    main()