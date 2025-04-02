import boto3
import os
from pathlib import Path
import logging
from botocore.exceptions import ClientError
from typing import Optional, List, Tuple
from io import BytesIO
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/aws_s3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class S3Handler:
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.region = region
        
        # Define standard paths
        self.pdf_prefix = 'raw/pdfs/'
        self.image_prefix = 'raw/images/'
        self.parsed_prefix = 'parsed/'
        self.parsed_images_prefix = 'parsed/images/'

    def upload_file(self, file_path: str, s3_key: str) -> bool:
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {file_path}: {str(e)}")
            return False

    def upload_fileobj(self, file_obj: BytesIO, s3_key: str, content_type: Optional[str] = None) -> bool:
        """Upload a file object to S3"""
        try:
            extra_args = {'ContentType': content_type} if content_type else {}
            self.s3_client.upload_fileobj(file_obj, self.bucket_name, s3_key, ExtraArgs=extra_args)
            logger.info(f"Successfully uploaded file object to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload file object: {str(e)}")
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a file from S3"""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {str(e)}")
            return False

    def download_fileobj(self, s3_key: str) -> Optional[BytesIO]:
        """Download a file from S3 as a file object"""
        try:
            file_obj = BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, s3_key, file_obj)
            file_obj.seek(0)
            return file_obj
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {str(e)}")
            return None

    def list_files(self, prefix: str) -> List[str]:
        """List files in S3 with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            logger.error(f"Failed to list files with prefix {prefix}: {str(e)}")
            return []

    def save_from_url(self, url: str, s3_key: str) -> bool:
        """Download file from URL and save directly to S3"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            file_obj = BytesIO(response.content)
            content_type = response.headers.get('content-type')
            return self.upload_fileobj(file_obj, s3_key, content_type)
        except Exception as e:
            logger.error(f"Failed to save {url} to S3: {str(e)}")
            return False


    def get_s3_url(self, s3_key: str) -> str:
        """Generate a publicly accessible S3 URL"""
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

    # Specific methods for our use case
    def save_pdf(self, title: str, pdf_url: str) -> str:
        """Save PDF from URL to S3 and return S3 path"""
        sanitized_title = self.sanitize_filename(title)
        s3_key = f"{self.pdf_prefix}{sanitized_title}.pdf"
        if self.save_from_url(pdf_url, s3_key):
            return self.get_s3_url(s3_key)
        return ""

    def save_image(self, title: str, image_url: str) -> str:
        """Save image from URL to S3 and return S3 path"""
        sanitized_title = self.sanitize_filename(title)
        s3_key = f"{self.image_prefix}{sanitized_title}.jpg"
        if self.save_from_url(image_url, s3_key):
            return self.get_s3_url(s3_key)
        return ""

    def save_parsed_json(self, filename: str, json_data: str) -> str:
        """Save parsed JSON to S3"""
        s3_key = f"{self.parsed_prefix}{filename}"
        file_obj = BytesIO(json_data.encode('utf-8'))
        if self.upload_fileobj(file_obj, s3_key, 'application/json'):
            return self.get_s3_url(s3_key)
        return ""

    def save_parsed_image(self, filename: str, image_data: bytes) -> str:
        """Save parsed image to S3"""
        s3_key = f"{self.parsed_images_prefix}{filename}"
        file_obj = BytesIO(image_data)
        if self.upload_fileobj(file_obj, s3_key, 'image/png'):
            return self.get_s3_url(s3_key)
        return ""

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Sanitize filename for S3"""
        return "".join(c if c.isalnum() or c in ".-_" else "_" for c in name)