import boto3
from botocore.exceptions import ClientError
import os
import io
import zipfile
from fastapi import HTTPException
from urllib.parse import urlparse, unquote
from pathlib import Path
 
FILE_NOT_FOUND_MSG = "File not found in S3"
NO_FILE_KEY_IN_URL_MSG = "No file key found in URL"
 
 
def get_s3_key_from_url(url: str) -> str:
    """
    Extract the S3 object key (path in the bucket) from the S3 URL.
    Accepts: https://<bucket>.s3.amazonaws.com/<key>, s3://<bucket>/<key>
    """
    url = unquote(url)
    if url.startswith('https://'):
        parsed = urlparse(url)
        object_key = parsed.path.lstrip('/')
    elif url.startswith('s3://'):
        path_parts = url[5:].split('/', 1)
        object_key = path_parts[1] if len(path_parts) > 1 else ""
    else:
        object_key = ""
    object_key = object_key.replace("+"," ")
    return object_key
 
 
class S3Utility:
    def __init__(self):
        
        self.s3_client = boto3.client(
            's3',
            region_name=os.getenv("AWS_REGION_LAMBDA") or os.getenv("AWS_REGION")
            
        )
        self.bucket_name = os.getenv("S3_BUCKET_NAME")

        
 
    def upload_file(self, file_content: bytes, file_name: str, folder: str) -> str:
        try:
            s3_key = f"{folder}/{file_name}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content
            )
            file_url = f"s3://{self.bucket_name}/{s3_key}"
            return file_url
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {str(e)}")
 
    def upload_file_by_url(self, file_content: bytes, url: str) -> str:
        try:
            key = get_s3_key_from_url(url)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_content
            )
            file_url = f"https://{self.bucket_name}.s3.amazonaws.com/{key}"
            return file_url
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {str(e)}")
 
    def get_data_from_s3_by_url(self, url: str) -> bytes:
        try:
            url = unquote(url)
            key = get_s3_key_from_url(url)
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['NoSuchKey', 'AccessDenied']:
                raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_MSG)
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file from S3: {str(e)}")
 
    def get_file(self, file_name: str, folder: str) -> bytes:
        try:
            s3_key = f"{folder}/{file_name}"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_MSG)
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file from S3: {str(e)}")
    
    def _get_s3_object(self, s3_key: str) -> bytes:
        """Retrieve an object from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            self.logger.error(f"Error retrieving S3 object {s3_key}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve S3 object: {str(e)}")
    
    def generate_presigned_url(self, file_url: str, expiration: int = 604800) -> str:
        try:
            s3_key = get_s3_key_from_url(file_url)
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return presigned_url
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_MSG)
            raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}")
 
    def delete_file(self, file_name: str, folder: str) -> dict:
        try:
            s3_key = f"{folder}/{file_name}"
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return {"status": "success", "message": f"File {file_name} deleted successfully"}
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_MSG)
            raise HTTPException(status_code=500, detail=f"Failed to delete file from S3: {str(e)}")
 
    def delete_file_by_url(self, url: str) -> dict:
        try:
            s3_key = get_s3_key_from_url(url)
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return {"status": "success", "message": f"File {s3_key} deleted successfully"}
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_MSG)
            raise HTTPException(status_code=500, detail=f"Failed to delete file from S3: {str(e)}")
 
    def create_zip_and_upload_for_urls(self, file_urls: list[str], zip_folder: str, zip_filename: str) -> str:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_url in file_urls:
                file_bytes = self.get_data_from_s3_by_url(file_url)
                arcname = file_url.split("/")[-1]
                zipf.writestr(arcname, file_bytes)
        zip_buffer.seek(0)
        return self.upload_file(zip_buffer.getvalue(), zip_filename, zip_folder)
 
    def extract_filename_from_s3_url(self, s3_url):
        s3_url = unquote(s3_url)
        parsed_url = urlparse(s3_url)
        full_filename = Path(parsed_url.path).name

        import re
        cleaned_filename = re.sub(r'_[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}(\.\w+)$', r'\1', full_filename)
        
        return cleaned_filename
 
    def copy_s3_file_to_new_path(self, s3_url: str, new_folder: str) -> tuple[str, str]:
        try:
            s3_url = unquote(s3_url)
            bucket_name, source_key = self._parse_s3_url_for_copy(s3_url)
 
            filename = source_key.split('/')[-1].replace(" ", "_")
            new_key = f"{new_folder}/{filename}"
 
            copy_source = {'Bucket': str(bucket_name), 'Key': str(source_key)}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=new_key
            )
            presigned_url = self.s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket_name, "Key": new_key},
                ExpiresIn=3600
            )
            return presigned_url, filename
        except Exception as e:
            raise ValueError(f"S3 copy operation failed: {str(e)}")
 
    def _parse_virtual_hosted_style(self, parsed_url) -> tuple[str, str]:
        """
        Parse bucket and key from virtual-hosted style URLs like bucket.s3.amazonaws.com/key
        """
        bucket_name = self.bucket_name  # Assuming you already know your bucket here
        key = parsed_url.path.lstrip('/')
        if not key:
            raise ValueError(NO_FILE_KEY_IN_URL_MSG)
        return bucket_name, key
 
    def _parse_path_style(self, parsed_url) -> tuple[str, str]:
        """
        Parse bucket and key from path-style URLs like s3.region.amazonaws.com/bucket/key
        """
        path_parts = parsed_url.path.lstrip('/').split('/', 1)
        bucket_name = path_parts[0] if len(path_parts) > 0 else self.bucket_name
        key = path_parts[1] if len(path_parts) > 1 else ''
        if not key:
            raise ValueError(NO_FILE_KEY_IN_URL_MSG)
        return bucket_name, key
 
    def _parse_s3_url_for_copy(self, s3_url: str) -> tuple[str, str]:
        if s3_url.startswith('https://'):
            parsed = urlparse(s3_url)
            if not parsed.netloc.endswith('.amazonaws.com'):
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
 
            if '.s3.' in parsed.netloc or '.s3-' in parsed.netloc:
                return self._parse_virtual_hosted_style(parsed)
 
            if parsed.netloc.startswith('s3.') or parsed.netloc.startswith('s3-'):
                return self._parse_path_style(parsed)
 
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
 
        elif s3_url.startswith('s3://'):
            path_parts = s3_url[5:].split('/', 1)
            bucket_name = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ''
            if not key:
                raise ValueError(NO_FILE_KEY_IN_URL_MSG)
            return bucket_name, key
 
        else:
            raise ValueError(f"Unsupported URL format: {s3_url}")
