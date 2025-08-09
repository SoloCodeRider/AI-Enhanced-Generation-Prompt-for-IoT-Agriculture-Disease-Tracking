"""
Cloud Storage Manager for IoT Crop Disease Tracking

Implements scalable, secure cloud data storage for:
- Raw IoT sensor data and images
- Generated graph structures and real-time updates
- Disease mutation pattern deviation logs
- Historical graph snapshots
"""

import asyncio
import json
import time
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import uuid
import logging

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from pydantic import BaseModel, Field
import aiofiles
import aiohttp


@dataclass
class StorageConfig:
    """Configuration for cloud storage providers"""
    provider: str  # 'aws', 'azure', 'gcp'
    bucket_name: str
    region: str = 'us-east-1'
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    connection_string: Optional[str] = None
    project_id: Optional[str] = None
    credentials_path: Optional[str] = None


@dataclass
class StorageBucket:
    """Storage bucket configuration"""
    name: str
    purpose: str  # 'raw_data', 'processed_graphs', 'mutation_logs', 'historical_snapshots'
    retention_days: int = 365
    encryption_enabled: bool = True
    versioning_enabled: bool = True
    access_level: str = 'private'  # 'private', 'public-read', 'public'


class DataRecord(BaseModel):
    """Base data record for storage"""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    data_type: str
    source: str
    location: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = None
    size_bytes: int = 0


class IoTDataRecord(DataRecord):
    """IoT sensor data record"""
    data_type: str = "iot_sensor_data"
    sensor_id: str
    sensor_type: str
    disease_detected: bool = False
    disease_confidence: float = 0.0
    disease_type: Optional[str] = None
    environmental_data: Dict[str, float] = Field(default_factory=dict)


class GraphDataRecord(DataRecord):
    """Graph data record"""
    data_type: str = "graph_data"
    graph_id: str
    node_count: int = 0
    edge_count: int = 0
    disease_centers: List[Dict] = Field(default_factory=list)
    mutation_events: List[Dict] = Field(default_factory=list)


class MutationLogRecord(DataRecord):
    """Disease mutation log record"""
    data_type: str = "mutation_log"
    mutation_id: str
    mutation_type: str
    severity: float = 0.0
    affected_area: List[Tuple[float, float]] = Field(default_factory=list)
    description: str = ""
    recommended_action: str = ""


class CloudStorageManager:
    """Main cloud storage manager for IoT crop disease tracking"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.buckets: Dict[str, StorageBucket] = {}
        self.clients = {}
        self.upload_queue = asyncio.Queue()
        self.processing_times = []
        
        # Initialize storage clients
        self._initialize_clients()
        
        # Initialize storage buckets
        self._initialize_buckets()
        
        # Start background processing
        self.background_task = asyncio.create_task(self._process_upload_queue())
        
    def _initialize_clients(self):
        """Initialize cloud storage clients"""
        try:
            if self.config.provider == 'aws':
                self.clients['aws'] = boto3.client(
                    's3',
                    aws_access_key_id=self.config.access_key,
                    aws_secret_access_key=self.config.secret_key,
                    region_name=self.config.region
                )
                
            elif self.config.provider == 'azure':
                self.clients['azure'] = BlobServiceClient.from_connection_string(
                    self.config.connection_string
                )
                
            elif self.config.provider == 'gcp':
                self.clients['gcp'] = storage.Client.from_service_account_json(
                    self.config.credentials_path
                )
                
        except Exception as e:
            logging.error(f"Failed to initialize {self.config.provider} client: {e}")
    
    def _initialize_buckets(self):
        """Initialize storage buckets for different data types"""
        bucket_configs = [
            StorageBucket(
                name=f"{self.config.bucket_name}-raw-data",
                purpose="raw_data",
                retention_days=90
            ),
            StorageBucket(
                name=f"{self.config.bucket_name}-processed-graphs",
                purpose="processed_graphs",
                retention_days=365
            ),
            StorageBucket(
                name=f"{self.config.bucket_name}-mutation-logs",
                purpose="mutation_logs",
                retention_days=730
            ),
            StorageBucket(
                name=f"{self.config.bucket_name}-historical-snapshots",
                purpose="historical_snapshots",
                retention_days=1825  # 5 years
            )
        ]
        
        for bucket_config in bucket_configs:
            self.buckets[bucket_config.purpose] = bucket_config
            self._create_bucket_if_not_exists(bucket_config)
    
    def _create_bucket_if_not_exists(self, bucket_config: StorageBucket):
        """Create bucket if it doesn't exist"""
        try:
            if self.config.provider == 'aws':
                try:
                    self.clients['aws'].head_bucket(Bucket=bucket_config.name)
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        self.clients['aws'].create_bucket(
                            Bucket=bucket_config.name,
                            CreateBucketConfiguration={
                                'LocationConstraint': self.config.region
                            }
                        )
                        
            elif self.config.provider == 'azure':
                container_client = self.clients['azure'].get_container_client(bucket_config.name)
                try:
                    container_client.get_container_properties()
                except:
                    self.clients['azure'].create_container(bucket_config.name)
                    
            elif self.config.provider == 'gcp':
                bucket = self.clients['gcp'].bucket(bucket_config.name)
                if not bucket.exists():
                    bucket.create()
                    
        except Exception as e:
            logging.error(f"Failed to create bucket {bucket_config.name}: {e}")
    
    async def store_iot_data(self, sensor_data: Dict) -> str:
        """Store IoT sensor data in cloud storage"""
        start_time = time.time()
        
        # Create data record
        record = IoTDataRecord(
            sensor_id=sensor_data.get('sensor_id', 'unknown'),
            sensor_type=sensor_data.get('sensor_type', 'unknown'),
            source='iot_sensor',
            location=sensor_data.get('location', {}),
            disease_detected=sensor_data.get('disease_detected', False),
            disease_confidence=sensor_data.get('disease_confidence', 0.0),
            disease_type=sensor_data.get('disease_type'),
            environmental_data={
                'temperature': sensor_data.get('temperature', 0.0),
                'humidity': sensor_data.get('humidity', 0.0),
                'soil_moisture': sensor_data.get('soil_moisture', 0.0),
                'light_intensity': sensor_data.get('light_intensity', 0.0),
                'wind_speed': sensor_data.get('wind_speed', 0.0)
            },
            data=sensor_data,
            size_bytes=len(json.dumps(sensor_data))
        )
        
        # Calculate checksum
        record.checksum = self._calculate_checksum(record.data)
        
        # Store in raw data bucket
        bucket_name = self.buckets['raw_data'].name
        key = f"iot_data/{record.timestamp.strftime('%Y/%m/%d')}/{record.record_id}.json"
        
        await self._upload_record(record, bucket_name, key)
        
        # Record performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return record.record_id
    
    async def store_graph_data(self, graph_data: Dict) -> str:
        """Store graph data in cloud storage"""
        start_time = time.time()
        
        # Create data record
        record = GraphDataRecord(
            graph_id=graph_data.get('graph_id', str(uuid.uuid4())),
            source='graph_engine',
            location=None,  # Graph covers entire field
            node_count=len(graph_data.get('nodes', [])),
            edge_count=len(graph_data.get('edges', [])),
            disease_centers=graph_data.get('disease_centers', []),
            mutation_events=graph_data.get('mutation_events', []),
            data=graph_data,
            size_bytes=len(json.dumps(graph_data))
        )
        
        # Calculate checksum
        record.checksum = self._calculate_checksum(record.data)
        
        # Store in processed graphs bucket
        bucket_name = self.buckets['processed_graphs'].name
        key = f"graph_data/{record.timestamp.strftime('%Y/%m/%d')}/{record.record_id}.json"
        
        await self._upload_record(record, bucket_name, key)
        
        # Record performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return record.record_id
    
    async def store_mutation_log(self, mutation_data: Dict) -> str:
        """Store disease mutation log in cloud storage"""
        start_time = time.time()
        
        # Create data record
        record = MutationLogRecord(
            mutation_id=mutation_data.get('id', str(uuid.uuid4())),
            source='ai_analysis',
            location=mutation_data.get('location'),
            mutation_type=mutation_data.get('mutation_type', 'unknown'),
            severity=mutation_data.get('severity', 0.0),
            affected_area=mutation_data.get('affected_area', []),
            description=mutation_data.get('description', ''),
            recommended_action=mutation_data.get('recommended_action', ''),
            data=mutation_data,
            size_bytes=len(json.dumps(mutation_data))
        )
        
        # Calculate checksum
        record.checksum = self._calculate_checksum(record.data)
        
        # Store in mutation logs bucket
        bucket_name = self.buckets['mutation_logs'].name
        key = f"mutation_logs/{record.timestamp.strftime('%Y/%m/%d')}/{record.record_id}.json"
        
        await self._upload_record(record, bucket_name, key)
        
        # Record performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return record.record_id
    
    async def store_historical_snapshot(self, snapshot_data: Dict) -> str:
        """Store historical snapshot in cloud storage"""
        start_time = time.time()
        
        # Create data record
        record = DataRecord(
            data_type="historical_snapshot",
            source='system_backup',
            location=None,
            data=snapshot_data,
            size_bytes=len(json.dumps(snapshot_data))
        )
        
        # Calculate checksum
        record.checksum = self._calculate_checksum(record.data)
        
        # Store in historical snapshots bucket
        bucket_name = self.buckets['historical_snapshots'].name
        key = f"snapshots/{record.timestamp.strftime('%Y/%m')}/{record.record_id}.json"
        
        await self._upload_record(record, bucket_name, key)
        
        # Record performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return record.record_id
    
    async def _upload_record(self, record: DataRecord, bucket_name: str, key: str):
        """Upload data record to cloud storage"""
        try:
            # Prepare data for upload
            upload_data = record.model_dump_json()
            
            if self.config.provider == 'aws':
                await self._upload_to_aws(bucket_name, key, upload_data)
                
            elif self.config.provider == 'azure':
                await self._upload_to_azure(bucket_name, key, upload_data)
                
            elif self.config.provider == 'gcp':
                await self._upload_to_gcp(bucket_name, key, upload_data)
                
            logging.info(f"Successfully uploaded {key} to {bucket_name}")
            
        except Exception as e:
            logging.error(f"Failed to upload {key} to {bucket_name}: {e}")
            # Add to retry queue
            await self.upload_queue.put((record, bucket_name, key))
    
    async def _upload_to_aws(self, bucket_name: str, key: str, data: str):
        """Upload to AWS S3"""
        try:
            self.clients['aws'].put_object(
                Bucket=bucket_name,
                Key=key,
                Body=data,
                ContentType='application/json',
                ServerSideEncryption='AES256'
            )
        except Exception as e:
            raise Exception(f"AWS upload failed: {e}")
    
    async def _upload_to_azure(self, bucket_name: str, key: str, data: str):
        """Upload to Azure Blob Storage"""
        try:
            blob_client = self.clients['azure'].get_blob_client(
                container=bucket_name, blob=key
            )
            blob_client.upload_blob(data, overwrite=True)
        except Exception as e:
            raise Exception(f"Azure upload failed: {e}")
    
    async def _upload_to_gcp(self, bucket_name: str, key: str, data: str):
        """Upload to Google Cloud Storage"""
        try:
            bucket = self.clients['gcp'].bucket(bucket_name)
            blob = bucket.blob(key)
            blob.upload_from_string(data, content_type='application/json')
        except Exception as e:
            raise Exception(f"GCP upload failed: {e}")
    
    async def retrieve_data(self, bucket_purpose: str, record_id: str) -> Optional[Dict]:
        """Retrieve data from cloud storage"""
        try:
            bucket_name = self.buckets[bucket_purpose].name
            
            # Construct key based on record_id
            key = self._construct_key_from_record_id(record_id, bucket_purpose)
            
            if self.config.provider == 'aws':
                return await self._retrieve_from_aws(bucket_name, key)
                
            elif self.config.provider == 'azure':
                return await self._retrieve_from_azure(bucket_name, key)
                
            elif self.config.provider == 'gcp':
                return await self._retrieve_from_gcp(bucket_name, key)
                
        except Exception as e:
            logging.error(f"Failed to retrieve data {record_id}: {e}")
            return None
    
    async def _retrieve_from_aws(self, bucket_name: str, key: str) -> Optional[Dict]:
        """Retrieve from AWS S3"""
        try:
            response = self.clients['aws'].get_object(Bucket=bucket_name, Key=key)
            data = response['Body'].read().decode('utf-8')
            return json.loads(data)
        except Exception as e:
            logging.error(f"AWS retrieval failed: {e}")
            return None
    
    async def _retrieve_from_azure(self, bucket_name: str, key: str) -> Optional[Dict]:
        """Retrieve from Azure Blob Storage"""
        try:
            blob_client = self.clients['azure'].get_blob_client(
                container=bucket_name, blob=key
            )
            data = blob_client.download_blob().readall().decode('utf-8')
            return json.loads(data)
        except Exception as e:
            logging.error(f"Azure retrieval failed: {e}")
            return None
    
    async def _retrieve_from_gcp(self, bucket_name: str, key: str) -> Optional[Dict]:
        """Retrieve from Google Cloud Storage"""
        try:
            bucket = self.clients['gcp'].bucket(bucket_name)
            blob = bucket.blob(key)
            data = blob.download_as_text()
            return json.loads(data)
        except Exception as e:
            logging.error(f"GCP retrieval failed: {e}")
            return None
    
    def _construct_key_from_record_id(self, record_id: str, bucket_purpose: str) -> str:
        """Construct storage key from record ID"""
        # This is a simplified key construction
        # In practice, you might want to use a more sophisticated approach
        return f"{bucket_purpose}/{record_id[:8]}/{record_id}.json"
    
    def _calculate_checksum(self, data: Dict) -> str:
        """Calculate checksum for data integrity"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def _process_upload_queue(self):
        """Background task to process upload queue"""
        while True:
            try:
                record, bucket_name, key = await self.upload_queue.get()
                await self._upload_record(record, bucket_name, key)
                self.upload_queue.task_done()
            except Exception as e:
                logging.error(f"Upload queue processing error: {e}")
            await asyncio.sleep(1)
    
    async def list_data_by_date_range(self, bucket_purpose: str, 
                                    start_date: datetime, end_date: datetime) -> List[str]:
        """List data records within a date range"""
        try:
            bucket_name = self.buckets[bucket_purpose].name
            records = []
            
            if self.config.provider == 'aws':
                records = await self._list_aws_objects(bucket_name, start_date, end_date)
            elif self.config.provider == 'azure':
                records = await self._list_azure_objects(bucket_name, start_date, end_date)
            elif self.config.provider == 'gcp':
                records = await self._list_gcp_objects(bucket_name, start_date, end_date)
            
            return records
            
        except Exception as e:
            logging.error(f"Failed to list data by date range: {e}")
            return []
    
    async def _list_aws_objects(self, bucket_name: str, start_date: datetime, 
                               end_date: datetime) -> List[str]:
        """List AWS S3 objects in date range"""
        try:
            response = self.clients['aws'].list_objects_v2(
                Bucket=bucket_name,
                StartAfter=start_date.strftime('%Y/%m/%d'),
                MaxKeys=1000
            )
            
            records = []
            for obj in response.get('Contents', []):
                if start_date <= obj['LastModified'].replace(tzinfo=None) <= end_date:
                    records.append(obj['Key'])
            
            return records
            
        except Exception as e:
            logging.error(f"AWS list objects failed: {e}")
            return []
    
    async def _list_azure_objects(self, bucket_name: str, start_date: datetime,
                                 end_date: datetime) -> List[str]:
        """List Azure Blob objects in date range"""
        try:
            container_client = self.clients['azure'].get_container_client(bucket_name)
            blobs = container_client.list_blobs()
            
            records = []
            for blob in blobs:
                if start_date <= blob.last_modified.replace(tzinfo=None) <= end_date:
                    records.append(blob.name)
            
            return records
            
        except Exception as e:
            logging.error(f"Azure list objects failed: {e}")
            return []
    
    async def _list_gcp_objects(self, bucket_name: str, start_date: datetime,
                               end_date: datetime) -> List[str]:
        """List GCP objects in date range"""
        try:
            bucket = self.clients['gcp'].bucket(bucket_name)
            blobs = bucket.list_blobs()
            
            records = []
            for blob in blobs:
                if start_date <= blob.updated.replace(tzinfo=None) <= end_date:
                    records.append(blob.name)
            
            return records
            
        except Exception as e:
            logging.error(f"GCP list objects failed: {e}")
            return []
    
    def get_storage_metrics(self) -> Dict:
        """Get storage performance metrics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_upload_time': sum(self.processing_times) / len(self.processing_times),
            'total_uploads': len(self.processing_times),
            'queue_size': self.upload_queue.qsize(),
            'bucket_info': {
                purpose: {
                    'name': bucket.name,
                    'retention_days': bucket.retention_days,
                    'encryption_enabled': bucket.encryption_enabled
                }
                for purpose, bucket in self.buckets.items()
            }
        }
    
    async def cleanup_old_data(self, retention_days: int = 365):
        """Clean up old data based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for bucket_purpose, bucket in self.buckets.items():
            if bucket.retention_days <= retention_days:
                old_records = await self.list_data_by_date_range(
                    bucket_purpose, 
                    datetime.min, 
                    cutoff_date
                )
                
                for record_key in old_records:
                    await self._delete_record(bucket.name, record_key)
    
    async def _delete_record(self, bucket_name: str, key: str):
        """Delete a record from cloud storage"""
        try:
            if self.config.provider == 'aws':
                self.clients['aws'].delete_object(Bucket=bucket_name, Key=key)
            elif self.config.provider == 'azure':
                blob_client = self.clients['azure'].get_blob_client(
                    container=bucket_name, blob=key
                )
                blob_client.delete_blob()
            elif self.config.provider == 'gcp':
                bucket = self.clients['gcp'].bucket(bucket_name)
                blob = bucket.blob(key)
                blob.delete()
                
            logging.info(f"Deleted {key} from {bucket_name}")
            
        except Exception as e:
            logging.error(f"Failed to delete {key} from {bucket_name}: {e}")


# Example usage
async def main():
    """Example usage of the cloud storage manager"""
    # Initialize storage configuration
    config = StorageConfig(
        provider='aws',  # or 'azure', 'gcp'
        bucket_name='crop-disease-tracking',
        region='us-east-1',
        access_key='your-access-key',
        secret_key='your-secret-key'
    )
    
    # Initialize storage manager
    storage_manager = CloudStorageManager(config)
    
    # Store IoT data
    sensor_data = {
        'sensor_id': 'sensor_001',
        'sensor_type': 'leaf_camera',
        'location': {'latitude': 100.0, 'longitude': 150.0},
        'disease_detected': True,
        'disease_confidence': 0.8,
        'disease_type': 'fungal_infection',
        'temperature': 25.5,
        'humidity': 65.0,
        'soil_moisture': 45.0
    }
    
    record_id = await storage_manager.store_iot_data(sensor_data)
    print(f"Stored IoT data with record ID: {record_id}")
    
    # Get storage metrics
    metrics = storage_manager.get_storage_metrics()
    print(f"Storage metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
