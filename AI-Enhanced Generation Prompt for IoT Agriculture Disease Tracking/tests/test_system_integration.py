"""Comprehensive Test Suite for IoT-Enhanced Crop Disease Tracking System

Tests all major components and their integration:
- IoT sensor simulation
- Graph engine (ST-GCN model and visualization)
- AI models (disease detection and pattern analysis)
- Cloud storage
- Alert system (alerts and notifications)
- Dashboard
- System integration
"""

import pytest
import asyncio
import json
import time
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from unittest.mock import MagicMock, patch

# Import system components
from iot_sensors.sensor_simulator import IoTSensorSimulator
from graph_engine.graph_model import DiseaseTrackingModel, GraphAttentionLayer, TemporalAttention, STGCN, create_graph_from_sensor_data
from graph_engine.visualization import DiseaseGraphVisualizer, RealTimeVisualizer
from ai_models.disease_detection import DiseaseDetectionModel, DiseaseImageDataset, DiseaseResNet
from ai_models.pattern_analysis import PatternAnalysisService, AnomalyDetector, MutationDetector, SpatialPatternAnalyzer
from cloud_storage.storage_manager import CloudStorageManager, StorageConfig, DataRecord, IoTDataRecord
from alert_system.alert_manager import AlertManager, Alert, AlertSeverity, AlertType
from alert_system.notification_service import NotificationService, NotificationChannel, NotificationPriority
from alert_system.dashboard import DashboardService, DashboardConfig, DashboardUI
from integration import IntegratedSystem, SystemConfig


@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing"""
    return [
        {
            "sensor_id": "sensor_001",
            "sensor_type": "leaf_camera",
            "timestamp": datetime.now(),
            "location": {"latitude": 37.5, "longitude": -122.1},
            "field_id": "field_001",
            "section_id": "section_A",
            "battery_level": 0.85,
            "disease_detected": True,
            "disease_type": "Fungal Leaf Spot",
            "disease_confidence": 0.78,
            "leaf_health_score": 0.65,
            "image_quality": 0.92,
            "image_data": "base64_encoded_image_data_placeholder"
        },
        {
            "sensor_id": "sensor_002",
            "sensor_type": "environmental",
            "timestamp": datetime.now(),
            "location": {"latitude": 37.52, "longitude": -122.15},
            "field_id": "field_001",
            "section_id": "section_B",
            "battery_level": 0.92,
            "temperature": 25.4,
            "humidity": 68.7,
            "soil_moisture": 0.42,
            "light_level": 0.85,
            "wind_speed": 3.2,
            "rainfall": 0.0
        }
    ]


@pytest.fixture
def mock_graph_data():
    """Generate mock graph data for testing"""
    # Create a simple graph with 5 nodes and edges
    num_nodes = 5
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                              [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    
    # Node features: position (x,y), disease confidence, health score
    node_features = torch.rand((num_nodes, 4))
    
    # Edge features: distance, disease transmission probability
    edge_features = torch.rand((edge_index.size(1), 2))
    
    # Time series data for each node (3 time steps)
    time_series = torch.rand((3, num_nodes, 4))
    
    return {
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "node_features": node_features,
        "edge_features": edge_features,
        "time_series": time_series
    }


class TestIoTSensorSimulator:
    """Test IoT sensor simulator functionality"""
    
    def test_sensor_initialization(self):
        """Test sensor network initialization"""
        simulator = IoTSensorSimulator(
            num_sensors=25,
            field_dimensions=(1000, 1000),
            update_interval=5
        )
        
        assert len(simulator.sensors) == 25
        assert simulator.field_dimensions == (1000, 1000)
        assert simulator.update_interval == 5
        
        # Check sensor distribution
        leaf_cameras = [s for s in simulator.sensors if s["type"] == "leaf_camera"]
        environmental = [s for s in simulator.sensors if s["type"] == "environmental"]
        
        assert len(leaf_cameras) > 0
        assert len(environmental) > 0
    
    @pytest.mark.asyncio
    async def test_data_generation(self):
        """Test sensor data generation"""
        simulator = IoTSensorSimulator(num_sensors=5)
        
        # Generate data for one sensor
        sensor = simulator.sensors[0]
        
        if sensor["type"] == "leaf_camera":
            data = simulator._generate_leaf_image_data(sensor)
            assert "disease_detected" in data
            assert "disease_confidence" in data
            assert "leaf_health_score" in data
        else:  # environmental
            data = simulator._generate_environmental_data(sensor)
            assert "temperature" in data
            assert "humidity" in data
            assert "soil_moisture" in data
    
    @pytest.mark.asyncio
    async def test_disease_pattern_simulation(self):
        """Test disease pattern simulation"""
        simulator = IoTSensorSimulator(num_sensors=10)
        
        # Initial state
        initial_centers = len(simulator.disease_centers)
        
        # Simulate disease spread
        simulator._update_disease_spread()
        
        # Check if disease spread was simulated
        assert simulator.disease_centers is not None
        
        # Run a short simulation
        with patch.object(simulator, "_transmit_data_mqtt", return_value=None):
            with patch.object(simulator, "_transmit_data_http", return_value=None):
                await simulator.simulate_sensor_network(duration=1)  # 1 second simulation
                
                # Should have generated some data
                assert simulator.get_latest_data() is not None


class TestGraphEngine:
    """Test graph engine functionality"""
    
    def test_graph_attention_layer(self):
        """Test graph attention layer"""
        in_features = 4
        out_features = 8
        num_heads = 2
        dropout = 0.1
        alpha = 0.2
        
        layer = GraphAttentionLayer(in_features, out_features, num_heads, dropout, alpha)
        
        # Test forward pass with mock data
        num_nodes = 5
        x = torch.rand((num_nodes, in_features))
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                  [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (num_nodes, out_features * num_heads)
    
    def test_temporal_attention(self):
        """Test temporal attention mechanism"""
        in_features = 8
        num_time_steps = 3
        
        layer = TemporalAttention(in_features, num_time_steps)
        
        # Test forward pass with mock data
        num_nodes = 5
        x = torch.rand((num_time_steps, num_nodes, in_features))
        
        output = layer(x)
        
        assert output.shape == (num_nodes, in_features)
    
    def test_stgcn_model(self, mock_graph_data):
        """Test ST-GCN model"""
        in_features = 4
        hidden_features = 8
        out_features = 2
        num_time_steps = 3
        
        model = STGCN(in_features, hidden_features, out_features, num_time_steps)
        
        # Test forward pass with mock data
        x = mock_graph_data["node_features"]
        edge_index = mock_graph_data["edge_index"]
        time_series = mock_graph_data["time_series"]
        
        output = model(x, edge_index, time_series)
        
        assert output.shape == (mock_graph_data["num_nodes"], out_features)
    
    def test_disease_tracking_model(self, sample_sensor_data):
        """Test disease tracking model"""
        model = DiseaseTrackingModel()
        
        # Create graph from sensor data
        graph_data = create_graph_from_sensor_data(sample_sensor_data)
        
        # Update model with graph data
        model.update(graph_data)
        
        # Get current state
        state = model.get_current_state()
        
        assert state is not None
        assert "timestamp" in state
        assert "nodes" in state
        assert "edges" in state
        
        # Test prediction
        prediction = model.predict_spread(hours=24)
        
        assert prediction is not None
        assert "predicted_nodes" in prediction
        assert "confidence" in prediction
    
    def test_graph_visualizer(self, mock_graph_data):
        """Test graph visualizer"""
        visualizer = DiseaseGraphVisualizer()
        
        # Create a simple graph state
        graph_state = {
            "timestamp": datetime.now(),
            "nodes": [
                {"id": "node_1", "position": (100, 150), "disease_confidence": 0.8, "health_score": 0.3},
                {"id": "node_2", "position": (200, 250), "disease_confidence": 0.2, "health_score": 0.9}
            ],
            "edges": [
                {"source": "node_1", "target": "node_2", "weight": 0.5}
            ]
        }
        
        # Test visualization methods
        with patch("matplotlib.pyplot.show"):
            # Create heatmap
            heatmap = visualizer.create_disease_heatmap(graph_state)
            assert heatmap is not None
            
            # Create network graph
            network = visualizer.create_network_graph(graph_state)
            assert network is not None
            
            # Create temporal analysis
            temporal = visualizer.create_temporal_analysis([graph_state] * 3)
            assert temporal is not None


class TestAIModels:
    """Test AI models functionality"""
    
    def test_disease_image_dataset(self):
        """Test disease image dataset"""
        # Create a mock dataset
        with patch("os.listdir", return_value=["img1.jpg", "img2.jpg"]):
            with patch("os.path.isdir", return_value=True):
                with patch("PIL.Image.open"):
                    with patch("torchvision.transforms.ToTensor"):
                        dataset = DiseaseImageDataset(
                            root_dir="dummy_path",
                            transform=None,
                            classes=["healthy", "diseased"]
                        )
                        
                        assert len(dataset) == 2
                        assert dataset.classes == ["healthy", "diseased"]
    
    def test_disease_resnet(self):
        """Test custom ResNet model"""
        model = DiseaseResNet(num_classes=3, use_attention=True)
        
        # Test forward pass
        x = torch.rand((2, 3, 224, 224))  # Batch of 2 RGB images
        output = model(x)
        
        assert output.shape == (2, 3)  # 2 samples, 3 classes
    
    def test_disease_detection_model(self):
        """Test disease detection model"""
        with patch.object(DiseaseDetectionModel, "_load_model", return_value=None):
            model = DiseaseDetectionModel(model_type="custom_resnet")
            
            # Test prediction with mock image
            with patch.object(model, "model", MagicMock()):
                with patch("PIL.Image.open"):
                    with patch("torchvision.transforms.ToTensor"):
                        prediction = model.predict("dummy_image.jpg")
                        
                        assert prediction is not None
    
    def test_anomaly_detector(self, sample_sensor_data):
        """Test anomaly detector"""
        detector = AnomalyDetector()
        
        # Convert to DataFrame for testing
        df = pd.DataFrame(sample_sensor_data)
        
        # Test anomaly detection
        anomalies = detector.detect_anomalies(df)
        
        assert anomalies is not None
        assert isinstance(anomalies, dict)
        assert "anomalies" in anomalies
    
    def test_mutation_detector(self, sample_sensor_data):
        """Test mutation detector"""
        detector = MutationDetector()
        
        # Create time series data
        time_series = []
        for i in range(5):
            data_point = sample_sensor_data[0].copy()
            data_point["timestamp"] = datetime.now() - timedelta(hours=i)
            data_point["disease_confidence"] = 0.5 + (i * 0.1)
            time_series.append(data_point)
        
        # Test mutation detection
        mutations = detector.detect_mutations(time_series)
        
        assert mutations is not None
        assert isinstance(mutations, list)
    
    def test_pattern_analysis_service(self, sample_sensor_data):
        """Test pattern analysis service"""
        service = PatternAnalysisService()
        
        # Test analysis
        analysis_results = service.analyze(sample_sensor_data)
        
        assert analysis_results is not None
        assert "anomalies" in analysis_results
        assert "mutations" in analysis_results
        assert "spatial_patterns" in analysis_results


class TestCloudStorage:
    """Test cloud storage functionality"""
    
    def test_storage_config(self):
        """Test storage configuration"""
        config = StorageConfig(
            aws_enabled=True,
            azure_enabled=False,
            gcp_enabled=True,
            aws_bucket_name="test-bucket",
            aws_region="us-east-1"
        )
        
        assert config.aws_enabled is True
        assert config.azure_enabled is False
        assert config.gcp_enabled is True
        assert config.aws_bucket_name == "test-bucket"
    
    @pytest.mark.asyncio
    async def test_storage_manager(self):
        """Test storage manager"""
        # Create a mock storage manager
        with patch.object(CloudStorageManager, "_initialize_aws_client", return_value=None):
            with patch.object(CloudStorageManager, "_initialize_azure_client", return_value=None):
                with patch.object(CloudStorageManager, "_initialize_gcp_client", return_value=None):
                    config = StorageConfig(aws_enabled=True)
                    manager = CloudStorageManager(config=config)
                    
                    # Initialize
                    await manager.initialize()
                    
                    # Test storing IoT data
                    with patch.object(manager, "_upload_record", return_value=None):
                        data = sample_sensor_data()[0]
                        await manager.store_iot_data(data)
                        
                        # Test retrieving data
                        with patch.object(manager, "_download_from_aws", return_value=json.dumps(data)):
                            retrieved = await manager.retrieve_data("iot_data", "test_id")
                            assert retrieved is not None


class TestAlertSystem:
    """Test alert system functionality"""
    
    def test_alert_manager(self):
        """Test alert manager"""
        manager = AlertManager()
        
        # Create an alert
        alert = manager.create_disease_detection_alert(
            sensor_id="sensor_001",
            location={"latitude": 37.5, "longitude": -122.1, "field_id": "field_001", "section_id": "section_A"},
            disease_type="Fungal Leaf Spot",
            confidence=0.85,
            severity=0.7
        )
        
        assert alert is not None
        assert alert.alert_type == AlertType.DISEASE_DETECTION
        assert alert.severity == AlertSeverity.HIGH
        assert alert.status == "active"
        
        # Get active alerts
        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) > 0
        
        # Update alert status
        manager.update_alert_status(alert.alert_id, "acknowledged")
        updated_alert = manager.get_alert(alert.alert_id)
        assert updated_alert.status == "acknowledged"
    
    @pytest.mark.asyncio
    async def test_notification_service(self):
        """Test notification service"""
        service = NotificationService()
        
        # Register a recipient
        service.add_recipient(
            name="Test User",
            email="test@example.com",
            phone="+1234567890",
            channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
        )
        
        # Create a notification
        with patch.object(service, "_send_email", return_value=True):
            with patch.object(service, "_send_sms", return_value=True):
                result = await service.send_notification(
                    title="Test Alert",
                    message="This is a test alert",
                    priority=NotificationPriority.HIGH,
                    recipients=["Test User"],
                    channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
                )
                
                assert result is True
    
    def test_dashboard_service(self, sample_sensor_data):
        """Test dashboard service"""
        config = DashboardConfig()
        service = DashboardService(config)
        
        # Register components
        alert_manager = AlertManager()
        service.register_alert_manager(alert_manager)
        
        # Update caches
        asyncio.run(service.update_all_caches())
        
        # Check cache updates
        assert service.alerts_cache is not None
        assert service.disease_data_cache is not None
        assert service.sensor_data_cache is not None


class TestSystemIntegration:
    """Test system integration"""
    
    @pytest.mark.asyncio
    async def test_integrated_system_initialization(self):
        """Test integrated system initialization"""
        config = SystemConfig(
            simulation_duration=60,  # 1 minute
            update_interval=5,
            num_sensors=10,
            cloud_storage_enabled=False,  # Disable for testing
            dashboard_enabled=False  # Disable for testing
        )
        
        system = IntegratedSystem(config)
        
        # Initialize system
        await system.initialize()
        
        assert system.is_initialized is True
        assert system.sensor_simulator is not None
        assert system.disease_tracking_model is not None
        assert system.alert_manager is not None
    
    @pytest.mark.asyncio
    async def test_data_flow(self):
        """Test data flow through the system"""
        config = SystemConfig(
            simulation_duration=10,  # 10 seconds
            update_interval=1,
            num_sensors=5,
            cloud_storage_enabled=False,  # Disable for testing
            dashboard_enabled=False  # Disable for testing
        )
        
        system = IntegratedSystem(config)
        await system.initialize()
        
        # Mock sensor data
        sensor_data = sample_sensor_data()
        
        # Process sensor data
        with patch.object(system.sensor_simulator, "get_latest_data", return_value=sensor_data):
            await system._process_sensor_data()
            
            # Check if data was added to buffer
            assert len(system.sensor_data_buffer) > 0
            
            # Process graph data
            await system._process_graph_data()
            
            # Check if graph data was processed
            assert len(system.graph_data_buffer) > 0
            
            # Process AI analysis
            with patch.object(system.pattern_analysis_service, "analyze", return_value={"mutations": [{"id": "test"}]}):
                await system._process_ai_analysis()
                
                # Check if mutations were detected
                assert len(system.mutation_data_buffer) > 0
                
                # Process alerts
                with patch.object(system.alert_manager, "process_alert", return_value=None):
                    await system._process_alerts()
                    
                    # Check if alerts were processed
                    assert len(system.mutation_data_buffer) == 0  # Buffer should be cleared


# Performance tests
class TestPerformance:
    """Test system performance"""
    
    @pytest.mark.asyncio
    async def test_processing_latency(self):
        """Test processing latency"""
        model = DiseaseTrackingModel()
        
        start_time = time.time()
        
        # Process multiple updates
        for i in range(5):
            # Create graph data
            graph_data = create_graph_from_sensor_data(sample_sensor_data())
            
            # Update model
            model.update(graph_data)
            
            # Get current state
            state = model.get_current_state()
            assert state is not None
        
        total_time = time.time() - start_time
        assert total_time < 5.0  # Should be under 5 seconds for 5 updates
    
    def test_memory_usage(self):
        """Test memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Create multiple components
            components = []
            for i in range(5):
                simulator = IoTSensorSimulator(num_sensors=10)
                model = DiseaseTrackingModel()
                components.append((simulator, model))
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 50MB)
            assert memory_increase < 50 * 1024 * 1024
        except ImportError:
            pytest.skip("psutil not available")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
