"""Test suite for the integration module.

Tests the functionality of the IntegratedSystem class, including:
- System configuration and initialization
- Component integration and lifecycle management
- Data flow between components
- System-wide operations
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

from integration import SystemConfig, IntegratedSystem


@pytest.fixture
def mock_components():
    """Create mock components for testing the integrated system"""
    # Mock IoT sensor network
    mock_sensor_network = MagicMock()
    mock_sensor_network.initialize_sensors = AsyncMock()
    mock_sensor_network.start_data_collection = AsyncMock()
    mock_sensor_network.get_sensor_data = AsyncMock(return_value=pd.DataFrame({
        "sensor_id": [f"sensor_{i}" for i in range(5)],
        "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="H"),
        "latitude": np.random.uniform(37.4, 37.6, 5),
        "longitude": np.random.uniform(-122.2, -122.0, 5),
        "field_id": ["field_001" if i < 3 else "field_002" for i in range(5)],
        "section_id": [f"section_{chr(65+i%3)}" for i in range(5)],
        "temperature": np.random.uniform(20, 30, 5),
        "humidity": np.random.uniform(40, 80, 5),
        "soil_moisture": np.random.uniform(0.2, 0.5, 5),
        "light_intensity": np.random.uniform(5000, 10000, 5),
        "wind_speed": np.random.uniform(0, 15, 5),
        "rainfall": np.random.uniform(0, 10, 5)
    }))
    mock_sensor_network.stop_data_collection = AsyncMock()
    
    # Mock graph engine
    mock_graph_engine = MagicMock()
    mock_graph_engine.initialize_graph = AsyncMock()
    mock_graph_engine.update_node_data = AsyncMock()
    mock_graph_engine.process_graph = AsyncMock()
    mock_graph_engine.get_disease_spread_prediction = AsyncMock(return_value=pd.DataFrame({
        "node_id": [f"node_{i}" for i in range(5)],
        "field_id": ["field_001" if i < 3 else "field_002" for i in range(5)],
        "section_id": [f"section_{chr(65+i%3)}" for i in range(5)],
        "disease_risk_t0": np.random.uniform(0, 0.3, 5),
        "disease_risk_t1": np.random.uniform(0, 0.5, 5),
        "disease_risk_t2": np.random.uniform(0, 0.7, 5)
    }))
    
    # Mock AI models
    mock_ai_models = MagicMock()
    mock_ai_models.initialize_models = AsyncMock()
    mock_ai_models.analyze_sensor_data = AsyncMock(return_value={
        "disease_detections": [
            {
                "sensor_id": "sensor_001",
                "field_id": "field_001",
                "section_id": "section_A",
                "disease_type": "Fungal Leaf Spot",
                "confidence": 0.85,
                "severity": 0.7,
                "image_url": "https://example.com/images/leaf_spot.jpg"
            }
        ],
        "environmental_anomalies": [
            {
                "sensor_id": "sensor_002",
                "field_id": "field_001",
                "section_id": "section_B",
                "anomaly_type": "High Temperature",
                "value": 35.8,
                "expected_range": (20.0, 30.0),
                "deviation": 0.35
            }
        ],
        "mutations": [
            {
                "mutation_id": "mutation_001",
                "field_id": "field_001",
                "section_id": "section_A",
                "disease_type": "Fungal Leaf Spot",
                "mutation_type": "Resistance Development",
                "confidence": 0.92,
                "previous_treatment": "Fungicide A",
                "resistance_level": 0.75
            }
        ]
    })
    
    # Mock cloud storage
    mock_storage = MagicMock()
    mock_storage.initialize_storage = AsyncMock()
    mock_storage.store_sensor_data = AsyncMock()
    mock_storage.store_analysis_results = AsyncMock()
    mock_storage.store_alert_data = AsyncMock()
    
    # Mock alert system
    mock_alert_system = MagicMock()
    mock_alert_system.initialize = AsyncMock()
    mock_alert_system.process_alerts = AsyncMock(return_value=[
        {
            "alert_id": "alert_001",
            "alert_type": "DISEASE_DETECTION",
            "severity": "HIGH",
            "status": "ACTIVE",
            "sensor_id": "sensor_001",
            "field_id": "field_001",
            "section_id": "section_A",
            "disease_type": "Fungal Leaf Spot",
            "confidence": 0.85,
            "description": "High confidence detection of Fungal Leaf Spot in field_001, section_A"
        },
        {
            "alert_id": "alert_002",
            "alert_type": "ENVIRONMENTAL_ANOMALY",
            "severity": "HIGH",
            "status": "ACTIVE",
            "sensor_id": "sensor_002",
            "field_id": "field_001",
            "section_id": "section_B",
            "anomaly_type": "High Temperature",
            "value": 35.8,
            "expected_range": (20.0, 30.0),
            "deviation": 0.35,
            "description": "Temperature anomaly detected in field_001, section_B"
        }
    ])
    
    # Mock dashboard
    mock_dashboard = MagicMock()
    mock_dashboard.initialize = AsyncMock()
    mock_dashboard.update_data = AsyncMock()
    mock_dashboard.start = AsyncMock()
    
    return {
        "sensor_network": mock_sensor_network,
        "graph_engine": mock_graph_engine,
        "ai_models": mock_ai_models,
        "storage": mock_storage,
        "alert_system": mock_alert_system,
        "dashboard": mock_dashboard
    }


@pytest.fixture
def system_config():
    """Create a system configuration for testing"""
    return SystemConfig(
        sensor_network_config={
            "num_sensors": 50,
            "update_interval": 5,  # seconds
            "fields": [
                {
                    "field_id": "field_001",
                    "name": "North Field",
                    "center": {"latitude": 37.5, "longitude": -122.1},
                    "radius": 1000,  # meters
                    "sections": 5
                },
                {
                    "field_id": "field_002",
                    "name": "South Field",
                    "center": {"latitude": 37.4, "longitude": -122.0},
                    "radius": 800,  # meters
                    "sections": 4
                }
            ]
        },
        graph_engine_config={
            "model_type": "STGCN",
            "time_steps": 24,
            "hidden_dim": 64,
            "spatial_kernel_size": 3,
            "temporal_kernel_size": 3
        },
        ai_models_config={
            "disease_detection": {
                "model_type": "ResNet50",
                "confidence_threshold": 0.7
            },
            "anomaly_detection": {
                "algorithm": "isolation_forest",
                "contamination": 0.05
            },
            "mutation_tracking": {
                "similarity_threshold": 0.8,
                "history_window": 30  # days
            }
        },
        storage_config={
            "storage_type": "local",
            "base_path": "./data",
            "backup_interval": 86400  # seconds (1 day)
        },
        alert_system_config={
            "notification_channels": ["email", "dashboard"],
            "severity_thresholds": {
                "disease_confidence": {
                    "LOW": 0.3,
                    "MEDIUM": 0.5,
                    "HIGH": 0.7,
                    "CRITICAL": 0.9
                },
                "environmental_deviation": {
                    "LOW": 0.1,
                    "MEDIUM": 0.2,
                    "HIGH": 0.3,
                    "CRITICAL": 0.5
                }
            }
        },
        dashboard_config={
            "update_interval": 60,  # seconds
            "cache_expiry": 300,  # seconds
            "map_center": {"lat": 37.5, "lng": -122.1},
            "map_zoom": 12
        },
        system_settings={
            "processing_interval": 10,  # seconds
            "log_level": "INFO",
            "enable_performance_tracking": True
        }
    )


class TestSystemConfig:
    """Test system configuration"""
    
    def test_initialization(self):
        """Test config initialization with default and custom values"""
        # Test with minimal values
        minimal_config = SystemConfig()
        assert minimal_config.sensor_network_config is not None
        assert minimal_config.graph_engine_config is not None
        assert minimal_config.ai_models_config is not None
        assert minimal_config.storage_config is not None
        assert minimal_config.alert_system_config is not None
        assert minimal_config.dashboard_config is not None
        assert minimal_config.system_settings is not None
        
        # Test with custom values
        custom_config = system_config()
        assert custom_config.sensor_network_config["num_sensors"] == 50
        assert custom_config.graph_engine_config["model_type"] == "STGCN"
        assert custom_config.ai_models_config["disease_detection"]["model_type"] == "ResNet50"
        assert custom_config.storage_config["storage_type"] == "local"
        assert "email" in custom_config.alert_system_config["notification_channels"]
        assert custom_config.dashboard_config["update_interval"] == 60
        assert custom_config.system_settings["processing_interval"] == 10
    
    def test_to_dict(self, system_config):
        """Test conversion to dictionary"""
        config_dict = system_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "sensor_network_config" in config_dict
        assert "graph_engine_config" in config_dict
        assert "ai_models_config" in config_dict
        assert "storage_config" in config_dict
        assert "alert_system_config" in config_dict
        assert "dashboard_config" in config_dict
        assert "system_settings" in config_dict
    
    def test_from_dict(self):
        """Test creation from dictionary"""
        config_dict = {
            "sensor_network_config": {
                "num_sensors": 30,
                "update_interval": 10
            },
            "system_settings": {
                "processing_interval": 15,
                "log_level": "DEBUG"
            }
        }
        
        config = SystemConfig.from_dict(config_dict)
        
        assert config.sensor_network_config["num_sensors"] == 30
        assert config.sensor_network_config["update_interval"] == 10
        assert config.system_settings["processing_interval"] == 15
        assert config.system_settings["log_level"] == "DEBUG"
        
        # Other configs should have default values
        assert config.graph_engine_config is not None
        assert config.ai_models_config is not None


class TestIntegratedSystem:
    """Test integrated system functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, system_config, mock_components):
        """Test system initialization"""
        system = IntegratedSystem(system_config)
        
        # Set mock components
        system.sensor_network = mock_components["sensor_network"]
        system.graph_engine = mock_components["graph_engine"]
        system.ai_models = mock_components["ai_models"]
        system.storage_manager = mock_components["storage"]
        system.alert_system = mock_components["alert_system"]
        system.dashboard = mock_components["dashboard"]
        
        # Initialize system
        await system.initialize()
        
        # Verify all components were initialized
        mock_components["sensor_network"].initialize_sensors.assert_called_once()
        mock_components["graph_engine"].initialize_graph.assert_called_once()
        mock_components["ai_models"].initialize_models.assert_called_once()
        mock_components["storage"].initialize_storage.assert_called_once()
        mock_components["alert_system"].initialize.assert_called_once()
        mock_components["dashboard"].initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_system(self, system_config, mock_components):
        """Test system startup"""
        system = IntegratedSystem(system_config)
        
        # Set mock components
        system.sensor_network = mock_components["sensor_network"]
        system.graph_engine = mock_components["graph_engine"]
        system.ai_models = mock_components["ai_models"]
        system.storage_manager = mock_components["storage"]
        system.alert_system = mock_components["alert_system"]
        system.dashboard = mock_components["dashboard"]
        
        # Start system (non-blocking)
        with patch("asyncio.create_task") as mock_create_task:
            await system.start(blocking=False)
            
            # Verify sensor data collection was started
            mock_components["sensor_network"].start_data_collection.assert_called_once()
            
            # Verify processing loop was started
            mock_create_task.assert_called()
    
    @pytest.mark.asyncio
    async def test_stop_system(self, system_config, mock_components):
        """Test system shutdown"""
        system = IntegratedSystem(system_config)
        
        # Set mock components
        system.sensor_network = mock_components["sensor_network"]
        system.graph_engine = mock_components["graph_engine"]
        system.ai_models = mock_components["ai_models"]
        system.storage_manager = mock_components["storage"]
        system.alert_system = mock_components["alert_system"]
        system.dashboard = mock_components["dashboard"]
        
        # Set running flag
        system._running = True
        
        # Stop system
        await system.stop()
        
        # Verify running flag was set to False
        assert system._running is False
        
        # Verify sensor data collection was stopped
        mock_components["sensor_network"].stop_data_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_processing_cycle(self, system_config, mock_components):
        """Test a single processing cycle"""
        system = IntegratedSystem(system_config)
        
        # Set mock components
        system.sensor_network = mock_components["sensor_network"]
        system.graph_engine = mock_components["graph_engine"]
        system.ai_models = mock_components["ai_models"]
        system.storage_manager = mock_components["storage"]
        system.alert_system = mock_components["alert_system"]
        system.dashboard = mock_components["dashboard"]
        
        # Run a single processing cycle
        await system._processing_cycle()
        
        # Verify data flow through components
        mock_components["sensor_network"].get_sensor_data.assert_called_once()
        mock_components["storage"].store_sensor_data.assert_called_once()
        mock_components["graph_engine"].update_node_data.assert_called_once()
        mock_components["graph_engine"].process_graph.assert_called_once()
        mock_components["ai_models"].analyze_sensor_data.assert_called_once()
        mock_components["storage"].store_analysis_results.assert_called_once()
        mock_components["alert_system"].process_alerts.assert_called_once()
        mock_components["storage"].store_alert_data.assert_called_once()
        mock_components["dashboard"].update_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_dashboard(self, system_config, mock_components):
        """Test dashboard startup"""
        system = IntegratedSystem(system_config)
        
        # Set mock dashboard
        system.dashboard = mock_components["dashboard"]
        
        # Start dashboard
        await system.start_dashboard()
        
        # Verify dashboard was started
        mock_components["dashboard"].start.assert_called_once()
    
    def test_get_system_status(self, system_config, mock_components):
        """Test system status reporting"""
        system = IntegratedSystem(system_config)
        
        # Set mock components
        system.sensor_network = mock_components["sensor_network"]
        system.graph_engine = mock_components["graph_engine"]
        system.ai_models = mock_components["ai_models"]
        system.storage_manager = mock_components["storage"]
        system.alert_system = mock_components["alert_system"]
        system.dashboard = mock_components["dashboard"]
        
        # Set running flag
        system._running = True
        
        # Get system status
        status = system.get_system_status()
        
        assert status is not None
        assert "running" in status
        assert status["running"] is True
        assert "components" in status
        assert "performance_metrics" in status
    
    def test_get_performance_metrics(self, system_config, mock_components):
        """Test performance metrics collection"""
        system = IntegratedSystem(system_config)
        
        # Set mock components
        system.sensor_network = mock_components["sensor_network"]
        system.graph_engine = mock_components["graph_engine"]
        system.ai_models = mock_components["ai_models"]
        system.storage_manager = mock_components["storage"]
        system.alert_system = mock_components["alert_system"]
        system.dashboard = mock_components["dashboard"]
        
        # Add some performance data
        system._performance_metrics = {
            "processing_cycles": 10,
            "avg_cycle_time": 0.5,  # seconds
            "sensor_data_points": 500,
            "alerts_generated": 5,
            "component_times": {
                "sensor_data_collection": 0.1,
                "graph_processing": 0.2,
                "ai_analysis": 0.3,
                "alert_processing": 0.1
            }
        }
        
        # Get performance metrics
        metrics = system.get_performance_metrics()
        
        assert metrics is not None
        assert metrics["processing_cycles"] == 10
        assert metrics["avg_cycle_time"] == 0.5
        assert metrics["sensor_data_points"] == 500
        assert metrics["alerts_generated"] == 5
        assert "component_times" in metrics


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])