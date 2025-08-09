"""Test suite for the dashboard component of the alert system.

Tests the functionality of the DashboardService and DashboardUI classes, including:
- Dashboard configuration and initialization
- Data caching and retrieval
- Module registration
- UI rendering and component functionality
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import streamlit as st

from alert_system.dashboard import DashboardConfig, DashboardService, DashboardUI
from alert_system.alert_manager import Alert, AlertType, AlertSeverity, AlertStatus, GeoLocation


@pytest.fixture
def mock_alert_manager():
    """Create a mock alert manager for testing"""
    mock = MagicMock()
    
    # Mock alerts data
    mock_alerts = [
        Alert(
            alert_id="alert_001",
            alert_type=AlertType.DISEASE_DETECTION,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            sensor_id="sensor_001",
            location=GeoLocation(
                latitude=37.5,
                longitude=-122.1,
                field_id="field_001",
                section_id="section_A"
            ),
            disease_type="Fungal Leaf Spot",
            confidence=0.85,
            description="High confidence detection of Fungal Leaf Spot in field_001, section_A"
        ),
        Alert(
            alert_id="alert_002",
            alert_type=AlertType.ENVIRONMENTAL_ANOMALY,
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.ACTIVE,
            sensor_id="sensor_002",
            location=GeoLocation(
                latitude=37.52,
                longitude=-122.15,
                field_id="field_001",
                section_id="section_B"
            ),
            anomaly_type="High Temperature",
            value=35.8,
            expected_range=(20.0, 30.0),
            deviation=0.25,
            description="Temperature anomaly detected in field_001, section_B"
        )
    ]
    
    # Mock alert manager methods
    mock.get_all_alerts.return_value = mock_alerts
    mock.get_active_alerts.return_value = mock_alerts
    mock.get_alerts_by_type.return_value = [mock_alerts[0]]
    mock.get_alerts_by_severity.return_value = [mock_alerts[0]]
    mock.get_alerts_by_field.return_value = mock_alerts
    mock.get_alert_metrics.return_value = {
        "total_alerts": 2,
        "active_alerts": 2,
        "acknowledged_alerts": 0,
        "resolved_alerts": 0,
        "by_type": {
            "DISEASE_DETECTION": 1,
            "ENVIRONMENTAL_ANOMALY": 1
        },
        "by_severity": {
            "HIGH": 1,
            "MEDIUM": 1
        },
        "by_field": {
            "field_001": 2
        }
    }
    
    return mock


@pytest.fixture
def mock_graph_engine():
    """Create a mock graph engine for testing"""
    mock = MagicMock()
    
    # Mock graph data
    mock_nodes = pd.DataFrame({
        "node_id": [f"node_{i}" for i in range(10)],
        "latitude": np.random.uniform(37.4, 37.6, 10),
        "longitude": np.random.uniform(-122.2, -122.0, 10),
        "field_id": ["field_001" if i < 6 else "field_002" for i in range(10)],
        "section_id": [f"section_{chr(65+i%5)}" for i in range(10)],
        "disease_risk": np.random.uniform(0, 1, 10),
        "temperature": np.random.uniform(20, 30, 10),
        "humidity": np.random.uniform(40, 80, 10)
    })
    
    mock_edges = pd.DataFrame({
        "source": [f"node_{i}" for i in range(8)],
        "target": [f"node_{i+1}" for i in range(8)],
        "weight": np.random.uniform(0, 1, 8),
        "disease_spread_risk": np.random.uniform(0, 1, 8)
    })
    
    # Mock graph engine methods
    mock.get_graph_data.return_value = (mock_nodes, mock_edges)
    mock.get_disease_spread_prediction.return_value = pd.DataFrame({
        "node_id": [f"node_{i}" for i in range(10)],
        "field_id": ["field_001" if i < 6 else "field_002" for i in range(10)],
        "section_id": [f"section_{chr(65+i%5)}" for i in range(10)],
        "disease_risk_t0": np.random.uniform(0, 0.3, 10),
        "disease_risk_t1": np.random.uniform(0, 0.5, 10),
        "disease_risk_t2": np.random.uniform(0, 0.7, 10)
    })
    
    return mock


@pytest.fixture
def mock_ai_models():
    """Create a mock AI models service for testing"""
    mock = MagicMock()
    
    # Mock mutation data
    mock_mutations = pd.DataFrame({
        "mutation_id": [f"mutation_{i}" for i in range(5)],
        "disease_type": ["Fungal Leaf Spot", "Powdery Mildew", "Rust", "Fungal Leaf Spot", "Bacterial Blight"],
        "mutation_type": ["Resistance", "Virulence", "Host Range", "Resistance", "Virulence"],
        "detection_date": pd.date_range(start="2023-01-01", periods=5),
        "field_id": ["field_001", "field_001", "field_002", "field_002", "field_001"],
        "confidence": np.random.uniform(0.7, 0.95, 5),
        "previous_treatment": ["Fungicide A", "Fungicide B", "Fungicide A", "Fungicide C", "Antibiotic A"],
        "resistance_level": np.random.uniform(0.5, 0.9, 5)
    })
    
    # Mock AI models methods
    mock.get_detected_mutations.return_value = mock_mutations
    mock.get_disease_classification_results.return_value = pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(10)],
        "sensor_id": [f"sensor_{i}" for i in range(10)],
        "field_id": ["field_001" if i < 6 else "field_002" for i in range(10)],
        "disease_type": ["Fungal Leaf Spot", "Powdery Mildew", "Rust", "Bacterial Blight", "Healthy"] * 2,
        "confidence": np.random.uniform(0.7, 0.99, 10),
        "detection_date": pd.date_range(start="2023-01-01", periods=10)
    })
    
    return mock


@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager for testing"""
    mock = MagicMock()
    
    # Mock sensor data
    mock_sensor_data = pd.DataFrame({
        "sensor_id": [f"sensor_{i}" for i in range(20)],
        "timestamp": pd.date_range(start="2023-01-01", periods=20, freq="H"),
        "latitude": np.random.uniform(37.4, 37.6, 20),
        "longitude": np.random.uniform(-122.2, -122.0, 20),
        "field_id": ["field_001" if i < 12 else "field_002" for i in range(20)],
        "section_id": [f"section_{chr(65+i%5)}" for i in range(20)],
        "temperature": np.random.uniform(20, 30, 20),
        "humidity": np.random.uniform(40, 80, 20),
        "soil_moisture": np.random.uniform(0.2, 0.5, 20),
        "light_intensity": np.random.uniform(5000, 10000, 20),
        "wind_speed": np.random.uniform(0, 15, 20),
        "rainfall": np.random.uniform(0, 10, 20)
    })
    
    # Mock storage manager methods
    mock.get_sensor_data.return_value = mock_sensor_data
    mock.get_field_boundaries.return_value = {
        "field_001": [
            {"lat": 37.45, "lng": -122.15},
            {"lat": 37.45, "lng": -122.05},
            {"lat": 37.55, "lng": -122.05},
            {"lat": 37.55, "lng": -122.15}
        ],
        "field_002": [
            {"lat": 37.56, "lng": -122.15},
            {"lat": 37.56, "lng": -122.05},
            {"lat": 37.65, "lng": -122.05},
            {"lat": 37.65, "lng": -122.15}
        ]
    }
    
    return mock


@pytest.fixture
def dashboard_service(mock_alert_manager, mock_graph_engine, mock_ai_models, mock_storage_manager):
    """Create a dashboard service instance for testing"""
    config = DashboardConfig(
        update_interval=60,  # seconds
        cache_expiry=300,  # seconds
        map_center={"lat": 37.5, "lng": -122.1},
        map_zoom=12
    )
    
    service = DashboardService(config)
    
    # Register mock components
    service.register_alert_manager(mock_alert_manager)
    service.register_graph_engine(mock_graph_engine)
    service.register_ai_models(mock_ai_models)
    service.register_storage_manager(mock_storage_manager)
    
    return service


class TestDashboardConfig:
    """Test dashboard configuration"""
    
    def test_initialization(self):
        """Test config initialization with default and custom values"""
        # Test with default values
        default_config = DashboardConfig()
        assert default_config.update_interval == 60
        assert default_config.cache_expiry == 300
        assert default_config.map_center == {"lat": 37.5, "lng": -122.1}
        assert default_config.map_zoom == 10
        
        # Test with custom values
        custom_config = DashboardConfig(
            update_interval=30,
            cache_expiry=600,
            map_center={"lat": 38.0, "lng": -123.0},
            map_zoom=8
        )
        assert custom_config.update_interval == 30
        assert custom_config.cache_expiry == 600
        assert custom_config.map_center == {"lat": 38.0, "lng": -123.0}
        assert custom_config.map_zoom == 8


class TestDashboardService:
    """Test dashboard service functionality"""
    
    def test_initialization(self, dashboard_service):
        """Test service initialization"""
        assert dashboard_service.config is not None
        assert dashboard_service.alert_manager is not None
        assert dashboard_service.graph_engine is not None
        assert dashboard_service.ai_models is not None
        assert dashboard_service.storage_manager is not None
        assert dashboard_service._alert_cache == {}
        assert dashboard_service._disease_data_cache == {}
        assert dashboard_service._sensor_data_cache == {}
        assert dashboard_service._mutation_data_cache == {}
    
    def test_register_components(self):
        """Test component registration"""
        config = DashboardConfig()
        service = DashboardService(config)
        
        # Create mock components
        mock_alert_manager = MagicMock()
        mock_graph_engine = MagicMock()
        mock_ai_models = MagicMock()
        mock_storage_manager = MagicMock()
        
        # Register components
        service.register_alert_manager(mock_alert_manager)
        service.register_graph_engine(mock_graph_engine)
        service.register_ai_models(mock_ai_models)
        service.register_storage_manager(mock_storage_manager)
        
        # Verify registration
        assert service.alert_manager == mock_alert_manager
        assert service.graph_engine == mock_graph_engine
        assert service.ai_models == mock_ai_models
        assert service.storage_manager == mock_storage_manager
    
    def test_get_alerts(self, dashboard_service):
        """Test alert data retrieval"""
        # Get alerts (should fetch from mock and cache)
        alerts = dashboard_service.get_alerts()
        
        assert len(alerts) == 2
        assert alerts[0].alert_id == "alert_001"
        assert alerts[1].alert_id == "alert_002"
        assert "alerts" in dashboard_service._alert_cache
        
        # Get alerts by type
        disease_alerts = dashboard_service.get_alerts_by_type(AlertType.DISEASE_DETECTION)
        assert len(disease_alerts) == 1
        assert disease_alerts[0].alert_id == "alert_001"
        
        # Get alerts by severity
        high_alerts = dashboard_service.get_alerts_by_severity(AlertSeverity.HIGH)
        assert len(high_alerts) == 1
        assert high_alerts[0].alert_id == "alert_001"
        
        # Get alerts by field
        field_alerts = dashboard_service.get_alerts_by_field("field_001")
        assert len(field_alerts) == 2
    
    def test_get_alert_metrics(self, dashboard_service):
        """Test alert metrics retrieval"""
        metrics = dashboard_service.get_alert_metrics()
        
        assert metrics is not None
        assert metrics["total_alerts"] == 2
        assert metrics["active_alerts"] == 2
        assert metrics["by_type"]["DISEASE_DETECTION"] == 1
        assert metrics["by_severity"]["HIGH"] == 1
        assert metrics["by_field"]["field_001"] == 2
    
    def test_get_disease_data(self, dashboard_service):
        """Test disease data retrieval"""
        # Get disease spread prediction
        prediction = dashboard_service.get_disease_spread_prediction()
        
        assert prediction is not None
        assert len(prediction) == 10
        assert "disease_spread_prediction" in dashboard_service._disease_data_cache
        
        # Get disease classification results
        results = dashboard_service.get_disease_classification_results()
        
        assert results is not None
        assert len(results) == 10
        assert "disease_classification" in dashboard_service._disease_data_cache
    
    def test_get_sensor_data(self, dashboard_service):
        """Test sensor data retrieval"""
        # Get sensor data
        data = dashboard_service.get_sensor_data()
        
        assert data is not None
        assert len(data) == 20
        assert "sensor_data" in dashboard_service._sensor_data_cache
        
        # Get field boundaries
        boundaries = dashboard_service.get_field_boundaries()
        
        assert boundaries is not None
        assert len(boundaries) == 2
        assert "field_001" in boundaries
        assert "field_boundaries" in dashboard_service._sensor_data_cache
    
    def test_get_mutation_data(self, dashboard_service):
        """Test mutation data retrieval"""
        mutations = dashboard_service.get_detected_mutations()
        
        assert mutations is not None
        assert len(mutations) == 5
        assert "detected_mutations" in dashboard_service._mutation_data_cache
    
    def test_cache_expiry(self, dashboard_service):
        """Test cache expiry functionality"""
        # Get data to populate cache
        dashboard_service.get_alerts()
        dashboard_service.get_disease_spread_prediction()
        dashboard_service.get_sensor_data()
        dashboard_service.get_detected_mutations()
        
        # Verify cache is populated
        assert len(dashboard_service._alert_cache) > 0
        assert len(dashboard_service._disease_data_cache) > 0
        assert len(dashboard_service._sensor_data_cache) > 0
        assert len(dashboard_service._mutation_data_cache) > 0
        
        # Manually expire cache
        dashboard_service._expire_cache()
        
        # Verify cache is cleared
        assert len(dashboard_service._alert_cache) == 0
        assert len(dashboard_service._disease_data_cache) == 0
        assert len(dashboard_service._sensor_data_cache) == 0
        assert len(dashboard_service._mutation_data_cache) == 0


class TestDashboardUI:
    """Test dashboard UI functionality"""
    
    @patch("streamlit.title")
    @patch("streamlit.sidebar.title")
    def test_initialization(self, mock_sidebar_title, mock_title, dashboard_service):
        """Test UI initialization"""
        ui = DashboardUI(dashboard_service)
        
        assert ui.dashboard_service == dashboard_service
        assert ui.config == dashboard_service.config
    
    @patch("streamlit.title")
    @patch("streamlit.sidebar.title")
    @patch("streamlit.sidebar.radio")
    def test_render_sidebar(self, mock_radio, mock_sidebar_title, mock_title, dashboard_service):
        """Test sidebar rendering"""
        ui = DashboardUI(dashboard_service)
        
        # Mock radio button selection
        mock_radio.return_value = "Disease Map"
        
        # Render sidebar
        selected = ui._render_sidebar()
        
        assert selected == "Disease Map"
        mock_sidebar_title.assert_called_once()
        mock_radio.assert_called_once()
    
    @patch("streamlit.title")
    @patch("streamlit.sidebar.title")
    @patch("streamlit.map")
    @patch("streamlit.dataframe")
    def test_render_disease_map(self, mock_dataframe, mock_map, mock_sidebar_title, mock_title, dashboard_service):
        """Test disease map rendering"""
        ui = DashboardUI(dashboard_service)
        
        # Render disease map
        ui._render_disease_map()
        
        # Verify map was rendered
        mock_map.assert_called()
    
    @patch("streamlit.title")
    @patch("streamlit.sidebar.title")
    @patch("streamlit.dataframe")
    @patch("streamlit.bar_chart")
    def test_render_alerts_section(self, mock_bar_chart, mock_dataframe, mock_sidebar_title, mock_title, dashboard_service):
        """Test alerts section rendering"""
        ui = DashboardUI(dashboard_service)
        
        # Render alerts section
        ui._render_alerts_section()
        
        # Verify dataframe and chart were rendered
        mock_dataframe.assert_called()
        mock_bar_chart.assert_called()
    
    @patch("streamlit.title")
    @patch("streamlit.sidebar.title")
    @patch("streamlit.dataframe")
    @patch("streamlit.line_chart")
    def test_render_mutations_section(self, mock_line_chart, mock_dataframe, mock_sidebar_title, mock_title, dashboard_service):
        """Test mutations section rendering"""
        ui = DashboardUI(dashboard_service)
        
        # Render mutations section
        ui._render_mutations_section()
        
        # Verify dataframe was rendered
        mock_dataframe.assert_called()
    
    @patch("streamlit.title")
    @patch("streamlit.sidebar.title")
    @patch("streamlit.dataframe")
    @patch("streamlit.line_chart")
    def test_render_sensor_data_section(self, mock_line_chart, mock_dataframe, mock_sidebar_title, mock_title, dashboard_service):
        """Test sensor data section rendering"""
        ui = DashboardUI(dashboard_service)
        
        # Render sensor data section
        ui._render_sensor_data_section()
        
        # Verify dataframe and chart were rendered
        mock_dataframe.assert_called()
        mock_line_chart.assert_called()
    
    @patch("streamlit.title")
    @patch("streamlit.sidebar.title")
    @patch("streamlit.sidebar.radio")
    @patch("streamlit.map")
    @patch("streamlit.dataframe")
    def test_render(self, mock_dataframe, mock_map, mock_radio, mock_sidebar_title, mock_title, dashboard_service):
        """Test full UI rendering"""
        ui = DashboardUI(dashboard_service)
        
        # Mock radio button selection
        mock_radio.return_value = "Disease Map"
        
        # Render UI
        ui.render()
        
        # Verify title and map were rendered
        mock_title.assert_called()
        mock_map.assert_called()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])