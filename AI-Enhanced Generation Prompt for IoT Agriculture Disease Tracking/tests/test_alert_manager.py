"""Test suite for the alert manager component of the alert system.

Tests the functionality of the AlertManager class, including:
- Alert creation for different types (disease detection, environmental anomaly)
- Alert severity calculation
- Alert description and recommendation generation
- Alert status management
- Alert subscription and notification
- Alert metrics collection
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from alert_system.alert_manager import (
    AlertManager,
    Alert,
    AlertType,
    AlertSeverity,
    AlertStatus,
    GeoLocation,
    InterventionRecommendation
)


@pytest.fixture
def alert_manager():
    """Create an alert manager instance for testing"""
    manager = AlertManager()
    
    # Add some test thresholds
    manager.severity_thresholds = {
        "disease_confidence": {
            AlertSeverity.LOW: 0.3,
            AlertSeverity.MEDIUM: 0.5,
            AlertSeverity.HIGH: 0.7,
            AlertSeverity.CRITICAL: 0.9
        },
        "environmental_deviation": {
            AlertSeverity.LOW: 0.1,
            AlertSeverity.MEDIUM: 0.2,
            AlertSeverity.HIGH: 0.3,
            AlertSeverity.CRITICAL: 0.5
        }
    }
    
    return manager


class TestAlertManager:
    """Test alert manager functionality"""
    
    def test_initialization(self, alert_manager):
        """Test manager initialization"""
        assert alert_manager.alerts == {}
        assert alert_manager.subscribers == []
        assert alert_manager.severity_thresholds is not None
        assert len(alert_manager.severity_thresholds) > 0
    
    def test_create_disease_detection_alert(self, alert_manager):
        """Test disease detection alert creation"""
        alert = alert_manager.create_disease_detection_alert(
            sensor_id="sensor_001",
            location={
                "latitude": 37.5,
                "longitude": -122.1,
                "field_id": "field_001",
                "section_id": "section_A"
            },
            disease_type="Fungal Leaf Spot",
            confidence=0.85,
            severity=0.7,
            image_url="https://example.com/images/leaf_spot.jpg"
        )
        
        assert alert is not None
        assert alert.alert_id in alert_manager.alerts
        assert alert.alert_type == AlertType.DISEASE_DETECTION
        assert alert.severity == AlertSeverity.HIGH  # Based on confidence 0.85
        assert alert.status == AlertStatus.ACTIVE
        assert alert.sensor_id == "sensor_001"
        assert alert.location.field_id == "field_001"
        assert alert.disease_type == "Fungal Leaf Spot"
        assert alert.confidence == 0.85
        assert alert.image_url == "https://example.com/images/leaf_spot.jpg"
        assert alert.created_at is not None
        assert alert.description is not None
        assert "Fungal Leaf Spot" in alert.description
    
    def test_create_environmental_anomaly_alert(self, alert_manager):
        """Test environmental anomaly alert creation"""
        alert = alert_manager.create_environmental_anomaly_alert(
            sensor_id="sensor_002",
            location={
                "latitude": 37.52,
                "longitude": -122.15,
                "field_id": "field_001",
                "section_id": "section_B"
            },
            anomaly_type="High Temperature",
            value=35.8,
            expected_range=(20.0, 30.0),
            deviation=0.35  # 35% deviation from expected range
        )
        
        assert alert is not None
        assert alert.alert_id in alert_manager.alerts
        assert alert.alert_type == AlertType.ENVIRONMENTAL_ANOMALY
        assert alert.severity == AlertSeverity.HIGH  # Based on deviation 0.35
        assert alert.status == AlertStatus.ACTIVE
        assert alert.sensor_id == "sensor_002"
        assert alert.location.field_id == "field_001"
        assert alert.anomaly_type == "High Temperature"
        assert alert.value == 35.8
        assert alert.expected_range == (20.0, 30.0)
        assert alert.deviation == 0.35
        assert alert.created_at is not None
        assert alert.description is not None
        assert "High Temperature" in alert.description
        assert "35.8" in alert.description
    
    def test_create_mutation_detection_alert(self, alert_manager):
        """Test mutation detection alert creation"""
        alert = alert_manager.create_mutation_detection_alert(
            mutation_id="mutation_001",
            location={
                "latitude": 37.5,
                "longitude": -122.1,
                "field_id": "field_001",
                "section_id": "section_A"
            },
            disease_type="Fungal Leaf Spot",
            mutation_type="Resistance Development",
            confidence=0.92,
            previous_treatment="Fungicide A",
            resistance_level=0.75
        )
        
        assert alert is not None
        assert alert.alert_id in alert_manager.alerts
        assert alert.alert_type == AlertType.MUTATION_DETECTION
        assert alert.severity == AlertSeverity.CRITICAL  # Based on confidence 0.92
        assert alert.status == AlertStatus.ACTIVE
        assert alert.mutation_id == "mutation_001"
        assert alert.location.field_id == "field_001"
        assert alert.disease_type == "Fungal Leaf Spot"
        assert alert.mutation_type == "Resistance Development"
        assert alert.confidence == 0.92
        assert alert.previous_treatment == "Fungicide A"
        assert alert.resistance_level == 0.75
        assert alert.created_at is not None
        assert alert.description is not None
        assert "Resistance Development" in alert.description
        assert "Fungal Leaf Spot" in alert.description
    
    def test_create_spread_risk_alert(self, alert_manager):
        """Test spread risk alert creation"""
        alert = alert_manager.create_spread_risk_alert(
            source_field_id="field_001",
            at_risk_fields=["field_002", "field_003"],
            disease_type="Fungal Leaf Spot",
            spread_probability=0.65,
            time_to_spread=48,  # hours
            environmental_factors=["High Humidity", "Wind Direction"]
        )
        
        assert alert is not None
        assert alert.alert_id in alert_manager.alerts
        assert alert.alert_type == AlertType.SPREAD_RISK
        assert alert.severity == AlertSeverity.MEDIUM  # Based on spread_probability 0.65
        assert alert.status == AlertStatus.ACTIVE
        assert alert.source_field_id == "field_001"
        assert alert.at_risk_fields == ["field_002", "field_003"]
        assert alert.disease_type == "Fungal Leaf Spot"
        assert alert.spread_probability == 0.65
        assert alert.time_to_spread == 48
        assert alert.environmental_factors == ["High Humidity", "Wind Direction"]
        assert alert.created_at is not None
        assert alert.description is not None
        assert "Fungal Leaf Spot" in alert.description
        assert "48 hours" in alert.description
    
    def test_calculate_severity(self, alert_manager):
        """Test severity calculation"""
        # Test disease confidence severity calculation
        severity = alert_manager._calculate_severity("disease_confidence", 0.45)
        assert severity == AlertSeverity.LOW
        
        severity = alert_manager._calculate_severity("disease_confidence", 0.65)
        assert severity == AlertSeverity.MEDIUM
        
        severity = alert_manager._calculate_severity("disease_confidence", 0.85)
        assert severity == AlertSeverity.HIGH
        
        severity = alert_manager._calculate_severity("disease_confidence", 0.95)
        assert severity == AlertSeverity.CRITICAL
        
        # Test environmental deviation severity calculation
        severity = alert_manager._calculate_severity("environmental_deviation", 0.05)
        assert severity == AlertSeverity.LOW
        
        severity = alert_manager._calculate_severity("environmental_deviation", 0.15)
        assert severity == AlertSeverity.MEDIUM
        
        severity = alert_manager._calculate_severity("environmental_deviation", 0.25)
        assert severity == AlertSeverity.HIGH
        
        severity = alert_manager._calculate_severity("environmental_deviation", 0.55)
        assert severity == AlertSeverity.CRITICAL
        
        # Test unknown metric
        with pytest.raises(ValueError):
            alert_manager._calculate_severity("unknown_metric", 0.5)
    
    def test_generate_disease_description(self, alert_manager):
        """Test disease description generation"""
        description = alert_manager._generate_disease_description(
            disease_type="Fungal Leaf Spot",
            confidence=0.85,
            severity=AlertSeverity.HIGH,
            location=GeoLocation(
                latitude=37.5,
                longitude=-122.1,
                field_id="field_001",
                section_id="section_A"
            )
        )
        
        assert description is not None
        assert "Fungal Leaf Spot" in description
        assert "field_001" in description
        assert "section_A" in description
        assert "85%" in description or "0.85" in description
        assert "HIGH" in description
    
    def test_generate_environmental_description(self, alert_manager):
        """Test environmental anomaly description generation"""
        description = alert_manager._generate_environmental_description(
            anomaly_type="High Temperature",
            value=35.8,
            expected_range=(20.0, 30.0),
            deviation=0.35,
            severity=AlertSeverity.HIGH,
            location=GeoLocation(
                latitude=37.52,
                longitude=-122.15,
                field_id="field_001",
                section_id="section_B"
            )
        )
        
        assert description is not None
        assert "High Temperature" in description
        assert "field_001" in description
        assert "section_B" in description
        assert "35.8" in description
        assert "20.0" in description and "30.0" in description
        assert "35%" in description or "0.35" in description
        assert "HIGH" in description
    
    def test_generate_mutation_description(self, alert_manager):
        """Test mutation description generation"""
        description = alert_manager._generate_mutation_description(
            disease_type="Fungal Leaf Spot",
            mutation_type="Resistance Development",
            confidence=0.92,
            previous_treatment="Fungicide A",
            resistance_level=0.75,
            severity=AlertSeverity.CRITICAL,
            location=GeoLocation(
                latitude=37.5,
                longitude=-122.1,
                field_id="field_001",
                section_id="section_A"
            )
        )
        
        assert description is not None
        assert "Fungal Leaf Spot" in description
        assert "Resistance Development" in description
        assert "field_001" in description
        assert "section_A" in description
        assert "92%" in description or "0.92" in description
        assert "Fungicide A" in description
        assert "75%" in description or "0.75" in description
        assert "CRITICAL" in description
    
    def test_generate_spread_risk_description(self, alert_manager):
        """Test spread risk description generation"""
        description = alert_manager._generate_spread_risk_description(
            disease_type="Fungal Leaf Spot",
            source_field_id="field_001",
            at_risk_fields=["field_002", "field_003"],
            spread_probability=0.65,
            time_to_spread=48,
            environmental_factors=["High Humidity", "Wind Direction"],
            severity=AlertSeverity.MEDIUM
        )
        
        assert description is not None
        assert "Fungal Leaf Spot" in description
        assert "field_001" in description
        assert "field_002" in description and "field_003" in description
        assert "65%" in description or "0.65" in description
        assert "48 hours" in description
        assert "High Humidity" in description
        assert "Wind Direction" in description
        assert "MEDIUM" in description
    
    def test_generate_intervention_recommendation(self, alert_manager):
        """Test intervention recommendation generation"""
        # Disease detection alert
        disease_alert = alert_manager.create_disease_detection_alert(
            sensor_id="sensor_001",
            location={
                "latitude": 37.5,
                "longitude": -122.1,
                "field_id": "field_001",
                "section_id": "section_A"
            },
            disease_type="Fungal Leaf Spot",
            confidence=0.85,
            severity=0.7
        )
        
        recommendation = alert_manager.generate_intervention_recommendation(disease_alert.alert_id)
        
        assert recommendation is not None
        assert recommendation.alert_id == disease_alert.alert_id
        assert recommendation.recommendation_id is not None
        assert recommendation.created_at is not None
        assert recommendation.intervention_type is not None
        assert len(recommendation.steps) > 0
        assert recommendation.priority is not None
        assert recommendation.estimated_resources is not None
        assert recommendation.expected_outcomes is not None
        
        # Environmental anomaly alert
        env_alert = alert_manager.create_environmental_anomaly_alert(
            sensor_id="sensor_002",
            location={
                "latitude": 37.52,
                "longitude": -122.15,
                "field_id": "field_001",
                "section_id": "section_B"
            },
            anomaly_type="High Temperature",
            value=35.8,
            expected_range=(20.0, 30.0),
            deviation=0.35
        )
        
        recommendation = alert_manager.generate_intervention_recommendation(env_alert.alert_id)
        
        assert recommendation is not None
        assert recommendation.alert_id == env_alert.alert_id
        assert recommendation.recommendation_id is not None
        assert recommendation.intervention_type is not None
        assert len(recommendation.steps) > 0
        
        # Test with invalid alert ID
        with pytest.raises(ValueError):
            alert_manager.generate_intervention_recommendation("invalid_id")
    
    def test_update_alert_status(self, alert_manager):
        """Test alert status update"""
        # Create an alert
        alert = alert_manager.create_disease_detection_alert(
            sensor_id="sensor_001",
            location={
                "latitude": 37.5,
                "longitude": -122.1,
                "field_id": "field_001",
                "section_id": "section_A"
            },
            disease_type="Fungal Leaf Spot",
            confidence=0.85,
            severity=0.7
        )
        
        # Update status to acknowledged
        alert_manager.update_alert_status(alert.alert_id, AlertStatus.ACKNOWLEDGED)
        updated_alert = alert_manager.get_alert(alert.alert_id)
        
        assert updated_alert.status == AlertStatus.ACKNOWLEDGED
        assert updated_alert.acknowledged_at is not None
        
        # Update status to resolved
        alert_manager.update_alert_status(alert.alert_id, AlertStatus.RESOLVED, resolution_notes="Applied fungicide treatment")
        updated_alert = alert_manager.get_alert(alert.alert_id)
        
        assert updated_alert.status == AlertStatus.RESOLVED
        assert updated_alert.resolved_at is not None
        assert updated_alert.resolution_notes == "Applied fungicide treatment"
        
        # Test with invalid alert ID
        with pytest.raises(ValueError):
            alert_manager.update_alert_status("invalid_id", AlertStatus.ACKNOWLEDGED)
    
    def test_get_alerts_by_criteria(self, alert_manager):
        """Test getting alerts by various criteria"""
        # Create multiple alerts
        alert1 = alert_manager.create_disease_detection_alert(
            sensor_id="sensor_001",
            location={
                "latitude": 37.5,
                "longitude": -122.1,
                "field_id": "field_001",
                "section_id": "section_A"
            },
            disease_type="Fungal Leaf Spot",
            confidence=0.85,
            severity=0.7
        )
        
        alert2 = alert_manager.create_environmental_anomaly_alert(
            sensor_id="sensor_002",
            location={
                "latitude": 37.52,
                "longitude": -122.15,
                "field_id": "field_001",
                "section_id": "section_B"
            },
            anomaly_type="High Temperature",
            value=35.8,
            expected_range=(20.0, 30.0),
            deviation=0.35
        )
        
        alert3 = alert_manager.create_disease_detection_alert(
            sensor_id="sensor_003",
            location={
                "latitude": 37.55,
                "longitude": -122.2,
                "field_id": "field_002",
                "section_id": "section_A"
            },
            disease_type="Powdery Mildew",
            confidence=0.95,
            severity=0.9
        )
        
        # Update status of one alert
        alert_manager.update_alert_status(alert1.alert_id, AlertStatus.ACKNOWLEDGED)
        
        # Test get_active_alerts
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 2
        assert alert2.alert_id in [a.alert_id for a in active_alerts]
        assert alert3.alert_id in [a.alert_id for a in active_alerts]
        
        # Test get_alerts_by_type
        disease_alerts = alert_manager.get_alerts_by_type(AlertType.DISEASE_DETECTION)
        assert len(disease_alerts) == 2
        assert alert1.alert_id in [a.alert_id for a in disease_alerts]
        assert alert3.alert_id in [a.alert_id for a in disease_alerts]
        
        # Test get_alerts_by_severity
        critical_alerts = alert_manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].alert_id == alert3.alert_id
        
        # Test get_alerts_by_field
        field1_alerts = alert_manager.get_alerts_by_field("field_001")
        assert len(field1_alerts) == 2
        assert alert1.alert_id in [a.alert_id for a in field1_alerts]
        assert alert2.alert_id in [a.alert_id for a in field1_alerts]
        
        # Test get_alerts_by_status
        acknowledged_alerts = alert_manager.get_alerts_by_status(AlertStatus.ACKNOWLEDGED)
        assert len(acknowledged_alerts) == 1
        assert acknowledged_alerts[0].alert_id == alert1.alert_id
        
        # Test get_alerts_by_time_range
        now = datetime.now()
        recent_alerts = alert_manager.get_alerts_by_time_range(
            start_time=now - timedelta(minutes=5),
            end_time=now + timedelta(minutes=5)
        )
        assert len(recent_alerts) == 3  # All alerts should be recent
    
    @pytest.mark.asyncio
    async def test_subscribe_to_alerts(self, alert_manager):
        """Test alert subscription"""
        # Create a mock callback
        mock_callback = AsyncMock()
        
        # Subscribe to alerts
        alert_manager.subscribe_to_alerts(mock_callback)
        assert len(alert_manager.subscribers) == 1
        
        # Create an alert to trigger the callback
        alert = alert_manager.create_disease_detection_alert(
            sensor_id="sensor_001",
            location={
                "latitude": 37.5,
                "longitude": -122.1,
                "field_id": "field_001",
                "section_id": "section_A"
            },
            disease_type="Fungal Leaf Spot",
            confidence=0.85,
            severity=0.7
        )
        
        # Verify callback was called
        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0][0]
        assert call_args.alert_id == alert.alert_id
        
        # Unsubscribe
        alert_manager.unsubscribe_from_alerts(mock_callback)
        assert len(alert_manager.subscribers) == 0
    
    def test_process_alert(self, alert_manager):
        """Test alert processing"""
        # Create a mock notification service
        mock_notification_service = MagicMock()
        mock_notification_service.send_notification = AsyncMock(return_value=True)
        
        # Set notification service
        alert_manager.notification_service = mock_notification_service
        
        # Process an alert
        alert_data = {
            "sensor_id": "sensor_001",
            "alert_type": "disease_detection",
            "location": {
                "latitude": 37.5,
                "longitude": -122.1,
                "field_id": "field_001",
                "section_id": "section_A"
            },
            "disease_type": "Fungal Leaf Spot",
            "confidence": 0.85,
            "severity": 0.7
        }
        
        alert = alert_manager.process_alert(alert_data)
        
        assert alert is not None
        assert alert.alert_id in alert_manager.alerts
        assert alert.alert_type == AlertType.DISEASE_DETECTION
        
        # Verify notification was sent
        mock_notification_service.send_notification.assert_called_once()
    
    def test_get_alert_metrics(self, alert_manager):
        """Test alert metrics collection"""
        # Create multiple alerts of different types and severities
        alert_manager.create_disease_detection_alert(
            sensor_id="sensor_001",
            location={
                "latitude": 37.5,
                "longitude": -122.1,
                "field_id": "field_001",
                "section_id": "section_A"
            },
            disease_type="Fungal Leaf Spot",
            confidence=0.85,
            severity=0.7
        )
        
        alert_manager.create_environmental_anomaly_alert(
            sensor_id="sensor_002",
            location={
                "latitude": 37.52,
                "longitude": -122.15,
                "field_id": "field_001",
                "section_id": "section_B"
            },
            anomaly_type="High Temperature",
            value=35.8,
            expected_range=(20.0, 30.0),
            deviation=0.35
        )
        
        alert_manager.create_disease_detection_alert(
            sensor_id="sensor_003",
            location={
                "latitude": 37.55,
                "longitude": -122.2,
                "field_id": "field_002",
                "section_id": "section_A"
            },
            disease_type="Powdery Mildew",
            confidence=0.95,
            severity=0.9
        )
        
        # Update status of one alert
        alert_ids = list(alert_manager.alerts.keys())
        alert_manager.update_alert_status(alert_ids[0], AlertStatus.ACKNOWLEDGED)
        
        # Get metrics
        metrics = alert_manager.get_alert_metrics()
        
        assert metrics is not None
        assert metrics["total_alerts"] == 3
        assert metrics["active_alerts"] == 2
        assert metrics["acknowledged_alerts"] == 1
        assert metrics["resolved_alerts"] == 0
        
        assert metrics["by_type"][AlertType.DISEASE_DETECTION.value] == 2
        assert metrics["by_type"][AlertType.ENVIRONMENTAL_ANOMALY.value] == 1
        
        assert metrics["by_severity"][AlertSeverity.HIGH.value] == 2
        assert metrics["by_severity"][AlertSeverity.CRITICAL.value] == 1
        
        assert metrics["by_field"]["field_001"] == 2
        assert metrics["by_field"]["field_002"] == 1
        
        assert "avg_time_to_acknowledge" in metrics
        assert "avg_time_to_resolve" in metrics


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])