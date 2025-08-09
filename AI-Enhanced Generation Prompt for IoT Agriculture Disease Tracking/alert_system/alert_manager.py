"""Alert Manager for IoT Crop Disease Tracking

This module provides alert management capabilities for disease detection,
mutation tracking, and intervention recommendations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts that can be generated"""
    DISEASE_DETECTED = "disease_detected"
    DISEASE_MUTATION = "disease_mutation"
    DISEASE_SPREAD = "disease_spread"
    ENVIRONMENTAL_ANOMALY = "environmental_anomaly"
    SYSTEM_HEALTH = "system_health"
    INTERVENTION_RECOMMENDATION = "intervention_recommendation"


class AlertStatus(str, Enum):
    """Status of an alert"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class GeoLocation(BaseModel):
    """Geographic location model"""
    latitude: float
    longitude: float
    field_id: Optional[str] = None
    section_id: Optional[str] = None


class InterventionRecommendation(BaseModel):
    """Recommended intervention actions"""
    action_type: str
    description: str
    urgency: AlertSeverity
    target_area: Optional[List[GeoLocation]] = None
    estimated_effectiveness: Optional[float] = None
    estimated_cost: Optional[float] = None


class Alert(BaseModel):
    """Alert model for disease and system notifications"""
    alert_id: str = Field(..., description="Unique identifier for the alert")
    timestamp: datetime = Field(default_factory=datetime.now)
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus = Field(default=AlertStatus.NEW)
    title: str
    description: str
    source: str = Field(..., description="Source of the alert (sensor ID, system component, etc.)")
    location: Optional[GeoLocation] = None
    affected_area_size: Optional[float] = None  # in square meters or hectares
    disease_type: Optional[str] = None
    confidence: Optional[float] = None  # 0.0 to 1.0
    recommendations: Optional[List[InterventionRecommendation]] = None
    metadata: Dict = Field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def acknowledge(self, user_id: str):
        """Mark alert as acknowledged"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.now()
    
    def resolve(self, user_id: str):
        """Mark alert as resolved"""
        self.status = AlertStatus.RESOLVED
        self.resolved_by = user_id
        self.resolved_at = datetime.now()
    
    def dismiss(self, user_id: str):
        """Mark alert as dismissed"""
        self.status = AlertStatus.DISMISSED
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.now()


class AlertThreshold(BaseModel):
    """Threshold configuration for generating alerts"""
    alert_type: AlertType
    min_confidence: float = 0.7  # Minimum confidence level to trigger alert
    min_severity: AlertSeverity = AlertSeverity.MEDIUM
    cooldown_period: int = 3600  # Seconds between similar alerts
    aggregation_window: int = 300  # Seconds to aggregate similar alerts
    affected_area_threshold: Optional[float] = None  # Min area size to trigger alert


class AlertManager:
    """Manager for handling alerts in the crop disease tracking system"""
    
    def __init__(self, notification_service=None):
        """Initialize the alert manager
        
        Args:
            notification_service: Service for sending notifications
        """
        self.alerts: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}  # alert_id -> Alert
        self.alert_history: List[Alert] = []
        self.thresholds: Dict[AlertType, AlertThreshold] = {}
        self.notification_service = notification_service
        self.alert_cooldowns: Dict[Tuple[str, str], datetime] = {}  # (source, alert_type) -> last_alert_time
        self.alert_subscribers: Dict[AlertType, Set[str]] = {}
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        # Performance tracking
        self.processing_times: List[float] = []
        
        logger.info("Alert Manager initialized")
    
    def _initialize_default_thresholds(self):
        """Initialize default alert thresholds"""
        for alert_type in AlertType:
            if alert_type == AlertType.DISEASE_MUTATION:
                # Higher sensitivity for mutations
                self.thresholds[alert_type] = AlertThreshold(
                    alert_type=alert_type,
                    min_confidence=0.6,
                    min_severity=AlertSeverity.MEDIUM,
                    cooldown_period=1800  # 30 minutes
                )
            elif alert_type == AlertType.DISEASE_SPREAD:
                # Track disease spread closely
                self.thresholds[alert_type] = AlertThreshold(
                    alert_type=alert_type,
                    min_confidence=0.65,
                    min_severity=AlertSeverity.MEDIUM,
                    cooldown_period=3600,  # 1 hour
                    affected_area_threshold=100  # square meters
                )
            else:
                # Default thresholds
                self.thresholds[alert_type] = AlertThreshold(
                    alert_type=alert_type
                )
    
    def update_threshold(self, alert_type: AlertType, **kwargs):
        """Update threshold configuration for an alert type
        
        Args:
            alert_type: Type of alert to update threshold for
            **kwargs: Threshold parameters to update
        """
        if alert_type in self.thresholds:
            current = self.thresholds[alert_type]
            self.thresholds[alert_type] = AlertThreshold(
                alert_type=alert_type,
                min_confidence=kwargs.get('min_confidence', current.min_confidence),
                min_severity=kwargs.get('min_severity', current.min_severity),
                cooldown_period=kwargs.get('cooldown_period', current.cooldown_period),
                aggregation_window=kwargs.get('aggregation_window', current.aggregation_window),
                affected_area_threshold=kwargs.get('affected_area_threshold', current.affected_area_threshold)
            )
            logger.info(f"Updated threshold for {alert_type}")
        else:
            self.thresholds[alert_type] = AlertThreshold(alert_type=alert_type, **kwargs)
            logger.info(f"Created new threshold for {alert_type}")
    
    def subscribe_to_alerts(self, alert_type: AlertType, subscriber_id: str):
        """Subscribe to alerts of a specific type
        
        Args:
            alert_type: Type of alert to subscribe to
            subscriber_id: ID of the subscriber (user, system, etc.)
        """
        if alert_type not in self.alert_subscribers:
            self.alert_subscribers[alert_type] = set()
        
        self.alert_subscribers[alert_type].add(subscriber_id)
        logger.info(f"Subscriber {subscriber_id} added for {alert_type} alerts")
    
    def unsubscribe_from_alerts(self, alert_type: AlertType, subscriber_id: str):
        """Unsubscribe from alerts of a specific type
        
        Args:
            alert_type: Type of alert to unsubscribe from
            subscriber_id: ID of the subscriber
        """
        if alert_type in self.alert_subscribers and subscriber_id in self.alert_subscribers[alert_type]:
            self.alert_subscribers[alert_type].remove(subscriber_id)
            logger.info(f"Subscriber {subscriber_id} removed from {alert_type} alerts")
    
    async def process_disease_detection(self, detection_data: Dict):
        """Process disease detection data and generate alerts if needed
        
        Args:
            detection_data: Disease detection data from AI models
        """
        start_time = time.time()
        
        # Extract relevant information
        disease_type = detection_data.get('disease_type')
        confidence = detection_data.get('confidence', 0.0)
        location = detection_data.get('location')
        source = detection_data.get('source', 'unknown')
        is_mutation = detection_data.get('is_mutation', False)
        affected_area = detection_data.get('affected_area', 0.0)
        
        # Determine alert type
        alert_type = AlertType.DISEASE_MUTATION if is_mutation else AlertType.DISEASE_DETECTED
        
        # Check if we should generate an alert based on thresholds
        threshold = self.thresholds.get(alert_type)
        if not threshold or confidence < threshold.min_confidence:
            logger.debug(f"Detection below confidence threshold: {confidence} < {threshold.min_confidence}")
            return
        
        # Check cooldown period
        cooldown_key = (source, str(alert_type))
        if cooldown_key in self.alert_cooldowns:
            last_alert_time = self.alert_cooldowns[cooldown_key]
            time_since_last = (datetime.now() - last_alert_time).total_seconds()
            if time_since_last < threshold.cooldown_period:
                logger.debug(f"Alert in cooldown period: {time_since_last}s < {threshold.cooldown_period}s")
                return
        
        # Check affected area threshold if applicable
        if threshold.affected_area_threshold and affected_area < threshold.affected_area_threshold:
            logger.debug(f"Affected area below threshold: {affected_area} < {threshold.affected_area_threshold}")
            return
        
        # Determine severity based on confidence and affected area
        severity = self._calculate_severity(confidence, affected_area, is_mutation)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(disease_type, severity, affected_area, is_mutation)
        
        # Create alert
        alert_id = f"{alert_type.value}_{int(time.time())}_{source}"
        geo_location = GeoLocation(**location) if location else None
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=f"{disease_type} {'Mutation' if is_mutation else 'Detection'}",
            description=self._generate_description(disease_type, confidence, affected_area, is_mutation),
            source=source,
            location=geo_location,
            affected_area_size=affected_area,
            disease_type=disease_type,
            confidence=confidence,
            recommendations=recommendations,
            metadata=detection_data
        )
        
        # Store alert
        self.alerts.append(alert)
        self.active_alerts[alert_id] = alert
        self.alert_cooldowns[cooldown_key] = datetime.now()
        
        # Send notification if service is available
        if self.notification_service:
            await self.notification_service.send_alert_notification(alert)
        
        # Record performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        logger.info(f"Generated {severity} alert for {disease_type} {'mutation' if is_mutation else 'detection'}")
        return alert_id
    
    def _calculate_severity(self, confidence: float, affected_area: float, is_mutation: bool) -> AlertSeverity:
        """Calculate alert severity based on detection parameters
        
        Args:
            confidence: Detection confidence (0.0 to 1.0)
            affected_area: Size of affected area
            is_mutation: Whether this is a disease mutation
            
        Returns:
            Appropriate alert severity level
        """
        # Base severity on confidence
        if confidence > 0.9:
            base_severity = AlertSeverity.CRITICAL
        elif confidence > 0.8:
            base_severity = AlertSeverity.HIGH
        elif confidence > 0.7:
            base_severity = AlertSeverity.MEDIUM
        else:
            base_severity = AlertSeverity.LOW
        
        # Adjust for mutations (increase severity)
        if is_mutation and base_severity != AlertSeverity.CRITICAL:
            severity_levels = list(AlertSeverity)
            current_index = severity_levels.index(base_severity)
            if current_index < len(severity_levels) - 1:
                return severity_levels[current_index + 1]
        
        # Adjust for large affected areas
        if affected_area > 1000 and base_severity != AlertSeverity.CRITICAL:  # 1000 sq meters
            severity_levels = list(AlertSeverity)
            current_index = severity_levels.index(base_severity)
            if current_index < len(severity_levels) - 1:
                return severity_levels[current_index + 1]
        
        return base_severity
    
    def _generate_description(self, disease_type: str, confidence: float, 
                             affected_area: float, is_mutation: bool) -> str:
        """Generate a human-readable alert description
        
        Args:
            disease_type: Type of detected disease
            confidence: Detection confidence
            affected_area: Size of affected area
            is_mutation: Whether this is a disease mutation
            
        Returns:
            Formatted description string
        """
        confidence_pct = int(confidence * 100)
        area_str = f"{affected_area:.1f} square meters"
        
        if is_mutation:
            return (f"A mutation of {disease_type} has been detected with {confidence_pct}% confidence. "
                   f"The affected area is approximately {area_str}. Immediate attention is recommended "
                   f"as mutations may spread more rapidly or be resistant to standard treatments.")
        else:
            return (f"{disease_type} has been detected with {confidence_pct}% confidence. "
                   f"The affected area is approximately {area_str}. "
                   f"Please review the recommended interventions.")
    
    def _generate_recommendations(self, disease_type: str, severity: AlertSeverity, 
                                affected_area: float, is_mutation: bool) -> List[InterventionRecommendation]:
        """Generate intervention recommendations based on detection
        
        Args:
            disease_type: Type of detected disease
            severity: Alert severity level
            affected_area: Size of affected area
            is_mutation: Whether this is a disease mutation
            
        Returns:
            List of intervention recommendations
        """
        recommendations = []
        
        # Add inspection recommendation
        recommendations.append(InterventionRecommendation(
            action_type="inspection",
            description="Conduct detailed field inspection to confirm detection and assess spread",
            urgency=severity
        ))
        
        # Add treatment recommendation based on disease type and severity
        if "fungal" in disease_type.lower():
            if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                recommendations.append(InterventionRecommendation(
                    action_type="treatment",
                    description=f"Apply broad-spectrum fungicide to affected area{' and surrounding buffer zone' if is_mutation else ''}",
                    urgency=severity,
                    estimated_effectiveness=0.85 if not is_mutation else 0.7
                ))
            else:
                recommendations.append(InterventionRecommendation(
                    action_type="treatment",
                    description="Apply targeted fungicide to affected plants",
                    urgency=severity,
                    estimated_effectiveness=0.9 if not is_mutation else 0.75
                ))
        
        elif "bacterial" in disease_type.lower():
            if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                recommendations.append(InterventionRecommendation(
                    action_type="treatment",
                    description=f"Apply copper-based bactericide to affected area{' and surrounding buffer zone' if is_mutation else ''}",
                    urgency=severity,
                    estimated_effectiveness=0.8 if not is_mutation else 0.65
                ))
            else:
                recommendations.append(InterventionRecommendation(
                    action_type="treatment",
                    description="Apply targeted bactericide to affected plants",
                    urgency=severity,
                    estimated_effectiveness=0.85 if not is_mutation else 0.7
                ))
        
        elif "viral" in disease_type.lower():
            # No direct treatment for viral diseases, focus on containment
            recommendations.append(InterventionRecommendation(
                action_type="containment",
                description="Remove and destroy infected plants to prevent spread",
                urgency=severity,
                estimated_effectiveness=0.9
            ))
        
        else:  # Generic or unknown disease type
            recommendations.append(InterventionRecommendation(
                action_type="treatment",
                description="Apply appropriate treatment based on field inspection results",
                urgency=severity
            ))
        
        # Add quarantine recommendation for severe cases or mutations
        if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] or is_mutation:
            buffer_size = 10 if affected_area < 100 else 20  # meters
            recommendations.append(InterventionRecommendation(
                action_type="quarantine",
                description=f"Establish {buffer_size}m buffer zone around affected area to prevent spread",
                urgency=severity
            ))
        
        # Add monitoring recommendation
        monitoring_frequency = "daily" if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] else "weekly"
        recommendations.append(InterventionRecommendation(
            action_type="monitoring",
            description=f"Implement {monitoring_frequency} monitoring of affected and surrounding areas",
            urgency=AlertSeverity.MEDIUM
        ))
        
        return recommendations
    
    async def process_environmental_anomaly(self, anomaly_data: Dict):
        """Process environmental anomaly data and generate alerts if needed
        
        Args:
            anomaly_data: Environmental anomaly data
        """
        start_time = time.time()
        
        # Extract relevant information
        anomaly_type = anomaly_data.get('anomaly_type', 'unknown')
        confidence = anomaly_data.get('confidence', 0.0)
        location = anomaly_data.get('location')
        source = anomaly_data.get('source', 'unknown')
        affected_area = anomaly_data.get('affected_area', 0.0)
        severity_score = anomaly_data.get('severity_score', 0.5)  # 0.0 to 1.0
        
        # Map severity score to alert severity
        if severity_score > 0.8:
            severity = AlertSeverity.CRITICAL
        elif severity_score > 0.6:
            severity = AlertSeverity.HIGH
        elif severity_score > 0.4:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        # Check thresholds
        threshold = self.thresholds.get(AlertType.ENVIRONMENTAL_ANOMALY)
        if not threshold or confidence < threshold.min_confidence:
            return
        
        # Check cooldown period
        cooldown_key = (source, str(AlertType.ENVIRONMENTAL_ANOMALY))
        if cooldown_key in self.alert_cooldowns:
            last_alert_time = self.alert_cooldowns[cooldown_key]
            time_since_last = (datetime.now() - last_alert_time).total_seconds()
            if time_since_last < threshold.cooldown_period:
                return
        
        # Create alert
        alert_id = f"env_anomaly_{int(time.time())}_{source}"
        geo_location = GeoLocation(**location) if location else None
        
        # Generate recommendations based on anomaly type
        recommendations = self._generate_environmental_recommendations(anomaly_type, severity)
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=AlertType.ENVIRONMENTAL_ANOMALY,
            severity=severity,
            title=f"Environmental Anomaly: {anomaly_type}",
            description=self._generate_environmental_description(anomaly_type, confidence, severity_score),
            source=source,
            location=geo_location,
            affected_area_size=affected_area,
            confidence=confidence,
            recommendations=recommendations,
            metadata=anomaly_data
        )
        
        # Store alert
        self.alerts.append(alert)
        self.active_alerts[alert_id] = alert
        self.alert_cooldowns[cooldown_key] = datetime.now()
        
        # Send notification if service is available
        if self.notification_service:
            await self.notification_service.send_alert_notification(alert)
        
        # Record performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        logger.info(f"Generated {severity} alert for environmental anomaly: {anomaly_type}")
        return alert_id
    
    def _generate_environmental_description(self, anomaly_type: str, confidence: float, 
                                          severity_score: float) -> str:
        """Generate a description for environmental anomaly alerts
        
        Args:
            anomaly_type: Type of environmental anomaly
            confidence: Detection confidence
            severity_score: Severity score of the anomaly
            
        Returns:
            Formatted description string
        """
        confidence_pct = int(confidence * 100)
        severity_pct = int(severity_score * 100)
        
        if "moisture" in anomaly_type.lower():
            return (f"Abnormal soil moisture levels detected with {confidence_pct}% confidence. "
                   f"Severity level: {severity_pct}%. This condition may affect plant health "
                   f"and potentially increase susceptibility to diseases.")
        
        elif "temperature" in anomaly_type.lower():
            return (f"Abnormal temperature conditions detected with {confidence_pct}% confidence. "
                   f"Severity level: {severity_pct}%. Extreme temperatures can stress plants "
                   f"and create favorable conditions for certain pathogens.")
        
        elif "humidity" in anomaly_type.lower():
            return (f"Abnormal humidity levels detected with {confidence_pct}% confidence. "
                   f"Severity level: {severity_pct}%. High humidity can promote fungal growth "
                   f"and disease spread.")
        
        else:  # Generic environmental anomaly
            return (f"Environmental anomaly ({anomaly_type}) detected with {confidence_pct}% confidence. "
                   f"Severity level: {severity_pct}%. Please review the recommended actions.")
    
    def _generate_environmental_recommendations(self, anomaly_type: str, 
                                              severity: AlertSeverity) -> List[InterventionRecommendation]:
        """Generate recommendations for environmental anomalies
        
        Args:
            anomaly_type: Type of environmental anomaly
            severity: Alert severity level
            
        Returns:
            List of intervention recommendations
        """
        recommendations = []
        
        # Add monitoring recommendation
        recommendations.append(InterventionRecommendation(
            action_type="monitoring",
            description=f"Increase monitoring frequency for {anomaly_type} conditions",
            urgency=severity
        ))
        
        # Add specific recommendations based on anomaly type
        if "moisture" in anomaly_type.lower():
            if "high" in anomaly_type.lower():
                recommendations.append(InterventionRecommendation(
                    action_type="irrigation",
                    description="Reduce irrigation schedule and check for drainage issues",
                    urgency=severity
                ))
            elif "low" in anomaly_type.lower():
                recommendations.append(InterventionRecommendation(
                    action_type="irrigation",
                    description="Increase irrigation frequency and check system functionality",
                    urgency=severity
                ))
        
        elif "temperature" in anomaly_type.lower():
            if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                recommendations.append(InterventionRecommendation(
                    action_type="protection",
                    description="Consider temporary shading or protective measures",
                    urgency=severity
                ))
        
        elif "humidity" in anomaly_type.lower():
            if "high" in anomaly_type.lower() and severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                recommendations.append(InterventionRecommendation(
                    action_type="preventive_treatment",
                    description="Consider preventive fungicide application to reduce disease risk",
                    urgency=severity
                ))
        
        # Add disease risk assessment for all environmental anomalies
        recommendations.append(InterventionRecommendation(
            action_type="risk_assessment",
            description="Conduct disease risk assessment based on current environmental conditions",
            urgency=AlertSeverity.MEDIUM
        ))
        
        return recommendations
    
    def get_active_alerts(self, alert_type: Optional[AlertType] = None, 
                         min_severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active (unresolved) alerts with optional filtering
        
        Args:
            alert_type: Filter by alert type
            min_severity: Filter by minimum severity level
            
        Returns:
            List of matching active alerts
        """
        filtered_alerts = list(self.active_alerts.values())
        
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type]
        
        if min_severity:
            severity_levels = list(AlertSeverity)
            min_severity_index = severity_levels.index(min_severity)
            filtered_alerts = [a for a in filtered_alerts if 
                              severity_levels.index(a.severity) >= min_severity_index]
        
        return filtered_alerts
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by its ID
        
        Args:
            alert_id: ID of the alert to retrieve
            
        Returns:
            Alert if found, None otherwise
        """
        return self.active_alerts.get(alert_id)
    
    def update_alert_status(self, alert_id: str, status: AlertStatus, user_id: str) -> bool:
        """Update the status of an alert
        
        Args:
            alert_id: ID of the alert to update
            status: New status for the alert
            user_id: ID of the user making the change
            
        Returns:
            True if successful, False otherwise
        """
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False
        
        if status == AlertStatus.ACKNOWLEDGED:
            alert.acknowledge(user_id)
        elif status == AlertStatus.RESOLVED:
            alert.resolve(user_id)
            # Move to history and remove from active alerts
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
        elif status == AlertStatus.DISMISSED:
            alert.dismiss(user_id)
            # Move to history and remove from active alerts
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
        else:
            alert.status = status
        
        logger.info(f"Alert {alert_id} status updated to {status} by {user_id}")
        return True
    
    def get_alert_metrics(self) -> Dict:
        """Get metrics about alerts and processing performance
        
        Returns:
            Dictionary of alert metrics
        """
        active_by_type = {}
        active_by_severity = {}
        
        for alert in self.active_alerts.values():
            # Count by type
            alert_type = str(alert.alert_type)
            if alert_type not in active_by_type:
                active_by_type[alert_type] = 0
            active_by_type[alert_type] += 1
            
            # Count by severity
            severity = str(alert.severity)
            if severity not in active_by_severity:
                active_by_severity[severity] = 0
            active_by_severity[severity] += 1
        
        # Calculate processing performance
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            "total_active_alerts": len(self.active_alerts),
            "total_historical_alerts": len(self.alert_history),
            "alerts_by_type": active_by_type,
            "alerts_by_severity": active_by_severity,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "thresholds": {str(k): v.model_dump() for k, v in self.thresholds.items()}
        }


# Example usage
async def main():
    """Example usage of the alert manager"""
    # Create alert manager
    alert_manager = AlertManager()
    
    # Process a disease detection
    detection_data = {
        "disease_type": "Fungal Leaf Spot",
        "confidence": 0.85,
        "is_mutation": True,
        "location": {"latitude": 34.052235, "longitude": -118.243683, "field_id": "field_001"},
        "affected_area": 150.0,
        "source": "sensor_123"
    }
    
    alert_id = await alert_manager.process_disease_detection(detection_data)
    print(f"Generated alert: {alert_id}")
    
    # Get active alerts
    active_alerts = alert_manager.get_active_alerts(min_severity=AlertSeverity.MEDIUM)
    print(f"Active alerts: {len(active_alerts)}")
    
    # Get metrics
    metrics = alert_manager.get_alert_metrics()
    print(f"Alert metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())