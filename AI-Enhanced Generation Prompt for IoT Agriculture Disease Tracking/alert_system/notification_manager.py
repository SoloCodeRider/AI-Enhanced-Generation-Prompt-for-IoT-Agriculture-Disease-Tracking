"""
Alert and Notification System for IoT Crop Disease Tracking

Implements real-time alerts for:
- Disease mutations and emerging threats
- Pattern deviations and anomalies
- Intervention recommendations
- Mobile/web dashboard notifications
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import uuid
import logging

from pydantic import BaseModel, Field
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests


@dataclass
class AlertConfig:
    """Alert configuration settings"""
    email_enabled: bool = True
    sms_enabled: bool = False
    webhook_enabled: bool = True
    dashboard_enabled: bool = True
    alert_threshold: float = 0.7
    notification_cooldown_minutes: int = 30
    max_alerts_per_hour: int = 10


@dataclass
class AlertRecipient:
    """Alert recipient configuration"""
    user_id: str
    name: str
    email: str
    phone: Optional[str] = None
    alert_preferences: Dict[str, bool] = field(default_factory=dict)
    location: Optional[Tuple[float, float]] = None
    field_id: Optional[str] = None


class AlertSeverity:
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType:
    """Alert type definitions"""
    DISEASE_DETECTED = "disease_detected"
    MUTATION_DETECTED = "mutation_detected"
    PATTERN_DEVIATION = "pattern_deviation"
    SPREAD_WARNING = "spread_warning"
    INTERVENTION_REQUIRED = "intervention_required"
    SYSTEM_ALERT = "system_alert"


class Alert(BaseModel):
    """Alert data model"""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    alert_type: str
    severity: str
    title: str
    message: str
    location: Optional[Tuple[float, float]] = None
    affected_area: List[Tuple[float, float]] = Field(default_factory=list)
    disease_type: Optional[str] = None
    confidence: float = 0.0
    recommended_action: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class InterventionRecommendation(BaseModel):
    """Intervention recommendation model"""
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    alert_id: str
    intervention_type: str
    priority: str
    description: str
    steps: List[str] = Field(default_factory=list)
    estimated_duration: str = ""
    required_resources: List[str] = Field(default_factory=list)
    cost_estimate: Optional[float] = None
    effectiveness_estimate: float = 0.0
    risk_assessment: str = ""


class NotificationChannel:
    """Base notification channel"""
    
    async def send_notification(self, alert: Alert, recipients: List[AlertRecipient]) -> bool:
        """Send notification to recipients"""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_config: Dict[str, str]):
        self.smtp_config = smtp_config
        
    async def send_notification(self, alert: Alert, recipients: List[AlertRecipient]) -> bool:
        """Send email notification"""
        try:
            for recipient in recipients:
                if not recipient.alert_preferences.get('email', True):
                    continue
                    
                # Create email message
                msg = MIMEMultipart()
                msg['From'] = self.smtp_config['from_email']
                msg['To'] = recipient.email
                msg['Subject'] = f"ALERT: {alert.title}"
                
                # Create email body
                body = self._create_email_body(alert, recipient)
                msg.attach(MIMEText(body, 'html'))
                
                # Send email
                with smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port']) as server:
                    if self.smtp_config.get('use_tls'):
                        server.starttls()
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                    server.send_message(msg)
                
                logging.info(f"Email alert sent to {recipient.email}")
                
            return True
            
        except Exception as e:
            logging.error(f"Email notification failed: {e}")
            return False
    
    def _create_email_body(self, alert: Alert, recipient: AlertRecipient) -> str:
        """Create HTML email body"""
        severity_colors = {
            AlertSeverity.LOW: "#28a745",
            AlertSeverity.MEDIUM: "#ffc107",
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545"
        }
        
        color = severity_colors.get(alert.severity, "#6c757d")
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert {{ border-left: 4px solid {color}; padding: 10px; margin: 10px 0; }}
                .severity {{ color: {color}; font-weight: bold; }}
                .action {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h2>Crop Disease Alert</h2>
            <div class="alert">
                <p><strong>Severity:</strong> <span class="severity">{alert.severity.upper()}</span></p>
                <p><strong>Type:</strong> {alert.alert_type.replace('_', ' ').title()}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                {f'<p><strong>Disease Type:</strong> {alert.disease_type}</p>' if alert.disease_type else ''}
                {f'<p><strong>Confidence:</strong> {alert.confidence:.1%}</p>' if alert.confidence else ''}
            </div>
            {f'<div class="action"><h3>Recommended Action:</h3><p>{alert.recommended_action}</p></div>' if alert.recommended_action else ''}
            <p>Please log into your dashboard for more details and to acknowledge this alert.</p>
        </body>
        </html>
        """
        
        return html


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel"""
    
    def __init__(self, webhook_urls: List[str]):
        self.webhook_urls = webhook_urls
        
    async def send_notification(self, alert: Alert, recipients: List[AlertRecipient]) -> bool:
        """Send webhook notification"""
        try:
            payload = {
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp.isoformat(),
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'title': alert.title,
                'message': alert.message,
                'location': alert.location,
                'affected_area': alert.affected_area,
                'disease_type': alert.disease_type,
                'confidence': alert.confidence,
                'recommended_action': alert.recommended_action,
                'metadata': alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                for webhook_url in self.webhook_urls:
                    try:
                        async with session.post(webhook_url, json=payload) as response:
                            if response.status == 200:
                                logging.info(f"Webhook notification sent to {webhook_url}")
                            else:
                                logging.warning(f"Webhook notification failed for {webhook_url}: {response.status}")
                    except Exception as e:
                        logging.error(f"Webhook notification error for {webhook_url}: {e}")
            
            return True
            
        except Exception as e:
            logging.error(f"Webhook notification failed: {e}")
            return False


class SMSNotificationChannel(NotificationChannel):
    """SMS notification channel"""
    
    def __init__(self, sms_config: Dict[str, str]):
        self.sms_config = sms_config
        
    async def send_notification(self, alert: Alert, recipients: List[AlertRecipient]) -> bool:
        """Send SMS notification"""
        try:
            for recipient in recipients:
                if not recipient.phone or not recipient.alert_preferences.get('sms', False):
                    continue
                
                # Create SMS message
                message = self._create_sms_message(alert)
                
                # Send SMS (implementation depends on SMS provider)
                # This is a placeholder for SMS sending logic
                logging.info(f"SMS alert sent to {recipient.phone}: {message}")
            
            return True
            
        except Exception as e:
            logging.error(f"SMS notification failed: {e}")
            return False
    
    def _create_sms_message(self, alert: Alert) -> str:
        """Create SMS message"""
        return f"ALERT: {alert.title} - {alert.message[:100]}..."


class DashboardNotificationChannel(NotificationChannel):
    """Dashboard notification channel"""
    
    def __init__(self, dashboard_url: str):
        self.dashboard_url = dashboard_url
        
    async def send_notification(self, alert: Alert, recipients: List[AlertRecipient]) -> bool:
        """Send dashboard notification"""
        try:
            # Send notification to dashboard
            payload = {
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp.isoformat(),
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'title': alert.title,
                'message': alert.message,
                'location': alert.location,
                'affected_area': alert.affected_area,
                'disease_type': alert.disease_type,
                'confidence': alert.confidence,
                'recommended_action': alert.recommended_action,
                'metadata': alert.metadata,
                'recipients': [r.user_id for r in recipients]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.dashboard_url}/api/alerts", json=payload) as response:
                    if response.status == 200:
                        logging.info(f"Dashboard notification sent")
                    else:
                        logging.warning(f"Dashboard notification failed: {response.status}")
            
            return True
            
        except Exception as e:
            logging.error(f"Dashboard notification failed: {e}")
            return False


class AlertManager:
    """Main alert manager for IoT crop disease tracking"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.alert_history: List[Alert] = []
        self.recipients: Dict[str, AlertRecipient] = {}
        self.notification_channels: List[NotificationChannel] = []
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.hourly_alert_counts: Dict[str, int] = {}
        
        # Initialize notification channels
        self._initialize_channels()
        
        # Initialize default recipients
        self._initialize_default_recipients()
        
    def _initialize_channels(self):
        """Initialize notification channels"""
        # Email channel
        smtp_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'use_tls': True,
            'username': 'alerts@cropdisease.com',
            'password': 'your-password',
            'from_email': 'alerts@cropdisease.com'
        }
        self.notification_channels.append(EmailNotificationChannel(smtp_config))
        
        # Webhook channel
        webhook_urls = ['https://api.cropdisease.com/webhooks/alerts']
        self.notification_channels.append(WebhookNotificationChannel(webhook_urls))
        
        # Dashboard channel
        dashboard_url = 'http://localhost:8000'
        self.notification_channels.append(DashboardNotificationChannel(dashboard_url))
        
        # SMS channel (if enabled)
        if self.config.sms_enabled:
            sms_config = {
                'provider': 'twilio',
                'account_sid': 'your-account-sid',
                'auth_token': 'your-auth-token',
                'from_number': '+1234567890'
            }
            self.notification_channels.append(SMSNotificationChannel(sms_config))
    
    def _initialize_default_recipients(self):
        """Initialize default alert recipients"""
        default_recipients = [
            AlertRecipient(
                user_id='farmer_001',
                name='John Farmer',
                email='john.farmer@example.com',
                phone='+1234567890',
                alert_preferences={
                    'email': True,
                    'sms': True,
                    'dashboard': True
                },
                location=(100.0, 150.0),
                field_id='field_001'
            ),
            AlertRecipient(
                user_id='agronomist_001',
                name='Dr. Sarah Agronomist',
                email='sarah.agronomist@example.com',
                alert_preferences={
                    'email': True,
                    'sms': False,
                    'dashboard': True
                },
                location=None,
                field_id=None
            )
        ]
        
        for recipient in default_recipients:
            self.recipients[recipient.user_id] = recipient
    
    async def create_alert(self, alert_type: str, severity: str, title: str, 
                          message: str, location: Optional[Tuple[float, float]] = None,
                          disease_type: Optional[str] = None, confidence: float = 0.0,
                          recommended_action: str = "", metadata: Dict[str, Any] = None) -> Alert:
        """Create and send a new alert"""
        
        # Check alert cooldown
        if not self._should_send_alert(alert_type, location):
            logging.info(f"Alert {alert_type} skipped due to cooldown")
            return None
        
        # Create alert
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            location=location,
            disease_type=disease_type,
            confidence=confidence,
            recommended_action=recommended_action,
            metadata=metadata or {}
        )
        
        # Add to history
        self.alert_history.append(alert)
        
        # Update cooldown and count
        self._update_alert_tracking(alert_type, location)
        
        # Send notifications
        await self._send_notifications(alert)
        
        return alert
    
    def _should_send_alert(self, alert_type: str, location: Optional[Tuple[float, float]]) -> bool:
        """Check if alert should be sent based on cooldown and rate limits"""
        # Check hourly limit
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        if current_hour not in self.hourly_alert_counts:
            self.hourly_alert_counts[current_hour] = 0
        
        if self.hourly_alert_counts[current_hour] >= self.config.max_alerts_per_hour:
            return False
        
        # Check cooldown
        cooldown_key = f"{alert_type}_{location}" if location else alert_type
        if cooldown_key in self.alert_cooldowns:
            cooldown_time = self.alert_cooldowns[cooldown_key]
            if datetime.now() - cooldown_time < timedelta(minutes=self.config.notification_cooldown_minutes):
                return False
        
        return True
    
    def _update_alert_tracking(self, alert_type: str, location: Optional[Tuple[float, float]]):
        """Update alert tracking for cooldown and rate limiting"""
        # Update hourly count
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        self.hourly_alert_counts[current_hour] = self.hourly_alert_counts.get(current_hour, 0) + 1
        
        # Update cooldown
        cooldown_key = f"{alert_type}_{location}" if location else alert_type
        self.alert_cooldowns[cooldown_key] = datetime.now()
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications through all channels"""
        # Determine recipients based on alert location and type
        recipients = self._get_alert_recipients(alert)
        
        # Send through all channels
        for channel in self.notification_channels:
            try:
                await channel.send_notification(alert, recipients)
            except Exception as e:
                logging.error(f"Notification channel failed: {e}")
    
    def _get_alert_recipients(self, alert: Alert) -> List[AlertRecipient]:
        """Get recipients for an alert based on location and preferences"""
        recipients = []
        
        for recipient in self.recipients.values():
            # Check if recipient should receive this alert
            if self._should_recipient_receive_alert(recipient, alert):
                recipients.append(recipient)
        
        return recipients
    
    def _should_recipient_receive_alert(self, recipient: AlertRecipient, alert: Alert) -> bool:
        """Check if recipient should receive this alert"""
        # Check location-based filtering
        if alert.location and recipient.location:
            # Calculate distance between alert and recipient
            distance = self._calculate_distance(alert.location, recipient.location)
            if distance > 1000:  # 1km radius
                return False
        
        # Check field-based filtering
        if alert.metadata.get('field_id') and recipient.field_id:
            if alert.metadata['field_id'] != recipient.field_id:
                return False
        
        # Check alert type preferences
        alert_type_pref = recipient.alert_preferences.get(alert.alert_type, True)
        if not alert_type_pref:
            return False
        
        return True
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points"""
        import math
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    async def create_intervention_recommendation(self, alert: Alert) -> InterventionRecommendation:
        """Create intervention recommendation based on alert"""
        
        # Generate recommendation based on alert type and severity
        recommendation = self._generate_recommendation(alert)
        
        return recommendation
    
    def _generate_recommendation(self, alert: Alert) -> InterventionRecommendation:
        """Generate intervention recommendation"""
        
        if alert.alert_type == AlertType.DISEASE_DETECTED:
            return InterventionRecommendation(
                alert_id=alert.alert_id,
                intervention_type="immediate_treatment",
                priority="high" if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] else "medium",
                description=f"Immediate treatment required for {alert.disease_type}",
                steps=[
                    "Isolate affected area",
                    "Apply appropriate fungicide/bactericide",
                    "Monitor spread for 24-48 hours",
                    "Adjust environmental conditions if needed"
                ],
                estimated_duration="2-4 hours",
                required_resources=["protective equipment", "treatment chemicals", "spray equipment"],
                cost_estimate=500.0,
                effectiveness_estimate=0.85,
                risk_assessment="Low risk when proper PPE is used"
            )
        
        elif alert.alert_type == AlertType.MUTATION_DETECTED:
            return InterventionRecommendation(
                alert_id=alert.alert_id,
                intervention_type="quarantine_and_research",
                priority="critical",
                description="New disease mutation detected - quarantine required",
                steps=[
                    "Immediate quarantine of affected area",
                    "Collect samples for laboratory analysis",
                    "Notify agricultural authorities",
                    "Implement enhanced monitoring",
                    "Consider alternative treatment strategies"
                ],
                estimated_duration="24-48 hours",
                required_resources=["quarantine materials", "sampling equipment", "laboratory access"],
                cost_estimate=2000.0,
                effectiveness_estimate=0.95,
                risk_assessment="High effectiveness, moderate cost"
            )
        
        elif alert.alert_type == AlertType.PATTERN_DEVIATION:
            return InterventionRecommendation(
                alert_id=alert.alert_id,
                intervention_type="investigation_and_adjustment",
                priority="medium",
                description="Investigate unusual disease pattern",
                steps=[
                    "Analyze environmental conditions",
                    "Review recent treatment history",
                    "Check for new pest introductions",
                    "Adjust monitoring frequency",
                    "Consider preventive measures"
                ],
                estimated_duration="4-8 hours",
                required_resources=["monitoring equipment", "data analysis tools"],
                cost_estimate=300.0,
                effectiveness_estimate=0.70,
                risk_assessment="Low risk investigation"
            )
        
        else:
            return InterventionRecommendation(
                alert_id=alert.alert_id,
                intervention_type="general_monitoring",
                priority="low",
                description="General monitoring and observation",
                steps=[
                    "Continue regular monitoring",
                    "Document observations",
                    "Report any changes",
                    "Maintain current protocols"
                ],
                estimated_duration="1-2 hours",
                required_resources=["monitoring equipment"],
                cost_estimate=50.0,
                effectiveness_estimate=0.60,
                risk_assessment="Very low risk"
            )
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user_id
                alert.acknowledged_at = datetime.now()
                return True
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        if not self.alert_history:
            return {}
        
        total_alerts = len(self.alert_history)
        acknowledged_alerts = sum(1 for a in self.alert_history if a.acknowledged)
        
        severity_counts = {}
        type_counts = {}
        
        for alert in self.alert_history:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'acknowledged_alerts': acknowledged_alerts,
            'acknowledgment_rate': acknowledged_alerts / total_alerts if total_alerts > 0 else 0,
            'severity_distribution': severity_counts,
            'type_distribution': type_counts,
            'recent_alerts': len([a for a in self.alert_history 
                                if a.timestamp > datetime.now() - timedelta(hours=24)])
        }


# Example usage
async def main():
    """Example usage of the alert manager"""
    # Initialize alert configuration
    config = AlertConfig(
        email_enabled=True,
        sms_enabled=False,
        webhook_enabled=True,
        dashboard_enabled=True,
        alert_threshold=0.7,
        notification_cooldown_minutes=30,
        max_alerts_per_hour=10
    )
    
    # Initialize alert manager
    alert_manager = AlertManager(config)
    
    # Create a disease detection alert
    alert = await alert_manager.create_alert(
        alert_type=AlertType.DISEASE_DETECTED,
        severity=AlertSeverity.HIGH,
        title="Fungal Infection Detected",
        message="High-confidence fungal infection detected in section A-3",
        location=(100.0, 150.0),
        disease_type="fungal_infection",
        confidence=0.85,
        recommended_action="Apply fungicide treatment immediately and isolate affected area"
    )
    
    if alert:
        print(f"Alert created: {alert.alert_id}")
        
        # Create intervention recommendation
        recommendation = await alert_manager.create_intervention_recommendation(alert)
        print(f"Intervention recommendation: {recommendation.intervention_type}")
        
        # Get statistics
        stats = alert_manager.get_alert_statistics()
        print(f"Alert statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
