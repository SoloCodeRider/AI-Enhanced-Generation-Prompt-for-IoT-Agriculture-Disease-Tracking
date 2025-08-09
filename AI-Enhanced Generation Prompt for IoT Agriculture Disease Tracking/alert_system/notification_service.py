"""Notification Service for IoT Crop Disease Tracking

This module provides notification capabilities for sending alerts through
various channels including email, SMS, mobile push notifications, and web dashboards.
"""

import asyncio
import json
import logging
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Available notification channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    DASHBOARD = "dashboard"
    WEBHOOK = "webhook"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationTemplate(BaseModel):
    """Template for formatting notifications"""
    template_id: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    html_template: Optional[str] = None


class NotificationRecipient(BaseModel):
    """Recipient for notifications"""
    recipient_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    device_tokens: Optional[List[str]] = None  # For push notifications
    webhook_url: Optional[str] = None
    preferred_channels: List[NotificationChannel] = Field(default_factory=list)
    notification_preferences: Dict[str, bool] = Field(default_factory=dict)  # alert_type -> bool


class NotificationMessage(BaseModel):
    """Message to be sent through notification channels"""
    message_id: str = Field(..., description="Unique identifier for the message")
    timestamp: datetime = Field(default_factory=datetime.now)
    subject: str
    body: str
    html_body: Optional[str] = None
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel]
    recipients: List[str]  # recipient_ids
    alert_id: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    status: str = "pending"  # pending, sent, failed
    sent_timestamp: Optional[datetime] = None


class EmailConfig(BaseModel):
    """Configuration for email notifications"""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_email: str
    use_tls: bool = True


class SMSConfig(BaseModel):
    """Configuration for SMS notifications"""
    provider: str  # 'twilio', 'sns', etc.
    account_sid: Optional[str] = None  # For Twilio
    auth_token: Optional[str] = None  # For Twilio
    from_number: Optional[str] = None
    aws_region: Optional[str] = None  # For AWS SNS


class PushConfig(BaseModel):
    """Configuration for push notifications"""
    provider: str  # 'firebase', 'apns', etc.
    api_key: Optional[str] = None  # For Firebase
    app_bundle_id: Optional[str] = None  # For APNS


class WebhookConfig(BaseModel):
    """Configuration for webhook notifications"""
    default_url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    retry_count: int = 3


class NotificationServiceConfig(BaseModel):
    """Configuration for the notification service"""
    email: Optional[EmailConfig] = None
    sms: Optional[SMSConfig] = None
    push: Optional[PushConfig] = None
    webhook: Optional[WebhookConfig] = None
    dashboard_url: Optional[str] = None


class NotificationService:
    """Service for sending notifications through various channels"""
    
    def __init__(self, config: NotificationServiceConfig):
        """Initialize the notification service
        
        Args:
            config: Configuration for notification channels
        """
        self.config = config
        self.templates: Dict[str, NotificationTemplate] = {}
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.message_history: List[NotificationMessage] = []
        self.pending_messages: List[NotificationMessage] = []
        
        # Initialize default templates
        self._initialize_default_templates()
        
        # Performance tracking
        self.send_times: Dict[NotificationChannel, List[float]] = {
            channel: [] for channel in NotificationChannel
        }
        
        logger.info("Notification Service initialized")
    
    def _initialize_default_templates(self):
        """Initialize default notification templates"""
        # Email template for disease alerts
        self.templates["disease_alert_email"] = NotificationTemplate(
            template_id="disease_alert_email",
            channel=NotificationChannel.EMAIL,
            subject_template="[{severity}] Crop Disease Alert: {title}",
            body_template="""Disease Alert: {title}\n\n"
                          "Severity: {severity}\n"
                          "Description: {description}\n\n"
                          "Location: Field {field_id}, Section {section_id}\n"
                          "Coordinates: {latitude}, {longitude}\n\n"
                          "Recommendations:\n{recommendations}\n\n"
                          "View details: {dashboard_url}/alerts/{alert_id}"""",
            html_template="""<h2>Disease Alert: {title}</h2>"
                           "<p><strong>Severity:</strong> <span style='color:{severity_color}'>{severity}</span></p>"
                           "<p><strong>Description:</strong> {description}</p>"
                           "<p><strong>Location:</strong> Field {field_id}, Section {section_id}<br>"
                           "Coordinates: {latitude}, {longitude}</p>"
                           "<h3>Recommendations:</h3>"
                           "<ul>{html_recommendations}</ul>"
                           "<p><a href='{dashboard_url}/alerts/{alert_id}'>View details on dashboard</a></p>""""
        )
        
        # SMS template for disease alerts
        self.templates["disease_alert_sms"] = NotificationTemplate(
            template_id="disease_alert_sms",
            channel=NotificationChannel.SMS,
            subject_template="",  # SMS doesn't use subject
            body_template="""ALERT: {severity} {title} detected in Field {field_id}. "
                          "View details: {dashboard_url}/alerts/{alert_id}""""
        )
        
        # Push notification template for disease alerts
        self.templates["disease_alert_push"] = NotificationTemplate(
            template_id="disease_alert_push",
            channel=NotificationChannel.PUSH,
            subject_template="{severity} Crop Disease Alert",
            body_template="{title} detected in Field {field_id}. Tap to view details."
        )
        
        # Dashboard notification template
        self.templates["disease_alert_dashboard"] = NotificationTemplate(
            template_id="disease_alert_dashboard",
            channel=NotificationChannel.DASHBOARD,
            subject_template="{severity} Alert: {title}",
            body_template="{description}"
        )
        
        # Webhook notification template
        self.templates["disease_alert_webhook"] = NotificationTemplate(
            template_id="disease_alert_webhook",
            channel=NotificationChannel.WEBHOOK,
            subject_template="",  # Webhook doesn't use subject in the same way
            body_template=""  # Webhook sends the full JSON payload
        )
    
    def add_template(self, template: NotificationTemplate):
        """Add or update a notification template
        
        Args:
            template: Template to add or update
        """
        self.templates[template.template_id] = template
        logger.info(f"Added template: {template.template_id}")
    
    def add_recipient(self, recipient: NotificationRecipient):
        """Add or update a notification recipient
        
        Args:
            recipient: Recipient to add or update
        """
        self.recipients[recipient.recipient_id] = recipient
        logger.info(f"Added recipient: {recipient.recipient_id}")
    
    def _format_message(self, template_id: str, alert_data: Dict) -> Dict[str, str]:
        """Format a message using a template and alert data
        
        Args:
            template_id: ID of the template to use
            alert_data: Data to populate the template
            
        Returns:
            Dictionary with formatted subject, body, and html_body
        """
        if template_id not in self.templates:
            logger.error(f"Template not found: {template_id}")
            return {"subject": "", "body": "", "html_body": ""}
        
        template = self.templates[template_id]
        
        # Prepare data for formatting
        format_data = alert_data.copy()
        
        # Add severity color
        severity = alert_data.get("severity", "").lower()
        if severity == "critical":
            format_data["severity_color"] = "#FF0000"  # Red
        elif severity == "high":
            format_data["severity_color"] = "#FF9900"  # Orange
        elif severity == "medium":
            format_data["severity_color"] = "#FFCC00"  # Yellow
        else:
            format_data["severity_color"] = "#00CC00"  # Green
        
        # Format recommendations for plain text
        if "recommendations" in alert_data and isinstance(alert_data["recommendations"], list):
            recommendations_text = ""
            html_recommendations = ""
            
            for i, rec in enumerate(alert_data["recommendations"]):
                if isinstance(rec, dict):
                    action = rec.get("action_type", "")
                    desc = rec.get("description", "")
                    urgency = rec.get("urgency", "")
                    
                    recommendations_text += f"{i+1}. [{action.upper()}] {desc} (Urgency: {urgency})\n"
                    html_recommendations += f"<li><strong>{action.upper()}</strong>: {desc} <em>(Urgency: {urgency})</em></li>"
            
            format_data["recommendations"] = recommendations_text
            format_data["html_recommendations"] = html_recommendations
        
        # Add dashboard URL if not present
        if "dashboard_url" not in format_data and self.config.dashboard_url:
            format_data["dashboard_url"] = self.config.dashboard_url
        
        # Format the message
        try:
            subject = template.subject_template.format(**format_data) if template.subject_template else ""
            body = template.body_template.format(**format_data) if template.body_template else ""
            html_body = template.html_template.format(**format_data) if template.html_template else ""
            
            return {"subject": subject, "body": body, "html_body": html_body}
        except KeyError as e:
            logger.error(f"Missing data for template formatting: {e}")
            return {"subject": "Alert Notification", "body": str(alert_data), "html_body": ""}
    
    async def send_alert_notification(self, alert):
        """Send notification for an alert through appropriate channels
        
        Args:
            alert: Alert object to send notification for
        """
        # Convert alert to dictionary if it's not already
        if hasattr(alert, "model_dump"):
            alert_data = alert.model_dump()
        else:
            alert_data = dict(alert)
        
        # Extract alert properties
        alert_id = alert_data.get("alert_id", "unknown")
        alert_type = alert_data.get("alert_type", "")
        severity = alert_data.get("severity", "")
        
        # Determine notification priority based on alert severity
        priority_map = {
            "critical": NotificationPriority.URGENT,
            "high": NotificationPriority.HIGH,
            "medium": NotificationPriority.NORMAL,
            "low": NotificationPriority.LOW
        }
        priority = priority_map.get(severity.lower(), NotificationPriority.NORMAL)
        
        # Determine channels based on severity
        channels = [NotificationChannel.DASHBOARD]  # Always send to dashboard
        
        if severity.lower() in ["critical", "high"]:
            # High priority alerts go to all channels
            channels.extend([NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PUSH])
        elif severity.lower() == "medium":
            # Medium priority alerts go to email and push
            channels.extend([NotificationChannel.EMAIL, NotificationChannel.PUSH])
        else:
            # Low priority alerts go to dashboard and email
            channels.append(NotificationChannel.EMAIL)
        
        # Add webhook if configured
        if self.config.webhook and self.config.webhook.default_url:
            channels.append(NotificationChannel.WEBHOOK)
        
        # Find recipients who should receive this alert
        recipient_ids = []
        for recipient_id, recipient in self.recipients.items():
            # Check if recipient has preferences for this alert type
            alert_type_key = f"alert_{alert_type.lower()}"
            if alert_type_key in recipient.notification_preferences:
                if recipient.notification_preferences[alert_type_key]:
                    recipient_ids.append(recipient_id)
            else:
                # Default to sending if no specific preference
                recipient_ids.append(recipient_id)
        
        if not recipient_ids:
            logger.warning(f"No recipients found for alert {alert_id}")
            return
        
        # Create message
        message_id = f"msg_{int(time.time())}_{alert_id}"
        
        # Format message for email
        formatted = self._format_message("disease_alert_email", alert_data)
        
        message = NotificationMessage(
            message_id=message_id,
            subject=formatted["subject"],
            body=formatted["body"],
            html_body=formatted["html_body"],
            priority=priority,
            channels=channels,
            recipients=recipient_ids,
            alert_id=alert_id,
            metadata=alert_data
        )
        
        # Send through each channel
        for channel in channels:
            try:
                start_time = time.time()
                
                if channel == NotificationChannel.EMAIL:
                    await self._send_email(message)
                elif channel == NotificationChannel.SMS:
                    await self._send_sms(message)
                elif channel == NotificationChannel.PUSH:
                    await self._send_push(message)
                elif channel == NotificationChannel.DASHBOARD:
                    await self._send_to_dashboard(message)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook(message)
                
                # Record performance
                send_time = time.time() - start_time
                self.send_times[channel].append(send_time)
                
                logger.info(f"Sent {channel} notification for alert {alert_id}")
                
            except Exception as e:
                logger.error(f"Failed to send {channel} notification: {e}")
                # Add to pending messages for retry
                if message not in self.pending_messages:
                    self.pending_messages.append(message)
        
        # Update message status and store in history
        message.status = "sent"
        message.sent_timestamp = datetime.now()
        self.message_history.append(message)
    
    async def _send_email(self, message: NotificationMessage):
        """Send email notification
        
        Args:
            message: Message to send
        """
        if not self.config.email:
            logger.error("Email configuration not provided")
            return
        
        # Get recipient email addresses
        to_emails = []
        for recipient_id in message.recipients:
            if recipient_id in self.recipients and self.recipients[recipient_id].email:
                to_emails.append(self.recipients[recipient_id].email)
        
        if not to_emails:
            logger.warning("No valid email recipients found")
            return
        
        try:
            # Create email message
            email_msg = MIMEMultipart('alternative')
            email_msg['Subject'] = message.subject
            email_msg['From'] = self.config.email.from_email
            email_msg['To'] = ", ".join(to_emails)
            
            # Attach plain text and HTML parts
            email_msg.attach(MIMEText(message.body, 'plain'))
            if message.html_body:
                email_msg.attach(MIMEText(message.html_body, 'html'))
            
            # Connect to SMTP server and send
            with smtplib.SMTP(self.config.email.smtp_server, self.config.email.smtp_port) as server:
                if self.config.email.use_tls:
                    server.starttls()
                
                server.login(self.config.email.username, self.config.email.password)
                server.send_message(email_msg)
            
            logger.info(f"Email sent to {len(to_emails)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise
    
    async def _send_sms(self, message: NotificationMessage):
        """Send SMS notification
        
        Args:
            message: Message to send
        """
        if not self.config.sms:
            logger.error("SMS configuration not provided")
            return
        
        # Get recipient phone numbers
        to_phones = []
        for recipient_id in message.recipients:
            if recipient_id in self.recipients and self.recipients[recipient_id].phone:
                to_phones.append(self.recipients[recipient_id].phone)
        
        if not to_phones:
            logger.warning("No valid SMS recipients found")
            return
        
        # Format SMS message using template
        sms_formatted = self._format_message("disease_alert_sms", message.metadata)
        sms_text = sms_formatted["body"]
        
        try:
            if self.config.sms.provider.lower() == 'twilio':
                # Placeholder for Twilio integration
                logger.info(f"Would send Twilio SMS to {len(to_phones)} recipients: {sms_text}")
                # In a real implementation, you would use the Twilio SDK here
                
            elif self.config.sms.provider.lower() == 'sns':
                # Placeholder for AWS SNS integration
                logger.info(f"Would send AWS SNS SMS to {len(to_phones)} recipients: {sms_text}")
                # In a real implementation, you would use boto3 here
                
            else:
                logger.warning(f"Unsupported SMS provider: {self.config.sms.provider}")
            
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            raise
    
    async def _send_push(self, message: NotificationMessage):
        """Send push notification
        
        Args:
            message: Message to send
        """
        if not self.config.push:
            logger.error("Push notification configuration not provided")
            return
        
        # Get recipient device tokens
        device_tokens = []
        for recipient_id in message.recipients:
            if recipient_id in self.recipients and self.recipients[recipient_id].device_tokens:
                device_tokens.extend(self.recipients[recipient_id].device_tokens)
        
        if not device_tokens:
            logger.warning("No valid push notification recipients found")
            return
        
        # Format push notification using template
        push_formatted = self._format_message("disease_alert_push", message.metadata)
        
        try:
            if self.config.push.provider.lower() == 'firebase':
                # Placeholder for Firebase Cloud Messaging integration
                logger.info(f"Would send Firebase push to {len(device_tokens)} devices")
                # In a real implementation, you would use the Firebase Admin SDK here
                
            elif self.config.push.provider.lower() == 'apns':
                # Placeholder for Apple Push Notification Service integration
                logger.info(f"Would send APNS push to {len(device_tokens)} devices")
                # In a real implementation, you would use a library like aioapns here
                
            else:
                logger.warning(f"Unsupported push provider: {self.config.push.provider}")
            
        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
            raise
    
    async def _send_to_dashboard(self, message: NotificationMessage):
        """Send notification to dashboard
        
        Args:
            message: Message to send
        """
        if not self.config.dashboard_url:
            logger.warning("Dashboard URL not configured")
        
        # Format dashboard notification using template
        dashboard_formatted = self._format_message("disease_alert_dashboard", message.metadata)
        
        # In a real implementation, this might publish to a WebSocket or store in a database
        # that the dashboard UI polls for new notifications
        logger.info(f"Dashboard notification ready: {dashboard_formatted['subject']}")
        
        # Simulate sending to dashboard
        await asyncio.sleep(0.1)  # Simulate processing time
    
    async def _send_webhook(self, message: NotificationMessage):
        """Send webhook notification
        
        Args:
            message: Message to send
        """
        if not self.config.webhook or not self.config.webhook.default_url:
            logger.error("Webhook configuration not provided")
            return
        
        # Prepare webhook payload
        payload = {
            "message_id": message.message_id,
            "timestamp": message.timestamp.isoformat(),
            "alert_id": message.alert_id,
            "alert_data": message.metadata
        }
        
        # Get custom webhook URLs if available, otherwise use default
        webhook_urls = []
        for recipient_id in message.recipients:
            if recipient_id in self.recipients and self.recipients[recipient_id].webhook_url:
                webhook_urls.append(self.recipients[recipient_id].webhook_url)
        
        if not webhook_urls and self.config.webhook.default_url:
            webhook_urls.append(self.config.webhook.default_url)
        
        if not webhook_urls:
            logger.warning("No webhook URLs available")
            return
        
        # In a real implementation, you would use aiohttp or httpx to make async HTTP requests
        logger.info(f"Would send webhook to {len(webhook_urls)} endpoints")
        
        # Simulate webhook sending
        await asyncio.sleep(0.1)  # Simulate processing time
    
    async def process_pending_messages(self):
        """Process pending messages that failed to send"""
        if not self.pending_messages:
            return
        
        logger.info(f"Processing {len(self.pending_messages)} pending messages")
        
        # Make a copy of the list to avoid modification during iteration
        messages_to_process = self.pending_messages.copy()
        self.pending_messages = []
        
        for message in messages_to_process:
            try:
                # Attempt to resend the message
                for channel in message.channels:
                    if channel == NotificationChannel.EMAIL:
                        await self._send_email(message)
                    elif channel == NotificationChannel.SMS:
                        await self._send_sms(message)
                    elif channel == NotificationChannel.PUSH:
                        await self._send_push(message)
                    elif channel == NotificationChannel.DASHBOARD:
                        await self._send_to_dashboard(message)
                    elif channel == NotificationChannel.WEBHOOK:
                        await self._send_webhook(message)
                
                # Update message status
                message.status = "sent"
                message.sent_timestamp = datetime.now()
                
                logger.info(f"Successfully resent message {message.message_id}")
                
            except Exception as e:
                logger.error(f"Failed to resend message {message.message_id}: {e}")
                # Add back to pending messages
                self.pending_messages.append(message)
    
    def get_notification_metrics(self) -> Dict:
        """Get metrics about notification processing
        
        Returns:
            Dictionary of notification metrics
        """
        metrics = {
            "total_messages_sent": len(self.message_history),
            "pending_messages": len(self.pending_messages),
            "channel_performance": {}
        }
        
        # Calculate average send time per channel
        for channel in NotificationChannel:
            times = self.send_times.get(channel, [])
            if times:
                avg_time = sum(times) / len(times)
                metrics["channel_performance"][str(channel)] = {
                    "count": len(times),
                    "avg_send_time_ms": avg_time * 1000
                }
        
        return metrics


# Example usage
async def main():
    """Example usage of the notification service"""
    # Create notification service configuration
    config = NotificationServiceConfig(
        email=EmailConfig(
            smtp_server="smtp.example.com",
            smtp_port=587,
            username="alerts@example.com",
            password="password",
            from_email="alerts@example.com"
        ),
        dashboard_url="https://dashboard.example.com"
    )
    
    # Initialize notification service
    notification_service = NotificationService(config)
    
    # Add a recipient
    recipient = NotificationRecipient(
        recipient_id="user_001",
        name="John Doe",
        email="john@example.com",
        phone="+1234567890",
        preferred_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
        notification_preferences={
            "alert_disease_detected": True,
            "alert_disease_mutation": True
        }
    )
    notification_service.add_recipient(recipient)
    
    # Create a sample alert
    alert_data = {
        "alert_id": "alert_123",
        "alert_type": "disease_mutation",
        "severity": "high",
        "title": "Fungal Leaf Spot Mutation",
        "description": "A mutation of Fungal Leaf Spot has been detected with high confidence.",
        "field_id": "field_001",
        "section_id": "section_A3",
        "latitude": 34.052235,
        "longitude": -118.243683,
        "recommendations": [
            {
                "action_type": "inspection",
                "description": "Conduct detailed field inspection",
                "urgency": "high"
            },
            {
                "action_type": "treatment",
                "description": "Apply broad-spectrum fungicide",
                "urgency": "high"
            }
        ]
    }
    
    # Send notification
    await notification_service.send_alert_notification(alert_data)
    
    # Get metrics
    metrics = notification_service.get_notification_metrics()
    print(f"Notification metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())