"""Test suite for the notification service component of the alert system.

Tests the functionality of the NotificationService class, including:
- Recipient management
- Template management
- Message formatting
- Notification sending through various channels
- Message queue processing
- Metrics collection
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from alert_system.notification_service import (
    NotificationService,
    NotificationChannel,
    NotificationPriority,
    NotificationTemplate,
    NotificationRecipient,
    NotificationMessage,
    EmailConfig,
    SMSConfig,
    PushConfig,
    WebhookConfig
)


@pytest.fixture
def notification_service():
    """Create a notification service instance for testing"""
    service = NotificationService(
        email_config=EmailConfig(
            smtp_server="smtp.example.com",
            smtp_port=587,
            username="test@example.com",
            password="password",
            from_email="alerts@example.com"
        ),
        sms_config=SMSConfig(
            api_key="sms-api-key",
            from_number="+12345678901"
        ),
        push_config=PushConfig(
            api_key="push-api-key",
            app_id="app-id"
        ),
        webhook_config=WebhookConfig(
            endpoints=["https://webhook.example.com/alerts"]
        )
    )
    
    # Add test templates
    service.add_template(
        template_id="disease_alert",
        subject="Disease Alert: {disease_type}",
        body="A {severity} severity {disease_type} has been detected in {location}. Confidence: {confidence}.",
        channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PUSH]
    )
    
    service.add_template(
        template_id="environmental_alert",
        subject="Environmental Alert: {alert_type}",
        body="An environmental anomaly ({alert_type}) has been detected in {location}. Value: {value}, Threshold: {threshold}.",
        channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
    )
    
    # Add test recipients
    service.add_recipient(
        name="Farmer John",
        email="john@example.com",
        phone="+12345678901",
        device_tokens=["device-token-1"],
        channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PUSH]
    )
    
    service.add_recipient(
        name="Field Manager Alice",
        email="alice@example.com",
        phone="+19876543210",
        channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
    )
    
    service.add_recipient(
        name="Research Team",
        email="research@example.com",
        channels=[NotificationChannel.EMAIL]
    )
    
    return service


class TestNotificationService:
    """Test notification service functionality"""
    
    def test_initialization(self, notification_service):
        """Test service initialization"""
        assert notification_service.email_config is not None
        assert notification_service.sms_config is not None
        assert notification_service.push_config is not None
        assert notification_service.webhook_config is not None
        assert len(notification_service.templates) == 2
        assert len(notification_service.recipients) == 3
        assert len(notification_service.message_queue) == 0
    
    def test_template_management(self, notification_service):
        """Test template management"""
        # Add a new template
        notification_service.add_template(
            template_id="mutation_alert",
            subject="Mutation Alert: {mutation_type}",
            body="A mutation of type {mutation_type} has been detected with confidence {confidence}.",
            channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD]
        )
        
        assert len(notification_service.templates) == 3
        assert "mutation_alert" in notification_service.templates
        
        # Update an existing template
        notification_service.add_template(
            template_id="disease_alert",
            subject="Updated: Disease Alert for {disease_type}",
            body="Updated alert for {disease_type} in {location}.",
            channels=[NotificationChannel.EMAIL]
        )
        
        assert notification_service.templates["disease_alert"].subject == "Updated: Disease Alert for {disease_type}"
        assert notification_service.templates["disease_alert"].channels == [NotificationChannel.EMAIL]
    
    def test_recipient_management(self, notification_service):
        """Test recipient management"""
        # Add a new recipient
        notification_service.add_recipient(
            name="New Farmer",
            email="new@example.com",
            phone="+15551234567",
            channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
        )
        
        assert len(notification_service.recipients) == 4
        assert "New Farmer" in notification_service.recipients
        
        # Update an existing recipient
        notification_service.add_recipient(
            name="Farmer John",
            email="john_updated@example.com",
            phone="+12345678901",
            channels=[NotificationChannel.EMAIL]
        )
        
        assert notification_service.recipients["Farmer John"].email == "john_updated@example.com"
        assert notification_service.recipients["Farmer John"].channels == [NotificationChannel.EMAIL]
        
        # Remove a recipient
        notification_service.remove_recipient("Research Team")
        assert "Research Team" not in notification_service.recipients
        assert len(notification_service.recipients) == 3
    
    def test_format_message(self, notification_service):
        """Test message formatting"""
        # Format a message using a template
        message = notification_service.format_message(
            template_id="disease_alert",
            context={
                "disease_type": "Fungal Leaf Spot",
                "severity": "high",
                "location": "Field A, Section 3",
                "confidence": 0.85
            }
        )
        
        assert message.subject == "Disease Alert: Fungal Leaf Spot"
        assert "high severity Fungal Leaf Spot" in message.body
        assert "Field A, Section 3" in message.body
        assert "Confidence: 0.85" in message.body
        
        # Test with missing template
        with pytest.raises(ValueError):
            notification_service.format_message(
                template_id="non_existent_template",
                context={}
            )
        
        # Test with missing context variables
        with pytest.raises(KeyError):
            notification_service.format_message(
                template_id="disease_alert",
                context={"disease_type": "Fungal Leaf Spot"}
            )
    
    @pytest.mark.asyncio
    async def test_send_notification_email(self, notification_service):
        """Test sending notifications via email"""
        with patch.object(notification_service, "_send_email", return_value=True) as mock_send_email:
            result = await notification_service.send_notification(
                title="Test Disease Alert",
                message="A disease has been detected in Field A.",
                priority=NotificationPriority.HIGH,
                recipients=["Farmer John", "Field Manager Alice"],
                channels=[NotificationChannel.EMAIL]
            )
            
            assert result is True
            assert mock_send_email.call_count == 2
            
            # Check message queue
            assert len(notification_service.message_queue) == 0  # All sent successfully
    
    @pytest.mark.asyncio
    async def test_send_notification_sms(self, notification_service):
        """Test sending notifications via SMS"""
        with patch.object(notification_service, "_send_sms", return_value=True) as mock_send_sms:
            result = await notification_service.send_notification(
                title="Test Alert",
                message="Environmental anomaly detected.",
                priority=NotificationPriority.MEDIUM,
                recipients=["Farmer John"],
                channels=[NotificationChannel.SMS]
            )
            
            assert result is True
            assert mock_send_sms.call_count == 1
    
    @pytest.mark.asyncio
    async def test_send_notification_push(self, notification_service):
        """Test sending notifications via push"""
        with patch.object(notification_service, "_send_push", return_value=True) as mock_send_push:
            result = await notification_service.send_notification(
                title="Urgent Alert",
                message="Critical disease detected.",
                priority=NotificationPriority.CRITICAL,
                recipients=["Farmer John"],  # Only Farmer John has device tokens
                channels=[NotificationChannel.PUSH]
            )
            
            assert result is True
            assert mock_send_push.call_count == 1
    
    @pytest.mark.asyncio
    async def test_send_notification_dashboard(self, notification_service):
        """Test sending notifications to dashboard"""
        with patch.object(notification_service, "_send_to_dashboard", return_value=True) as mock_send_dashboard:
            result = await notification_service.send_notification(
                title="Dashboard Alert",
                message="New data available.",
                priority=NotificationPriority.LOW,
                recipients=["Farmer John", "Field Manager Alice"],
                channels=[NotificationChannel.DASHBOARD]
            )
            
            assert result is True
            assert mock_send_dashboard.call_count == 1  # Only called once for all recipients
    
    @pytest.mark.asyncio
    async def test_send_notification_webhook(self, notification_service):
        """Test sending notifications via webhook"""
        with patch.object(notification_service, "_send_webhook", return_value=True) as mock_send_webhook:
            result = await notification_service.send_notification(
                title="Webhook Alert",
                message="Data for external systems.",
                priority=NotificationPriority.MEDIUM,
                recipients=["Farmer John"],  # Recipients don't matter for webhooks
                channels=[NotificationChannel.WEBHOOK]
            )
            
            assert result is True
            assert mock_send_webhook.call_count == 1
    
    @pytest.mark.asyncio
    async def test_send_notification_multiple_channels(self, notification_service):
        """Test sending notifications via multiple channels"""
        with patch.object(notification_service, "_send_email", return_value=True) as mock_send_email:
            with patch.object(notification_service, "_send_sms", return_value=True) as mock_send_sms:
                with patch.object(notification_service, "_send_push", return_value=True) as mock_send_push:
                    result = await notification_service.send_notification(
                        title="Multi-channel Alert",
                        message="Important alert on multiple channels.",
                        priority=NotificationPriority.HIGH,
                        recipients=["Farmer John"],
                        channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PUSH]
                    )
                    
                    assert result is True
                    assert mock_send_email.call_count == 1
                    assert mock_send_sms.call_count == 1
                    assert mock_send_push.call_count == 1
    
    @pytest.mark.asyncio
    async def test_send_notification_failure(self, notification_service):
        """Test handling of notification sending failures"""
        with patch.object(notification_service, "_send_email", side_effect=Exception("SMTP error")):
            result = await notification_service.send_notification(
                title="Failed Alert",
                message="This alert will fail to send.",
                priority=NotificationPriority.HIGH,
                recipients=["Farmer John"],
                channels=[NotificationChannel.EMAIL]
            )
            
            assert result is False
            assert len(notification_service.message_queue) == 1  # Message added to queue for retry
    
    @pytest.mark.asyncio
    async def test_process_message_queue(self, notification_service):
        """Test processing of message queue"""
        # Add a message to the queue
        message = NotificationMessage(
            id="test-message-id",
            title="Queued Alert",
            body="This is a queued alert.",
            priority=NotificationPriority.HIGH,
            recipient=notification_service.recipients["Farmer John"],
            channel=NotificationChannel.EMAIL,
            created_at=datetime.now(),
            status="pending",
            retry_count=0
        )
        notification_service.message_queue.append(message)
        
        # Process the queue with successful sending
        with patch.object(notification_service, "_send_email", return_value=True) as mock_send_email:
            await notification_service.process_message_queue()
            
            assert mock_send_email.call_count == 1
            assert len(notification_service.message_queue) == 0  # Queue should be empty
        
        # Test with failed sending and retry
        message.retry_count = 0
        notification_service.message_queue.append(message)
        
        with patch.object(notification_service, "_send_email", side_effect=Exception("SMTP error")):
            await notification_service.process_message_queue()
            
            assert len(notification_service.message_queue) == 1  # Message still in queue
            assert notification_service.message_queue[0].retry_count == 1  # Retry count incremented
        
        # Test max retries
        notification_service.message_queue[0].retry_count = notification_service.max_retries
        
        with patch.object(notification_service, "_send_email", side_effect=Exception("SMTP error")):
            await notification_service.process_message_queue()
            
            assert len(notification_service.message_queue) == 0  # Message removed after max retries
    
    def test_get_notification_metrics(self, notification_service):
        """Test notification metrics collection"""
        # Add some sent messages to history
        notification_service.notification_history = [
            NotificationMessage(
                id="msg1",
                title="Alert 1",
                body="Message 1",
                priority=NotificationPriority.HIGH,
                recipient=notification_service.recipients["Farmer John"],
                channel=NotificationChannel.EMAIL,
                created_at=datetime.now() - timedelta(hours=2),
                sent_at=datetime.now() - timedelta(hours=2),
                status="sent"
            ),
            NotificationMessage(
                id="msg2",
                title="Alert 2",
                body="Message 2",
                priority=NotificationPriority.MEDIUM,
                recipient=notification_service.recipients["Field Manager Alice"],
                channel=NotificationChannel.SMS,
                created_at=datetime.now() - timedelta(hours=1),
                sent_at=datetime.now() - timedelta(hours=1),
                status="sent"
            ),
            NotificationMessage(
                id="msg3",
                title="Alert 3",
                body="Message 3",
                priority=NotificationPriority.CRITICAL,
                recipient=notification_service.recipients["Farmer John"],
                channel=NotificationChannel.PUSH,
                created_at=datetime.now() - timedelta(minutes=30),
                status="failed"
            )
        ]
        
        metrics = notification_service.get_notification_metrics()
        
        assert metrics["total_sent"] == 2
        assert metrics["total_failed"] == 1
        assert metrics["by_channel"][NotificationChannel.EMAIL.value] == 1
        assert metrics["by_channel"][NotificationChannel.SMS.value] == 1
        assert metrics["by_priority"][NotificationPriority.HIGH.value] == 1
        assert metrics["by_priority"][NotificationPriority.MEDIUM.value] == 1
        assert metrics["by_recipient"]["Farmer John"] == 2  # 1 sent, 1 failed
        assert metrics["by_recipient"]["Field Manager Alice"] == 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])