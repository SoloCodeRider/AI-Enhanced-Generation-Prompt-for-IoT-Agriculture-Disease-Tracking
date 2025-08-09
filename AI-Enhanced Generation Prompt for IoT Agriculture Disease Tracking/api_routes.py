"""
API Routes for IoT-Enhanced Crop Disease Mutation Tracking System

This module defines all the API endpoints for the system, including:
- Dashboard routes
- Data API endpoints
- WebSocket endpoints for real-time updates
- Authentication endpoints
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create API routers
dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])
api_router = APIRouter(prefix="/api", tags=["api"])
auth_router = APIRouter(prefix="/auth", tags=["auth"])
ws_router = APIRouter(tags=["websocket"])

# Simple in-memory user database (replace with proper auth in production)
users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
        "disabled": False,
        "role": "admin"
    },
    "user": {
        "username": "user",
        "full_name": "Regular User",
        "email": "user@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
        "disabled": False,
        "role": "user"
    }
}

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str = "user"

class UserInDB(User):
    hashed_password: str

class SensorData(BaseModel):
    sensor_id: str
    sensor_type: str
    timestamp: datetime
    location: Dict[str, float]
    data: Dict[str, Any]

class Alert(BaseModel):
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str
    title: str
    message: str
    location: Optional[Dict[str, float]] = None
    disease_type: Optional[str] = None
    confidence: Optional[float] = None
    recommended_action: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SystemConfig(BaseModel):
    field_size: List[float] = Field(default=[500.0, 300.0], description="Field size in meters [width, height]")
    num_sensors: int = Field(default=25, description="Number of sensors in the field")
    simulation_duration_hours: float = Field(default=24.0, description="Simulation duration in hours")
    alert_threshold: float = Field(default=0.7, description="Alert threshold (0-1)")
    enable_real_time_processing: bool = Field(default=True, description="Enable real-time processing")
    enable_ai_analysis: bool = Field(default=True, description="Enable AI analysis")
    enable_cloud_storage: bool = Field(default=True, description="Enable cloud storage")
    enable_alerts: bool = Field(default=True, description="Enable alerts")

class StorageConfig(BaseModel):
    provider: str = Field(default="local", description="Storage provider (aws, azure, gcp, local)")
    bucket_name: str = Field(default="crop-disease-tracking", description="Bucket name")
    region: str = Field(default="us-east-1", description="Region")
    access_key: str = Field(default="", description="Access key")
    secret_key: str = Field(default="", description="Secret key")

class SMTPConfig(BaseModel):
    smtp_server: str = Field(default="smtp.gmail.com", description="SMTP server")
    smtp_port: int = Field(default=587, description="SMTP port")
    username: str = Field(default="", description="SMTP username")
    password: str = Field(default="", description="SMTP password")

class AlertConfig(BaseModel):
    email_enabled: bool = Field(default=True, description="Enable email alerts")
    sms_enabled: bool = Field(default=False, description="Enable SMS alerts")
    webhook_enabled: bool = Field(default=True, description="Enable webhook alerts")
    dashboard_enabled: bool = Field(default=True, description="Enable dashboard alerts")
    smtp_config: SMTPConfig = Field(default_factory=SMTPConfig, description="SMTP configuration")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Authentication functions
def verify_password(plain_password, hashed_password):
    # In a real app, use proper password hashing (e.g., bcrypt)
    return plain_password == "password"  # Simplified for demo

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # In a real app, validate JWT token
    username = token  # Simplified for demo
    user = get_user(users_db, username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Auth routes
@auth_router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # In a real app, generate JWT token
    access_token = user.username  # Simplified for demo
    return {"access_token": access_token, "token_type": "bearer"}

# Dashboard routes
@dashboard_router.get("/")
async def get_dashboard():
    """Serve the main dashboard page"""
    return FileResponse("dashboard/index.html")

@dashboard_router.get("/login")
async def get_login_page():
    """Serve the login page"""
    return FileResponse("dashboard/login.html")

@dashboard_router.get("/settings")
async def get_settings_page(current_user: User = Depends(get_current_active_user)):
    """Serve the settings page (requires authentication)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    return FileResponse("dashboard/settings.html")

# API routes
@api_router.get("/system-status")
async def get_system_status(current_user: User = Depends(get_current_active_user)):
    """Get system status"""
    # In a real app, get this from the system instance
    return {
        "status": "running",
        "uptime": "12:34:56",
        "active_sensors": 25,
        "disease_centers": 3,
        "mutation_events": 1,
        "last_update": datetime.now().isoformat()
    }

@api_router.get("/graph-data")
async def get_graph_data(current_user: User = Depends(get_current_active_user)):
    """Get current graph data"""
    # In a real app, get this from the system instance
    return {
        "nodes": [
            {"id": "node1", "x": 100, "y": 150, "health_status": "healthy", "disease_confidence": 0.0},
            {"id": "node2", "x": 200, "y": 250, "health_status": "infected", "disease_confidence": 0.8},
            {"id": "node3", "x": 300, "y": 150, "health_status": "mutation", "disease_confidence": 0.9}
        ],
        "edges": [
            {"source": "node1", "target": "node2"},
            {"source": "node2", "target": "node3"}
        ],
        "disease_centers": [{"x": 200, "y": 250, "radius": 50, "disease_type": "fungal_infection"}],
        "mutation_events": [{"x": 300, "y": 150, "timestamp": datetime.now().isoformat(), "mutation_type": "resistance"}]
    }

@api_router.get("/alerts")
async def get_alerts(current_user: User = Depends(get_current_active_user)):
    """Get recent alerts"""
    # In a real app, get this from the system instance
    return {
        "alerts": [
            {
                "alert_id": "alert1",
                "timestamp": datetime.now().isoformat(),
                "alert_type": "DISEASE_DETECTED",
                "severity": "HIGH",
                "title": "Disease Detected: Fungal Infection",
                "message": "High-confidence disease detection: Fungal Infection with 85% confidence",
                "location": {"latitude": 100.0, "longitude": 150.0},
                "disease_type": "fungal_infection",
                "confidence": 0.85,
                "recommended_action": "Apply appropriate treatment immediately and monitor spread"
            }
        ],
        "statistics": {
            "total_alerts": 1,
            "by_severity": {"HIGH": 1, "MEDIUM": 0, "LOW": 0},
            "by_type": {"DISEASE_DETECTED": 1, "PATTERN_DEVIATION": 0, "MUTATION_DETECTED": 0}
        }
    }

@api_router.post("/sensor-data")
async def receive_sensor_data(data: SensorData):
    """Receive sensor data from IoT devices"""
    # In a real app, process this data through the system
    logger.info(f"Received sensor data: {data}")
    return {"status": "received", "timestamp": datetime.now().isoformat()}

@api_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, current_user: User = Depends(get_current_active_user)):
    """Acknowledge an alert"""
    # In a real app, update the alert status in the system
    logger.info(f"Alert {alert_id} acknowledged by {current_user.username}")
    return {"status": "acknowledged", "alert_id": alert_id, "timestamp": datetime.now().isoformat()}

@api_router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, current_user: User = Depends(get_current_active_user)):
    """Resolve an alert"""
    # In a real app, update the alert status in the system
    logger.info(f"Alert {alert_id} resolved by {current_user.username}")
    return {"status": "resolved", "alert_id": alert_id, "timestamp": datetime.now().isoformat()}

@api_router.post("/system-config")
async def update_system_config(config: SystemConfig, current_user: User = Depends(get_current_active_user)):
    """Update system configuration"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # In a real app, update the system configuration
    logger.info(f"System configuration updated by {current_user.username}: {config}")
    return {"status": "updated", "timestamp": datetime.now().isoformat()}

@api_router.get("/system-config")
async def get_system_config(current_user: User = Depends(get_current_active_user)):
    """Get system configuration"""
    # In a real app, get this from the system instance
    return SystemConfig()

@api_router.post("/storage-config")
async def update_storage_config(config: StorageConfig, current_user: User = Depends(get_current_active_user)):
    """Update storage configuration"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # In a real app, update the storage configuration
    logger.info(f"Storage configuration updated by {current_user.username}: {config}")
    return {"status": "updated", "timestamp": datetime.now().isoformat()}

@api_router.get("/storage-config")
async def get_storage_config(current_user: User = Depends(get_current_active_user)):
    """Get storage configuration"""
    # In a real app, get this from the system instance
    return StorageConfig()

@api_router.post("/alert-config")
async def update_alert_config(config: AlertConfig, current_user: User = Depends(get_current_active_user)):
    """Update alert configuration"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # In a real app, update the alert configuration
    logger.info(f"Alert configuration updated by {current_user.username}: {config}")
    return {"status": "updated", "timestamp": datetime.now().isoformat()}

@api_router.get("/alert-config")
async def get_alert_config(current_user: User = Depends(get_current_active_user)):
    """Get alert configuration"""
    # In a real app, get this from the system instance
    return AlertConfig()

# WebSocket routes
@ws_router.websocket("/ws/graph-updates")
async def websocket_graph_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time graph updates"""
    await websocket.accept()
    
    try:
        while True:
            # In a real app, get this from the system instance
            graph_data = {
                "nodes": [
                    {"id": "node1", "x": 100, "y": 150, "health_status": "healthy", "disease_confidence": 0.0},
                    {"id": "node2", "x": 200, "y": 250, "health_status": "infected", "disease_confidence": 0.8},
                    {"id": "node3", "x": 300, "y": 150, "health_status": "mutation", "disease_confidence": 0.9}
                ],
                "edges": [
                    {"source": "node1", "target": "node2"},
                    {"source": "node2", "target": "node3"}
                ],
                "disease_centers": [{"x": 200, "y": 250, "radius": 50, "disease_type": "fungal_infection"}],
                "mutation_events": [{"x": 300, "y": 150, "timestamp": datetime.now().isoformat(), "mutation_type": "resistance"}]
            }
            
            await websocket.send_json(graph_data)
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@ws_router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts"""
    await websocket.accept()
    
    try:
        while True:
            # In a real app, get this from the system instance
            alerts = [
                {
                    "alert_id": "alert1",
                    "timestamp": datetime.now().isoformat(),
                    "alert_type": "DISEASE_DETECTED",
                    "severity": "HIGH",
                    "title": "Disease Detected: Fungal Infection",
                    "message": "High-confidence disease detection: Fungal Infection with 85% confidence",
                    "location": {"latitude": 100.0, "longitude": 150.0},
                    "disease_type": "fungal_infection",
                    "confidence": 0.85,
                    "recommended_action": "Apply appropriate treatment immediately and monitor spread"
                }
            ]
            
            await websocket.send_json({"alerts": alerts})
            await asyncio.sleep(10)  # Update every 10 seconds
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")