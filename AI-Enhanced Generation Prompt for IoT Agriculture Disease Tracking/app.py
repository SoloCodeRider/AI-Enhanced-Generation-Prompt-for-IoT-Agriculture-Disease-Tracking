"""
Main Web Application for IoT-Enhanced Crop Disease Mutation Tracking System

This module serves as the entry point for the web application, integrating:
- FastAPI web framework
- API routes for data access
- Dashboard for visualization
- WebSocket connections for real-time updates
- Authentication and authorization
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Import our custom modules
from api_routes import dashboard_router, api_router, auth_router, ws_router
from main import CropDiseaseTrackingSystem, SystemConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crop_disease_tracking.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="IoT-Enhanced Crop Disease Tracking System",
    description="Real-time crop disease mutation tracking with AI-powered analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for dashboard
if os.path.exists("dashboard"):
    app.mount("/dashboard", StaticFiles(directory="dashboard"), name="dashboard")

# Include routers
app.include_router(dashboard_router)
app.include_router(api_router)
app.include_router(auth_router)
app.include_router(ws_router)

# Global system instance
system = None
system_task = None

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global system, system_task
    
    # Initialize system configuration
    config = SystemConfig()
    
    # Initialize system
    system = CropDiseaseTrackingSystem(config)
    
    # Start system in background
    system_task = asyncio.create_task(system.start_system())
    logger.info("System started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global system, system_task
    
    if system:
        system.stop_system()
    
    if system_task:
        system_task.cancel()
        try:
            await system_task
        except asyncio.CancelledError:
            pass
    
    logger.info("System shut down successfully")

@app.get("/")
async def root():
    """Root endpoint - redirect to dashboard"""
    return RedirectResponse(url="/dashboard")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not system:
        return {"status": "initializing"}
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "system_status": system.system_status["status"] if system else "unknown"
    }

# Make system instance available to routes
@app.middleware("http")
async def add_system_to_request(request: Request, call_next):
    """Add system instance to request state"""
    request.state.system = system
    response = await call_next(request)
    return response

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # For development
        log_level="info"
    )