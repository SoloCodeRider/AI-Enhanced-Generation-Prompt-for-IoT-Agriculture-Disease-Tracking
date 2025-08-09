"""
Main Application for IoT-Enhanced Crop Disease Mutation Tracking System

Integrates all components:
- IoT sensor network simulation
- Real-time graph processing
- AI-powered disease detection
- Cloud storage management
- Alert and notification system
- Web dashboard
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import threading
import os

# Import our custom modules
from iot_sensors.sensor_simulator import IoTSensorSimulator
from graph_engine.spatial_temporal_graph import RealTimeGraphProcessor, GraphVisualizer
from ai_models.disease_detection_models import AIAnalysisEngine
from cloud_storage.storage_manager import CloudStorageManager, StorageConfig
from alert_system.notification_manager import AlertManager, AlertConfig, AlertType, AlertSeverity


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


class SystemConfig(BaseModel):
    """System configuration"""
    field_size: Tuple[float, float] = (500.0, 300.0)
    num_sensors: int = 25
    simulation_duration_hours: float = 24.0
    cloud_storage_provider: str = "aws"
    alert_threshold: float = 0.7
    enable_real_time_processing: bool = True
    enable_ai_analysis: bool = True
    enable_cloud_storage: bool = True
    enable_alerts: bool = True


class CropDiseaseTrackingSystem:
    """Main system class that integrates all components"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.is_running = False
        self.start_time = None
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_metrics = {
            'total_sensor_readings': 0,
            'total_disease_detections': 0,
            'total_mutations_detected': 0,
            'total_alerts_sent': 0,
            'system_uptime': 0,
            'processing_latency': []
        }
        
        # Real-time data storage
        self.current_graph_data = {}
        self.recent_alerts = []
        self.system_status = {
            'status': 'initialized',
            'last_update': datetime.now(),
            'active_sensors': 0,
            'disease_centers': 0,
            'mutation_events': 0
        }
    
    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing Crop Disease Tracking System...")
        
        try:
            # Initialize IoT sensor simulator
            self.iot_simulator = IoTSensorSimulator(
                field_size=self.config.field_size,
                num_sensors=self.config.num_sensors,
                mqtt_broker="localhost",
                http_endpoint="http://localhost:8000/api/sensor-data"
            )
            logger.info("IoT sensor simulator initialized")
            
            # Initialize graph processor
            self.graph_processor = RealTimeGraphProcessor(self.config.field_size)
            self.graph_visualizer = GraphVisualizer()
            logger.info("Graph processor initialized")
            
            # Initialize AI analysis engine
            self.ai_engine = AIAnalysisEngine(device='cpu')
            logger.info("AI analysis engine initialized")
            
            # Initialize cloud storage (if enabled)
            if self.config.enable_cloud_storage:
                storage_config = StorageConfig(
                    provider=self.config.cloud_storage_provider,
                    bucket_name='crop-disease-tracking',
                    region='us-east-1',
                    access_key='your-access-key',
                    secret_key='your-secret-key'
                )
                self.storage_manager = CloudStorageManager(storage_config)
                logger.info("Cloud storage manager initialized")
            else:
                self.storage_manager = None
            
            # Initialize alert system (if enabled)
            if self.config.enable_alerts:
                alert_config = AlertConfig(
                    email_enabled=True,
                    sms_enabled=False,
                    webhook_enabled=True,
                    dashboard_enabled=True,
                    alert_threshold=self.config.alert_threshold
                )
                self.alert_manager = AlertManager(alert_config)
                logger.info("Alert manager initialized")
            else:
                self.alert_manager = None
            
            logger.info("All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system components: {e}")
            raise
    
    async def start_system(self):
        """Start the complete system"""
        logger.info("Starting Crop Disease Tracking System...")
        
        self.is_running = True
        self.start_time = datetime.now()
        self.system_status['status'] = 'running'
        
        # Start background tasks
        tasks = []
        
        # Start IoT sensor simulation
        if self.config.enable_real_time_processing:
            tasks.append(self._run_iot_simulation())
        
        # Start real-time processing
        if self.config.enable_real_time_processing:
            tasks.append(self._run_real_time_processing())
        
        # Start AI analysis
        if self.config.enable_ai_analysis:
            tasks.append(self._run_ai_analysis())
        
        # Start cloud storage management
        if self.config.enable_cloud_storage and self.storage_manager:
            tasks.append(self._run_cloud_storage_management())
        
        # Start alert monitoring
        if self.config.enable_alerts and self.alert_manager:
            tasks.append(self._run_alert_monitoring())
        
        # Start system monitoring
        tasks.append(self._run_system_monitoring())
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
    
    async def _run_iot_simulation(self):
        """Run IoT sensor simulation"""
        logger.info("Starting IoT sensor simulation...")
        
        try:
            # Run simulation for specified duration
            await self.iot_simulator.simulate_sensor_network(
                duration_hours=self.config.simulation_duration_hours
            )
        except Exception as e:
            logger.error(f"IoT simulation error: {e}")
    
    async def _run_real_time_processing(self):
        """Run real-time data processing"""
        logger.info("Starting real-time data processing...")
        
        while self.is_running:
            try:
                # Simulate incoming sensor data
                sensor_data = self._generate_simulated_sensor_data()
                
                # Process sensor data through graph engine
                start_time = time.time()
                result = await self.graph_processor.process_sensor_data(sensor_data)
                processing_time = time.time() - start_time
                
                # Update performance metrics
                self.performance_metrics['processing_latency'].append(processing_time)
                self.performance_metrics['total_sensor_readings'] += 1
                
                # Update current graph data
                self.current_graph_data = result['graph_data']
                
                # Update system status
                self.system_status['last_update'] = datetime.now()
                self.system_status['active_sensors'] = len([
                    node for node in result['graph_data']['nodes'] 
                    if node['health_status'] != 'healthy'
                ])
                self.system_status['disease_centers'] = len(result['graph_data']['disease_centers'])
                self.system_status['mutation_events'] = len(result['graph_data']['mutation_events'])
                
                # Store in cloud storage
                if self.config.enable_cloud_storage and self.storage_manager:
                    await self.storage_manager.store_iot_data(sensor_data)
                    await self.storage_manager.store_graph_data(result['graph_data'])
                
                # Wait before next processing cycle
                await asyncio.sleep(5)  # 5-second intervals
                
            except Exception as e:
                logger.error(f"Real-time processing error: {e}")
                await asyncio.sleep(10)
    
    async def _run_ai_analysis(self):
        """Run AI analysis on sensor data"""
        logger.info("Starting AI analysis...")
        
        while self.is_running:
            try:
                # Simulate image analysis
                image_path = "dummy_image.jpg"  # In real system, this would be actual image
                location = (100.0, 150.0)
                
                # Run AI analysis
                prediction = await self.ai_engine.analyze_image(image_path, location)
                
                if prediction and prediction.disease_detected:
                    self.performance_metrics['total_disease_detections'] += 1
                    
                    # Create alert if confidence is high enough
                    if prediction.confidence >= self.config.alert_threshold:
                        await self._create_disease_alert(prediction)
                
                # Run pattern deviation detection
                sensor_sequence = self._generate_sensor_sequence()
                deviations = await self.ai_engine.detect_pattern_deviations(sensor_sequence)
                
                for deviation in deviations:
                    await self._create_pattern_deviation_alert(deviation)
                
                # Run mutation tracking
                disease_data = self._generate_disease_data()
                mutations = await self.ai_engine.track_mutations(disease_data)
                
                for mutation in mutations:
                    self.performance_metrics['total_mutations_detected'] += 1
                    await self._create_mutation_alert(mutation)
                
                # Wait before next analysis cycle
                await asyncio.sleep(30)  # 30-second intervals
                
            except Exception as e:
                logger.error(f"AI analysis error: {e}")
                await asyncio.sleep(60)
    
    async def _run_cloud_storage_management(self):
        """Run cloud storage management tasks"""
        logger.info("Starting cloud storage management...")
        
        while self.is_running:
            try:
                # Clean up old data
                await self.storage_manager.cleanup_old_data(retention_days=90)
                
                # Wait before next cleanup cycle
                await asyncio.sleep(3600)  # 1-hour intervals
                
            except Exception as e:
                logger.error(f"Cloud storage management error: {e}")
                await asyncio.sleep(3600)
    
    async def _run_alert_monitoring(self):
        """Run alert monitoring and management"""
        logger.info("Starting alert monitoring...")
        
        while self.is_running:
            try:
                # Check for new alerts and send notifications
                # This would typically be triggered by events rather than polling
                await asyncio.sleep(60)  # 1-minute intervals
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _run_system_monitoring(self):
        """Run system monitoring and health checks"""
        logger.info("Starting system monitoring...")
        
        while self.is_running:
            try:
                # Update system uptime
                if self.start_time:
                    self.performance_metrics['system_uptime'] = (
                        datetime.now() - self.start_time
                    ).total_seconds()
                
                # Log system status
                logger.info(f"System Status: {self.system_status}")
                
                # Wait before next monitoring cycle
                await asyncio.sleep(300)  # 5-minute intervals
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(300)
    
    def _generate_simulated_sensor_data(self) -> Dict:
        """Generate simulated sensor data"""
        return {
            'sensor_id': f'sensor_{int(time.time()) % 1000:03d}',
            'sensor_type': 'leaf_camera',
            'location': {
                'latitude': 100.0 + (time.time() % 100),
                'longitude': 150.0 + (time.time() % 100)
            },
            'disease_detected': time.time() % 10 < 3,  # 30% chance of disease
            'disease_confidence': (time.time() % 100) / 100,
            'disease_type': 'fungal_infection' if time.time() % 10 < 3 else None,
            'leaf_health_score': 0.7 + (time.time() % 30) / 100,
            'temperature': 20.0 + (time.time() % 20),
            'humidity': 60.0 + (time.time() % 20),
            'soil_moisture': 50.0 + (time.time() % 30),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_sensor_sequence(self) -> List[Dict]:
        """Generate sensor data sequence for pattern analysis"""
        sequence = []
        for i in range(20):
            sequence.append(self._generate_simulated_sensor_data())
        return sequence
    
    def _generate_disease_data(self) -> List[Dict]:
        """Generate disease data for mutation tracking"""
        data = []
        for i in range(10):
            data.append({
                'disease_confidence': (time.time() + i) % 100 / 100,
                'severity': (time.time() + i) % 100 / 100,
                'spread_rate': (time.time() + i) % 50 / 100,
                'resistance_level': (time.time() + i) % 30 / 100,
                'environmental_stress': (time.time() + i) % 20 / 100,
                'treatment_effectiveness': (time.time() + i) % 80 / 100
            })
        return data
    
    async def _create_disease_alert(self, prediction):
        """Create disease detection alert"""
        if not self.alert_manager:
            return
        
        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.DISEASE_DETECTED,
            severity=AlertSeverity.HIGH if prediction.confidence > 0.8 else AlertSeverity.MEDIUM,
            title=f"Disease Detected: {prediction.disease_type}",
            message=f"High-confidence disease detection: {prediction.disease_type} with {prediction.confidence:.1%} confidence",
            location=prediction.location,
            disease_type=prediction.disease_type,
            confidence=prediction.confidence,
            recommended_action="Apply appropriate treatment immediately and monitor spread"
        )
        
        if alert:
            self.performance_metrics['total_alerts_sent'] += 1
            self.recent_alerts.append(alert)
    
    async def _create_pattern_deviation_alert(self, deviation):
        """Create pattern deviation alert"""
        if not self.alert_manager:
            return
        
        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.PATTERN_DEVIATION,
            severity=AlertSeverity.MEDIUM,
            title=f"Pattern Deviation: {deviation.deviation_type}",
            message=deviation.description,
            location=None,
            confidence=deviation.confidence,
            recommended_action=deviation.recommended_action
        )
        
        if alert:
            self.performance_metrics['total_alerts_sent'] += 1
            self.recent_alerts.append(alert)
    
    async def _create_mutation_alert(self, mutation):
        """Create mutation detection alert"""
        if not self.alert_manager:
            return
        
        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.MUTATION_DETECTED,
            severity=AlertSeverity.CRITICAL,
            title=f"Disease Mutation: {mutation['mutation_type']}",
            message=mutation['description'],
            location=None,
            confidence=mutation['confidence'],
            recommended_action="Immediate quarantine and research required"
        )
        
        if alert:
            self.performance_metrics['total_alerts_sent'] += 1
            self.recent_alerts.append(alert)
    
    def stop_system(self):
        """Stop the system"""
        logger.info("Stopping Crop Disease Tracking System...")
        self.is_running = False
        self.system_status['status'] = 'stopped'
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'system_status': self.system_status,
            'performance_metrics': self.performance_metrics,
            'recent_alerts': [alert.dict() for alert in self.recent_alerts[-10:]],  # Last 10 alerts
            'current_graph_data': self.current_graph_data
        }


# FastAPI application
app = FastAPI(
    title="IoT-Enhanced Crop Disease Tracking System",
    description="Real-time crop disease mutation tracking with AI-powered analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for dashboard
if os.path.exists("dashboard"):
    app.mount("/dashboard", StaticFiles(directory="dashboard"), name="dashboard")

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


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "IoT-Enhanced Crop Disease Tracking System",
        "version": "1.0.0",
        "status": "running",
        "dashboard_url": "/dashboard/index.html"
    }


@app.get("/dashboard")
async def dashboard():
    """Serve dashboard"""
    return FileResponse("dashboard/index.html")


@app.get("/api/system-status")
async def get_system_status():
    """Get system status"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return system.get_system_status()


@app.get("/api/graph-data")
async def get_graph_data():
    """Get current graph data"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return system.current_graph_data


@app.get("/api/alerts")
async def get_alerts():
    """Get recent alerts"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "alerts": [alert.dict() for alert in system.recent_alerts[-20:]],  # Last 20 alerts
        "statistics": system.alert_manager.get_alert_statistics() if system.alert_manager else {}
    }


@app.post("/api/sensor-data")
async def receive_sensor_data(data: Dict):
    """Receive sensor data from IoT devices"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Process sensor data
        # This would integrate with the real-time processing pipeline
        logger.info(f"Received sensor data: {data}")
        
        return {"status": "received", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")
        raise HTTPException(status_code=500, detail="Error processing sensor data")


@app.websocket("/ws/graph-updates")
async def websocket_graph_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time graph updates"""
    await websocket.accept()
    
    try:
        while True:
            if system and system.current_graph_data:
                # Send current graph data
                await websocket.send_json(system.current_graph_data)
            
            # Wait before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts"""
    await websocket.accept()
    
    try:
        while True:
            if system and system.recent_alerts:
                # Send recent alerts
                alerts = [alert.dict() for alert in system.recent_alerts[-5:]]  # Last 5 alerts
                await websocket.send_json({"alerts": alerts})
            
            # Wait before next update
            await asyncio.sleep(10)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
