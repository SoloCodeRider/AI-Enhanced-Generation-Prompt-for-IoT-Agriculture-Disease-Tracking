"""Integration Module for IoT Crop Disease Tracking System

This module connects all components of the system together, including:
- IoT Sensor Network
- Graph Engine
- AI Models
- Cloud Storage
- Alert System
- Dashboard
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Import components from each module
from iot_sensors.sensor_simulator import IoTSensorSimulator
from graph_engine.graph_model import DiseaseTrackingModel, create_graph_from_sensor_data
from graph_engine.visualization import DiseaseGraphVisualizer, RealTimeVisualizer
from ai_models.disease_detection import DiseaseDetectionModel
from ai_models.pattern_analysis import PatternAnalysisService, AnomalyDetector, MutationDetector
from cloud_storage.storage_manager import CloudStorageManager, StorageConfig
from alert_system.alert_manager import AlertManager
from alert_system.notification_service import NotificationService
from alert_system.dashboard import DashboardService, DashboardConfig, DashboardUI

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemConfig:
    """Configuration for the integrated system"""
    
    def __init__(self,
                 simulation_duration: int = 3600,  # seconds
                 update_interval: int = 5,  # seconds
                 num_sensors: int = 50,
                 field_dimensions: Tuple[int, int] = (1000, 1000),  # meters
                 cloud_storage_enabled: bool = True,
                 cloud_providers: List[str] = None,
                 alert_thresholds: Dict[str, float] = None,
                 dashboard_enabled: bool = True):
        """Initialize system configuration
        
        Args:
            simulation_duration: Duration of the simulation in seconds
            update_interval: Interval between sensor updates in seconds
            num_sensors: Number of sensors in the network
            field_dimensions: Dimensions of the field in meters (width, height)
            cloud_storage_enabled: Whether cloud storage is enabled
            cloud_providers: List of cloud providers to use (aws, azure, gcp)
            alert_thresholds: Thresholds for triggering alerts
            dashboard_enabled: Whether the dashboard is enabled
        """
        self.simulation_duration = simulation_duration
        self.update_interval = update_interval
        self.num_sensors = num_sensors
        self.field_dimensions = field_dimensions
        self.cloud_storage_enabled = cloud_storage_enabled
        self.cloud_providers = cloud_providers or ["aws", "azure", "gcp"]
        self.alert_thresholds = alert_thresholds or {
            "disease_confidence": 0.7,
            "mutation_confidence": 0.8,
            "environmental_anomaly": 0.75
        }
        self.dashboard_enabled = dashboard_enabled


class IntegratedSystem:
    """Integrated system for IoT crop disease tracking"""
    
    def __init__(self, config: SystemConfig = None):
        """Initialize the integrated system
        
        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()
        
        # Initialize components
        self.sensor_simulator = None
        self.disease_tracking_model = None
        self.graph_visualizer = None
        self.real_time_visualizer = None
        self.disease_detection_model = None
        self.pattern_analysis_service = None
        self.storage_manager = None
        self.alert_manager = None
        self.notification_service = None
        self.dashboard_service = None
        
        # Data buffers
        self.sensor_data_buffer = []
        self.graph_data_buffer = []
        self.mutation_data_buffer = []
        
        # Status flags
        self.is_running = False
        self.is_initialized = False
        
        logger.info("Integrated system created")
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing integrated system...")
        
        # Initialize IoT sensor simulator
        self.sensor_simulator = IoTSensorSimulator(
            num_sensors=self.config.num_sensors,
            field_dimensions=self.config.field_dimensions,
            update_interval=self.config.update_interval
        )
        
        # Initialize graph engine components
        self.disease_tracking_model = DiseaseTrackingModel()
        self.graph_visualizer = DiseaseGraphVisualizer()
        self.real_time_visualizer = RealTimeVisualizer()
        
        # Initialize AI models
        self.disease_detection_model = DiseaseDetectionModel(model_type="custom_resnet")
        
        # Initialize pattern analysis service
        self.pattern_analysis_service = PatternAnalysisService(
            anomaly_detector=AnomalyDetector(),
            mutation_detector=MutationDetector()
        )
        
        # Initialize cloud storage if enabled
        if self.config.cloud_storage_enabled:
            storage_config = StorageConfig(
                aws_enabled="aws" in self.config.cloud_providers,
                azure_enabled="azure" in self.config.cloud_providers,
                gcp_enabled="gcp" in self.config.cloud_providers
            )
            self.storage_manager = CloudStorageManager(config=storage_config)
            await self.storage_manager.initialize()
        
        # Initialize alert system
        self.alert_manager = AlertManager(thresholds=self.config.alert_thresholds)
        self.notification_service = NotificationService()
        
        # Register notification service with alert manager
        self.alert_manager.register_notification_service(self.notification_service)
        
        # Initialize dashboard if enabled
        if self.config.dashboard_enabled:
            dashboard_config = DashboardConfig()
            self.dashboard_service = DashboardService(config=dashboard_config)
            
            # Register components with dashboard
            self.dashboard_service.register_alert_manager(self.alert_manager)
            self.dashboard_service.register_graph_engine(self.disease_tracking_model)
            if self.config.cloud_storage_enabled:
                self.dashboard_service.register_storage_manager(self.storage_manager)
        
        self.is_initialized = True
        logger.info("Integrated system initialized successfully")
    
    async def start(self):
        """Start the integrated system"""
        if not self.is_initialized:
            await self.initialize()
        
        logger.info("Starting integrated system...")
        self.is_running = True
        
        # Start main processing loop
        try:
            await self._run_main_loop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.is_running = False
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the integrated system"""
        logger.info("Stopping integrated system...")
        self.is_running = False
        
        # Stop sensor simulator
        if self.sensor_simulator:
            await self.sensor_simulator.stop()
        
        # Close cloud storage connections
        if self.storage_manager:
            await self.storage_manager.close()
        
        logger.info("Integrated system stopped")
    
    async def _run_main_loop(self):
        """Run the main processing loop"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=self.config.simulation_duration)
        
        logger.info(f"Starting main loop, will run until {end_time}")
        
        # Start sensor simulator in background
        sensor_task = asyncio.create_task(
            self.sensor_simulator.simulate_sensor_network(
                duration=self.config.simulation_duration
            )
        )
        
        # Main processing loop
        while self.is_running and datetime.now() < end_time:
            try:
                # Process sensor data
                await self._process_sensor_data()
                
                # Process graph data
                await self._process_graph_data()
                
                # Process AI analysis
                await self._process_ai_analysis()
                
                # Process alerts
                await self._process_alerts()
                
                # Update dashboard
                if self.dashboard_service:
                    await self.dashboard_service.update_all_caches()
                
                # Wait for next update interval
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
        
        # Wait for sensor simulator to complete
        await sensor_task
        
        logger.info("Main loop completed")
    
    async def _process_sensor_data(self):
        """Process sensor data from the simulator"""
        # Get latest sensor data
        new_data = self.sensor_simulator.get_latest_data()
        
        if not new_data:
            return
        
        # Add to buffer
        self.sensor_data_buffer.extend(new_data)
        
        # Store in cloud storage if enabled
        if self.storage_manager:
            for data_point in new_data:
                await self.storage_manager.store_iot_data(data_point)
        
        logger.debug(f"Processed {len(new_data)} new sensor data points")
    
    async def _process_graph_data(self):
        """Process graph data from sensor readings"""
        if not self.sensor_data_buffer:
            return
        
        # Create graph from sensor data
        graph_data = create_graph_from_sensor_data(self.sensor_data_buffer)
        
        # Update disease tracking model
        self.disease_tracking_model.update(graph_data)
        
        # Get latest graph state
        graph_state = self.disease_tracking_model.get_current_state()
        
        # Add to buffer
        self.graph_data_buffer.append(graph_state)
        
        # Store in cloud storage if enabled
        if self.storage_manager:
            await self.storage_manager.store_graph_data(graph_state)
        
        # Update visualization
        self.real_time_visualizer.update(graph_state)
        
        # Clear sensor data buffer after processing
        self.sensor_data_buffer = []
        
        logger.debug("Processed graph data")
    
    async def _process_ai_analysis(self):
        """Process AI analysis on graph data"""
        if not self.graph_data_buffer:
            return
        
        # Get latest graph state
        latest_graph = self.graph_data_buffer[-1]
        
        # Run pattern analysis
        analysis_results = self.pattern_analysis_service.analyze(latest_graph)
        
        # Check for mutations
        mutations = analysis_results.get("mutations", [])
        
        if mutations:
            # Add to buffer
            self.mutation_data_buffer.extend(mutations)
            
            # Store in cloud storage if enabled
            if self.storage_manager:
                for mutation in mutations:
                    await self.storage_manager.store_mutation_log(mutation)
            
            logger.info(f"Detected {len(mutations)} new mutations")
        
        # Clear graph data buffer after processing
        self.graph_data_buffer = []
        
        logger.debug("Processed AI analysis")
    
    async def _process_alerts(self):
        """Process alerts based on mutations and anomalies"""
        if not self.mutation_data_buffer:
            return
        
        # Process each mutation for alerts
        for mutation in self.mutation_data_buffer:
            # Create alert for mutation
            alert = self.alert_manager.create_disease_mutation_alert(
                mutation_data=mutation,
                confidence=mutation.get("confidence", 0.0)
            )
            
            # Process the alert
            await self.alert_manager.process_alert(alert)
        
        # Clear mutation buffer after processing
        self.mutation_data_buffer = []
        
        logger.debug("Processed alerts")


async def run_system(config: SystemConfig = None):
    """Run the integrated system
    
    Args:
        config: System configuration
    """
    # Create and initialize system
    system = IntegratedSystem(config)
    
    try:
        # Start the system
        await system.start()
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        # Ensure system is stopped
        await system.stop()


def run_dashboard():
    """Run the dashboard application"""
    from alert_system.dashboard import main as dashboard_main
    dashboard_main()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IoT Crop Disease Tracking System")
    parser.add_argument("--dashboard-only", action="store_true", help="Run only the dashboard")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration in seconds")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")
    parser.add_argument("--sensors", type=int, default=50, help="Number of sensors")
    parser.add_argument("--no-cloud", action="store_true", help="Disable cloud storage")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    
    args = parser.parse_args()
    
    if args.dashboard_only:
        run_dashboard()
    else:
        # Create system configuration
        config = SystemConfig(
            simulation_duration=args.duration,
            update_interval=args.interval,
            num_sensors=args.sensors,
            cloud_storage_enabled=not args.no_cloud,
            dashboard_enabled=not args.no_dashboard
        )
        
        # Run the system
        asyncio.run(run_system(config))