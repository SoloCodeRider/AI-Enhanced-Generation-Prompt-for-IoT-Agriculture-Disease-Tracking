"""
IoT Sensor Simulator for Crop Disease Tracking

Simulates distributed IoT sensors including:
- Leaf imaging cameras with disease detection
- Environmental sensors (temperature, humidity, soil moisture)
- GPS-enabled geo-tagging
- Real-time data transmission via MQTT/HTTP
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid
import math

import paho.mqtt.client as mqtt
import requests
from pydantic import BaseModel, Field
import numpy as np
from dataclasses import dataclass


@dataclass
class SensorLocation:
    """GPS coordinates for sensor placement"""
    latitude: float
    longitude: float
    altitude: float = 0.0


class SensorData(BaseModel):
    """Base sensor data model"""
    sensor_id: str
    timestamp: datetime
    location: SensorLocation
    sensor_type: str
    data: Dict


class LeafImageData(SensorData):
    """Leaf imaging camera data with disease detection"""
    sensor_type: str = "leaf_camera"
    image_path: str = ""
    disease_detected: bool = False
    disease_confidence: float = 0.0
    disease_type: Optional[str] = None
    leaf_health_score: float = Field(ge=0.0, le=1.0)
    image_quality: float = Field(ge=0.0, le=1.0)


class EnvironmentalData(SensorData):
    """Environmental sensor data"""
    sensor_type: str = "environmental"
    temperature: float = Field(ge=-50.0, le=60.0)
    humidity: float = Field(ge=0.0, le=100.0)
    soil_moisture: float = Field(ge=0.0, le=100.0)
    light_intensity: float = Field(ge=0.0, le=100000.0)
    wind_speed: float = Field(ge=0.0, le=50.0)
    wind_direction: float = Field(ge=0.0, le=360.0)


class DiseasePattern:
    """Simulates disease spread patterns across the field"""
    
    def __init__(self, field_size: Tuple[float, float], initial_infection_rate: float = 0.05):
        self.field_size = field_size
        self.infection_rate = initial_infection_rate
        self.disease_centers = []
        self.spread_rate = 0.1  # meters per hour
        self.mutation_probability = 0.01
        
    def update_disease_spread(self, elapsed_hours: float):
        """Update disease spread across the field"""
        # Simulate disease mutation and spread
        for center in self.disease_centers:
            center['radius'] += self.spread_rate * elapsed_hours
            
        # Random mutation events
        if random.random() < self.mutation_probability * elapsed_hours:
            self._create_mutation_event()
            
    def _create_mutation_event(self):
        """Create a new disease mutation event"""
        new_center = {
            'lat': random.uniform(0, self.field_size[0]),
            'lng': random.uniform(0, self.field_size[1]),
            'radius': 0.0,
            'mutation_type': random.choice(['resistant', 'aggressive', 'latent']),
            'severity': random.uniform(0.1, 1.0)
        }
        self.disease_centers.append(new_center)


class IoTSensorSimulator:
    """Main IoT sensor simulator for crop disease tracking"""
    
    def __init__(self, 
                 field_size: Tuple[float, float] = (1000.0, 1000.0),
                 num_sensors: int = 50,
                 mqtt_broker: str = "localhost",
                 mqtt_port: int = 1883,
                 http_endpoint: str = "http://localhost:8000/api/sensor-data"):
        
        self.field_size = field_size
        self.num_sensors = num_sensors
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.http_endpoint = http_endpoint
        
        # Initialize disease pattern simulation
        self.disease_pattern = DiseasePattern(field_size)
        
        # Sensor network configuration
        self.sensors = self._initialize_sensors()
        
        # MQTT client setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_publish = self._on_mqtt_publish
        
        # Performance tracking
        self.data_transmission_count = 0
        self.start_time = time.time()
        
    def _initialize_sensors(self) -> List[Dict]:
        """Initialize distributed sensor network"""
        sensors = []
        
        for i in range(self.num_sensors):
            sensor_type = "leaf_camera" if i % 3 == 0 else "environmental"
            
            sensor = {
                'id': f"sensor_{uuid.uuid4().hex[:8]}",
                'type': sensor_type,
                'location': SensorLocation(
                    latitude=random.uniform(0, self.field_size[0]),
                    longitude=random.uniform(0, self.field_size[1]),
                    altitude=random.uniform(0, 2.0)
                ),
                'last_update': datetime.now(),
                'health_status': 'healthy',
                'battery_level': random.uniform(0.3, 1.0)
            }
            sensors.append(sensor)
            
        return sensors
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        print(f"Connected to MQTT broker with result code {rc}")
        client.subscribe("crop-disease/sensor-data")
        
    def _on_mqtt_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        self.data_transmission_count += 1
        
    def _generate_leaf_image_data(self, sensor: Dict) -> LeafImageData:
        """Generate realistic leaf imaging data with disease detection"""
        location = sensor['location']
        
        # Check if sensor is in disease-affected area
        disease_detected = False
        disease_confidence = 0.0
        disease_type = None
        leaf_health_score = random.uniform(0.7, 1.0)
        
        for center in self.disease_pattern.disease_centers:
            distance = math.sqrt(
                (location.latitude - center['lat'])**2 + 
                (location.longitude - center['lng'])**2
            )
            
            if distance <= center['radius']:
                disease_detected = True
                disease_confidence = max(0.1, 1.0 - distance / center['radius'])
                disease_type = center['mutation_type']
                leaf_health_score = max(0.1, leaf_health_score - disease_confidence * 0.8)
                break
        
        return LeafImageData(
            sensor_id=sensor['id'],
            timestamp=datetime.now(),
            location=location,
            disease_detected=disease_detected,
            disease_confidence=disease_confidence,
            disease_type=disease_type,
            leaf_health_score=leaf_health_score,
            image_quality=random.uniform(0.8, 1.0),
            data={
                'battery_level': sensor['battery_level'],
                'health_status': sensor['health_status']
            }
        )
    
    def _generate_environmental_data(self, sensor: Dict) -> EnvironmentalData:
        """Generate environmental sensor data"""
        location = sensor['location']
        
        # Simulate realistic environmental conditions
        base_temp = 20.0 + 10.0 * math.sin(time.time() / 86400)  # Daily cycle
        temp_variation = random.uniform(-2.0, 2.0)
        
        humidity = 60.0 + 20.0 * math.sin(time.time() / 43200)  # 12-hour cycle
        humidity += random.uniform(-5.0, 5.0)
        
        soil_moisture = max(0.0, min(100.0, 
            50.0 + 30.0 * math.sin(time.time() / 86400) + random.uniform(-10.0, 10.0)
        ))
        
        return EnvironmentalData(
            sensor_id=sensor['id'],
            timestamp=datetime.now(),
            location=location,
            temperature=base_temp + temp_variation,
            humidity=max(0.0, min(100.0, humidity)),
            soil_moisture=soil_moisture,
            light_intensity=random.uniform(0, 100000),
            wind_speed=random.uniform(0, 10),
            wind_direction=random.uniform(0, 360),
            data={
                'battery_level': sensor['battery_level'],
                'health_status': sensor['health_status']
            }
        )
    
    async def _transmit_data_mqtt(self, data: SensorData):
        """Transmit sensor data via MQTT"""
        try:
            topic = f"crop-disease/{data.sensor_type}/{data.sensor_id}"
            payload = data.model_dump_json()
            self.mqtt_client.publish(topic, payload, qos=1)
        except Exception as e:
            print(f"MQTT transmission error: {e}")
    
    async def _transmit_data_http(self, data: SensorData):
        """Transmit sensor data via HTTP"""
        try:
            payload = data.model_dump()
            response = requests.post(
                self.http_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            if response.status_code != 200:
                print(f"HTTP transmission error: {response.status_code}")
        except Exception as e:
            print(f"HTTP transmission error: {e}")
    
    async def simulate_sensor_network(self, duration_hours: float = 24.0):
        """Run the complete sensor network simulation"""
        print(f"Starting IoT sensor simulation for {duration_hours} hours...")
        print(f"Field size: {self.field_size[0]}x{self.field_size[1]} meters")
        print(f"Number of sensors: {self.num_sensors}")
        
        # Connect to MQTT broker
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"MQTT connection failed: {e}")
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        while time.time() < end_time:
            # Update disease pattern
            elapsed_hours = (time.time() - start_time) / 3600
            self.disease_pattern.update_disease_spread(elapsed_hours)
            
            # Generate and transmit data from each sensor
            for sensor in self.sensors:
                # Update sensor battery and health
                sensor['battery_level'] = max(0.0, sensor['battery_level'] - 0.001)
                if sensor['battery_level'] < 0.1:
                    sensor['health_status'] = 'low_battery'
                
                # Generate sensor data
                if sensor['type'] == 'leaf_camera':
                    data = self._generate_leaf_image_data(sensor)
                else:
                    data = self._generate_environmental_data(sensor)
                
                # Transmit data
                await self._transmit_data_mqtt(data)
                await self._transmit_data_http(data)
                
                # Update sensor timestamp
                sensor['last_update'] = datetime.now()
            
            # Print status every 10 minutes
            if int(time.time() - start_time) % 600 == 0:
                self._print_status()
            
            # Wait before next cycle (simulate 5-minute intervals)
            await asyncio.sleep(300)  # 5 minutes
        
        # Cleanup
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        
        print(f"\nSimulation completed!")
        print(f"Total data transmissions: {self.data_transmission_count}")
        print(f"Average transmission rate: {self.data_transmission_count / duration_hours:.2f} per hour")
    
    def _print_status(self):
        """Print current simulation status"""
        elapsed = time.time() - self.start_time
        healthy_sensors = sum(1 for s in self.sensors if s['health_status'] == 'healthy')
        disease_centers = len(self.disease_pattern.disease_centers)
        
        print(f"\n--- Simulation Status ---")
        print(f"Elapsed time: {elapsed/3600:.2f} hours")
        print(f"Healthy sensors: {healthy_sensors}/{self.num_sensors}")
        print(f"Active disease centers: {disease_centers}")
        print(f"Data transmissions: {self.data_transmission_count}")
        print(f"Infection rate: {self.disease_pattern.infection_rate:.3f}")


async def main():
    """Main function to run the IoT sensor simulation"""
    # Initialize simulator with realistic field parameters
    simulator = IoTSensorSimulator(
        field_size=(500.0, 300.0),  # 500m x 300m field
        num_sensors=25,  # 25 sensors distributed across the field
        mqtt_broker="localhost",
        http_endpoint="http://localhost:8000/api/sensor-data"
    )
    
    # Run simulation for 2 hours (for demonstration)
    await simulator.simulate_sensor_network(duration_hours=2.0)


if __name__ == "__main__":
    asyncio.run(main())
