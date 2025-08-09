"""Web Dashboard for IoT Crop Disease Tracking

This module provides a web-based dashboard for visualizing disease tracking data,
alerts, and intervention recommendations using Streamlit.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DashboardConfig:
    """Configuration for the web dashboard"""
    
    def __init__(self, 
                 title: str = "Crop Disease Tracking Dashboard",
                 refresh_interval: int = 60,  # seconds
                 max_alerts: int = 100,
                 map_center: Tuple[float, float] = (39.8283, -98.5795),  # US center
                 map_zoom: int = 4):
        """Initialize dashboard configuration
        
        Args:
            title: Dashboard title
            refresh_interval: Data refresh interval in seconds
            max_alerts: Maximum number of alerts to display
            map_center: Default map center coordinates (latitude, longitude)
            map_zoom: Default map zoom level
        """
        self.title = title
        self.refresh_interval = refresh_interval
        self.max_alerts = max_alerts
        self.map_center = map_center
        self.map_zoom = map_zoom


class DashboardService:
    """Service for managing the web dashboard"""
    
    def __init__(self, config: DashboardConfig = None):
        """Initialize the dashboard service
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.alert_manager = None
        self.graph_engine = None
        self.ai_models = None
        self.storage_manager = None
        
        # Cache for dashboard data
        self.alerts_cache = []
        self.disease_data_cache = []
        self.sensor_data_cache = []
        self.mutation_data_cache = []
        
        # Last update timestamps
        self.last_update = {}
        
        logger.info("Dashboard Service initialized")
    
    def register_alert_manager(self, alert_manager):
        """Register alert manager with the dashboard
        
        Args:
            alert_manager: Alert manager instance
        """
        self.alert_manager = alert_manager
        logger.info("Alert Manager registered with Dashboard")
    
    def register_graph_engine(self, graph_engine):
        """Register graph engine with the dashboard
        
        Args:
            graph_engine: Graph engine instance
        """
        self.graph_engine = graph_engine
        logger.info("Graph Engine registered with Dashboard")
    
    def register_ai_models(self, ai_models):
        """Register AI models with the dashboard
        
        Args:
            ai_models: AI models instance
        """
        self.ai_models = ai_models
        logger.info("AI Models registered with Dashboard")
    
    def register_storage_manager(self, storage_manager):
        """Register storage manager with the dashboard
        
        Args:
            storage_manager: Storage manager instance
        """
        self.storage_manager = storage_manager
        logger.info("Storage Manager registered with Dashboard")
    
    async def update_alerts_cache(self):
        """Update the alerts cache from the alert manager"""
        if not self.alert_manager:
            logger.warning("Alert Manager not registered")
            return
        
        try:
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            
            # Convert to list of dictionaries if not already
            alerts = []
            for alert in active_alerts:
                if hasattr(alert, "model_dump"):
                    alerts.append(alert.model_dump())
                else:
                    alerts.append(dict(alert))
            
            # Sort by timestamp (newest first) and limit to max_alerts
            alerts.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
            self.alerts_cache = alerts[:self.config.max_alerts]
            
            self.last_update["alerts"] = datetime.now()
            logger.info(f"Updated alerts cache with {len(self.alerts_cache)} alerts")
            
        except Exception as e:
            logger.error(f"Failed to update alerts cache: {e}")
    
    async def update_disease_data_cache(self):
        """Update the disease data cache from the graph engine"""
        if not self.graph_engine:
            logger.warning("Graph Engine not registered")
            return
        
        try:
            # In a real implementation, this would get data from the graph engine
            # For now, we'll use placeholder data
            
            # Placeholder for disease tracking data
            self.disease_data_cache = self._generate_demo_disease_data()
            
            self.last_update["disease_data"] = datetime.now()
            logger.info("Updated disease data cache")
            
        except Exception as e:
            logger.error(f"Failed to update disease data cache: {e}")
    
    async def update_sensor_data_cache(self):
        """Update the sensor data cache"""
        try:
            # In a real implementation, this would get data from the IoT sensors or storage
            # For now, we'll use placeholder data
            
            # Placeholder for sensor data
            self.sensor_data_cache = self._generate_demo_sensor_data()
            
            self.last_update["sensor_data"] = datetime.now()
            logger.info("Updated sensor data cache")
            
        except Exception as e:
            logger.error(f"Failed to update sensor data cache: {e}")
    
    async def update_mutation_data_cache(self):
        """Update the mutation data cache from the AI models"""
        if not self.ai_models:
            logger.warning("AI Models not registered")
            return
        
        try:
            # In a real implementation, this would get data from the AI models
            # For now, we'll use placeholder data
            
            # Placeholder for mutation data
            self.mutation_data_cache = self._generate_demo_mutation_data()
            
            self.last_update["mutation_data"] = datetime.now()
            logger.info("Updated mutation data cache")
            
        except Exception as e:
            logger.error(f"Failed to update mutation data cache: {e}")
    
    async def update_all_caches(self):
        """Update all data caches"""
        await asyncio.gather(
            self.update_alerts_cache(),
            self.update_disease_data_cache(),
            self.update_sensor_data_cache(),
            self.update_mutation_data_cache()
        )
        logger.info("All caches updated")
    
    def _generate_demo_disease_data(self) -> List[Dict]:
        """Generate demo disease tracking data
        
        Returns:
            List of disease data points
        """
        # Generate 100 data points over the last 30 days
        now = datetime.now()
        data = []
        
        disease_types = ["Fungal Leaf Spot", "Bacterial Blight", "Powdery Mildew", "Rust"]
        field_ids = [f"field_{i:03d}" for i in range(1, 6)]
        
        for i in range(100):
            # Random timestamp within last 30 days
            days_ago = i % 30
            timestamp = now - timedelta(days=days_ago, 
                                       hours=(i * 7) % 24, 
                                       minutes=(i * 13) % 60)
            
            # Select disease type and field
            disease_type = disease_types[i % len(disease_types)]
            field_id = field_ids[i % len(field_ids)]
            
            # Generate location (US bounds)
            latitude = 37.0 + ((i * 0.7) % 10) - 5
            longitude = -95.0 + ((i * 0.9) % 20) - 10
            
            # Disease metrics
            severity = 0.2 + (i % 8) * 0.1  # 0.2 to 0.9
            spread_rate = 0.05 + (i % 5) * 0.03  # 0.05 to 0.17
            affected_area = 50 + (i % 20) * 25  # 50 to 525 sq meters
            
            # Is this a mutation?
            is_mutation = (i % 10) == 0
            
            data.append({
                "timestamp": timestamp,
                "disease_type": disease_type,
                "field_id": field_id,
                "latitude": latitude,
                "longitude": longitude,
                "severity": severity,
                "spread_rate": spread_rate,
                "affected_area": affected_area,
                "is_mutation": is_mutation,
                "confidence": 0.7 + (i % 3) * 0.1  # 0.7 to 0.9
            })
        
        # Sort by timestamp
        data.sort(key=lambda x: x["timestamp"])
        
        return data
    
    def _generate_demo_sensor_data(self) -> List[Dict]:
        """Generate demo sensor data
        
        Returns:
            List of sensor data points
        """
        # Generate 200 sensor readings over the last 7 days
        now = datetime.now()
        data = []
        
        sensor_types = ["leaf_camera", "environmental"]
        sensor_ids = [f"sensor_{i:03d}" for i in range(1, 21)]
        field_ids = [f"field_{i:03d}" for i in range(1, 6)]
        
        for i in range(200):
            # Random timestamp within last 7 days
            hours_ago = i % 168  # 7 days * 24 hours
            timestamp = now - timedelta(hours=hours_ago, minutes=(i * 7) % 60)
            
            # Select sensor type, ID, and field
            sensor_type = sensor_types[i % len(sensor_types)]
            sensor_id = sensor_ids[i % len(sensor_ids)]
            field_id = field_ids[(i // 4) % len(field_ids)]
            
            # Generate location (US bounds)
            latitude = 37.0 + ((i * 0.5) % 10) - 5
            longitude = -95.0 + ((i * 0.7) % 20) - 10
            
            # Base data for all sensors
            sensor_data = {
                "timestamp": timestamp,
                "sensor_id": sensor_id,
                "sensor_type": sensor_type,
                "field_id": field_id,
                "latitude": latitude,
                "longitude": longitude,
                "battery_level": 0.3 + (i % 7) * 0.1  # 0.3 to 0.9
            }
            
            # Add sensor-specific data
            if sensor_type == "leaf_camera":
                sensor_data.update({
                    "disease_detected": (i % 5) == 0,
                    "disease_type": "Fungal Leaf Spot" if (i % 5) == 0 else None,
                    "disease_confidence": 0.75 + (i % 3) * 0.08 if (i % 5) == 0 else 0.0,
                    "leaf_health_score": 0.5 + (i % 5) * 0.1  # 0.5 to 0.9
                })
            else:  # environmental
                sensor_data.update({
                    "temperature": 15 + (i % 20),  # 15 to 34 °C
                    "humidity": 40 + (i % 50),  # 40 to 89 %
                    "soil_moisture": 0.2 + (i % 8) * 0.1,  # 0.2 to 0.9
                    "light_level": 0.3 + (i % 7) * 0.1  # 0.3 to 0.9
                })
            
            data.append(sensor_data)
        
        # Sort by timestamp
        data.sort(key=lambda x: x["timestamp"])
        
        return data
    
    def _generate_demo_mutation_data(self) -> List[Dict]:
        """Generate demo mutation data
        
        Returns:
            List of mutation data points
        """
        # Generate 20 mutation events over the last 90 days
        now = datetime.now()
        data = []
        
        base_diseases = ["Fungal Leaf Spot", "Bacterial Blight", "Powdery Mildew"]
        mutation_types = ["Resistance", "Virulence", "Host Range", "Environmental Adaptation"]
        field_ids = [f"field_{i:03d}" for i in range(1, 6)]
        
        for i in range(20):
            # Random timestamp within last 90 days
            days_ago = i * 4 + (i % 10)  # Spread out over 90 days
            timestamp = now - timedelta(days=days_ago)
            
            # Select disease and mutation type
            base_disease = base_diseases[i % len(base_diseases)]
            mutation_type = mutation_types[i % len(mutation_types)]
            field_id = field_ids[i % len(field_ids)]
            
            # Generate location (US bounds)
            latitude = 37.0 + ((i * 0.8) % 10) - 5
            longitude = -95.0 + ((i * 1.2) % 20) - 10
            
            # Mutation metrics
            severity = 0.5 + (i % 5) * 0.1  # 0.5 to 0.9
            confidence = 0.7 + (i % 3) * 0.1  # 0.7 to 0.9
            resistance_factor = 1.0 + (i % 5) * 0.5 if mutation_type == "Resistance" else None
            
            data.append({
                "mutation_id": f"mutation_{i:03d}",
                "timestamp": timestamp,
                "base_disease": base_disease,
                "mutation_type": mutation_type,
                "field_id": field_id,
                "latitude": latitude,
                "longitude": longitude,
                "severity": severity,
                "confidence": confidence,
                "resistance_factor": resistance_factor,
                "affected_area": 100 + (i % 10) * 50,  # 100 to 550 sq meters
                "estimated_spread_rate": 0.1 + (i % 5) * 0.05  # 0.1 to 0.3 per day
            })
        
        # Sort by timestamp
        data.sort(key=lambda x: x["timestamp"])
        
        return data


class DashboardUI:
    """Streamlit-based UI for the disease tracking dashboard"""
    
    def __init__(self, dashboard_service: DashboardService):
        """Initialize the dashboard UI
        
        Args:
            dashboard_service: Dashboard service instance
        """
        self.dashboard_service = dashboard_service
        self.config = dashboard_service.config
    
    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(page_title=self.config.title, layout="wide")
        
        # Set up the dashboard layout
        self._setup_layout()
        
        # Schedule data updates
        self._schedule_updates()
    
    def _setup_layout(self):
        """Set up the dashboard layout"""
        st.title(self.config.title)
        
        # Top row: Key metrics and refresh button
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col1:
            st.metric("Active Alerts", len(self.dashboard_service.alerts_cache))
        
        with col2:
            disease_count = sum(1 for d in self.dashboard_service.disease_data_cache 
                              if d.get("is_mutation", False))
            st.metric("Detected Mutations", disease_count)
        
        with col3:
            sensor_count = len(set(s.get("sensor_id") for s in self.dashboard_service.sensor_data_cache))
            st.metric("Active Sensors", sensor_count)
        
        with col4:
            field_count = len(set(d.get("field_id") for d in self.dashboard_service.disease_data_cache))
            st.metric("Monitored Fields", field_count)
        
        with col5:
            if st.button("Refresh Data"):
                st.experimental_rerun()
        
        # Tabs for different dashboard sections
        tab1, tab2, tab3, tab4 = st.tabs(["Disease Map", "Alerts", "Mutations", "Sensors"])
        
        with tab1:
            self._render_disease_map()
        
        with tab2:
            self._render_alerts_section()
        
        with tab3:
            self._render_mutations_section()
        
        with tab4:
            self._render_sensors_section()
    
    def _render_disease_map(self):
        """Render the disease map section"""
        st.header("Disease Spread Map")
        
        # Controls for map filtering
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get unique disease types
            disease_types = list(set(d.get("disease_type") for d in self.dashboard_service.disease_data_cache 
                                if d.get("disease_type")))
            selected_diseases = st.multiselect("Disease Types", disease_types, default=disease_types)
        
        with col2:
            # Get unique field IDs
            field_ids = list(set(d.get("field_id") for d in self.dashboard_service.disease_data_cache 
                              if d.get("field_id")))
            selected_fields = st.multiselect("Fields", field_ids, default=field_ids)
        
        with col3:
            # Date range selector
            all_dates = [d.get("timestamp") for d in self.dashboard_service.disease_data_cache 
                       if d.get("timestamp")]
            min_date = min(all_dates) if all_dates else datetime.now() - timedelta(days=30)
            max_date = max(all_dates) if all_dates else datetime.now()
            
            date_range = st.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
        
        # Filter data based on selections
        filtered_data = []
        for d in self.dashboard_service.disease_data_cache:
            disease_type = d.get("disease_type")
            field_id = d.get("field_id")
            timestamp = d.get("timestamp")
            
            if (disease_type in selected_diseases and 
                field_id in selected_fields and 
                timestamp and 
                len(date_range) == 2 and 
                date_range[0] <= timestamp.date() <= date_range[1]):
                filtered_data.append(d)
        
        # Create map
        if filtered_data:
            # Convert to DataFrame for Plotly
            df = pd.DataFrame(filtered_data)
            
            # Create scatter mapbox
            fig = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                color="disease_type",
                size="affected_area",
                hover_name="disease_type",
                hover_data=["field_id", "severity", "confidence", "is_mutation"],
                zoom=self.config.map_zoom,
                center={"lat": self.config.map_center[0], "lon": self.config.map_center[1]},
                mapbox_style="carto-positron",
                title="Disease Spread Map"
            )
            
            # Highlight mutations with different marker symbol
            fig.update_traces(
                marker=dict(
                    symbol=["circle" if not m else "star" for m in df["is_mutation"]],
                    opacity=0.7
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add legend for mutations
            st.markdown("**Note:** ⭐ Star markers indicate detected mutations")
        else:
            st.info("No disease data available for the selected filters")
        
        # Disease spread over time
        st.subheader("Disease Spread Over Time")
        
        if filtered_data:
            # Group by date and disease type
            df = pd.DataFrame(filtered_data)
            df["date"] = df["timestamp"].dt.date
            
            # Sum affected area by date and disease type
            area_by_date = df.groupby(["date", "disease_type"])["affected_area"].sum().reset_index()
            
            # Create line chart
            fig = px.line(
                area_by_date,
                x="date",
                y="affected_area",
                color="disease_type",
                markers=True,
                title="Affected Area Over Time"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No disease data available for the selected filters")
    
    def _render_alerts_section(self):
        """Render the alerts section"""
        st.header("Active Alerts")
        
        # Controls for alert filtering
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique alert types
            alert_types = list(set(a.get("alert_type") for a in self.dashboard_service.alerts_cache 
                               if a.get("alert_type")))
            selected_alert_types = st.multiselect("Alert Types", alert_types, default=alert_types)
        
        with col2:
            # Get unique severity levels
            severity_levels = list(set(a.get("severity") for a in self.dashboard_service.alerts_cache 
                                  if a.get("severity")))
            selected_severities = st.multiselect("Severity Levels", severity_levels, default=severity_levels)
        
        # Filter alerts based on selections
        filtered_alerts = []
        for a in self.dashboard_service.alerts_cache:
            alert_type = a.get("alert_type")
            severity = a.get("severity")
            
            if alert_type in selected_alert_types and severity in selected_severities:
                filtered_alerts.append(a)
        
        # Display alerts
        if filtered_alerts:
            for i, alert in enumerate(filtered_alerts):
                with st.expander(f"{alert.get('severity', '').upper()}: {alert.get('title', 'Alert')} - {alert.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')}"): 
                    # Alert details
                    st.markdown(f"**Description:** {alert.get('description', '')}")
                    st.markdown(f"**Location:** Field {alert.get('location', {}).get('field_id', 'N/A')}, Section {alert.get('location', {}).get('section_id', 'N/A')}")
                    st.markdown(f"**Coordinates:** {alert.get('location', {}).get('latitude', 'N/A')}, {alert.get('location', {}).get('longitude', 'N/A')}")
                    st.markdown(f"**Status:** {alert.get('status', 'N/A')}")
                    
                    # Recommendations
                    if alert.get('recommendations'):
                        st.subheader("Recommendations")
                        for rec in alert.get('recommendations'):
                            st.markdown(f"**{rec.get('action_type', '').upper()}:** {rec.get('description', '')} (Urgency: {rec.get('urgency', 'N/A')})")
                    
                    # Action buttons (in a real app, these would trigger actions)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.button("Acknowledge", key=f"ack_{i}")
                    with col2:
                        st.button("Resolve", key=f"res_{i}")
                    with col3:
                        st.button("Dismiss", key=f"dis_{i}")
        else:
            st.info("No alerts match the selected filters")
        
        # Alert trends
        st.subheader("Alert Trends")
        
        if self.dashboard_service.alerts_cache:
            # Convert to DataFrame
            df = pd.DataFrame(self.dashboard_service.alerts_cache)
            
            # Add date column
            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                
                # Count alerts by date and type
                alerts_by_date = df.groupby(["date", "alert_type"]).size().reset_index(name="count")
                
                # Create bar chart
                fig = px.bar(
                    alerts_by_date,
                    x="date",
                    y="count",
                    color="alert_type",
                    title="Alerts by Date and Type"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Alert data does not contain timestamp information")
        else:
            st.info("No alert data available")
    
    def _render_mutations_section(self):
        """Render the mutations section"""
        st.header("Disease Mutations")
        
        # Controls for mutation filtering
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique base diseases
            base_diseases = list(set(m.get("base_disease") for m in self.dashboard_service.mutation_data_cache 
                                 if m.get("base_disease")))
            selected_diseases = st.multiselect("Base Diseases", base_diseases, default=base_diseases)
        
        with col2:
            # Get unique mutation types
            mutation_types = list(set(m.get("mutation_type") for m in self.dashboard_service.mutation_data_cache 
                                  if m.get("mutation_type")))
            selected_mutation_types = st.multiselect("Mutation Types", mutation_types, default=mutation_types)
        
        # Filter mutations based on selections
        filtered_mutations = []
        for m in self.dashboard_service.mutation_data_cache:
            base_disease = m.get("base_disease")
            mutation_type = m.get("mutation_type")
            
            if base_disease in selected_diseases and mutation_type in selected_mutation_types:
                filtered_mutations.append(m)
        
        # Display mutations
        if filtered_mutations:
            # Convert to DataFrame for visualization
            df = pd.DataFrame(filtered_mutations)
            
            # Create timeline
            fig = px.timeline(
                df,
                x_start="timestamp",
                x_end="timestamp",  # Same as start for point events
                y="base_disease",
                color="mutation_type",
                hover_name="mutation_id",
                hover_data=["severity", "confidence", "field_id"],
                title="Mutation Timeline"
            )
            
            # Adjust layout
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(xaxis_title="")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mutation details table
            st.subheader("Mutation Details")
            
            # Select columns to display
            display_cols = ["mutation_id", "timestamp", "base_disease", "mutation_type", 
                          "field_id", "severity", "confidence", "affected_area"]
            
            # Format DataFrame for display
            display_df = df[display_cols].copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            display_df["severity"] = display_df["severity"].apply(lambda x: f"{x:.2f}")
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.2f}")
            
            # Rename columns for display
            display_df.columns = [col.replace("_", " ").title() for col in display_df.columns]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Mutation severity distribution
            st.subheader("Mutation Severity Distribution")
            
            # Create histogram
            fig = px.histogram(
                df,
                x="severity",
                color="mutation_type",
                nbins=10,
                title="Distribution of Mutation Severity"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No mutations match the selected filters")
    
    def _render_sensors_section(self):
        """Render the sensors section"""
        st.header("Sensor Network")
        
        # Controls for sensor filtering
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique sensor types
            sensor_types = list(set(s.get("sensor_type") for s in self.dashboard_service.sensor_data_cache 
                                if s.get("sensor_type")))
            selected_sensor_types = st.multiselect("Sensor Types", sensor_types, default=sensor_types)
        
        with col2:
            # Get unique field IDs
            field_ids = list(set(s.get("field_id") for s in self.dashboard_service.sensor_data_cache 
                              if s.get("field_id")))
            selected_fields = st.multiselect("Fields", field_ids, default=field_ids)
        
        # Filter sensors based on selections
        filtered_sensors = []
        for s in self.dashboard_service.sensor_data_cache:
            sensor_type = s.get("sensor_type")
            field_id = s.get("field_id")
            
            if sensor_type in selected_sensor_types and field_id in selected_fields:
                filtered_sensors.append(s)
        
        # Display sensor map
        if filtered_sensors:
            st.subheader("Sensor Locations")
            
            # Convert to DataFrame for Plotly
            df = pd.DataFrame(filtered_sensors)
            
            # Get unique sensors (latest reading for each)
            latest_sensors = df.sort_values("timestamp").groupby("sensor_id").last().reset_index()
            
            # Create scatter mapbox
            fig = px.scatter_mapbox(
                latest_sensors,
                lat="latitude",
                lon="longitude",
                color="sensor_type",
                size="battery_level",
                hover_name="sensor_id",
                hover_data=["field_id", "battery_level"],
                zoom=self.config.map_zoom,
                center={"lat": self.config.map_center[0], "lon": self.config.map_center[1]},
                mapbox_style="carto-positron",
                title="Sensor Network Map"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sensor readings over time
            st.subheader("Sensor Readings Over Time")
            
            # Select a sensor for detailed view
            sensor_ids = list(set(s.get("sensor_id") for s in filtered_sensors))
            selected_sensor = st.selectbox("Select Sensor", sensor_ids)
            
            # Filter data for selected sensor
            sensor_data = [s for s in filtered_sensors if s.get("sensor_id") == selected_sensor]
            
            if sensor_data:
                # Convert to DataFrame
                sensor_df = pd.DataFrame(sensor_data).sort_values("timestamp")
                
                # Determine sensor type and create appropriate chart
                sensor_type = sensor_df["sensor_type"].iloc[0] if not sensor_df.empty else None
                
                if sensor_type == "environmental":
                    # Create multi-line chart for environmental sensors
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add temperature line
                    fig.add_trace(
                        go.Scatter(x=sensor_df["timestamp"], y=sensor_df["temperature"],
                                 mode="lines+markers", name="Temperature (°C)"),
                        secondary_y=False
                    )
                    
                    # Add humidity line
                    fig.add_trace(
                        go.Scatter(x=sensor_df["timestamp"], y=sensor_df["humidity"],
                                 mode="lines+markers", name="Humidity (%)"),
                        secondary_y=False
                    )
                    
                    # Add soil moisture line
                    fig.add_trace(
                        go.Scatter(x=sensor_df["timestamp"], y=sensor_df["soil_moisture"],
                                 mode="lines+markers", name="Soil Moisture"),
                        secondary_y=True
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Environmental Readings - {selected_sensor}",
                        xaxis_title="Time"
                    )
                    fig.update_yaxes(title_text="Temperature (°C) / Humidity (%)", secondary_y=False)
                    fig.update_yaxes(title_text="Soil Moisture", secondary_y=True)
                    
                elif sensor_type == "leaf_camera":
                    # Create chart for leaf camera sensors
                    fig = go.Figure()
                    
                    # Add leaf health score line
                    fig.add_trace(
                        go.Scatter(x=sensor_df["timestamp"], y=sensor_df["leaf_health_score"],
                                 mode="lines+markers", name="Leaf Health Score")
                    )
                    
                    # Add disease confidence if available
                    if "disease_confidence" in sensor_df.columns:
                        fig.add_trace(
                            go.Scatter(x=sensor_df["timestamp"], y=sensor_df["disease_confidence"],
                                     mode="lines+markers", name="Disease Confidence")
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Leaf Health Readings - {selected_sensor}",
                        xaxis_title="Time",
                        yaxis_title="Score (0-1)"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sensor details
                st.subheader("Sensor Details")
                
                # Get latest reading
                latest = sensor_df.iloc[-1] if not sensor_df.empty else None
                
                if latest is not None:
                    # Create two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Sensor ID:** {latest.get('sensor_id')}")
                        st.markdown(f"**Type:** {latest.get('sensor_type')}")
                        st.markdown(f"**Field:** {latest.get('field_id')}")
                        st.markdown(f"**Battery Level:** {latest.get('battery_level', 0):.2f}")
                    
                    with col2:
                        st.markdown(f"**Last Update:** {latest.get('timestamp').strftime('%Y-%m-%d %H:%M')}")
                        st.markdown(f"**Latitude:** {latest.get('latitude')}")
                        st.markdown(f"**Longitude:** {latest.get('longitude')}")
                    
                    # Sensor type specific details
                    if sensor_type == "environmental":
                        st.markdown("### Environmental Readings")
                        st.markdown(f"**Temperature:** {latest.get('temperature')} °C")
                        st.markdown(f"**Humidity:** {latest.get('humidity')} %")
                        st.markdown(f"**Soil Moisture:** {latest.get('soil_moisture'):.2f}")
                        st.markdown(f"**Light Level:** {latest.get('light_level'):.2f}")
                    
                    elif sensor_type == "leaf_camera":
                        st.markdown("### Leaf Health Readings")
                        st.markdown(f"**Leaf Health Score:** {latest.get('leaf_health_score'):.2f}")
                        st.markdown(f"**Disease Detected:** {latest.get('disease_detected', False)}")
                        
                        if latest.get('disease_detected', False):
                            st.markdown(f"**Disease Type:** {latest.get('disease_type', 'Unknown')}")
                            st.markdown(f"**Confidence:** {latest.get('disease_confidence', 0):.2f}")
            else:
                st.info(f"No data available for sensor {selected_sensor}")
        else:
            st.info("No sensors match the selected filters")
    
    def _schedule_updates(self):
        """Schedule periodic data updates"""
        # In a real Streamlit app, you would use st.experimental_rerun() with a timer
        # or implement server-side updates
        pass


# Example usage
def main():
    """Run the dashboard application"""
    # Create dashboard service
    config = DashboardConfig()
    dashboard_service = DashboardService(config)
    
    # Update data caches (in a real app, this would be done asynchronously)
    asyncio.run(dashboard_service.update_all_caches())
    
    # Create and run dashboard UI
    dashboard_ui = DashboardUI(dashboard_service)
    dashboard_ui.run()


if __name__ == "__main__":
    main()