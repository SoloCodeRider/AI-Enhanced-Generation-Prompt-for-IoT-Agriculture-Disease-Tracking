"""Graph Visualization Module

This module provides real-time visualization of crop disease spread and mutation
patterns using interactive dashboards and charts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import logging
from datetime import datetime, timedelta
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiseaseGraphVisualizer:
    """Visualizer for crop disease spread and mutation patterns"""
    
    def __init__(self, field_size: Tuple[float, float] = (1000.0, 1000.0),
                 field_sections: Optional[List[str]] = None,
                 section_coordinates: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize the disease graph visualizer.
        
        Args:
            field_size: Size of the field in meters (width, height)
            field_sections: List of field section identifiers
            section_coordinates: Dictionary mapping section IDs to coordinates
        """
        self.field_size = field_size
        
        # Initialize field sections if not provided
        if field_sections is None:
            # Create a grid of sections (e.g., A1, A2, B1, B2, etc.)
            rows = ['A', 'B', 'C', 'D', 'E']
            cols = list(range(1, 6))  # 1-5
            self.field_sections = [f"{row}{col}" for row in rows for col in cols]
        else:
            self.field_sections = field_sections
        
        # Initialize section coordinates if not provided
        if section_coordinates is None:
            self.section_coordinates = self._generate_grid_coordinates()
        else:
            self.section_coordinates = section_coordinates
        
        # Create NetworkX graph for visualization
        self.graph = nx.Graph()
        self._initialize_graph()
        
        # Track disease state history for temporal analysis
        self.disease_history = {}
        self.mutation_events = []
        
        logger.info(f"Initialized DiseaseGraphVisualizer with {len(self.field_sections)} field sections")
    
    def _generate_grid_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """Generate grid coordinates for field sections"""
        coordinates = {}
        
        # Determine grid dimensions
        num_sections = len(self.field_sections)
        grid_size = int(np.ceil(np.sqrt(num_sections)))
        
        # Calculate cell size
        cell_width = self.field_size[0] / grid_size
        cell_height = self.field_size[1] / grid_size
        
        # Assign coordinates to each section
        for i, section in enumerate(self.field_sections):
            row = i // grid_size
            col = i % grid_size
            
            # Calculate center of the cell
            x = col * cell_width + cell_width / 2
            y = row * cell_height + cell_height / 2
            
            coordinates[section] = (x, y)
        
        return coordinates
    
    def _initialize_graph(self):
        """Initialize the NetworkX graph with nodes and edges"""
        # Add nodes (field sections)
        for section in self.field_sections:
            self.graph.add_node(
                section,
                pos=self.section_coordinates[section],
                disease_state="none",
                disease_severity=0.0,
                last_updated=datetime.now()
            )
        
        # Add edges (connections between neighboring sections)
        # This is a simplified approach - in practice, you'd use actual field topology
        for i, section1 in enumerate(self.field_sections):
            pos1 = self.section_coordinates[section1]
            
            for j, section2 in enumerate(self.field_sections):
                if i != j:
                    pos2 = self.section_coordinates[section2]
                    
                    # Calculate distance between sections
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # Connect sections within a certain distance
                    if distance < max(self.field_size) / 4:  # Adjust threshold as needed
                        self.graph.add_edge(
                            section1,
                            section2,
                            weight=1.0 / (distance + 1e-6),  # Inverse distance as weight
                            disease_flow=0.0
                        )
    
    def update_disease_state(self, section: str, disease_state: str, 
                            disease_severity: float, timestamp: Optional[datetime] = None):
        """Update the disease state for a field section
        
        Args:
            section: Field section identifier
            disease_state: Current disease state
            disease_severity: Disease severity (0.0-1.0)
            timestamp: Timestamp of the update (default: current time)
        """
        if section not in self.graph.nodes:
            logger.warning(f"Section {section} not found in graph")
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get previous state
        prev_state = self.graph.nodes[section].get("disease_state", "none")
        prev_severity = self.graph.nodes[section].get("disease_severity", 0.0)
        
        # Update node attributes
        self.graph.nodes[section]["disease_state"] = disease_state
        self.graph.nodes[section]["disease_severity"] = disease_severity
        self.graph.nodes[section]["last_updated"] = timestamp
        
        # Record history
        if section not in self.disease_history:
            self.disease_history[section] = []
        
        self.disease_history[section].append({
            "timestamp": timestamp,
            "disease_state": disease_state,
            "disease_severity": disease_severity
        })
        
        # Check for mutation events
        if prev_state != disease_state and prev_state != "none" and disease_state != "none":
            # Record mutation event
            mutation = {
                "section": section,
                "timestamp": timestamp,
                "from_disease": prev_state,
                "to_disease": disease_state,
                "from_severity": prev_severity,
                "to_severity": disease_severity
            }
            self.mutation_events.append(mutation)
            logger.info(f"Mutation detected in section {section}: {prev_state} -> {disease_state}")
    
    def update_disease_flow(self):
        """Update disease flow along edges based on current disease states"""
        for u, v, data in self.graph.edges(data=True):
            u_severity = self.graph.nodes[u]["disease_severity"]
            v_severity = self.graph.nodes[v]["disease_severity"]
            
            # Calculate disease flow based on severity difference and edge weight
            severity_diff = abs(u_severity - v_severity)
            edge_weight = data["weight"]
            
            # Update edge attribute
            self.graph.edges[u, v]["disease_flow"] = severity_diff * edge_weight
    
    def create_field_heatmap(self, colorscale: str = "Viridis") -> go.Figure:
        """Create a heatmap visualization of disease severity across the field
        
        Args:
            colorscale: Plotly colorscale for the heatmap
            
        Returns:
            Plotly figure object
        """
        # Extract node positions and disease severity
        x = []
        y = []
        sections = []
        severity = []
        disease_types = []
        
        for section, data in self.graph.nodes(data=True):
            pos = self.section_coordinates[section]
            x.append(pos[0])
            y.append(pos[1])
            sections.append(section)
            severity.append(data["disease_severity"])
            disease_types.append(data["disease_state"])
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            "x": x,
            "y": y,
            "section": sections,
            "severity": severity,
            "disease": disease_types
        })
        
        # Create heatmap using Plotly
        fig = px.density_heatmap(
            df, x="x", y="y", z="severity",
            title="Disease Severity Heatmap",
            labels={"x": "Field X (m)", "y": "Field Y (m)", "severity": "Disease Severity"},
            range_color=[0, 1],
            color_continuous_scale=colorscale
        )
        
        # Add section labels
        for section, pos in self.section_coordinates.items():
            fig.add_annotation(
                x=pos[0],
                y=pos[1],
                text=section,
                showarrow=False,
                font=dict(color="white", size=10)
            )
        
        # Update layout
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    def create_network_graph(self) -> go.Figure:
        """Create a network graph visualization of disease spread
        
        Returns:
            Plotly figure object
        """
        # Get node positions
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge color based on disease flow
            edge_colors.append(edge[2]["disease_flow"])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node, data in self.graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node color based on disease type
            if data["disease_state"] == "none":
                color = 0  # Healthy
            elif data["disease_state"] == "leaf_rust":
                color = 1
            elif data["disease_state"] == "powdery_mildew":
                color = 2
            elif data["disease_state"] == "leaf_spot":
                color = 3
            elif data["disease_state"] == "early_blight":
                color = 4
            else:
                color = 5  # Other diseases
            
            node_colors.append(color)
            
            # Node size based on severity
            node_sizes.append(10 + data["disease_severity"] * 20)
            
            # Node hover text
            node_text.append(
                f"Section: {node}<br>"
                f"Disease: {data['disease_state']}<br>"
                f"Severity: {data['disease_severity']:.2f}<br>"
                f"Last Updated: {data['last_updated'].strftime('%Y-%m-%d %H:%M')}"
            )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    title='Disease Type',
                    tickvals=[0, 1, 2, 3, 4, 5],
                    ticktext=['Healthy', 'Leaf Rust', 'Powdery Mildew', 'Leaf Spot', 'Early Blight', 'Other']
                )
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Disease Spread Network Graph',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=800,
                height=600
            )
        )
        
        return fig
    
    def create_temporal_analysis(self, section: Optional[str] = None) -> go.Figure:
        """Create a temporal analysis of disease progression
        
        Args:
            section: Specific field section to analyze (None for all sections)
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=("Disease Severity Over Time", "Mutation Events"),
                           vertical_spacing=0.2)
        
        # Filter history data
        if section is not None:
            # Single section analysis
            if section in self.disease_history:
                history = self.disease_history[section]
                
                # Extract data
                timestamps = [entry["timestamp"] for entry in history]
                severities = [entry["disease_severity"] for entry in history]
                states = [entry["disease_state"] for entry in history]
                
                # Create severity line plot
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=severities,
                        mode='lines+markers',
                        name=f"Section {section}"
                    ),
                    row=1, col=1
                )
                
                # Add disease state changes as markers
                for i in range(1, len(states)):
                    if states[i] != states[i-1]:
                        fig.add_trace(
                            go.Scatter(
                                x=[timestamps[i]],
                                y=[severities[i]],
                                mode='markers',
                                marker=dict(size=12, symbol='star', color='red'),
                                name=f"State Change: {states[i-1]} → {states[i]}",
                                showlegend=True
                            ),
                            row=1, col=1
                        )
        else:
            # All sections analysis - show average severity by disease type
            all_data = []
            for section_id, history in self.disease_history.items():
                for entry in history:
                    all_data.append({
                        "section": section_id,
                        "timestamp": entry["timestamp"],
                        "disease_state": entry["disease_state"],
                        "disease_severity": entry["disease_severity"]
                    })
            
            if all_data:
                df = pd.DataFrame(all_data)
                
                # Group by timestamp and disease state
                grouped = df.groupby([pd.Grouper(key='timestamp', freq='1H'), 'disease_state'])['disease_severity'].mean().reset_index()
                
                # Plot each disease type
                for disease in grouped['disease_state'].unique():
                    if disease == "none":
                        continue  # Skip healthy plants
                        
                    disease_data = grouped[grouped['disease_state'] == disease]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=disease_data['timestamp'],
                            y=disease_data['disease_severity'],
                            mode='lines',
                            name=disease
                        ),
                        row=1, col=1
                    )
        
        # Plot mutation events
        if self.mutation_events:
            # Filter mutations for the specific section if provided
            mutations = [m for m in self.mutation_events if section is None or m["section"] == section]
            
            if mutations:
                # Extract data
                timestamps = [m["timestamp"] for m in mutations]
                sections = [m["section"] for m in mutations]
                from_diseases = [m["from_disease"] for m in mutations]
                to_diseases = [m["to_disease"] for m in mutations]
                severities = [m["to_severity"] for m in mutations]
                
                # Create text labels
                texts = [f"Section {s}: {f} → {t}" for s, f, t in zip(sections, from_diseases, to_diseases)]
                
                # Add mutation events as scatter points
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=range(len(timestamps)),
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color=severities,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title='Severity')
                        ),
                        text=texts,
                        textposition="top center",
                        name="Mutations"
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title="Temporal Disease Analysis" + (f" - Section {section}" if section else ""),
            xaxis_title="Time",
            yaxis_title="Disease Severity",
            xaxis2_title="Time",
            yaxis2_title="Mutation Events",
            height=800,
            width=1000,
            showlegend=True
        )
        
        return fig
    
    def create_mutation_sankey(self) -> go.Figure:
        """Create a Sankey diagram of disease mutations
        
        Returns:
            Plotly figure object
        """
        if not self.mutation_events:
            # No mutations to display
            return go.Figure()
        
        # Count mutations between disease types
        mutation_counts = {}
        for mutation in self.mutation_events:
            from_disease = mutation["from_disease"]
            to_disease = mutation["to_disease"]
            
            key = (from_disease, to_disease)
            if key in mutation_counts:
                mutation_counts[key] += 1
            else:
                mutation_counts[key] = 1
        
        # Create lists for Sankey diagram
        unique_diseases = set()
        for from_disease, to_disease in mutation_counts.keys():
            unique_diseases.add(from_disease)
            unique_diseases.add(to_disease)
        
        # Map diseases to indices
        disease_to_idx = {disease: i for i, disease in enumerate(unique_diseases)}
        
        # Create source, target, and value lists
        source = []
        target = []
        value = []
        
        for (from_disease, to_disease), count in mutation_counts.items():
            source.append(disease_to_idx[from_disease])
            target.append(disease_to_idx[to_disease])
            value.append(count)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(unique_diseases)
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        # Update layout
        fig.update_layout(
            title_text="Disease Mutation Patterns",
            font_size=10,
            height=600,
            width=800
        )
        
        return fig
    
    def create_dashboard(self) -> Dict[str, go.Figure]:
        """Create a complete dashboard with multiple visualizations
        
        Returns:
            Dictionary of Plotly figures for the dashboard
        """
        # Update disease flow before creating visualizations
        self.update_disease_flow()
        
        # Create visualizations
        heatmap = self.create_field_heatmap()
        network = self.create_network_graph()
        temporal = self.create_temporal_analysis()
        mutation = self.create_mutation_sankey()
        
        # Return all visualizations
        return {
            "heatmap": heatmap,
            "network": network,
            "temporal": temporal,
            "mutation": mutation
        }
    
    def export_graph_data(self, filepath: str):
        """Export graph data to JSON for persistence
        
        Args:
            filepath: Path to save the JSON file
        """
        # Convert graph to dictionary
        graph_data = nx.node_link_data(self.graph)
        
        # Add history and mutation events
        data = {
            "graph": graph_data,
            "disease_history": self.disease_history,
            "mutation_events": self.mutation_events,
            "field_sections": self.field_sections,
            "section_coordinates": self.section_coordinates,
            "field_size": self.field_size,
            "export_time": datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings
        data_serializable = json.loads(
            json.dumps(data, default=lambda obj: obj.isoformat() if isinstance(obj, datetime) else str(obj))
        )
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data_serializable, f, indent=2)
        
        logger.info(f"Graph data exported to {filepath}")
    
    @classmethod
    def import_graph_data(cls, filepath: str) -> 'DiseaseGraphVisualizer':
        """Import graph data from JSON file
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            DiseaseGraphVisualizer instance with imported data
        """
        # Load data from file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create visualizer instance
        visualizer = cls(
            field_size=tuple(data["field_size"]),
            field_sections=data["field_sections"],
            section_coordinates={k: tuple(v) for k, v in data["section_coordinates"].items()}
        )
        
        # Recreate graph from data
        visualizer.graph = nx.node_link_graph(data["graph"])
        
        # Convert string timestamps back to datetime objects in history
        for section, history in data["disease_history"].items():
            visualizer.disease_history[section] = [
                {**entry, "timestamp": datetime.fromisoformat(entry["timestamp"])}
                for entry in history
            ]
        
        # Convert string timestamps back to datetime objects in mutation events
        visualizer.mutation_events = [
            {**event, "timestamp": datetime.fromisoformat(event["timestamp"])}
            for event in data["mutation_events"]
        ]
        
        logger.info(f"Graph data imported from {filepath}")
        return visualizer


class RealTimeVisualizer:
    """Real-time visualization manager for disease tracking dashboard"""
    
    def __init__(self, graph_visualizer: DiseaseGraphVisualizer, update_interval: int = 300):
        """
        Initialize the real-time visualizer.
        
        Args:
            graph_visualizer: DiseaseGraphVisualizer instance
            update_interval: Dashboard update interval in seconds
        """
        self.graph_visualizer = graph_visualizer
        self.update_interval = update_interval
        self.last_update = datetime.now()
        
        # Dashboard components
        self.dashboard = {}
        
        logger.info(f"Initialized RealTimeVisualizer with update interval of {update_interval} seconds")
    
    def process_sensor_data(self, sensor_data: Dict):
        """Process incoming sensor data and update the graph
        
        Args:
            sensor_data: Sensor data dictionary
        """
        # Extract relevant information
        if "field_section" in sensor_data and "sensor_type" in sensor_data:
            section = sensor_data["field_section"]
            timestamp = datetime.fromisoformat(sensor_data["timestamp"]) if "timestamp" in sensor_data else datetime.now()
            
            # Process leaf imaging sensor data
            if sensor_data["sensor_type"] == "leaf_imaging" and "disease_analysis" in sensor_data:
                disease_analysis = sensor_data["disease_analysis"]
                
                # Update disease state in the graph
                self.graph_visualizer.update_disease_state(
                    section=section,
                    disease_state=disease_analysis["disease_type"],
                    disease_severity=disease_analysis["disease_severity"],
                    timestamp=timestamp
                )
                
                # Check if mutation was detected
                if sensor_data.get("mutation_detected", False) and "latest_mutation" in sensor_data:
                    logger.info(f"Mutation detected in section {section}: {sensor_data['latest_mutation']}")
    
    def update_dashboard(self) -> Dict[str, Any]:
        """Update the dashboard visualizations
        
        Returns:
            Updated dashboard components
        """
        current_time = datetime.now()
        time_diff = (current_time - self.last_update).total_seconds()
        
        # Only update if enough time has passed
        if time_diff >= self.update_interval or not self.dashboard:
            logger.info("Updating dashboard visualizations")
            self.dashboard = self.graph_visualizer.create_dashboard()
            self.last_update = current_time
        
        return self.dashboard
    
    def get_mutation_alerts(self, since: Optional[datetime] = None) -> List[Dict]:
        """Get recent mutation alerts
        
        Args:
            since: Only return mutations after this timestamp
            
        Returns:
            List of mutation events
        """
        if since is None:
            # Default to last hour
            since = datetime.now() - timedelta(hours=1)
        
        # Filter recent mutations
        recent_mutations = [
            mutation for mutation in self.graph_visualizer.mutation_events
            if mutation["timestamp"] >= since
        ]
        
        return recent_mutations
    
    def export_snapshot(self, filepath: str):
        """Export a snapshot of the current graph state
        
        Args:
            filepath: Path to save the snapshot
        """
        self.graph_visualizer.export_graph_data(filepath)
        logger.info(f"Dashboard snapshot exported to {filepath}")


def create_demo_visualization():
    """Create a demo visualization with simulated disease spread"""
    # Create field sections (5x5 grid: A1-E5)
    rows = ['A', 'B', 'C', 'D', 'E']
    cols = list(range(1, 6))  # 1-5
    field_sections = [f"{row}{col}" for row in rows for col in cols]
    
    # Initialize visualizer
    visualizer = DiseaseGraphVisualizer(
        field_size=(500.0, 500.0),
        field_sections=field_sections
    )
    
    # Simulate disease spread
    # Initial infection in one corner
    visualizer.update_disease_state("A1", "leaf_rust", 0.3)
    visualizer.update_disease_state("A2", "leaf_rust", 0.2)
    visualizer.update_disease_state("B1", "leaf_rust", 0.1)
    
    # Another disease type in another area
    visualizer.update_disease_state("D4", "powdery_mildew", 0.4)
    visualizer.update_disease_state("D5", "powdery_mildew", 0.3)
    visualizer.update_disease_state("E4", "powdery_mildew", 0.2)
    
    # Simulate progression over time
    for i in range(10):
        # Advance time
        timestamp = datetime.now() - timedelta(days=10-i)
        
        # Spread leaf rust
        if i >= 2:
            visualizer.update_disease_state("B2", "leaf_rust", 0.2 + i*0.05, timestamp)
        if i >= 4:
            visualizer.update_disease_state("C1", "leaf_rust", 0.1 + i*0.05, timestamp)
            visualizer.update_disease_state("C2", "leaf_rust", 0.1 + i*0.03, timestamp)
        if i >= 6:
            # Mutation event
            visualizer.update_disease_state("C2", "leaf_spot", 0.4, timestamp)
            visualizer.update_disease_state("C3", "leaf_spot", 0.3, timestamp)
        
        # Spread powdery mildew
        if i >= 3:
            visualizer.update_disease_state("E3", "powdery_mildew", 0.2 + i*0.04, timestamp)
        if i >= 5:
            visualizer.update_disease_state("D3", "powdery_mildew", 0.3 + i*0.03, timestamp)
        if i >= 7:
            # Mutation event
            visualizer.update_disease_state("D3", "early_blight", 0.5, timestamp)
            visualizer.update_disease_state("C3", "early_blight", 0.4, timestamp)
    
    # Create real-time visualizer
    real_time_viz = RealTimeVisualizer(visualizer)
    
    # Generate dashboard
    dashboard = real_time_viz.update_dashboard()
    
    return dashboard, real_time_viz


if __name__ == "__main__":
    # Create demo visualization
    dashboard, visualizer = create_demo_visualization()
    
    # Display some visualizations
    dashboard["heatmap"].show()
    dashboard["network"].show()
    
    # Export snapshot
    visualizer.export_snapshot("disease_graph_snapshot.json")