"""
Spatial-Temporal Graph Engine for Crop Disease Tracking

Implements real-time graph-based data workflow mapping crop sections as nodes
and tracking disease mutation and spread across spatial-temporal edges.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
import math
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import uuid


@dataclass
class GraphNode:
    """Represents a crop section node in the spatial-temporal graph"""
    node_id: str
    location: Tuple[float, float]  # (latitude, longitude)
    crop_type: str
    health_status: str = "healthy"
    disease_confidence: float = 0.0
    disease_type: Optional[str] = None
    last_update: datetime = field(default_factory=datetime.now)
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    mutation_events: List[Dict] = field(default_factory=list)


@dataclass
class GraphEdge:
    """Represents a spatial-temporal edge between crop sections"""
    source_id: str
    target_id: str
    edge_type: str = "spatial"  # spatial, temporal, disease_spread
    weight: float = 1.0
    disease_spread_probability: float = 0.0
    last_mutation: Optional[datetime] = None
    mutation_history: List[Dict] = field(default_factory=list)


class SpatialTemporalGraph:
    """Main spatial-temporal graph for disease tracking"""
    
    def __init__(self, field_size: Tuple[float, float], node_spacing: float = 20.0):
        self.field_size = field_size
        self.node_spacing = node_spacing
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.disease_centers: List[Dict] = []
        self.mutation_events: List[Dict] = []
        
        # Initialize graph structure
        self._initialize_graph()
        
    def _initialize_graph(self):
        """Initialize the spatial-temporal graph structure"""
        # Create nodes for each crop section
        num_rows = int(self.field_size[0] / self.node_spacing)
        num_cols = int(self.field_size[1] / self.node_spacing)
        
        for row in range(num_rows):
            for col in range(num_cols):
                node_id = f"section_{row}_{col}"
                location = (
                    row * self.node_spacing,
                    col * self.node_spacing
                )
                
                node = GraphNode(
                    node_id=node_id,
                    location=location,
                    crop_type="corn",  # Default crop type
                    health_status="healthy"
                )
                
                self.nodes[node_id] = node
                self.graph.add_node(node_id, **node.__dict__)
        
        # Create spatial edges between adjacent sections
        for row in range(num_rows):
            for col in range(num_cols):
                node_id = f"section_{row}_{col}"
                
                # Connect to adjacent sections
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < num_rows and 0 <= new_col < num_cols:
                        target_id = f"section_{new_row}_{new_col}"
                        edge_id = f"{node_id}_to_{target_id}"
                        
                        edge = GraphEdge(
                            source_id=node_id,
                            target_id=target_id,
                            edge_type="spatial"
                        )
                        
                        self.edges[edge_id] = edge
                        self.graph.add_edge(node_id, target_id, **edge.__dict__)
    
    def update_node_data(self, sensor_data: Dict):
        """Update node data with incoming sensor data"""
        sensor_location = sensor_data.get('location', {})
        lat, lng = sensor_location.get('latitude', 0), sensor_location.get('longitude', 0)
        
        # Find closest node to sensor location
        closest_node = self._find_closest_node(lat, lng)
        if not closest_node:
            return
        
        # Update node with sensor data
        node = self.nodes[closest_node]
        node.last_update = datetime.now()
        node.sensor_data.update(sensor_data)
        
        # Update health status based on disease detection
        if sensor_data.get('sensor_type') == 'leaf_camera':
            disease_detected = sensor_data.get('disease_detected', False)
            disease_confidence = sensor_data.get('disease_confidence', 0.0)
            disease_type = sensor_data.get('disease_type')
            
            if disease_detected:
                node.health_status = "infected"
                node.disease_confidence = disease_confidence
                node.disease_type = disease_type
                
                # Record mutation event
                mutation_event = {
                    'timestamp': datetime.now(),
                    'disease_type': disease_type,
                    'confidence': disease_confidence,
                    'sensor_id': sensor_data.get('sensor_id')
                }
                node.mutation_events.append(mutation_event)
                
                # Update disease centers
                self._update_disease_centers(node)
        
        # Update graph node attributes
        self.graph.nodes[closest_node].update(node.__dict__)
    
    def _find_closest_node(self, lat: float, lng: float) -> Optional[str]:
        """Find the closest graph node to a given location"""
        min_distance = float('inf')
        closest_node = None
        
        for node_id, node in self.nodes.items():
            distance = math.sqrt(
                (lat - node.location[0])**2 + (lng - node.location[1])**2
            )
            if distance < min_distance:
                min_distance = distance
                closest_node = node_id
        
        return closest_node
    
    def _update_disease_centers(self, infected_node: GraphNode):
        """Update disease centers when new infection is detected"""
        center = {
            'location': infected_node.location,
            'disease_type': infected_node.disease_type,
            'severity': infected_node.disease_confidence,
            'timestamp': datetime.now(),
            'affected_nodes': [infected_node.node_id]
        }
        self.disease_centers.append(center)
    
    def update_disease_spread(self, elapsed_hours: float):
        """Update disease spread across the graph"""
        # Update existing disease centers
        for center in self.disease_centers:
            center['severity'] = min(1.0, center['severity'] + 0.1 * elapsed_hours)
            
            # Spread to adjacent nodes
            affected_nodes = set(center['affected_nodes'])
            new_infections = []
            
            for node_id in affected_nodes:
                node = self.nodes[node_id]
                if node.health_status == "infected":
                    # Find adjacent nodes
                    for edge_id, edge in self.edges.items():
                        if edge.source_id == node_id:
                            target_node = self.nodes[edge.target_id]
                            if target_node.health_status == "healthy":
                                # Calculate spread probability
                                spread_prob = center['severity'] * 0.3 * elapsed_hours
                                if random.random() < spread_prob:
                                    target_node.health_status = "infected"
                                    target_node.disease_confidence = center['severity'] * 0.8
                                    target_node.disease_type = center['disease_type']
                                    new_infections.append(edge.target_id)
            
            center['affected_nodes'].extend(new_infections)
        
        # Create new mutation events
        if random.random() < 0.01 * elapsed_hours:  # 1% chance per hour
            self._create_mutation_event()
    
    def _create_mutation_event(self):
        """Create a new disease mutation event"""
        # Select random infected node
        infected_nodes = [n for n in self.nodes.values() if n.health_status == "infected"]
        if not infected_nodes:
            return
        
        source_node = random.choice(infected_nodes)
        
        mutation_event = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'location': source_node.location,
            'original_disease': source_node.disease_type,
            'mutation_type': random.choice(['resistant', 'aggressive', 'latent']),
            'severity': random.uniform(0.5, 1.0),
            'source_node': source_node.node_id
        }
        
        self.mutation_events.append(mutation_event)
        
        # Apply mutation to source node
        source_node.disease_type = f"{source_node.disease_type}_{mutation_event['mutation_type']}"
        source_node.disease_confidence = mutation_event['severity']
    
    def get_graph_data(self) -> Dict:
        """Get current graph data for visualization"""
        nodes_data = []
        edges_data = []
        
        for node_id, node in self.nodes.items():
            nodes_data.append({
                'id': node_id,
                'x': node.location[0],
                'y': node.location[1],
                'health_status': node.health_status,
                'disease_confidence': node.disease_confidence,
                'disease_type': node.disease_type,
                'last_update': node.last_update.isoformat(),
                'mutation_count': len(node.mutation_events)
            })
        
        for edge_id, edge in self.edges.items():
            edges_data.append({
                'source': edge.source_id,
                'target': edge.target_id,
                'edge_type': edge.edge_type,
                'weight': edge.weight,
                'disease_spread_probability': edge.disease_spread_probability
            })
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'disease_centers': self.disease_centers,
            'mutation_events': self.mutation_events,
            'timestamp': datetime.now().isoformat()
        }


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for disease tracking"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super(STGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.conv_layers.append(GCNConv(hidden_dim, output_dim))
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3)  # healthy, infected, mutation
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the ST-GCN"""
        x, edge_index = data.x, data.edge_index
        
        # Graph convolutions
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Final convolution
        x = self.conv_layers[-1](x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, data.batch)
        
        # Classification
        output = self.classifier(x)
        
        return output


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for disease pattern detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8):
        super(GraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Graph attention layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.2)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the GAT"""
        x, edge_index = data.x, data.edge_index
        
        # First GAT layer
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GAT layer
        x = F.elu(self.gat2(x, edge_index))
        
        # Global attention pooling
        x = global_mean_pool(x, data.batch)
        
        # Output
        output = torch.sigmoid(self.output_proj(x))
        
        return output


class RealTimeGraphProcessor:
    """Real-time graph processor for disease tracking"""
    
    def __init__(self, field_size: Tuple[float, float]):
        self.graph = SpatialTemporalGraph(field_size)
        self.stgcn = STGCN(input_dim=10, hidden_dim=64, output_dim=32)
        self.gat = GraphAttentionNetwork(input_dim=10, hidden_dim=64)
        
        # Performance tracking
        self.processing_times = []
        self.detection_accuracy = []
        
    async def process_sensor_data(self, sensor_data: Dict):
        """Process incoming sensor data and update graph"""
        start_time = time.time()
        
        # Update graph with sensor data
        self.graph.update_node_data(sensor_data)
        
        # Update disease spread
        self.graph.update_disease_spread(0.1)  # 6 minutes
        
        # Generate graph features for AI models
        graph_features = self._extract_graph_features()
        
        # Run AI analysis
        predictions = await self._run_ai_analysis(graph_features)
        
        # Record performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return {
            'predictions': predictions,
            'processing_time': processing_time,
            'graph_data': self.graph.get_graph_data()
        }
    
    def _extract_graph_features(self) -> torch.Tensor:
        """Extract features from the current graph state"""
        features = []
        
        for node_id, node in self.graph.nodes.items():
            # Node features: [health_status, disease_confidence, location_x, location_y, 
            #                mutation_count, time_since_update, crop_type_encoded]
            health_encoded = 1.0 if node['health_status'] == "infected" else 0.0
            time_since_update = (datetime.now() - datetime.fromisoformat(node['last_update'])).total_seconds() / 3600
            
            node_features = [
                health_encoded,
                node['disease_confidence'],
                node['location'][0] / self.graph.field_size[0],  # Normalized x
                node['location'][1] / self.graph.field_size[1],  # Normalized y
                len(node['mutation_events']),
                time_since_update,
                1.0,  # crop_type (corn)
                0.0,  # placeholder
                0.0,  # placeholder
                0.0   # placeholder
            ]
            features.append(node_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    async def _run_ai_analysis(self, features: torch.Tensor) -> Dict:
        """Run AI analysis on graph features"""
        # Prepare graph data for PyTorch Geometric
        edge_index = []
        for edge_id, edge in self.graph.edges.items():
            source_idx = list(self.graph.nodes.keys()).index(edge['source_id'])
            target_idx = list(self.graph.nodes.keys()).index(edge['target_id'])
            edge_index.append([source_idx, target_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create PyTorch Geometric data object
        data = Data(x=features, edge_index=edge_index)
        
        # Run ST-GCN
        with torch.no_grad():
            stgcn_output = self.stgcn(data)
            gat_output = self.gat(data)
        
        # Process outputs
        stgcn_probs = F.softmax(stgcn_output, dim=1)
        disease_probability = gat_output.item()
        
        return {
            'disease_probability': disease_probability,
            'health_classification': stgcn_probs.tolist(),
            'mutation_detected': disease_probability > 0.7
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'total_processing_count': len(self.processing_times)
        }


class GraphVisualizer:
    """Real-time graph visualization component"""
    
    def __init__(self):
        self.colors = {
            'healthy': '#2E8B57',      # Sea Green
            'infected': '#DC143C',      # Crimson
            'mutation': '#FF8C00',      # Dark Orange
            'spread': '#FFD700'         # Gold
        }
    
    def create_interactive_graph(self, graph_data: Dict) -> go.Figure:
        """Create interactive graph visualization"""
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        # Create node traces
        node_x = [node['x'] for node in nodes]
        node_y = [node['y'] for node in nodes]
        node_colors = [self.colors.get(node['health_status'], '#808080') for node in nodes]
        node_sizes = [10 + node['disease_confidence'] * 20 for node in nodes]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in edges:
            source_node = next(n for n in nodes if n['id'] == edge['source'])
            target_node = next(n for n in nodes if n['id'] == edge['target'])
            edge_x.extend([source_node['x'], target_node['x'], None])
            edge_y.extend([source_node['y'], target_node['y'], None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888888'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=[f"Section: {node['id']}<br>Health: {node['health_status']}<br>Disease: {node['disease_confidence']:.2f}" 
                  for node in nodes],
            hoverinfo='text',
            name='Crop Sections'
        ))
        
        # Update layout
        fig.update_layout(
            title='Real-Time Crop Disease Tracking Graph',
            xaxis_title='Field Width (meters)',
            yaxis_title='Field Length (meters)',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Green: Healthy | Red: Infected | Orange: Mutation",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0, y=1.1,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ]
        )
        
        return fig
    
    def create_disease_spread_animation(self, graph_data: Dict) -> go.Figure:
        """Create animated visualization of disease spread"""
        nodes = graph_data['nodes']
        
        # Create time series data
        timestamps = []
        infection_counts = []
        
        # Simulate time series (in real implementation, this would come from historical data)
        for i in range(24):  # 24 time points
            timestamps.append(i)
            infected_count = sum(1 for node in nodes if node['health_status'] == 'infected')
            infection_counts.append(infected_count)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=infection_counts,
            mode='lines+markers',
            name='Infected Sections',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Disease Spread Over Time',
            xaxis_title='Time (hours)',
            yaxis_title='Number of Infected Sections',
            showlegend=True
        )
        
        return fig


# Example usage
async def main():
    """Example usage of the graph engine"""
    # Initialize graph processor
    processor = RealTimeGraphProcessor(field_size=(500.0, 300.0))
    visualizer = GraphVisualizer()
    
    # Simulate sensor data
    sensor_data = {
        'sensor_id': 'sensor_001',
        'sensor_type': 'leaf_camera',
        'location': {'latitude': 100.0, 'longitude': 150.0},
        'disease_detected': True,
        'disease_confidence': 0.8,
        'disease_type': 'fungal_infection',
        'leaf_health_score': 0.3
    }
    
    # Process data
    result = await processor.process_sensor_data(sensor_data)
    
    # Create visualization
    fig = visualizer.create_interactive_graph(result['graph_data'])
    
    print(f"Processing time: {result['processing_time']:.3f}s")
    print(f"Disease probability: {result['predictions']['disease_probability']:.3f}")
    print(f"Mutation detected: {result['predictions']['mutation_detected']}")


if __name__ == "__main__":
    asyncio.run(main())
