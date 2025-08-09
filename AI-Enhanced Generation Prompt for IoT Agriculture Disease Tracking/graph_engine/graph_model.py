"""Spatial-Temporal Graph Convolutional Network Model

This module implements the ST-GCN with graph attention mechanisms for
tracking and analyzing crop disease mutation and spread patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for disease spread attention mechanism"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Learnable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # Leaky ReLU activation
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention layer
        
        Args:
            h: Node features of shape [N, in_features]
            adj: Adjacency matrix of shape [N, N]
            
        Returns:
            Updated node features of shape [N, out_features]
        """
        # Linear transformation
        Wh = torch.mm(h, self.W)  # [N, out_features]
        
        # Self-attention mechanism
        a_input = self._prepare_attentional_mechanism_input(Wh)  # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]
        
        # Masked attention (using adjacency matrix)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        attention = F.softmax(attention, dim=1)  # [N, N]
        attention = F.dropout(attention, self.dropout, training=self.training)  # [N, N]
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """Prepare the attention mechanism input"""
        N = Wh.size(0)
        
        # Repeat the features for each node
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # [N*N, out_features]
        Wh_repeated_alternating = Wh.repeat(N, 1)  # [N*N, out_features]
        
        # Concatenate the repeated features
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)  # [N*N, 2*out_features]
        
        # Reshape to [N, N, 2*out_features]
        return all_combinations_matrix.view(N, N, 2*self.out_features)


class TemporalAttention(nn.Module):
    """Temporal attention module for capturing disease progression over time"""
    
    def __init__(self, hidden_dim: int, num_time_steps: int):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_time_steps = num_time_steps
        
        # Temporal attention parameters
        self.W_t = nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim)))
        nn.init.xavier_uniform_(self.W_t.data, gain=1.414)
        
        self.b_t = nn.Parameter(torch.zeros(size=(1, hidden_dim)))
        
        self.u_t = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        nn.init.xavier_uniform_(self.u_t.data, gain=1.414)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of temporal attention
        
        Args:
            x: Input tensor of shape [batch_size, num_time_steps, num_nodes, hidden_dim]
            
        Returns:
            Weighted output of shape [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_time_steps, num_nodes, hidden_dim = x.size()
        
        # Reshape for attention calculation
        x_reshaped = x.view(-1, hidden_dim)  # [batch_size*num_time_steps*num_nodes, hidden_dim]
        
        # Calculate attention scores
        lhs = torch.matmul(x_reshaped, self.W_t)  # [batch*time*nodes, hidden]
        lhs = lhs.view(batch_size, num_time_steps, num_nodes, hidden_dim)  # [batch, time, nodes, hidden]
        lhs = torch.tanh(lhs + self.b_t)  # [batch, time, nodes, hidden]
        
        # Calculate attention weights
        lhs_reshaped = lhs.view(-1, hidden_dim)  # [batch*time*nodes, hidden]
        attn = torch.matmul(lhs_reshaped, self.u_t)  # [batch*time*nodes, 1]
        attn = attn.view(batch_size, num_time_steps, num_nodes)  # [batch, time, nodes]
        
        # Apply softmax to get attention distribution over time steps
        alpha = F.softmax(attn, dim=1)  # [batch, time, nodes]
        alpha = alpha.unsqueeze(-1)  # [batch, time, nodes, 1]
        
        # Apply attention weights to input
        weighted_sum = torch.sum(alpha * x, dim=1)  # [batch, nodes, hidden]
        
        return weighted_sum


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for disease tracking"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 num_nodes: int,
                 num_time_steps: int,
                 num_gnn_layers: int = 2,
                 dropout: float = 0.5):
        super(STGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph Convolutional layers with attention
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=8, dropout=dropout))
        
        for _ in range(num_gnn_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=dropout))
        
        # Temporal attention layer
        self.temporal_attention = TemporalAttention(hidden_dim * 8, num_time_steps)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data_list: List[Data]) -> Dict[str, torch.Tensor]:
        """Forward pass of the ST-GCN model
        
        Args:
            data_list: List of PyTorch Geometric Data objects containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_attr_dim]
                - time_step: Time step information
                
        Returns:
            Dictionary containing:
                - node_predictions: Disease predictions for each node
                - graph_embedding: Graph-level embedding for the entire field
                - attention_weights: Attention weights for interpretability
        """
        batch = Batch.from_data_list(data_list)
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        # Initial feature projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # Apply GAT layers
        attention_weights = []
        for i, gat_layer in enumerate(self.gat_layers):
            x, attention = gat_layer(x, edge_index, return_attention_weights=True)
            attention_weights.append(attention)
            
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        # Reshape for temporal attention
        batch_size = len(data_list) // self.num_time_steps
        x = x.view(batch_size, self.num_time_steps, self.num_nodes, -1)
        
        # Apply temporal attention
        x = self.temporal_attention(x)  # [batch_size, num_nodes, hidden_dim*8]
        
        # Final prediction layers
        x = x.view(-1, self.hidden_dim * 8)  # [batch_size*num_nodes, hidden_dim*8]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.batch_norm(x)
        x = self.dropout_layer(x)
        node_predictions = self.fc2(x)  # [batch_size*num_nodes, output_dim]
        
        # Reshape predictions back to [batch_size, num_nodes, output_dim]
        node_predictions = node_predictions.view(batch_size, self.num_nodes, self.output_dim)
        
        # Create graph-level embedding by pooling node features
        graph_embedding = torch.mean(x.view(batch_size, self.num_nodes, -1), dim=1)  # [batch_size, hidden_dim]
        
        return {
            "node_predictions": node_predictions,
            "graph_embedding": graph_embedding,
            "attention_weights": attention_weights
        }


class DiseaseTrackingModel:
    """High-level disease tracking model using ST-GCN"""
    
    def __init__(self, 
                 num_nodes: int,
                 num_time_steps: int = 10,
                 input_dim: int = 16,
                 hidden_dim: int = 32,
                 output_dim: int = 5,  # Number of disease states to predict
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the disease tracking model.
        
        Args:
            num_nodes: Number of nodes in the graph (field sections)
            num_time_steps: Number of time steps to consider for temporal patterns
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (number of disease states)
            device: Device to run the model on (cuda or cpu)
        """
        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # Initialize the ST-GCN model
        self.model = STGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            num_time_steps=num_time_steps,
            num_gnn_layers=2,
            dropout=0.3
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Track training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        
        logger.info(f"Initialized DiseaseTrackingModel with {num_nodes} nodes on {device}")
    
    def train(self, train_data: List[Data], val_data: List[Data], epochs: int = 100, patience: int = 10):
        """Train the disease tracking model
        
        Args:
            train_data: Training data as list of PyTorch Geometric Data objects
            val_data: Validation data as list of PyTorch Geometric Data objects
            epochs: Number of training epochs
            patience: Early stopping patience
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(train_data)
            node_predictions = outputs["node_predictions"]
            
            # Extract labels from data
            labels = torch.cat([data.y for data in train_data]).to(self.device)
            
            # Calculate loss
            loss = self.criterion(node_predictions.view(-1, self.output_dim), labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_data)
                val_predictions = val_outputs["node_predictions"]
                val_labels = torch.cat([data.y for data in val_data]).to(self.device)
                val_loss = self.criterion(val_predictions.view(-1, self.output_dim), val_labels)
            
            # Calculate accuracy
            train_acc = self._calculate_accuracy(node_predictions, labels)
            val_acc = self._calculate_accuracy(val_predictions, val_labels)
            
            # Update history
            self.history["train_loss"].append(loss.item())
            self.history["val_loss"].append(val_loss.item())
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {loss.item():.4f}, "
                           f"Val Loss: {val_loss.item():.4f}, "
                           f"Train Acc: {train_acc:.4f}, "
                           f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_disease_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load("best_disease_model.pt"))
        logger.info("Training completed")
        
        return self.history
    
    def predict(self, data: List[Data]) -> Dict[str, torch.Tensor]:
        """Make predictions with the trained model
        
        Args:
            data: Input data as list of PyTorch Geometric Data objects
            
        Returns:
            Dictionary containing predictions and attention weights
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
        
        return outputs
    
    def detect_mutations(self, current_data: List[Data], historical_data: List[Data], 
                        threshold: float = 0.3) -> List[Dict]:
        """Detect disease mutations by comparing current and historical predictions
        
        Args:
            current_data: Current graph data
            historical_data: Historical graph data for comparison
            threshold: Threshold for mutation detection
            
        Returns:
            List of detected mutations with details
        """
        # Get predictions for current and historical data
        current_outputs = self.predict(current_data)
        historical_outputs = self.predict(historical_data)
        
        current_preds = current_outputs["node_predictions"]
        historical_preds = historical_outputs["node_predictions"]
        
        # Calculate changes in disease patterns
        changes = torch.abs(current_preds - historical_preds)
        
        # Identify significant changes (potential mutations)
        mutations = []
        for node_idx in range(self.num_nodes):
            node_changes = changes[0, node_idx]  # Assuming batch size of 1
            max_change, new_disease = torch.max(node_changes, dim=0)
            
            if max_change.item() > threshold:
                # Get the previous disease type
                _, old_disease = torch.max(historical_preds[0, node_idx], dim=0)
                
                # Record mutation
                mutation = {
                    "node_id": node_idx,
                    "old_disease": old_disease.item(),
                    "new_disease": new_disease.item(),
                    "change_magnitude": max_change.item(),
                    "timestamp": current_data[0].timestamp  # Assuming timestamp is stored in data
                }
                mutations.append(mutation)
        
        return mutations
    
    def _calculate_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate prediction accuracy"""
        pred_classes = torch.argmax(predictions.view(-1, self.output_dim), dim=1)
        correct = (pred_classes == labels).sum().item()
        total = labels.size(0)
        return correct / total


def create_graph_from_sensor_data(sensor_data: List[Dict], 
                                field_sections: List[str],
                                time_steps: int = 10) -> List[Data]:
    """Create graph data objects from sensor readings for the ST-GCN model
    
    Args:
        sensor_data: List of sensor readings with location and disease information
        field_sections: List of field section identifiers
        time_steps: Number of time steps to include
        
    Returns:
        List of PyTorch Geometric Data objects for the model
    """
    # Map field sections to node indices
    section_to_idx = {section: i for i, section in enumerate(field_sections)}
    num_nodes = len(field_sections)
    
    # Group data by timestamp
    data_by_time = {}
    for reading in sensor_data:
        timestamp = reading["timestamp"]
        if timestamp not in data_by_time:
            data_by_time[timestamp] = []
        data_by_time[timestamp].append(reading)
    
    # Sort timestamps
    sorted_timestamps = sorted(data_by_time.keys())
    
    # Take the most recent time_steps
    if len(sorted_timestamps) > time_steps:
        sorted_timestamps = sorted_timestamps[-time_steps:]
    
    # Create graph data for each time step
    graph_data_list = []
    
    for t, timestamp in enumerate(sorted_timestamps):
        readings = data_by_time[timestamp]
        
        # Initialize node features and labels
        node_features = torch.zeros(num_nodes, 16)  # Adjust feature dimension as needed
        node_labels = torch.zeros(num_nodes, dtype=torch.long)
        
        # Fill in node features from sensor readings
        for reading in readings:
            if "field_section" in reading:
                section = reading["field_section"]
                if section in section_to_idx:
                    node_idx = section_to_idx[section]
                    
                    # Extract features based on sensor type
                    if reading["sensor_type"] == "leaf_imaging":
                        # Disease features
                        disease_type = reading["disease_analysis"]["disease_type"]
                        disease_severity = reading["disease_analysis"]["disease_severity"]
                        
                        # One-hot encode disease type
                        disease_idx = min(DISEASE_TYPES.index(disease_type) if disease_type in DISEASE_TYPES else 0, 4)
                        node_labels[node_idx] = disease_idx
                        
                        # Set features
                        node_features[node_idx, 0] = disease_severity
                        node_features[node_idx, 1:6] = F.one_hot(torch.tensor(disease_idx), num_classes=5)
                        
                    elif reading["sensor_type"] == "environmental":
                        # Environmental features
                        node_features[node_idx, 6] = reading["temperature"] / 40.0  # Normalize
                        node_features[node_idx, 7] = reading["humidity"] / 100.0
                        node_features[node_idx, 8] = reading["soil_moisture"] / 100.0
        
        # Create edges (connect neighboring field sections)
        # This is a simplified approach - in practice, you'd use actual field topology
        edge_index = [[], []]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Connect all nodes (fully connected graph)
                    edge_index[0].append(i)
                    edge_index[1].append(j)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=node_labels,
            timestamp=timestamp,
            time_step=t
        )
        
        graph_data_list.append(data)
    
    return graph_data_list


# Define disease types for reference
DISEASE_TYPES = [
    "none",  # Healthy
    "leaf_rust",
    "powdery_mildew",
    "leaf_spot",
    "early_blight"
]