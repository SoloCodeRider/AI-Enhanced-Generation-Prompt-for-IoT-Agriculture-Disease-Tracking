"""Disease Detection Models

This module implements deep learning models for crop disease detection
using convolutional neural networks and residual networks.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiseaseImageDataset(Dataset):
    """Dataset for crop disease images"""
    
    def __init__(self, image_dir: str, annotations_file: Optional[str] = None, 
                 transform=None, is_training: bool = True):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing images
            annotations_file: Path to annotations JSON file (optional)
            transform: Image transformations to apply
            is_training: Whether this is a training dataset
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Load annotations if provided, otherwise use directory structure
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
            
            self.image_paths = []
            self.labels = []
            self.metadata = []
            
            for item in self.annotations:
                self.image_paths.append(os.path.join(image_dir, item['image_path']))
                self.labels.append(item['disease_class'])
                self.metadata.append(item.get('metadata', {}))
        else:
            # Use directory structure (each disease in its own folder)
            self.image_paths = []
            self.labels = []
            self.metadata = []
            
            # Get disease classes from subdirectories
            disease_classes = [d for d in os.listdir(image_dir) 
                              if os.path.isdir(os.path.join(image_dir, d))]
            
            # Create class to index mapping
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(disease_classes))}
            
            # Collect images and labels
            for disease_class in disease_classes:
                class_dir = os.path.join(image_dir, disease_class)
                class_idx = self.class_to_idx[disease_class]
                
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_file))
                        self.labels.append(class_idx)
                        self.metadata.append({})
        
        logger.info(f"Loaded {len(self.image_paths)} images with {len(set(self.labels))} disease classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        # Get metadata
        metadata = self.metadata[idx]
        
        return {
            'image': image,
            'label': label,
            'metadata': metadata,
            'image_path': img_path
        }


class ResidualBlock(nn.Module):
    """Residual block for deep residual network"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DiseaseResNet(nn.Module):
    """Custom ResNet architecture for crop disease detection"""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(DiseaseResNet, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification layer
        x = self.fc(x)
        
        return x


class DiseaseDetectionModel:
    """High-level disease detection model using deep learning"""
    
    def __init__(self, 
                 num_classes: int,
                 model_type: str = "resnet",
                 pretrained: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the disease detection model.
        
        Args:
            num_classes: Number of disease classes to detect
            model_type: Type of model to use ("resnet", "custom_resnet", "efficientnet")
            pretrained: Whether to use pretrained weights
            device: Device to run the model on (cuda or cpu)
        """
        self.num_classes = num_classes
        self.model_type = model_type
        self.device = device
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        if model_type == "resnet":
            # Use pretrained ResNet50
            self.model = models.resnet50(pretrained=pretrained)
            # Replace the final fully connected layer
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        elif model_type == "custom_resnet":
            # Use custom ResNet implementation
            self.model = DiseaseResNet(num_classes=num_classes)
        
        elif model_type == "efficientnet":
            # Use EfficientNet
            self.model = models.efficientnet_b0(pretrained=pretrained)
            # Replace the final classifier
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Class names mapping
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        
        logger.info(f"Initialized {model_type} model with {num_classes} classes on {device}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
              epochs: int = 10, save_path: Optional[str] = None):
        """
        Train the disease detection model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            save_path: Path to save the trained model
            
        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            # Calculate epoch statistics
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                # Save best model
                if save_path and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'class_names': self.class_names
                    }, save_path)
                    logger.info(f"Saved best model with validation accuracy: {val_acc:.4f}")
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {epoch_train_loss:.4f}, "
                           f"Train Acc: {epoch_train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, "
                           f"Val Acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {epoch_train_loss:.4f}, "
                           f"Train Acc: {epoch_train_acc:.4f}")
                
                # Save model without validation
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_acc': epoch_train_acc,
                        'class_names': self.class_names
                    }, save_path)
        
        return history
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                eval_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        return eval_loss / total, correct / total
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict disease for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
            # Get top prediction
            top_prob, top_class = torch.max(probabilities, 0)
            
            # Get top-3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            
            # Convert to numpy for easier handling
            top3_probs = top3_probs.cpu().numpy()
            top3_indices = top3_indices.cpu().numpy()
            
            # Create result dictionary
            result = {
                "predicted_class": int(top_class.item()),
                "predicted_class_name": self.class_names[top_class.item()],
                "confidence": float(top_prob.item()),
                "top3_predictions": [
                    {
                        "class": int(idx),
                        "class_name": self.class_names[idx],
                        "probability": float(prob)
                    } for idx, prob in zip(top3_indices, top3_probs)
                ]
            }
            
            return result
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict diseases for multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result["image_path"] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e)
                })
        
        return results
    
    def load_model(self, model_path: str):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the saved model file
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load class names if available
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        
        logger.info(f"Loaded model from {model_path} with {len(self.class_names)} classes")
    
    def set_class_names(self, class_names: List[str]):
        """
        Set the class names for the model.
        
        Args:
            class_names: List of class names
        """
        if len(class_names) != self.num_classes:
            logger.warning(f"Number of class names ({len(class_names)}) "
                          f"does not match number of classes ({self.num_classes})")
        
        self.class_names = class_names


def create_data_loaders(image_dir: str, annotations_file: Optional[str] = None,
                       batch_size: int = 32, train_split: float = 0.8,
                       num_workers: int = 4):
    """
    Create DataLoaders for training and validation.
    
    Args:
        image_dir: Directory containing images
        annotations_file: Path to annotations file (optional)
        batch_size: Batch size for DataLoader
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes for DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = DiseaseImageDataset(
        image_dir=image_dir,
        annotations_file=annotations_file,
        transform=None,  # We'll apply transforms later
        is_training=True
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_disease_model(image_dir: str, annotations_file: Optional[str] = None,
                       num_classes: int = 5, model_type: str = "resnet",
                       batch_size: int = 32, epochs: int = 10,
                       save_path: str = "disease_model.pth"):
    """
    Train a disease detection model.
    
    Args:
        image_dir: Directory containing images
        annotations_file: Path to annotations file (optional)
        num_classes: Number of disease classes
        model_type: Type of model to use
        batch_size: Batch size for training
        epochs: Number of training epochs
        save_path: Path to save the trained model
        
    Returns:
        Trained model and training history
    """
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        image_dir=image_dir,
        annotations_file=annotations_file,
        batch_size=batch_size
    )
    
    # Initialize model
    model = DiseaseDetectionModel(
        num_classes=num_classes,
        model_type=model_type
    )
    
    # Train model
    history = model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path=save_path
    )
    
    return model, history