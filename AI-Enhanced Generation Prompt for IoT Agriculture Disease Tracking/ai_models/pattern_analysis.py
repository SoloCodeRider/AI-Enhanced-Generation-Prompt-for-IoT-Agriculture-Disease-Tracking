"""Pattern Analysis Module

This module implements algorithms for detecting disease mutation patterns,
tracking deviations from baseline health conditions, and analyzing
spatial-temporal patterns in crop disease spread.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import entropy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpatialTemporalData:
    """Class for handling spatial-temporal data from IoT sensors"""
    
    def __init__(self, data_source: Union[str, pd.DataFrame]):
        """
        Initialize with either a file path or a DataFrame.
        
        Args:
            data_source: Path to CSV/JSON file or pandas DataFrame with sensor data
        """
        if isinstance(data_source, str):
            # Load data from file
            if data_source.endswith('.csv'):
                self.data = pd.read_csv(data_source)
            elif data_source.endswith('.json'):
                self.data = pd.read_json(data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        elif isinstance(data_source, pd.DataFrame):
            # Use provided DataFrame
            self.data = data_source
        else:
            raise TypeError("data_source must be a file path or pandas DataFrame")
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'sensor_id', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Sort by timestamp
        self.data = self.data.sort_values('timestamp')
        
        logger.info(f"Loaded spatial-temporal data with {len(self.data)} records")
        logger.info(f"Time range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        logger.info(f"Number of unique sensors: {self.data['sensor_id'].nunique()}")
    
    def get_time_slice(self, start_time: Union[str, datetime], 
                      end_time: Union[str, datetime]) -> pd.DataFrame:
        """
        Get data slice for a specific time range.
        
        Args:
            start_time: Start time (inclusive)
            end_time: End time (inclusive)
            
        Returns:
            DataFrame with data in the specified time range
        """
        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        # Filter data
        time_slice = self.data[
            (self.data['timestamp'] >= start_time) & 
            (self.data['timestamp'] <= end_time)
        ]
        
        return time_slice
    
    def get_spatial_slice(self, min_lat: float, max_lat: float, 
                         min_lon: float, max_lon: float) -> pd.DataFrame:
        """
        Get data slice for a specific geographic area.
        
        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            
        Returns:
            DataFrame with data in the specified geographic area
        """
        # Filter data
        spatial_slice = self.data[
            (self.data['latitude'] >= min_lat) & 
            (self.data['latitude'] <= max_lat) &
            (self.data['longitude'] >= min_lon) & 
            (self.data['longitude'] <= max_lon)
        ]
        
        return spatial_slice
    
    def get_sensor_data(self, sensor_ids: Union[str, List[str]]) -> pd.DataFrame:
        """
        Get data for specific sensors.
        
        Args:
            sensor_ids: Single sensor ID or list of sensor IDs
            
        Returns:
            DataFrame with data for the specified sensors
        """
        if isinstance(sensor_ids, str):
            sensor_ids = [sensor_ids]
        
        # Filter data
        sensor_data = self.data[self.data['sensor_id'].isin(sensor_ids)]
        
        return sensor_data
    
    def resample_time_series(self, freq: str = '1H', 
                            agg_funcs: Dict[str, str] = None) -> pd.DataFrame:
        """
        Resample time series data to a specified frequency.
        
        Args:
            freq: Frequency string (e.g., '1H' for hourly, '1D' for daily)
            agg_funcs: Dictionary mapping column names to aggregation functions
            
        Returns:
            Resampled DataFrame
        """
        # Default aggregation functions if not provided
        if agg_funcs is None:
            # Identify numeric columns for aggregation
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols 
                           if col not in ['latitude', 'longitude', 'sensor_id']]
            
            agg_funcs = {col: 'mean' for col in numeric_cols}
        
        # Group by sensor_id and resample
        resampled_dfs = []
        
        for sensor_id, group in self.data.groupby('sensor_id'):
            # Set timestamp as index for resampling
            group = group.set_index('timestamp')
            
            # Resample
            resampled = group.resample(freq).agg(agg_funcs)
            
            # Add sensor_id back as a column
            resampled['sensor_id'] = sensor_id
            
            # Reset index to make timestamp a column again
            resampled = resampled.reset_index()
            
            resampled_dfs.append(resampled)
        
        # Combine all resampled DataFrames
        if resampled_dfs:
            result = pd.concat(resampled_dfs, ignore_index=True)
            return result
        else:
            return pd.DataFrame()
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute basic statistics for the dataset.
        
        Returns:
            Dictionary with statistics
        """
        # Identify numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols 
                       if col not in ['latitude', 'longitude', 'sensor_id']]
        
        stats = {
            'record_count': len(self.data),
            'sensor_count': self.data['sensor_id'].nunique(),
            'time_range': {
                'start': self.data['timestamp'].min().isoformat(),
                'end': self.data['timestamp'].max().isoformat(),
                'duration_hours': (self.data['timestamp'].max() - 
                                  self.data['timestamp'].min()).total_seconds() / 3600
            },
            'spatial_range': {
                'min_lat': float(self.data['latitude'].min()),
                'max_lat': float(self.data['latitude'].max()),
                'min_lon': float(self.data['longitude'].min()),
                'max_lon': float(self.data['longitude'].max())
            },
            'numeric_columns': {}
        }
        
        # Compute statistics for each numeric column
        for col in numeric_cols:
            stats['numeric_columns'][col] = {
                'mean': float(self.data[col].mean()),
                'std': float(self.data[col].std()),
                'min': float(self.data[col].min()),
                'max': float(self.data[col].max()),
                'median': float(self.data[col].median())
            }
        
        return stats


class PatternDetector:
    """Base class for pattern detection algorithms"""
    
    def __init__(self, data: SpatialTemporalData):
        """
        Initialize with spatial-temporal data.
        
        Args:
            data: SpatialTemporalData object
        """
        self.data = data
        self.results = None
    
    def detect_patterns(self):
        """
        Detect patterns in the data.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement detect_patterns()")
    
    def visualize_results(self):
        """
        Visualize detection results.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement visualize_results()")
    
    def export_results(self, output_path: str):
        """
        Export detection results to a file.
        
        Args:
            output_path: Path to save results
        """
        if self.results is None:
            raise ValueError("No results to export. Run detect_patterns() first.")
        
        if output_path.endswith('.csv'):
            self.results.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            self.results.to_json(output_path, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported file format: {output_path}")
        
        logger.info(f"Exported results to {output_path}")


class AnomalyDetector(PatternDetector):
    """Anomaly detection for sensor data"""
    
    def __init__(self, data: SpatialTemporalData, 
                 features: List[str], 
                 method: str = 'dbscan'):
        """
        Initialize anomaly detector.
        
        Args:
            data: SpatialTemporalData object
            features: List of feature columns to use for anomaly detection
            method: Detection method ('dbscan', 'isolation_forest', 'lof')
        """
        super().__init__(data)
        self.features = features
        self.method = method
        
        # Validate features
        missing_features = [f for f in features if f not in data.data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
    
    def detect_patterns(self, **kwargs) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Args:
            **kwargs: Additional parameters for the detection method
            
        Returns:
            DataFrame with anomaly detection results
        """
        # Extract feature data
        feature_data = self.data.data[self.features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.mean())
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Detect anomalies using the specified method
        if self.method == 'dbscan':
            # Default parameters
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(scaled_features)
            
            # Anomalies are labeled as -1
            is_anomaly = labels == -1
            
        elif self.method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            # Default parameters
            contamination = kwargs.get('contamination', 0.05)
            random_state = kwargs.get('random_state', 42)
            
            # Run Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination, 
                random_state=random_state
            )
            labels = iso_forest.fit_predict(scaled_features)
            
            # Anomalies are labeled as -1
            is_anomaly = labels == -1
            
        elif self.method == 'lof':
            from sklearn.neighbors import LocalOutlierFactor
            
            # Default parameters
            n_neighbors = kwargs.get('n_neighbors', 20)
            contamination = kwargs.get('contamination', 0.05)
            
            # Run Local Outlier Factor
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors, 
                contamination=contamination
            )
            labels = lof.fit_predict(scaled_features)
            
            # Anomalies are labeled as -1
            is_anomaly = labels == -1
            
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Create results DataFrame
        results = self.data.data.copy()
        results['anomaly'] = is_anomaly
        results['anomaly_score'] = np.nan  # Placeholder for anomaly scores
        
        # Calculate anomaly scores based on distance to nearest non-anomaly point
        if self.method == 'dbscan':
            # For DBSCAN, use distance to nearest cluster center
            cluster_centers = {}
            for cluster_id in set(labels) - {-1}:  # Exclude anomalies
                cluster_points = scaled_features[labels == cluster_id]
                cluster_centers[cluster_id] = np.mean(cluster_points, axis=0)
            
            # Calculate distances to nearest cluster center
            for i, point in enumerate(scaled_features):
                if is_anomaly[i]:
                    # Find distance to nearest cluster center
                    min_dist = float('inf')
                    for center in cluster_centers.values():
                        dist = np.linalg.norm(point - center)
                        min_dist = min(min_dist, dist)
                    
                    results.loc[results.index[i], 'anomaly_score'] = min_dist
        
        # Store results
        self.results = results
        
        # Log summary
        anomaly_count = results['anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies out of {len(results)} records "
                   f"({anomaly_count/len(results)*100:.2f}%)")
        
        return results
    
    def visualize_results(self, output_path: Optional[str] = None):
        """
        Visualize anomaly detection results.
        
        Args:
            output_path: Path to save visualization (optional)
        """
        if self.results is None:
            raise ValueError("No results to visualize. Run detect_patterns() first.")
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Scatter plot of first two features with anomalies highlighted
        if len(self.features) >= 2:
            feature1, feature2 = self.features[:2]
            axs[0, 0].scatter(
                self.results[~self.results['anomaly']][feature1],
                self.results[~self.results['anomaly']][feature2],
                c='blue', label='Normal', alpha=0.5
            )
            axs[0, 0].scatter(
                self.results[self.results['anomaly']][feature1],
                self.results[self.results['anomaly']][feature2],
                c='red', label='Anomaly', alpha=0.7
            )
            axs[0, 0].set_xlabel(feature1)
            axs[0, 0].set_ylabel(feature2)
            axs[0, 0].set_title('Feature Space with Anomalies')
            axs[0, 0].legend()
        
        # 2. Time series plot with anomalies highlighted
        if 'timestamp' in self.results.columns:
            # Sort by timestamp
            time_series = self.results.sort_values('timestamp')
            
            # Plot first feature over time
            feature = self.features[0]
            axs[0, 1].plot(
                time_series['timestamp'],
                time_series[feature],
                'b-', alpha=0.5, label=feature
            )
            
            # Highlight anomalies
            anomalies = time_series[time_series['anomaly']]
            axs[0, 1].scatter(
                anomalies['timestamp'],
                anomalies[feature],
                c='red', label='Anomaly', alpha=0.7
            )
            
            axs[0, 1].set_xlabel('Time')
            axs[0, 1].set_ylabel(feature)
            axs[0, 1].set_title('Time Series with Anomalies')
            axs[0, 1].legend()
            plt.setp(axs[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. PCA visualization if more than 2 features
        if len(self.features) > 2:
            # Extract feature data
            feature_data = self.results[self.features].copy()
            feature_data = feature_data.fillna(feature_data.mean())
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(feature_data)
            
            # Plot PCA results
            axs[1, 0].scatter(
                pca_result[~self.results['anomaly'], 0],
                pca_result[~self.results['anomaly'], 1],
                c='blue', label='Normal', alpha=0.5
            )
            axs[1, 0].scatter(
                pca_result[self.results['anomaly'], 0],
                pca_result[self.results['anomaly'], 1],
                c='red', label='Anomaly', alpha=0.7
            )
            axs[1, 0].set_xlabel('Principal Component 1')
            axs[1, 0].set_ylabel('Principal Component 2')
            axs[1, 0].set_title('PCA Visualization with Anomalies')
            axs[1, 0].legend()
        
        # 4. Anomaly score distribution
        if 'anomaly_score' in self.results.columns:
            # Filter out NaN scores
            scores = self.results['anomaly_score'].dropna()
            
            if len(scores) > 0:
                axs[1, 1].hist(scores, bins=30, alpha=0.7)
                axs[1, 1].set_xlabel('Anomaly Score')
                axs[1, 1].set_ylabel('Frequency')
                axs[1, 1].set_title('Anomaly Score Distribution')
        
        # Adjust layout and save if output_path is provided
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
        
        plt.show()


class MutationDetector(PatternDetector):
    """Detect disease mutations and pattern changes over time"""
    
    def __init__(self, data: SpatialTemporalData, 
                 disease_features: List[str],
                 time_window: str = '1D',
                 overlap: str = '12H'):
        """
        Initialize mutation detector.
        
        Args:
            data: SpatialTemporalData object
            disease_features: List of features related to disease measurements
            time_window: Size of time window for analysis (e.g., '1D' for 1 day)
            overlap: Overlap between consecutive windows (e.g., '12H' for 12 hours)
        """
        super().__init__(data)
        self.disease_features = disease_features
        self.time_window = pd.Timedelta(time_window)
        self.overlap = pd.Timedelta(overlap)
        
        # Validate disease features
        missing_features = [f for f in disease_features if f not in data.data.columns]
        if missing_features:
            raise ValueError(f"Missing disease features in data: {missing_features}")
    
    def detect_patterns(self, n_clusters: int = 3, 
                       mutation_threshold: float = 0.3) -> pd.DataFrame:
        """
        Detect disease mutations by clustering disease patterns in time windows
        and tracking changes in cluster assignments.
        
        Args:
            n_clusters: Number of clusters for KMeans
            mutation_threshold: Threshold for considering a change as a mutation
            
        Returns:
            DataFrame with mutation detection results
        """
        # Get time range
        start_time = self.data.data['timestamp'].min()
        end_time = self.data.data['timestamp'].max()
        
        # Create time windows with overlap
        windows = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = current_start + self.time_window
            windows.append((current_start, current_end))
            current_start = current_start + self.time_window - self.overlap
        
        logger.info(f"Created {len(windows)} time windows for analysis")
        
        # Initialize results
        window_results = []
        
        # Process each time window
        for i, (window_start, window_end) in enumerate(windows):
            # Get data for this window
            window_data = self.data.get_time_slice(window_start, window_end)
            
            if len(window_data) < n_clusters:  # Skip if not enough data
                continue
            
            # Extract disease features
            feature_data = window_data[self.disease_features].copy()
            feature_data = feature_data.fillna(feature_data.mean())
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Calculate cluster centers and sizes
            centers = kmeans.cluster_centers_
            sizes = np.bincount(cluster_labels, minlength=n_clusters)
            
            # Store window results
            window_result = {
                'window_id': i,
                'start_time': window_start,
                'end_time': window_end,
                'data_points': len(window_data),
                'cluster_centers': centers,
                'cluster_sizes': sizes,
                'cluster_labels': cluster_labels,
                'sensor_ids': window_data['sensor_id'].values,
                'timestamps': window_data['timestamp'].values
            }
            
            window_results.append(window_result)
        
        # Detect mutations between consecutive windows
        mutations = []
        
        for i in range(1, len(window_results)):
            prev_window = window_results[i-1]
            curr_window = window_results[i]
            
            # Calculate distance between cluster centers
            prev_centers = prev_window['cluster_centers']
            curr_centers = curr_window['cluster_centers']
            
            # Calculate pairwise distances between all clusters
            distances = cdist(prev_centers, curr_centers)
            
            # Find minimum distance for each previous cluster
            min_distances = np.min(distances, axis=1)
            
            # Identify mutations (clusters that changed significantly)
            mutation_clusters = np.where(min_distances > mutation_threshold)[0]
            
            if len(mutation_clusters) > 0:
                # Calculate mutation severity based on distance and cluster size
                for cluster_idx in mutation_clusters:
                    cluster_size = prev_window['cluster_sizes'][cluster_idx]
                    distance = min_distances[cluster_idx]
                    
                    # Calculate mutation severity (normalized by threshold)
                    severity = (distance - mutation_threshold) / mutation_threshold
                    severity = min(1.0, max(0.0, severity))  # Clip to [0, 1]
                    
                    # Find affected sensors
                    affected_indices = np.where(prev_window['cluster_labels'] == cluster_idx)[0]
                    affected_sensors = prev_window['sensor_ids'][affected_indices]
                    
                    # Create mutation record
                    mutation = {
                        'start_time': prev_window['start_time'],
                        'end_time': curr_window['end_time'],
                        'cluster_id': int(cluster_idx),
                        'distance': float(distance),
                        'severity': float(severity),
                        'affected_sensors': affected_sensors.tolist(),
                        'affected_count': int(len(affected_sensors))
                    }
                    
                    mutations.append(mutation)
        
        # Convert mutations to DataFrame
        if mutations:
            mutation_df = pd.DataFrame(mutations)
            
            # Add spatial information
            mutation_df['center_lat'] = np.nan
            mutation_df['center_lon'] = np.nan
            mutation_df['radius_km'] = np.nan
            
            for i, row in mutation_df.iterrows():
                # Get affected sensors
                affected_sensors = row['affected_sensors']
                
                # Get locations of affected sensors
                sensor_data = self.data.data[
                    self.data.data['sensor_id'].isin(affected_sensors)
                ]
                
                if len(sensor_data) > 0:
                    # Calculate center point
                    center_lat = sensor_data['latitude'].mean()
                    center_lon = sensor_data['longitude'].mean()
                    
                    # Calculate radius (maximum distance from center)
                    max_dist_km = 0
                    for _, sensor in sensor_data.iterrows():
                        # Approximate distance using Euclidean distance
                        # (this is a simplification, use haversine for more accuracy)
                        dist = np.sqrt(
                            (sensor['latitude'] - center_lat)**2 + 
                            (sensor['longitude'] - center_lon)**2
                        ) * 111  # Rough conversion to km
                        
                        max_dist_km = max(max_dist_km, dist)
                    
                    mutation_df.at[i, 'center_lat'] = center_lat
                    mutation_df.at[i, 'center_lon'] = center_lon
                    mutation_df.at[i, 'radius_km'] = max_dist_km
            
            self.results = mutation_df
            
            logger.info(f"Detected {len(mutation_df)} mutations across {len(window_results)} time windows")
        else:
            self.results = pd.DataFrame()
            logger.info("No mutations detected")
        
        return self.results
    
    def visualize_results(self, output_path: Optional[str] = None):
        """
        Visualize mutation detection results.
        
        Args:
            output_path: Path to save visualization (optional)
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results to visualize or no mutations detected.")
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Timeline of mutations
        axs[0, 0].scatter(
            self.results['start_time'],
            self.results['severity'],
            c=self.results['severity'],
            cmap='YlOrRd',
            s=self.results['affected_count'] * 5,  # Size based on affected count
            alpha=0.7
        )
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Mutation Severity')
        axs[0, 0].set_title('Timeline of Disease Mutations')
        plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Spatial distribution of mutations
        if 'center_lat' in self.results.columns and not self.results['center_lat'].isna().all():
            scatter = axs[0, 1].scatter(
                self.results['center_lon'],
                self.results['center_lat'],
                c=self.results['severity'],
                cmap='YlOrRd',
                s=self.results['radius_km'] * 20,  # Size based on radius
                alpha=0.7
            )
            axs[0, 1].set_xlabel('Longitude')
            axs[0, 1].set_ylabel('Latitude')
            axs[0, 1].set_title('Spatial Distribution of Mutations')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axs[0, 1])
            cbar.set_label('Severity')
        
        # 3. Histogram of mutation severities
        axs[1, 0].hist(self.results['severity'], bins=10, alpha=0.7)
        axs[1, 0].set_xlabel('Mutation Severity')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].set_title('Distribution of Mutation Severities')
        
        # 4. Scatter plot of severity vs. affected count
        axs[1, 1].scatter(
            self.results['affected_count'],
            self.results['severity'],
            alpha=0.7
        )
        axs[1, 1].set_xlabel('Number of Affected Sensors')
        axs[1, 1].set_ylabel('Mutation Severity')
        axs[1, 1].set_title('Severity vs. Affected Area')
        
        # Adjust layout and save if output_path is provided
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
        
        plt.show()


class SpatialPatternAnalyzer(PatternDetector):
    """Analyze spatial patterns and hotspots in disease spread"""
    
    def __init__(self, data: SpatialTemporalData, 
                 disease_feature: str,
                 grid_size: float = 0.01):
        """
        Initialize spatial pattern analyzer.
        
        Args:
            data: SpatialTemporalData object
            disease_feature: Feature representing disease severity/presence
            grid_size: Size of grid cells for spatial analysis (in degrees)
        """
        super().__init__(data)
        self.disease_feature = disease_feature
        self.grid_size = grid_size
        
        # Validate disease feature
        if disease_feature not in data.data.columns:
            raise ValueError(f"Disease feature '{disease_feature}' not found in data")
    
    def detect_patterns(self, time_point: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Detect spatial patterns at a specific time point.
        If time_point is None, use the latest data.
        
        Args:
            time_point: Time point for analysis (optional)
            
        Returns:
            DataFrame with spatial pattern analysis results
        """
        # Get data for the specified time point
        if time_point is not None:
            # Convert string time to datetime if needed
            if isinstance(time_point, str):
                time_point = pd.to_datetime(time_point)
            
            # Get closest data points to the specified time
            data_df = self.data.data.copy()
            data_df['time_diff'] = abs(data_df['timestamp'] - time_point)
            latest_data = data_df.sort_values('time_diff').groupby('sensor_id').first().reset_index()
        else:
            # Use the latest data point for each sensor
            latest_data = self.data.data.sort_values('timestamp').groupby('sensor_id').last().reset_index()
        
        # Create spatial grid
        min_lat = latest_data['latitude'].min()
        max_lat = latest_data['latitude'].max()
        min_lon = latest_data['longitude'].min()
        max_lon = latest_data['longitude'].max()
        
        # Add small buffer
        lat_buffer = (max_lat - min_lat) * 0.05
        lon_buffer = (max_lon - min_lon) * 0.05
        
        min_lat -= lat_buffer
        max_lat += lat_buffer
        min_lon -= lon_buffer
        max_lon += lon_buffer
        
        # Create grid
        lat_bins = np.arange(min_lat, max_lat + self.grid_size, self.grid_size)
        lon_bins = np.arange(min_lon, max_lon + self.grid_size, self.grid_size)
        
        # Assign data points to grid cells
        latest_data['lat_bin'] = np.digitize(latest_data['latitude'], lat_bins) - 1
        latest_data['lon_bin'] = np.digitize(latest_data['longitude'], lon_bins) - 1
        
        # Aggregate disease values by grid cell
        grid_data = latest_data.groupby(['lat_bin', 'lon_bin']).agg({
            self.disease_feature: ['mean', 'std', 'count'],
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        grid_data.columns = ['_'.join(col).strip('_') for col in grid_data.columns.values]
        
        # Rename columns for clarity
        grid_data = grid_data.rename(columns={
            f"{self.disease_feature}_mean": "disease_mean",
            f"{self.disease_feature}_std": "disease_std",
            f"{self.disease_feature}_count": "sensor_count",
            "latitude_mean": "center_lat",
            "longitude_mean": "center_lon"
        })
        
        # Calculate grid cell boundaries
        grid_data['min_lat'] = lat_bins[grid_data['lat_bin']]
        grid_data['max_lat'] = lat_bins[grid_data['lat_bin'] + 1]
        grid_data['min_lon'] = lon_bins[grid_data['lon_bin']]
        grid_data['max_lon'] = lon_bins[grid_data['lon_bin'] + 1]
        
        # Identify hotspots (cells with high disease values)
        disease_threshold = grid_data['disease_mean'].mean() + grid_data['disease_mean'].std()
        grid_data['is_hotspot'] = grid_data['disease_mean'] > disease_threshold
        
        # Calculate spatial autocorrelation (simplified)
        # For each cell, calculate correlation with neighboring cells
        grid_data['spatial_correlation'] = np.nan
        
        for i, cell in grid_data.iterrows():
            # Find neighboring cells (adjacent in grid)
            neighbors = grid_data[
                ((grid_data['lat_bin'] == cell['lat_bin']) & 
                 (grid_data['lon_bin'].isin([cell['lon_bin']-1, cell['lon_bin']+1]))) |
                ((grid_data['lon_bin'] == cell['lon_bin']) & 
                 (grid_data['lat_bin'].isin([cell['lat_bin']-1, cell['lat_bin']+1])))
            ]
            
            if len(neighbors) > 0:
                # Calculate correlation as similarity to neighbors
                neighbor_mean = neighbors['disease_mean'].mean()
                cell_value = cell['disease_mean']
                
                # Normalize correlation to [-1, 1]
                max_diff = grid_data['disease_mean'].max() - grid_data['disease_mean'].min()
                if max_diff > 0:
                    correlation = 1 - (2 * abs(cell_value - neighbor_mean) / max_diff)
                else:
                    correlation = 0
                
                grid_data.at[i, 'spatial_correlation'] = correlation
        
        # Store results
        self.results = grid_data
        
        logger.info(f"Analyzed spatial patterns across {len(grid_data)} grid cells")
        logger.info(f"Identified {grid_data['is_hotspot'].sum()} hotspot cells")
        
        return grid_data
    
    def visualize_results(self, output_path: Optional[str] = None):
        """
        Visualize spatial pattern analysis results.
        
        Args:
            output_path: Path to save visualization (optional)
        """
        if self.results is None:
            raise ValueError("No results to visualize. Run detect_patterns() first.")
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Heatmap of disease values
        scatter = axs[0, 0].scatter(
            self.results['center_lon'],
            self.results['center_lat'],
            c=self.results['disease_mean'],
            cmap='YlOrRd',
            s=self.results['sensor_count'] * 10,  # Size based on sensor count
            alpha=0.7
        )
        axs[0, 0].set_xlabel('Longitude')
        axs[0, 0].set_ylabel('Latitude')
        axs[0, 0].set_title('Disease Intensity Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axs[0, 0])
        cbar.set_label(f'{self.disease_feature} (Mean)')
        
        # 2. Hotspot map
        axs[0, 1].scatter(
            self.results[~self.results['is_hotspot']]['center_lon'],
            self.results[~self.results['is_hotspot']]['center_lat'],
            c='blue',
            s=30,
            alpha=0.3,
            label='Normal'
        )
        axs[0, 1].scatter(
            self.results[self.results['is_hotspot']]['center_lon'],
            self.results[self.results['is_hotspot']]['center_lat'],
            c='red',
            s=100,
            alpha=0.7,
            label='Hotspot'
        )
        axs[0, 1].set_xlabel('Longitude')
        axs[0, 1].set_ylabel('Latitude')
        axs[0, 1].set_title('Disease Hotspots')
        axs[0, 1].legend()
        
        # 3. Spatial correlation map
        if 'spatial_correlation' in self.results.columns and not self.results['spatial_correlation'].isna().all():
            scatter = axs[1, 0].scatter(
                self.results['center_lon'],
                self.results['center_lat'],
                c=self.results['spatial_correlation'],
                cmap='coolwarm',
                s=50,
                alpha=0.7
            )
            axs[1, 0].set_xlabel('Longitude')
            axs[1, 0].set_ylabel('Latitude')
            axs[1, 0].set_title('Spatial Correlation')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axs[1, 0])
            cbar.set_label('Correlation')
        
        # 4. Sensor density map
        scatter = axs[1, 1].scatter(
            self.results['center_lon'],
            self.results['center_lat'],
            c=self.results['sensor_count'],
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        axs[1, 1].set_xlabel('Longitude')
        axs[1, 1].set_ylabel('Latitude')
        axs[1, 1].set_title('Sensor Density')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axs[1, 1])
        cbar.set_label('Sensor Count')
        
        # Adjust layout and save if output_path is provided
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
        
        plt.show()


class PatternAnalysisService:
    """Service for running multiple pattern analysis algorithms"""
    
    def __init__(self, data_source: Union[str, pd.DataFrame]):
        """
        Initialize the pattern analysis service.
        
        Args:
            data_source: Path to data file or pandas DataFrame
        """
        # Load data
        self.data = SpatialTemporalData(data_source)
        
        # Initialize detectors
        self.anomaly_detector = None
        self.mutation_detector = None
        self.spatial_analyzer = None
        
        logger.info(f"Initialized pattern analysis service with {len(self.data.data)} records")
    
    def detect_anomalies(self, features: List[str], method: str = 'dbscan', **kwargs) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Args:
            features: List of features to use for anomaly detection
            method: Detection method
            **kwargs: Additional parameters for the detection method
            
        Returns:
            DataFrame with anomaly detection results
        """
        self.anomaly_detector = AnomalyDetector(
            data=self.data,
            features=features,
            method=method
        )
        
        results = self.anomaly_detector.detect_patterns(**kwargs)
        return results
    
    def detect_mutations(self, disease_features: List[str], 
                        time_window: str = '1D',
                        overlap: str = '12H',
                        **kwargs) -> pd.DataFrame:
        """
        Detect disease mutations over time.
        
        Args:
            disease_features: List of features related to disease measurements
            time_window: Size of time window for analysis
            overlap: Overlap between consecutive windows
            **kwargs: Additional parameters for mutation detection
            
        Returns:
            DataFrame with mutation detection results
        """
        self.mutation_detector = MutationDetector(
            data=self.data,
            disease_features=disease_features,
            time_window=time_window,
            overlap=overlap
        )
        
        results = self.mutation_detector.detect_patterns(**kwargs)
        return results
    
    def analyze_spatial_patterns(self, disease_feature: str,
                               grid_size: float = 0.01,
                               time_point: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Analyze spatial patterns in disease spread.
        
        Args:
            disease_feature: Feature representing disease severity/presence
            grid_size: Size of grid cells for spatial analysis
            time_point: Time point for analysis (optional)
            
        Returns:
            DataFrame with spatial pattern analysis results
        """
        self.spatial_analyzer = SpatialPatternAnalyzer(
            data=self.data,
            disease_feature=disease_feature,
            grid_size=grid_size
        )
        
        results = self.spatial_analyzer.detect_patterns(time_point=time_point)
        return results
    
    def generate_comprehensive_report(self, output_dir: str):
        """
        Generate a comprehensive report with results from all detectors.
        
        Args:
            output_dir: Directory to save report files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report components
        report_components = []
        
        # 1. Basic statistics
        stats = self.data.compute_statistics()
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        report_components.append({
            'title': 'Dataset Statistics',
            'type': 'statistics',
            'file': 'statistics.json',
            'summary': f"Dataset contains {stats['record_count']} records from "
                      f"{stats['sensor_count']} sensors over "
                      f"{stats['time_range']['duration_hours']:.1f} hours."
        })
        
        # 2. Anomaly detection results
        if self.anomaly_detector and self.anomaly_detector.results is not None:
            # Save results
            self.anomaly_detector.export_results(
                os.path.join(output_dir, 'anomalies.csv')
            )
            
            # Generate visualization
            self.anomaly_detector.visualize_results(
                os.path.join(output_dir, 'anomaly_visualization.png')
            )
            
            anomaly_count = self.anomaly_detector.results['anomaly'].sum()
            report_components.append({
                'title': 'Anomaly Detection',
                'type': 'anomalies',
                'data_file': 'anomalies.csv',
                'visualization': 'anomaly_visualization.png',
                'summary': f"Detected {anomaly_count} anomalies out of "
                          f"{len(self.anomaly_detector.results)} records "
                          f"({anomaly_count/len(self.anomaly_detector.results)*100:.2f}%)."
            })
        
        # 3. Mutation detection results
        if self.mutation_detector and self.mutation_detector.results is not None:
            # Save results
            self.mutation_detector.export_results(
                os.path.join(output_dir, 'mutations.csv')
            )
            
            # Generate visualization if mutations were detected
            if len(self.mutation_detector.results) > 0:
                self.mutation_detector.visualize_results(
                    os.path.join(output_dir, 'mutation_visualization.png')
                )
                
                report_components.append({
                    'title': 'Disease Mutation Detection',
                    'type': 'mutations',
                    'data_file': 'mutations.csv',
                    'visualization': 'mutation_visualization.png',
                    'summary': f"Detected {len(self.mutation_detector.results)} "
                              f"disease mutations with average severity "
                              f"{self.mutation_detector.results['severity'].mean():.2f}."
                })
            else:
                report_components.append({
                    'title': 'Disease Mutation Detection',
                    'type': 'mutations',
                    'data_file': 'mutations.csv',
                    'summary': "No disease mutations detected in the dataset."
                })
        
        # 4. Spatial pattern analysis results
        if self.spatial_analyzer and self.spatial_analyzer.results is not None:
            # Save results
            self.spatial_analyzer.export_results(
                os.path.join(output_dir, 'spatial_patterns.csv')
            )
            
            # Generate visualization
            self.spatial_analyzer.visualize_results(
                os.path.join(output_dir, 'spatial_visualization.png')
            )
            
            hotspot_count = self.spatial_analyzer.results['is_hotspot'].sum()
            report_components.append({
                'title': 'Spatial Pattern Analysis',
                'type': 'spatial',
                'data_file': 'spatial_patterns.csv',
                'visualization': 'spatial_visualization.png',
                'summary': f"Analyzed {len(self.spatial_analyzer.results)} spatial grid cells "
                          f"and identified {hotspot_count} disease hotspots."
            })
        
        # Create main report file
        report = {
            'generated_at': datetime.now().isoformat(),
            'dataset_info': {
                'record_count': len(self.data.data),
                'sensor_count': self.data.data['sensor_id'].nunique(),
                'time_range': {
                    'start': self.data.data['timestamp'].min().isoformat(),
                    'end': self.data.data['timestamp'].max().isoformat()
                }
            },
            'components': report_components
        }
        
        with open(os.path.join(output_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated comprehensive report in {output_dir}")
        
        return report


def create_demo_analysis(output_dir: str = "demo_analysis"):
    """
    Create a demo analysis with synthetic data.
    
    Args:
        output_dir: Directory to save demo files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Time range: 7 days with hourly data
    start_time = datetime(2023, 1, 1)
    end_time = start_time + timedelta(days=7)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')
    
    # Sensor grid: 5x5 grid
    n_sensors = 25
    sensor_ids = [f"sensor_{i:03d}" for i in range(n_sensors)]
    
    # Sensor locations: grid pattern
    grid_size = 5
    base_lat, base_lon = 40.0, -100.0
    grid_spacing = 0.1
    
    sensor_locations = []
    for i in range(grid_size):
        for j in range(grid_size):
            sensor_locations.append({
                'sensor_id': sensor_ids[i * grid_size + j],
                'latitude': base_lat + i * grid_spacing,
                'longitude': base_lon + j * grid_spacing
            })
    
    # Create disease pattern: starts at one corner and spreads
    disease_center_lat = base_lat
    disease_center_lon = base_lon
    disease_spread_rate = 0.05  # degrees per day
    
    # Generate data for each timestamp and sensor
    data_records = []
    
    for t, timestamp in enumerate(timestamps):
        # Update disease center (spreading)
        days_elapsed = t / 24  # hours to days
        
        # On day 3, introduce a mutation (change in spread direction)
        if days_elapsed >= 3 and days_elapsed < 5:
            disease_center_lat += disease_spread_rate / 24  # per hour
            disease_center_lon += disease_spread_rate / 24 * 2  # faster spread in lon
        else:
            disease_center_lat += disease_spread_rate / 24 * 0.5  # slower in lat
            disease_center_lon += disease_spread_rate / 24
        
        for sensor in sensor_locations:
            # Calculate distance from disease center
            distance = np.sqrt(
                (sensor['latitude'] - disease_center_lat)**2 + 
                (sensor['longitude'] - disease_center_lon)**2
            )
            
            # Disease severity decreases with distance
            base_severity = max(0, 1 - distance / (grid_spacing * 2))
            
            # Add time-based pattern and randomness
            time_factor = np.sin(days_elapsed * np.pi / 3.5) * 0.2 + 0.8  # 7-day cycle
            random_factor = np.random.normal(0, 0.05)  # small random variation
            
            disease_severity = base_severity * time_factor + random_factor
            disease_severity = max(0, min(1, disease_severity))  # clip to [0, 1]
            
            # Environmental factors
            temperature = 20 + 5 * np.sin(days_elapsed * np.pi / 3.5) + np.random.normal(0, 1)
            humidity = 60 + 10 * np.cos(days_elapsed * np.pi / 3.5) + np.random.normal(0, 2)
            soil_moisture = 0.3 + 0.1 * np.sin(days_elapsed * np.pi / 7) + np.random.normal(0, 0.02)
            
            # Create record
            record = {
                'timestamp': timestamp,
                'sensor_id': sensor['sensor_id'],
                'latitude': sensor['latitude'],
                'longitude': sensor['longitude'],
                'disease_severity': disease_severity,
                'leaf_health': 1 - disease_severity,  # inverse of disease severity
                'temperature': temperature,
                'humidity': humidity,
                'soil_moisture': soil_moisture
            }
            
            # Add anomalies
            if np.random.random() < 0.01:  # 1% chance of anomaly
                record['disease_severity'] += np.random.choice([-0.5, 0.5])  # large deviation
                record['disease_severity'] = max(0, min(1, record['disease_severity']))  # clip
                record['leaf_health'] = 1 - record['disease_severity']  # update leaf health
            
            data_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data_records)
    
    # Save synthetic data
    df.to_csv(os.path.join(output_dir, 'synthetic_sensor_data.csv'), index=False)
    
    # Run analysis
    analysis_service = PatternAnalysisService(df)
    
    # 1. Detect anomalies
    anomalies = analysis_service.detect_anomalies(
        features=['disease_severity', 'temperature', 'humidity', 'soil_moisture'],
        method='dbscan',
        eps=0.3,
        min_samples=3
    )
    
    # 2. Detect mutations
    mutations = analysis_service.detect_mutations(
        disease_features=['disease_severity', 'leaf_health'],
        time_window='1D',
        overlap='12H',
        n_clusters=3,
        mutation_threshold=0.2
    )
    
    # 3. Analyze spatial patterns
    spatial_patterns = analysis_service.analyze_spatial_patterns(
        disease_feature='disease_severity',
        grid_size=0.05
    )
    
    # Generate comprehensive report
    report = analysis_service.generate_comprehensive_report(output_dir)
    
    logger.info(f"Created demo analysis in {output_dir}")
    logger.info(f"Generated {len(df)} synthetic data points for {n_sensors} sensors")
    
    return report