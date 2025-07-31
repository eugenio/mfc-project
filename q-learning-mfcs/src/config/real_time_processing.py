"""
Real-time Data Streaming and Processing Framework

This module provides comprehensive real-time data streaming and processing capabilities
for MFC systems, including data acquisition, stream processing, and real-time analytics.

Classes:
- DataStream: Base class for data streaming
- MFCDataStream: Specialized MFC data stream with sensor integration
- StreamProcessor: Real-time data processing engine
- RealTimeAnalyzer: Real-time analytics and monitoring
- StreamBuffer: Circular buffer for efficient data storage
- AlertSystem: Real-time alerting and notification system

Features:
- Multi-threaded data acquisition from multiple sensors
- Real-time data processing with configurable pipelines
- Streaming statistics and trend analysis
- Anomaly detection and alerting
- Data quality monitoring and validation
- Configurable data retention policies
- Integration with control systems
- Real-time visualization support

Literature References:
1. Akidau, T., et al. (2018). "Streaming Systems: The What, Where, When, and How of Large-Scale Data Processing"
2. Chen, C., & Zhang, J. (2014). "Data-intensive applications, challenges, techniques and technologies: A survey on Big Data"
3. Bifet, A., et al. (2018). "Machine Learning for Data Streams"
4. Gama, J. (2010). "Knowledge Discovery from Data Streams"
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import warnings
import time
import threading
from queue import Queue, Empty
from collections import deque

# Async and threading support
try:
    import asyncio  # noqa: F401
    HAS_ASYNCIO = True
except ImportError:
    HAS_ASYNCIO = False
    warnings.warn("Asyncio not available. Some async features will be limited.")

# Statistical processing
try:
    from scipy import stats
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Some signal processing features will be limited.")

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose  # noqa: F401
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: F401
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("Statsmodels not available. Some time series features will be limited.")

# Anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available. Anomaly detection will be limited.")


class StreamingMode(Enum):
    """Data streaming modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"
    CONTINUOUS = "continuous"


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataPoint:
    """Individual data point in a stream."""
    timestamp: datetime
    sensor_id: str
    value: float
    quality: DataQuality = DataQuality.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'sensor_id': self.sensor_id,
            'value': self.value,
            'quality': self.quality.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataPoint':
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            sensor_id=data['sensor_id'],
            value=data['value'],
            quality=DataQuality(data.get('quality', 'good')),
            metadata=data.get('metadata', {})
        )


@dataclass
class StreamingStats:
    """Streaming statistics for a data stream."""
    sensor_id: str
    window_size: int
    
    # Basic statistics
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    
    # Trend analysis
    trend_slope: float = 0.0
    trend_r_squared: float = 0.0
    
    # Quality metrics
    quality_score: float = 1.0
    missing_data_rate: float = 0.0
    outlier_rate: float = 0.0
    
    # Timing
    last_update: datetime = field(default_factory=datetime.now)
    update_frequency: float = 0.0
    
    def update(self, value: float, quality: DataQuality = DataQuality.GOOD):
        """Update statistics with new value."""
        self.count += 1
        
        # Update basic statistics using Welford's online algorithm
        if self.count == 1:
            self.mean = value
            self.variance = 0.0
        else:
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.variance += delta * delta2
        
        # Update min/max
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        
        # Update quality metrics
        if quality != DataQuality.GOOD:
            self.quality_score = 0.9 * self.quality_score  # Exponential decay
        else:
            self.quality_score = min(1.0, self.quality_score + 0.01)
        
        # Update timing
        now = datetime.now()
        if hasattr(self, 'last_update') and self.last_update:
            time_diff = (now - self.last_update).total_seconds()
            if time_diff > 0:
                # Exponential moving average of update frequency
                new_freq = 1.0 / time_diff
                self.update_frequency = 0.9 * self.update_frequency + 0.1 * new_freq
        
        self.last_update = now
    
    def get_std(self) -> float:
        """Get standard deviation."""
        if self.count < 2:
            return 0.0
        return np.sqrt(self.variance / (self.count - 1))


@dataclass
class Alert:
    """Real-time alert."""
    timestamp: datetime
    level: AlertLevel
    sensor_id: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'sensor_id': self.sensor_id,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'metadata': self.metadata
        }


class StreamBuffer:
    """Circular buffer for efficient streaming data storage."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize stream buffer.
        
        Args:
            max_size: Maximum number of data points to store
        """
        self.max_size = max_size
        self.buffer: deque[DataPoint] = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.total_points = 0
    
    def add(self, data_point: DataPoint):
        """Add data point to buffer."""
        with self.lock:
            self.buffer.append(data_point)
            self.total_points += 1
    
    def get_recent(self, n: int = 100) -> List[DataPoint]:
        """Get n most recent data points."""
        with self.lock:
            return list(self.buffer)[-n:]
    
    def get_time_window(self, start_time: datetime, 
                       end_time: Optional[datetime] = None) -> List[DataPoint]:
        """Get data points within time window."""
        if end_time is None:
            end_time = datetime.now()
        
        with self.lock:
            return [dp for dp in self.buffer 
                   if start_time <= dp.timestamp <= end_time]
    
    def get_sensor_data(self, sensor_id: str, n: int = 100) -> List[DataPoint]:
        """Get recent data for specific sensor."""
        with self.lock:
            sensor_data = [dp for dp in self.buffer if dp.sensor_id == sensor_id]
            return sensor_data[-n:]
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)


class DataStream(ABC):
    """Abstract base class for data streams."""
    
    def __init__(self, stream_id: str, buffer_size: int = 10000):
        """
        Initialize data stream.
        
        Args:
            stream_id: Unique identifier for the stream
            buffer_size: Size of data buffer
        """
        self.stream_id = stream_id
        self.buffer = StreamBuffer(buffer_size)
        self.is_active = False
        self.logger = logging.getLogger(__name__)
        self.callbacks: List[Callable[[DataPoint], None]] = []
        self.stats: Dict[str, Any] = {}
    
    @abstractmethod
    def start(self):
        """Start the data stream."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the data stream."""
        pass
    
    def add_callback(self, callback: Callable[[DataPoint], None]):
        """Add callback function for new data points."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[DataPoint], None]):
        """Remove callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self, data_point: DataPoint):
        """Notify all callbacks of new data point."""
        for callback in self.callbacks:
            try:
                callback(data_point)
            except Exception as e:
                self.logger.warning(f"Callback error: {e}")
    
    def get_stats(self, sensor_id: str) -> Optional[StreamingStats]:
        """Get streaming statistics for sensor."""
        return self.stats.get(sensor_id)
    
    def update_stats(self, data_point: DataPoint):
        """Update streaming statistics."""
        sensor_id = data_point.sensor_id
        
        if sensor_id not in self.stats:
            self.stats[sensor_id] = StreamingStats(
                sensor_id=sensor_id,
                window_size=1000  # Default window size
            )
        
        self.stats[sensor_id].update(data_point.value, data_point.quality)


class MFCDataStream(DataStream):
    """Specialized data stream for MFC sensor data."""
    
    def __init__(self, stream_id: str, sensor_config: Dict[str, Any],
                 sampling_rate: float = 1.0, buffer_size: int = 10000):
        """
        Initialize MFC data stream.
        
        Args:
            stream_id: Unique identifier for the stream
            sensor_config: Configuration for sensors
            sampling_rate: Data sampling rate in Hz
            buffer_size: Size of data buffer
        """
        super().__init__(stream_id, buffer_size)
        self.sensor_config = sensor_config
        self.sampling_rate = sampling_rate
        self.sampling_interval = 1.0 / sampling_rate
        self.thread = None
        self.stop_event = threading.Event()
    
    def start(self):
        """Start the MFC data stream."""
        if not self.is_active:
            self.is_active = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._data_acquisition_loop)
            self.thread.daemon = True
            self.thread.start()
            self.logger.info(f"Started MFC data stream: {self.stream_id}")
    
    def stop(self):
        """Stop the MFC data stream."""
        if self.is_active:
            self.is_active = False
            self.stop_event.set()
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
            self.logger.info(f"Stopped MFC data stream: {self.stream_id}")
    
    def _data_acquisition_loop(self):
        """Main data acquisition loop."""
        while not self.stop_event.is_set():
            try:
                # Simulate data acquisition from multiple sensors
                for sensor_id, config in self.sensor_config.items():
                    data_point = self._read_sensor(sensor_id, config)
                    
                    # Add to buffer and update statistics
                    self.buffer.add(data_point)
                    self.update_stats(data_point)
                    
                    # Notify callbacks
                    self._notify_callbacks(data_point)
                
                # Wait for next sampling interval
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in data acquisition: {e}")
                time.sleep(0.1)  # Brief pause before retrying
    
    def _read_sensor(self, sensor_id: str, config: Dict[str, Any]) -> DataPoint:
        """
        Simulate reading from a sensor.
        
        In a real implementation, this would interface with actual hardware.
        """
        # Simulate sensor reading with some noise
        base_value = config.get('base_value', 0.0)
        noise_level = config.get('noise_level', 0.1)
        drift = config.get('drift', 0.0)
        
        # Add time-based drift
        time_factor = time.time() % 3600  # Hourly cycle
        drift_component = drift * np.sin(2 * np.pi * time_factor / 3600)
        
        # Generate value with noise
        value = base_value + drift_component + np.random.normal(0, noise_level)
        
        # Simulate occasional quality issues
        quality = DataQuality.GOOD
        if np.random.random() < config.get('error_rate', 0.01):
            quality = DataQuality.POOR
            value = np.nan
        
        return DataPoint(
            timestamp=datetime.now(),
            sensor_id=sensor_id,
            value=value,
            quality=quality,
            metadata={'config': config}
        )


class StreamProcessor:
    """Real-time data processing engine."""
    
    def __init__(self, processing_config: Dict[str, Any]):
        """
        Initialize stream processor.
        
        Args:
            processing_config: Configuration for processing pipeline
        """
        self.config = processing_config
        self.processors: List[Any] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize processing pipeline
        self._setup_processors()
    
    def _setup_processors(self):
        """Setup processing pipeline based on configuration."""
        pipeline_config = self.config.get('pipeline', [])
        
        for stage_config in pipeline_config:
            processor_type = stage_config.get('type')
            
            if processor_type == 'smoothing':
                self.processors.append(self._create_smoothing_processor(stage_config))
            elif processor_type == 'outlier_detection':
                self.processors.append(self._create_outlier_processor(stage_config))
            elif processor_type == 'trend_analysis':
                self.processors.append(self._create_trend_processor(stage_config))
            elif processor_type == 'anomaly_detection':
                self.processors.append(self._create_anomaly_processor(stage_config))
    
    def process(self, data_points: List[DataPoint]) -> List[DataPoint]:
        """
        Process a batch of data points through the pipeline.
        
        Args:
            data_points: List of data points to process
            
        Returns:
            Processed data points
        """
        processed_data = data_points.copy()
        
        for processor in self.processors:
            try:
                processed_data = processor(processed_data)
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
        
        return processed_data
    
    def _create_smoothing_processor(self, config: Dict[str, Any]) -> Callable:
        """Create smoothing processor."""
        method = config.get('method', 'moving_average')
        window_size = config.get('window_size', 5)
        
        def smooth_processor(data_points: List[DataPoint]) -> List[DataPoint]:
            if len(data_points) < window_size:
                return data_points
            
            # Group by sensor
            sensor_data: Dict[str, List[DataPoint]] = {}
            for dp in data_points:
                if dp.sensor_id not in sensor_data:
                    sensor_data[dp.sensor_id] = []
                sensor_data[dp.sensor_id].append(dp)
            
            processed_points = []
            
            for sensor_id, points in sensor_data.items():
                values = np.array([dp.value for dp in points])
                
                if method == 'moving_average':
                    smoothed = self._moving_average(values, window_size)
                elif method == 'savgol' and HAS_SCIPY:
                    smoothed = savgol_filter(values, window_size, 3)
                else:
                    smoothed = values
                
                # Create new data points with smoothed values
                for i, point in enumerate(points):
                    new_point = DataPoint(
                        timestamp=point.timestamp,
                        sensor_id=point.sensor_id,
                        value=smoothed[i] if not np.isnan(smoothed[i]) else point.value,
                        quality=point.quality,
                        metadata={**point.metadata, 'smoothed': True}
                    )
                    processed_points.append(new_point)
            
            return processed_points
        
        return smooth_processor
    
    def _create_outlier_processor(self, config: Dict[str, Any]) -> Callable:
        """Create outlier detection processor."""
        method = config.get('method', 'iqr')
        threshold = config.get('threshold', 3.0)
        
        def outlier_processor(data_points: List[DataPoint]) -> List[DataPoint]:
            # Group by sensor
            sensor_data: Dict[str, List[DataPoint]] = {}
            for dp in data_points:
                if dp.sensor_id not in sensor_data:
                    sensor_data[dp.sensor_id] = []
                sensor_data[dp.sensor_id].append(dp)
            
            processed_points = []
            
            for sensor_id, points in sensor_data.items():
                values = np.array([dp.value for dp in points if not np.isnan(dp.value)])
                
                if len(values) < 5:  # Need minimum data for outlier detection
                    processed_points.extend(points)
                    continue
                
                # Detect outliers
                if method == 'iqr':
                    q25, q75 = np.percentile(values, [25, 75])
                    iqr = q75 - q25
                    lower_bound = q25 - threshold * iqr
                    upper_bound = q75 + threshold * iqr
                    outlier_mask = (values < lower_bound) | (values > upper_bound)
                
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(values)) if HAS_SCIPY else np.abs((values - np.mean(values)) / np.std(values))
                    outlier_mask = z_scores > threshold
                
                else:
                    outlier_mask = np.zeros(len(values), dtype=bool)
                
                # Mark outliers
                valid_idx = 0
                for point in points:
                    if not np.isnan(point.value):
                        if outlier_mask[valid_idx]:
                            # Mark as poor quality
                            new_point = DataPoint(
                                timestamp=point.timestamp,
                                sensor_id=point.sensor_id,
                                value=point.value,
                                quality=DataQuality.POOR,
                                metadata={**point.metadata, 'outlier': True}
                            )
                        else:
                            new_point = point
                        valid_idx += 1
                    else:
                        new_point = point
                    
                    processed_points.append(new_point)
            
            return processed_points
        
        return outlier_processor
    
    def _create_trend_processor(self, config: Dict[str, Any]) -> Callable:
        """Create trend analysis processor."""
        window_size = config.get('window_size', 20)
        
        def trend_processor(data_points: List[DataPoint]) -> List[DataPoint]:
            # Group by sensor
            sensor_data: Dict[str, List[DataPoint]] = {}
            for dp in data_points:
                if dp.sensor_id not in sensor_data:
                    sensor_data[dp.sensor_id] = []
                sensor_data[dp.sensor_id].append(dp)
            
            processed_points = []
            
            for sensor_id, points in sensor_data.items():
                if len(points) < window_size:
                    processed_points.extend(points)
                    continue
                
                # Calculate trend for recent window
                recent_points = points[-window_size:]
                values = np.array([dp.value for dp in recent_points if not np.isnan(dp.value)])
                
                if len(values) < 3:
                    processed_points.extend(points)
                    continue
                
                # Linear regression for trend
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values) if HAS_SCIPY else (0, 0, 0, 1, 0)
                
                # Add trend information to metadata
                for point in points:
                    new_point = DataPoint(
                        timestamp=point.timestamp,
                        sensor_id=point.sensor_id,
                        value=point.value,
                        quality=point.quality,
                        metadata={
                            **point.metadata,
                            'trend_slope': slope,
                            'trend_r_squared': r_value**2,
                            'trend_p_value': p_value
                        }
                    )
                    processed_points.append(new_point)
            
            return processed_points
        
        return trend_processor
    
    def _create_anomaly_processor(self, config: Dict[str, Any]) -> Callable:
        """Create anomaly detection processor."""
        method = config.get('method', 'isolation_forest')
        contamination = config.get('contamination', 0.1)
        
        def anomaly_processor(data_points: List[DataPoint]) -> List[DataPoint]:
            if not HAS_SKLEARN:
                return data_points
            
            # Group by sensor
            sensor_data: Dict[str, List[DataPoint]] = {}
            for dp in data_points:
                if dp.sensor_id not in sensor_data:
                    sensor_data[dp.sensor_id] = []
                sensor_data[dp.sensor_id].append(dp)
            
            processed_points = []
            
            for sensor_id, points in sensor_data.items():
                values = np.array([dp.value for dp in points if not np.isnan(dp.value)])
                
                if len(values) < 10:  # Need minimum data for anomaly detection
                    processed_points.extend(points)
                    continue
                
                # Prepare features (can be extended to multivariate)
                X = values.reshape(-1, 1)
                
                # Detect anomalies
                if method == 'isolation_forest':
                    detector = IsolationForest(contamination=contamination, random_state=42)
                    anomaly_labels = detector.fit_predict(X)
                else:
                    anomaly_labels = np.ones(len(values))  # No anomalies
                
                # Mark anomalies
                valid_idx = 0
                for point in points:
                    if not np.isnan(point.value):
                        is_anomaly = anomaly_labels[valid_idx] == -1
                        new_point = DataPoint(
                            timestamp=point.timestamp,
                            sensor_id=point.sensor_id,
                            value=point.value,
                            quality=DataQuality.POOR if is_anomaly else point.quality,
                            metadata={**point.metadata, 'anomaly': is_anomaly}
                        )
                        valid_idx += 1
                    else:
                        new_point = point
                    
                    processed_points.append(new_point)
            
            return processed_points
        
        return anomaly_processor
    
    def _moving_average(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average."""
        if len(values) < window_size:
            return values
        
        # Pad the beginning with the first value
        padded = np.concatenate([np.full(window_size//2, values[0]), values, np.full(window_size//2, values[-1])])
        smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
        
        return smoothed


class RealTimeAnalyzer:
    """Real-time analytics and monitoring."""
    
    def __init__(self, analysis_config: Dict[str, Any]):
        """
        Initialize real-time analyzer.
        
        Args:
            analysis_config: Configuration for analysis
        """
        self.config = analysis_config
        self.alert_thresholds = analysis_config.get('alert_thresholds', {})
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def analyze_stream(self, data_stream: DataStream, 
                      analysis_window: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """
        Perform real-time analysis on data stream.
        
        Args:
            data_stream: Data stream to analyze
            analysis_window: Time window for analysis
            
        Returns:
            Analysis results
        """
        # Get recent data
        end_time = datetime.now()
        start_time = end_time - analysis_window
        recent_data = data_stream.buffer.get_time_window(start_time, end_time)
        
        if not recent_data:
            return {'status': 'no_data', 'timestamp': end_time.isoformat()}
        
        # Group by sensor
        sensor_data: Dict[str, List[DataPoint]] = {}
        for dp in recent_data:
            if dp.sensor_id not in sensor_data:
                sensor_data[dp.sensor_id] = []
            sensor_data[dp.sensor_id].append(dp)
        
        analysis_results = {
            'timestamp': end_time.isoformat(),
            'window_size': len(recent_data),
            'sensors': {}
        }
        
        # Analyze each sensor
        for sensor_id, points in sensor_data.items():
            sensor_analysis = self._analyze_sensor_data(sensor_id, points)
            analysis_results['sensors'][sensor_id] = sensor_analysis
            
            # Check for alerts
            self._check_alerts(sensor_id, sensor_analysis, points)
        
        return analysis_results
    
    def _analyze_sensor_data(self, sensor_id: str, data_points: List[DataPoint]) -> Dict[str, Any]:
        """Analyze data for a single sensor."""
        values = [dp.value for dp in data_points if not np.isnan(dp.value)]
        
        if not values:
            return {'status': 'no_valid_data'}
        
        values = np.array(values)
        
        analysis = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'quality_score': np.mean([1.0 if dp.quality == DataQuality.GOOD else 0.0 for dp in data_points])
        }
        
        # Trend analysis
        if len(values) >= 3:
            x = np.arange(len(values))
            if HAS_SCIPY:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                analysis['trend'] = {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                }
            else:
                # Simple trend estimation
                trend_slope = (values[-1] - values[0]) / len(values)
                analysis['trend'] = {
                    'slope': trend_slope,
                    'direction': 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable'
                }
        
        # Variability analysis
        if len(values) > 1:
            analysis['coefficient_of_variation'] = analysis['std'] / analysis['mean'] if analysis['mean'] != 0 else np.inf
        
        # Rate of change
        if len(values) >= 2:
            rate_of_change = np.diff(values)
            analysis['rate_of_change'] = {
                'mean': np.mean(rate_of_change),
                'std': np.std(rate_of_change),
                'max_increase': np.max(rate_of_change),
                'max_decrease': np.min(rate_of_change)
            }
        
        return analysis
    
    def _check_alerts(self, sensor_id: str, analysis: Dict[str, Any], 
                     data_points: List[DataPoint]):
        """Check for alert conditions."""
        if sensor_id not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[sensor_id]
        
        # Value-based alerts
        if 'min_value' in thresholds and analysis.get('min') < thresholds['min_value']:
            alert = Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                sensor_id=sensor_id,
                message=f"Value below minimum threshold: {analysis['min']:.3f} < {thresholds['min_value']}",
                value=analysis['min'],
                threshold=thresholds['min_value']
            )
            self._send_alert(alert)
        
        if 'max_value' in thresholds and analysis.get('max') > thresholds['max_value']:
            alert = Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                sensor_id=sensor_id,
                message=f"Value above maximum threshold: {analysis['max']:.3f} > {thresholds['max_value']}",
                value=analysis['max'],
                threshold=thresholds['max_value']
            )
            self._send_alert(alert)
        
        # Quality-based alerts
        if 'min_quality' in thresholds and analysis.get('quality_score', 1.0) < thresholds['min_quality']:
            alert = Alert(
                timestamp=datetime.now(),
                level=AlertLevel.ERROR,
                sensor_id=sensor_id,
                message=f"Data quality below threshold: {analysis['quality_score']:.3f} < {thresholds['min_quality']}",
                value=analysis['quality_score'],
                threshold=thresholds['min_quality']
            )
            self._send_alert(alert)
        
        # Trend-based alerts
        if 'max_trend_slope' in thresholds and 'trend' in analysis:
            slope = analysis['trend'].get('slope', 0)
            if abs(slope) > thresholds['max_trend_slope']:
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    sensor_id=sensor_id,
                    message=f"Rapid trend change detected: slope = {slope:.6f}",
                    value=slope,
                    threshold=thresholds['max_trend_slope']
                )
                self._send_alert(alert)
    
    def _send_alert(self, alert: Alert):
        """Send alert to all registered callbacks."""
        self.logger.warning(f"ALERT: {alert.message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")


class AlertSystem:
    """Real-time alerting and notification system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert system.
        
        Args:
            config: Alert system configuration
        """
        self.config = config
        self.alert_queue: Queue[Alert] = Queue()
        self.alert_history: deque[Alert] = deque(maxlen=config.get('max_history', 1000))
        self.is_active = False
        self.thread = None
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the alert system."""
        if not self.is_active:
            self.is_active = True
            self.thread = threading.Thread(target=self._alert_processing_loop)
            self.thread.daemon = True
            self.thread.start()
            self.logger.info("Alert system started")
    
    def stop(self):
        """Stop the alert system."""
        if self.is_active:
            self.is_active = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
            self.logger.info("Alert system stopped")
    
    def send_alert(self, alert: Alert):
        """Send alert through the system."""
        self.alert_queue.put(alert)
    
    def get_recent_alerts(self, n: int = 100) -> List[Alert]:
        """Get recent alerts."""
        return list(self.alert_history)[-n:]
    
    def _alert_processing_loop(self):
        """Main alert processing loop."""
        while self.is_active:
            try:
                # Get alert from queue (with timeout)
                alert = self.alert_queue.get(timeout=1.0)
                
                # Add to history
                self.alert_history.append(alert)
                
                # Process alert
                self._process_alert(alert)
                
                # Mark task as done
                self.alert_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
    
    def _process_alert(self, alert: Alert):
        """Process individual alert."""
        # Log alert
        level_map = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.ERROR: self.logger.error,
            AlertLevel.CRITICAL: self.logger.critical
        }
        
        log_func = level_map.get(alert.level, self.logger.info)
        log_func(f"Alert [{alert.sensor_id}]: {alert.message}")
        
        # Additional processing based on alert level
        if alert.level == AlertLevel.CRITICAL:
            self._handle_critical_alert(alert)
        elif alert.level == AlertLevel.ERROR:
            self._handle_error_alert(alert)
    
    def _handle_critical_alert(self, alert: Alert):
        """Handle critical alerts."""
        # In a real system, this might trigger emergency shutdowns,
        # send notifications to operators, etc.
        self.logger.critical(f"CRITICAL ALERT HANDLING: {alert.message}")
    
    def _handle_error_alert(self, alert: Alert):
        """Handle error alerts."""
        # In a real system, this might trigger automated responses,
        # send notifications, etc.
        self.logger.error(f"ERROR ALERT HANDLING: {alert.message}")


# Utility functions for real-time processing
def create_sample_mfc_config() -> Dict[str, Any]:
    """
    Create sample MFC sensor configuration.
    
    Returns:
        Sample sensor configuration
    """
    return {
        'power_sensor': {
            'base_value': 25.0,
            'noise_level': 0.5,
            'drift': 2.0,
            'error_rate': 0.005,
            'units': 'W'
        },
        'flow_rate_sensor': {
            'base_value': 15.0,
            'noise_level': 0.2,
            'drift': 1.0,
            'error_rate': 0.002,
            'units': 'mL/h'
        },
        'substrate_concentration': {
            'base_value': 12.5,
            'noise_level': 0.3,
            'drift': 0.5,
            'error_rate': 0.001,
            'units': 'mmol/L'
        },
        'biofilm_thickness': {
            'base_value': 30.0,
            'noise_level': 1.0,
            'drift': 3.0,
            'error_rate': 0.01,
            'units': 'μm'
        }
    }


def create_sample_processing_config() -> Dict[str, Any]:
    """
    Create sample processing configuration.
    
    Returns:
        Sample processing configuration
    """
    return {
        'pipeline': [
            {
                'type': 'outlier_detection',
                'method': 'iqr',
                'threshold': 2.5
            },
            {
                'type': 'smoothing',
                'method': 'moving_average',
                'window_size': 5
            },
            {
                'type': 'trend_analysis',
                'window_size': 20
            },
            {
                'type': 'anomaly_detection',
                'method': 'isolation_forest',
                'contamination': 0.05
            }
        ]
    }


def create_sample_alert_config() -> Dict[str, Any]:
    """
    Create sample alert configuration.
    
    Returns:
        Sample alert configuration
    """
    return {
        'alert_thresholds': {
            'power_sensor': {
                'min_value': 10.0,
                'max_value': 50.0,
                'min_quality': 0.8,
                'max_trend_slope': 0.1
            },
            'flow_rate_sensor': {
                'min_value': 5.0,
                'max_value': 30.0,
                'min_quality': 0.9,
                'max_trend_slope': 0.05
            },
            'substrate_concentration': {
                'min_value': 5.0,
                'max_value': 25.0,
                'min_quality': 0.85,
                'max_trend_slope': 0.02
            },
            'biofilm_thickness': {
                'min_value': 10.0,
                'max_value': 60.0,
                'min_quality': 0.7,
                'max_trend_slope': 0.2
            }
        },
        'max_history': 1000
    }