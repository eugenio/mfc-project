"""
Long-term Data Storage and Analysis for MFC Stability Studies

Comprehensive data management system for storing, indexing, and analyzing
long-term MFC operational data. Provides efficient storage, querying, and
analysis capabilities for stability studies and degradation tracking.

Created: 2025-07-28
"""
import numpy as np
import pandas as pd
import sqlite3
import h5py
import json
import gzip
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
from contextlib import contextmanager
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

from degradation_detector import DegradationPattern, DegradationType
from reliability_analyzer import ComponentReliability


class DataType(Enum):
    """Types of data stored in the system."""
    SENSOR_DATA = "sensor_data"
    SYSTEM_METRICS = "system_metrics"
    PERFORMANCE_DATA = "performance_data"
    DEGRADATION_PATTERNS = "degradation_patterns"
    MAINTENANCE_RECORDS = "maintenance_records"
    RELIABILITY_METRICS = "reliability_metrics"
    ENVIRONMENTAL_DATA = "environmental_data"
    OPERATIONAL_LOGS = "operational_logs"

class StorageFormat(Enum):
    """Storage formats for different data types."""
    HDF5 = "hdf5"
    SQLITE = "sqlite"
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class DataQuery:
    """Query parameters for data retrieval."""
    data_types: List[DataType]
    start_time: datetime
    end_time: datetime
    components: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    sampling_rate: Optional[str] = None  # '1H', '1D', etc.
    filters: Dict[str, Any] = field(default_factory=dict)
    aggregation: Optional[str] = None  # 'mean', 'max', 'min', 'std'

class DataSummary:
    """Summary statistics for stored data."""
    data_type: DataType
    start_time: datetime
    end_time: datetime
    record_count: int
    size_bytes: int
    metrics: List[str]
    components: List[str]
    sampling_intervals: List[str]


class AnalysisResult:
    """Result of data analysis operation."""
    analysis_type: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)

class LongTermDataManager:
    """
    Comprehensive long-term data storage and analysis system.
    
    Features:
    - Multi-format data storage (HDF5, SQLite, CSV, etc.)
    - Efficient querying and indexing
    - Automated data archiving and compression
    - Statistical analysis and trend detection
    - Integration with degradation and reliability analysis
    - Data quality monitoring and validation
    """
    
    def __init__(self, 
                 data_directory: str = "../data/stability_data",
                 max_memory_mb: int = 1000,
                 compression_level: int = 6,
                 auto_archive_days: int = 90):
        """
        Initialize the long-term data manager.
        
        Args:
            data_directory: Root directory for data storage
            max_memory_mb: Maximum memory usage for caching
            compression_level: Compression level (1-9)
            auto_archive_days: Days after which to auto-archive data
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_mb = max_memory_mb
        self.compression_level = compression_level
        self.auto_archive_days = auto_archive_days
        
        # Initialize storage backends
        self._init_storage_backends()
        
        # Data cache
        self._data_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_size_mb = 0
        
        # Metadata database
        self.metadata_db_path = self.data_directory / "metadata.db"
        self._init_metadata_db()
        
        # Storage format mapping
        self.format_mapping = {
            DataType.SENSOR_DATA: StorageFormat.HDF5,
            DataType.SYSTEM_METRICS: StorageFormat.PARQUET,
            DataType.PERFORMANCE_DATA: StorageFormat.PARQUET,
            DataType.DEGRADATION_PATTERNS: StorageFormat.JSON,
            DataType.MAINTENANCE_RECORDS: StorageFormat.SQLITE,
            DataType.RELIABILITY_METRICS: StorageFormat.JSON,
            DataType.ENVIRONMENTAL_DATA: StorageFormat.HDF5,
            DataType.OPERATIONAL_LOGS: StorageFormat.SQLITE
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Start background tasks
        self._start_background_tasks()
    
    def _init_storage_backends(self):
        """Initialize storage backend directories."""
        self.storage_paths = {
            StorageFormat.HDF5: self.data_directory / "hdf5",
            StorageFormat.SQLITE: self.data_directory / "sqlite",
            StorageFormat.CSV: self.data_directory / "csv",
            StorageFormat.JSON: self.data_directory / "json",
            StorageFormat.PARQUET: self.data_directory / "parquet"
        }
        
        for path in self.storage_paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _init_metadata_db(self):
        """Initialize metadata database."""
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    record_count INTEGER,
                    size_bytes INTEGER,
                    storage_format TEXT,
                    compression TEXT,
                    created_at TEXT,
                    archived BOOLEAN DEFAULT FALSE,
                    checksum TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER,
                    metric_name TEXT NOT NULL,
                    component TEXT,
                    data_type TEXT,
                    min_value REAL,
                    max_value REAL,
                    mean_value REAL,
                    std_value REAL,
                    null_count INTEGER,
                    FOREIGN KEY (file_id) REFERENCES data_files (id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_files_type_time 
                ON data_files (data_type, start_time, end_time)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_metrics_name_component 
                ON data_metrics (metric_name, component)
            """)
    
    def store_data(self, 
                   data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
                   data_type: DataType,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store data in the appropriate format.
        
        Args:
            data: Data to store
            data_type: Type of data
            metadata: Additional metadata
            
        Returns:
            File ID for the stored data
        """
        # Convert data to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Determine storage format
        storage_format = self.format_mapping.get(data_type, StorageFormat.PARQUET)
        
        # Generate filename
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type.value}_{timestamp_str}.{self._get_file_extension(storage_format)}"
        
        # Store data
        filepath = self.storage_paths[storage_format] / filename
        record_count, size_bytes = self._write_data_file(df, filepath, storage_format, data_type)
        
        # Update metadata database
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO data_files 
                (data_type, filename, start_time, end_time, record_count, size_bytes, 
                 storage_format, created_at, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data_type.value,
                filename,
                start_time.isoformat(),
                end_time.isoformat(),
                record_count,
                size_bytes,
                storage_format.value,
                datetime.now().isoformat(),
                self._calculate_checksum(filepath)
            ))
            
            file_id = cursor.lastrowid
            
            # Store metric metadata
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if column != 'timestamp':
                    values = df[column].dropna()
                    if len(values) > 0:
                        conn.execute("""
                            INSERT INTO data_metrics
                            (file_id, metric_name, data_type, min_value, max_value, 
                             mean_value, std_value, null_count)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            file_id,
                            column,
                            data_type.value,
                            float(values.min()),
                            float(values.max()),
                            float(values.mean()),
                            float(values.std()) if len(values) > 1 else 0.0,
                            int(df[column].isnull().sum())
                        ))
        
        self.logger.info(f"Stored {record_count} records of {data_type.value} in {filename}")
        return str(file_id)
    
    def _get_file_extension(self, storage_format: StorageFormat) -> str:
        """Get file extension for storage format."""
        extensions = {
            StorageFormat.HDF5: "h5",
            StorageFormat.SQLITE: "db",
            StorageFormat.CSV: "csv.gz",
            StorageFormat.JSON: "json.gz",
            StorageFormat.PARQUET: "parquet"
        }
        return extensions[storage_format]
    
    def _write_data_file(self, 
                        df: pd.DataFrame, 
                        filepath: Path, 
                        storage_format: StorageFormat,
                        data_type: DataType) -> Tuple[int, int]:
        """Write data file in specified format."""
        
        if storage_format == StorageFormat.HDF5:
            with h5py.File(filepath, 'w') as f:
                # Store as HDF5 dataset
                for column in df.columns:
                    if column == 'timestamp':
                        # Convert timestamp to Unix timestamp
                        timestamps = df[column].astype('datetime64[s]').astype('int64')
                        f.create_dataset(column, data=timestamps, compression='gzip')
                    elif df[column].dtype == 'object':
                        # String data
                        str_data = df[column].astype(str).values
                        f.create_dataset(column, data=str_data, compression='gzip')
                    else:
                        # Numeric data
                        f.create_dataset(column, data=df[column].values, compression='gzip')
                
                # Add metadata
                f.attrs['data_type'] = data_type.value
                f.attrs['record_count'] = len(df)
                f.attrs['created_at'] = datetime.now().isoformat()
        
        elif storage_format == StorageFormat.PARQUET:
            df.to_parquet(filepath, compression='snappy', index=False)
        
        elif storage_format == StorageFormat.CSV:
            with gzip.open(filepath, 'wt') as f:
                df.to_csv(f, index=False)
        
        elif storage_format == StorageFormat.JSON:
            data_dict = df.to_dict('records')
            with gzip.open(filepath, 'wt') as f:
                json.dump({
                    'data_type': data_type.value,
                    'created_at': datetime.now().isoformat(),
                    'records': data_dict
                }, f, default=str, indent=2)
        
        elif storage_format == StorageFormat.SQLITE:
            with sqlite3.connect(filepath) as conn:
                df.to_sql('data', conn, if_exists='replace', index=False)
        
        # Get file size
        size_bytes = filepath.stat().st_size
        record_count = len(df)
        
        return record_count, size_bytes
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        import hashlib
        
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()[:16]  # First 16 characters
    
    def query_data(self, query: DataQuery) -> pd.DataFrame:
        """
        Query data based on specified criteria.
        
        Args:
            query: Query parameters
            
        Returns:
            DataFrame with queried data
        """
        # Find relevant files from metadata
        with sqlite3.connect(self.metadata_db_path) as conn:
            # Build SQL query
            where_conditions = []
            params = []
            
            # Data type filter
            if query.data_types:
                placeholders = ','.join(['?' for _ in query.data_types])
                where_conditions.append(f"data_type IN ({placeholders})")
                params.extend([dt.value for dt in query.data_types])
            
            # Time range filter
            where_conditions.append("end_time >= ? AND start_time <= ?")
            params.extend([query.start_time.isoformat(), query.end_time.isoformat()])
            
            # Build full query
            where_clause = " AND ".join(where_conditions)
            sql_query = f"""
                SELECT * FROM data_files 
                WHERE {where_clause}
                ORDER BY start_time
            """
            
            file_metadata = pd.read_sql_query(sql_query, conn, params=params)
        
        if file_metadata.empty:
            return pd.DataFrame()
        
        # Load and combine data from files
        combined_data = []
        
        for _, file_info in file_metadata.iterrows():
            try:
                # Load data file
                data_type = DataType(file_info['data_type'])
                storage_format = StorageFormat(file_info['storage_format'])
                filepath = self.storage_paths[storage_format] / file_info['filename']
                
                df = self._read_data_file(filepath, storage_format, data_type)
                
                # Apply time filter
                if 'timestamp' in df.columns:
                    df = df[
                        (df['timestamp'] >= query.start_time) & 
                        (df['timestamp'] <= query.end_time)
                    ]
                
                # Apply component filter
                if query.components and 'component' in df.columns:
                    df = df[df['component'].isin(query.components)]
                
                # Apply metric filter
                if query.metrics:
                    available_metrics = [col for col in query.metrics if col in df.columns]
                    if available_metrics:
                        keep_columns = ['timestamp'] + available_metrics
                        if 'component' in df.columns:
                            keep_columns.append('component')
                        df = df[keep_columns]
                
                # Apply additional filters
                for filter_column, filter_value in query.filters.items():
                    if filter_column in df.columns:
                        if isinstance(filter_value, (list, tuple)):
                            df = df[df[filter_column].isin(filter_value)]
                        else:
                            df = df[df[filter_column] == filter_value]
                
                if not df.empty:
                    combined_data.append(df)
                
            except Exception as e:
                self.logger.warning(f"Error loading file {file_info['filename']}: {e}")
                continue
        
        if not combined_data:
            return pd.DataFrame()
        
        # Combine all data
        result_df = pd.concat(combined_data, ignore_index=True)
        
        # Sort by timestamp
        if 'timestamp' in result_df.columns:
            result_df = result_df.sort_values('timestamp')
        
        # Apply sampling rate
        if query.sampling_rate and 'timestamp' in result_df.columns:
            result_df = result_df.set_index('timestamp')
            result_df = result_df.resample(query.sampling_rate).agg(
                query.aggregation if query.aggregation else 'mean'
            ).reset_index()
        
        return result_df
    
    def _read_data_file(self, 
                       filepath: Path, 
                       storage_format: StorageFormat,
                       data_type: DataType) -> pd.DataFrame:
        """Read data file in specified format."""
        
        if storage_format == StorageFormat.HDF5:
            data_dict = {}
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    if key == 'timestamp':
                        # Convert Unix timestamp back to datetime
                        timestamps = f[key][:]
                        data_dict[key] = pd.to_datetime(timestamps, unit='s')
                    else:
                        data_dict[key] = f[key][:]
            
            return pd.DataFrame(data_dict)
        
        elif storage_format == StorageFormat.PARQUET:
            return pd.read_parquet(filepath)
        
        elif storage_format == StorageFormat.CSV:
            with gzip.open(filepath, 'rt') as f:
                df = pd.read_csv(f)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        elif storage_format == StorageFormat.JSON:
            with gzip.open(filepath, 'rt') as f:
                data = json.load(f)
                df = pd.DataFrame(data['records'])
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        elif storage_format == StorageFormat.SQLITE:
            with sqlite3.connect(filepath) as conn:
                df = pd.read_sql_query("SELECT * FROM data", conn)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        else:
            raise ValueError(f"Unsupported storage format: {storage_format}")
    
    def get_data_summary(self, data_types: Optional[List[DataType]] = None) -> List[DataSummary]:
        """Get summary of stored data."""
        with sqlite3.connect(self.metadata_db_path) as conn:
            if data_types:
                placeholders = ','.join(['?' for _ in data_types])
                query = f"""
                    SELECT df.*, GROUP_CONCAT(DISTINCT dm.metric_name) as metrics,
                           GROUP_CONCAT(DISTINCT dm.component) as components
                    FROM data_files df
                    LEFT JOIN data_metrics dm ON df.id = dm.file_id
                    WHERE df.data_type IN ({placeholders})
                    GROUP BY df.id
                    ORDER BY df.start_time
                """
                params = [dt.value for dt in data_types]
            else:
                query = """
                    SELECT df.*, GROUP_CONCAT(DISTINCT dm.metric_name) as metrics,
                           GROUP_CONCAT(DISTINCT dm.component) as components
                    FROM data_files df
                    LEFT JOIN data_metrics dm ON df.id = dm.file_id
                    GROUP BY df.id
                    ORDER BY df.start_time
                """
                params = []
            
            file_data = pd.read_sql_query(query, conn, params=params)
        
        summaries = []
        for _, row in file_data.iterrows():
            summary = DataSummary(
                data_type=DataType(row['data_type']),
                start_time=pd.to_datetime(row['start_time']),
                end_time=pd.to_datetime(row['end_time']),
                record_count=row['record_count'],
                size_bytes=row['size_bytes'],
                metrics=row['metrics'].split(',') if row['metrics'] else [],
                components=row['components'].split(',') if row['components'] else [],
                sampling_intervals=[]  # Could be computed from data
            )
            summaries.append(summary)
        
        return summaries
    
    def analyze_trends(self, 
                      metrics: List[str],
                      components: Optional[List[str]] = None,
                      time_window_days: int = 30) -> AnalysisResult:
        """Analyze long-term trends in specified metrics."""
        
        # Query recent data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_window_days)
        
        query = DataQuery(
            data_types=[DataType.SENSOR_DATA, DataType.SYSTEM_METRICS, DataType.PERFORMANCE_DATA],
            start_time=start_time,
            end_time=end_time,
            components=components,
            metrics=metrics,
            sampling_rate='1H',  # Hourly sampling
            aggregation='mean'
        )
        
        data = self.query_data(query)
        
        if data.empty:
            return AnalysisResult(
                analysis_type="trend_analysis",
                parameters={"metrics": metrics, "time_window_days": time_window_days},
                results={"error": "No data available for analysis"},
                metadata={"data_points": 0}
            )
        
        # Perform trend analysis for each metric
        trend_results = {}
        
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            values = data[metric].dropna()
            timestamps = data.loc[values.index, 'timestamp']
            
            if len(values) < 10:
                continue
            
            # Linear trend analysis
            time_numeric = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
            
            # Change point detection
            change_points = self._detect_trend_changes(values.values)
            
            # Seasonal decomposition (simplified)
            seasonality = self._analyze_seasonality(values.values, timestamps)
            
            # Statistical tests
            stationarity_test = self._test_stationarity(values.values)
            
            trend_results[metric] = {
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value,
                'trend_significance': 'significant' if p_value < 0.05 else 'not_significant',
                'change_points': change_points,
                'seasonality_strength': seasonality,
                'stationarity_p_value': stationarity_test,
                'mean_value': float(values.mean()),
                'std_value': float(values.std()),
                'min_value': float(values.min()),
                'max_value': float(values.max()),
                'data_points': len(values)
            }
        
        return AnalysisResult(
            analysis_type="trend_analysis",
            parameters={
                "metrics": metrics,
                "components": components,
                "time_window_days": time_window_days,
                "sampling_rate": "1H"
            },
            results=trend_results,
            metadata={
                "total_data_points": len(data),
                "time_range": f"{start_time.isoformat()} to {end_time.isoformat()}",
                "analysis_duration_seconds": 0  # Could time the analysis
            }
        )
    
    def _detect_trend_changes(self, values: np.ndarray) -> List[int]:
        """Detect significant changes in trend."""
        if len(values) < 20:
            return []
        
        # Use a simple moving window approach
        window_size = len(values) // 10
        change_points = []
        
        for i in range(window_size, len(values) - window_size):
            # Compare trends before and after potential change point
            before = values[max(0, i - window_size):i]
            after = values[i:min(len(values), i + window_size)]
            
            if len(before) < 5 or len(after) < 5:
                continue
            
            # Calculate trend slopes
            slope_before = np.polyfit(range(len(before)), before, 1)[0]
            slope_after = np.polyfit(range(len(after)), after, 1)[0]
            
            # Check for significant change in slope
            if abs(slope_after - slope_before) > np.std(values) * 0.1:
                change_points.append(i)
        
        return change_points
    
    def _analyze_seasonality(self, values: np.ndarray, timestamps: pd.Series) -> float:
        """Analyze seasonality strength in time series."""
        if len(values) < 48:  # Need at least 2 days of hourly data
            return 0.0
        
        # Simple approach: check for daily patterns
        try:
            # Group by hour of day
            hour_groups = timestamps.dt.hour
            hourly_means = []
            
            for hour in range(24):
                hour_mask = hour_groups == hour
                if hour_mask.any():
                    hourly_means.append(np.mean(values[hour_mask]))
                else:
                    hourly_means.append(np.mean(values))
            
            # Calculate coefficient of variation for hourly means
            hourly_means = np.array(hourly_means)
            cv = np.std(hourly_means) / (np.mean(hourly_means) + 1e-10)
            
            return min(cv, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0
    
    def _test_stationarity(self, values: np.ndarray) -> float:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(values)
            return result[1]  # p-value
        except ImportError:
            # Fallback: variance ratio test
            if len(values) < 20:
                return 1.0
            
            mid = len(values) // 2
            var1 = np.var(values[:mid])
            var2 = np.var(values[mid:])
            
            if var1 == 0 or var2 == 0:
                return 1.0
            
            f_ratio = max(var1, var2) / min(var1, var2)
            # Convert F-ratio to approximate p-value
            return 1.0 / (1.0 + f_ratio)
    
    def export_analysis_report(self, 
                              analysis_results: List[AnalysisResult],
                              filepath: str):
        """Export analysis results to comprehensive report."""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'long_term_stability_analysis',
            'data_manager_config': {
                'data_directory': str(self.data_directory),
                'max_memory_mb': self.max_memory_mb,
                'auto_archive_days': self.auto_archive_days
            },
            'data_summary': [
                {
                    'data_type': summary.data_type.value,
                    'start_time': summary.start_time.isoformat(),
                    'end_time': summary.end_time.isoformat(),
                    'record_count': summary.record_count,
                    'size_mb': summary.size_bytes / (1024 * 1024),
                    'metrics': summary.metrics,
                    'components': summary.components
                }
                for summary in self.get_data_summary()
            ],
            'analysis_results': [
                {
                    'analysis_type': result.analysis_type,
                    'parameters': result.parameters,
                    'results': result.results,
                    'metadata': result.metadata,
                    'generated_at': result.generated_at.isoformat()
                }
                for result in analysis_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Analysis report exported to {filepath}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # Could implement:
        # - Automatic data archiving
        # - Cache cleanup
        # - Data integrity checks
        # - Compression optimization
        pass
    
    def cleanup_cache(self):
        """Clean up data cache to free memory."""
        with self._cache_lock:
            self._data_cache.clear()
            self._cache_size_mb = 0
        
        self.logger.info("Data cache cleared")
    
    def archive_old_data(self, archive_days: Optional[int] = None):
        """Archive old data files."""
        if archive_days is None:
            archive_days = self.auto_archive_days
        
        cutoff_date = datetime.now() - timedelta(days=archive_days)
        
        with sqlite3.connect(self.metadata_db_path) as conn:
            # Find files to archive
            old_files = pd.read_sql_query("""
                SELECT * FROM data_files 
                WHERE end_time < ? AND archived = FALSE
            """, conn, params=[cutoff_date.isoformat()])
            
            archived_count = 0
            for _, file_info in old_files.iterrows():
                try:
                    # Move file to archive directory
                    storage_format = StorageFormat(file_info['storage_format'])
                    current_path = self.storage_paths[storage_format] / file_info['filename']
                    
                    archive_dir = self.data_directory / "archive" / storage_format.value
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    archive_path = archive_dir / file_info['filename']
                    
                    if current_path.exists():
                        current_path.rename(archive_path)
                        
                        # Update metadata
                        conn.execute("""
                            UPDATE data_files SET archived = TRUE 
                            WHERE id = ?
                        """, (file_info['id'],))
                        
                        archived_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error archiving file {file_info['filename']}: {e}")
        
        self.logger.info(f"Archived {archived_count} data files older than {archive_days} days")