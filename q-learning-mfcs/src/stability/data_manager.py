#!/usr/bin/env python3
"""Data Manager Module for MFC Stability Analysis.

This module provides comprehensive data management capabilities for MFC stability
analysis, including data collection, validation, storage, retrieval, preprocessing,
and quality assessment for stability monitoring systems.

Author: MFC Analysis Team
Created: 2025-07-31
Last Modified: 2025-07-31
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
DataValue = Union[float, int, str, bool, datetime, np.number]
TimestampType = Union[datetime, pd.Timestamp, float, int]
DataRecord = dict[str, DataValue]
QueryResult = Union[pd.DataFrame, list[DataRecord], dict[str, Any]]

# Generic types
T = TypeVar("T")
DataStorageType = TypeVar("DataStorageType", bound="DataStorage")


class DataType(Enum):
    """Types of data in MFC systems."""

    SENSOR_READING = auto()
    PERFORMANCE_METRIC = auto()
    MAINTENANCE_LOG = auto()
    ENVIRONMENTAL_DATA = auto()
    CONTROL_PARAMETER = auto()
    ALARM_EVENT = auto()
    ANALYSIS_RESULT = auto()
    CONFIGURATION_DATA = auto()

    def __str__(self) -> str:
        return self.name.lower().replace("_", " ")


class DataQuality(Enum):
    """Data quality levels."""

    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    POOR = auto()
    INVALID = auto()

    @property
    def numeric_value(self) -> float:
        """Return numeric representation of quality level."""
        return {
            DataQuality.EXCELLENT: 1.0,
            DataQuality.GOOD: 0.8,
            DataQuality.ACCEPTABLE: 0.6,
            DataQuality.POOR: 0.4,
            DataQuality.INVALID: 0.0,
        }[self]


class StorageFormat(Enum):
    """Data storage formats."""

    CSV = auto()
    JSON = auto()
    PARQUET = auto()
    HDF5 = auto()
    SQLITE = auto()
    PICKLE = auto()

    @property
    def file_extension(self) -> str:
        """Return file extension for format."""
        return {
            StorageFormat.CSV: ".csv",
            StorageFormat.JSON: ".json",
            StorageFormat.PARQUET: ".parquet",
            StorageFormat.HDF5: ".h5",
            StorageFormat.SQLITE: ".db",
            StorageFormat.PICKLE: ".pkl",
        }[self]


@dataclass(frozen=True)
class DataSchema:
    """Schema definition for data validation."""

    schema_name: str
    schema_version: str
    fields: dict[str, dict[str, Any]]  # field_name -> {type, required, constraints}
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""

    def validate_record(self, record: DataRecord) -> tuple[bool, list[str]]:
        """Validate a data record against this schema."""
        errors = []

        # Check required fields
        for field_name, field_config in self.fields.items():
            if field_config.get("required", False) and field_name not in record:
                errors.append(f"Missing required field: {field_name}")
                continue

            if field_name not in record:
                continue

            value = record[field_name]
            expected_type = field_config.get("type")

            # Type validation
            if expected_type and not isinstance(value, expected_type):
                if not self._is_compatible_type(value, expected_type):
                    errors.append(
                        f"Field {field_name} has incorrect type. "
                        f"Expected {expected_type.__name__}, got {type(value).__name__}",
                    )

            # Constraint validation
            constraints = field_config.get("constraints", {})
            field_errors = self._validate_constraints(field_name, value, constraints)
            errors.extend(field_errors)

        return len(errors) == 0, errors

    def _is_compatible_type(self, value: Any, expected_type: type) -> bool:
        """Check if value is compatible with expected type."""
        if expected_type is float:
            return isinstance(value, int | float | np.number)
        if expected_type is int:
            return isinstance(value, int | np.integer)
        if expected_type is str:
            return isinstance(value, str)
        if expected_type == datetime:
            return isinstance(value, datetime | pd.Timestamp)
        return False

    def _validate_constraints(
        self,
        field_name: str,
        value: Any,
        constraints: dict[str, Any],
    ) -> list[str]:
        """Validate field constraints."""
        errors = []

        if "min_value" in constraints and value < constraints["min_value"]:
            errors.append(
                f"Field {field_name} below minimum value {constraints['min_value']}",
            )

        if "max_value" in constraints and value > constraints["max_value"]:
            errors.append(
                f"Field {field_name} above maximum value {constraints['max_value']}",
            )

        if "min_length" in constraints and len(str(value)) < constraints["min_length"]:
            errors.append(
                f"Field {field_name} below minimum length {constraints['min_length']}",
            )

        if "max_length" in constraints and len(str(value)) > constraints["max_length"]:
            errors.append(
                f"Field {field_name} above maximum length {constraints['max_length']}",
            )

        if (
            "allowed_values" in constraints
            and value not in constraints["allowed_values"]
        ):
            errors.append(
                f"Field {field_name} not in allowed values {constraints['allowed_values']}",
            )

        return errors


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report."""

    # Basic quality metrics
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_data_percentage: float = 0.0
    duplicate_records: int = 0

    # Temporal quality
    temporal_coverage: timedelta = field(default=timedelta())
    temporal_gaps: list[tuple[datetime, datetime]] = field(default_factory=list)
    sampling_rate_consistency: float = 0.0

    # Value quality
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    value_range_violations: int = 0
    type_consistency: float = 0.0

    # Statistical quality
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    timeliness_score: float = 0.0

    # Overall assessment
    overall_quality_score: float = 0.0
    quality_level: DataQuality = DataQuality.ACCEPTABLE

    # Issues and recommendations
    quality_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    data_source: str = ""
    assessment_period: timedelta = field(default=timedelta(hours=1))

    def __post_init__(self) -> None:
        """Compute derived quality metrics."""
        self._compute_overall_score()
        self._determine_quality_level()
        self._generate_recommendations()

    def _compute_overall_score(self) -> None:
        """Compute overall quality score from individual metrics."""
        scores = [
            self.completeness_score,
            self.accuracy_score,
            self.consistency_score,
            self.timeliness_score,
        ]

        # Weight different aspects of quality
        weights = [0.3, 0.3, 0.2, 0.2]
        self.overall_quality_score = sum(
            score * weight for score, weight in zip(scores, weights, strict=False)
        )

    def _determine_quality_level(self) -> None:
        """Determine quality level based on overall score."""
        if self.overall_quality_score >= 0.9:
            self.quality_level = DataQuality.EXCELLENT
        elif self.overall_quality_score >= 0.8:
            self.quality_level = DataQuality.GOOD
        elif self.overall_quality_score >= 0.6:
            self.quality_level = DataQuality.ACCEPTABLE
        elif self.overall_quality_score >= 0.3:
            self.quality_level = DataQuality.POOR
        else:
            self.quality_level = DataQuality.INVALID

    def _generate_recommendations(self) -> None:
        """Generate quality improvement recommendations."""
        if self.missing_data_percentage > 10:
            self.recommendations.append("Investigate and fix data collection gaps")

        if self.outlier_percentage > 5:
            self.recommendations.append("Review outlier detection and data validation")

        if self.duplicate_records > 0:
            self.recommendations.append("Implement duplicate detection and removal")

        if self.temporal_gaps:
            self.recommendations.append("Address temporal gaps in data collection")

        if self.sampling_rate_consistency < 0.8:
            self.recommendations.append("Improve sampling rate consistency")

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "missing_data_percentage": self.missing_data_percentage,
            "duplicate_records": self.duplicate_records,
            "temporal_coverage": self.temporal_coverage.total_seconds(),
            "temporal_gaps": [
                (start.isoformat(), end.isoformat())
                for start, end in self.temporal_gaps
            ],
            "sampling_rate_consistency": self.sampling_rate_consistency,
            "outlier_count": self.outlier_count,
            "outlier_percentage": self.outlier_percentage,
            "value_range_violations": self.value_range_violations,
            "type_consistency": self.type_consistency,
            "completeness_score": self.completeness_score,
            "accuracy_score": self.accuracy_score,
            "consistency_score": self.consistency_score,
            "timeliness_score": self.timeliness_score,
            "overall_quality_score": self.overall_quality_score,
            "quality_level": self.quality_level.name,
            "quality_issues": self.quality_issues,
            "recommendations": self.recommendations,
            "assessment_timestamp": self.assessment_timestamp.isoformat(),
            "data_source": self.data_source,
            "assessment_period": self.assessment_period.total_seconds(),
        }


class DataStorage(Protocol):
    """Protocol for data storage implementations."""

    def store_data(
        self,
        data: pd.DataFrame | list[DataRecord] | DataRecord,
        data_type: DataType,
        **kwargs: Any,
    ) -> bool:
        """Store data."""
        ...

    def retrieve_data(
        self,
        data_type: DataType,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """Retrieve data."""
        ...

    def delete_data(
        self,
        data_type: DataType,
        conditions: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Delete data."""
        ...


class BaseDataStorage(ABC):
    """Base class for data storage implementations."""

    def __init__(
        self,
        storage_path: str | Path,
        compression: bool = True,
        backup_enabled: bool = True,
    ) -> None:
        """Initialize data storage.

        Args:
            storage_path: Path for data storage
            compression: Enable data compression
            backup_enabled: Enable automatic backups

        """
        self.storage_path = Path(storage_path)
        self.compression = compression
        self.backup_enabled = backup_enabled
        self._setup_logging()
        self._ensure_storage_directory()

    def _setup_logging(self) -> None:
        """Setup logging for the storage system."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def _ensure_storage_directory(self) -> None:
        """Ensure storage directory exists."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def store_data(
        self,
        data: pd.DataFrame | list[DataRecord] | DataRecord,
        data_type: DataType,
        **kwargs: Any,
    ) -> bool:
        """Store data."""

    @abstractmethod
    def retrieve_data(
        self,
        data_type: DataType,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """Retrieve data."""

    @abstractmethod
    def delete_data(
        self,
        data_type: DataType,
        conditions: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Delete data."""

    def create_backup(self, backup_path: Path | None = None) -> bool:
        """Create backup of stored data."""
        if not self.backup_enabled:
            return False

        try:
            backup_path = (
                backup_path
                or self.storage_path.parent
                / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            backup_path.mkdir(parents=True, exist_ok=True)

            # Copy all data files to backup location
            import shutil

            shutil.copytree(self.storage_path, backup_path / "data", dirs_exist_ok=True)

            self.logger.info(f"Backup created at {backup_path}")
            return True

        except Exception as e:
            self.logger.exception(f"Backup creation failed: {e!s}")
            return False


class SQLiteDataStorage(BaseDataStorage):
    """SQLite-based data storage implementation."""

    def __init__(
        self,
        storage_path: str | Path,
        compression: bool = True,
        backup_enabled: bool = True,
        database_name: str = "mfc_stability_data.db",
    ) -> None:
        """Initialize SQLite data storage.

        Args:
            storage_path: Path for data storage
            compression: Enable data compression
            backup_enabled: Enable automatic backups
            database_name: Name of the SQLite database file

        """
        super().__init__(storage_path, compression, backup_enabled)
        self.db_path = self.storage_path / database_name
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Create main data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mfc_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        data_type TEXT NOT NULL,
                        component_id TEXT,
                        metric_name TEXT,
                        value REAL,
                        unit TEXT,
                        quality_score REAL DEFAULT 1.0,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create indices for better query performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON mfc_data(timestamp)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_type_timestamp
                    ON mfc_data(data_type, timestamp)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_component_metric
                    ON mfc_data(component_id, metric_name)
                """)

                # Create data quality log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_quality_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        assessment_timestamp DATETIME NOT NULL,
                        data_source TEXT,
                        quality_score REAL,
                        quality_level TEXT,
                        total_records INTEGER,
                        valid_records INTEGER,
                        issues TEXT,
                        recommendations TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
                self.logger.info("SQLite database initialized successfully")

        except Exception as e:
            self.logger.exception(f"Database initialization failed: {e!s}")
            raise

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with proper resource management."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def store_data(
        self,
        data: pd.DataFrame | list[DataRecord] | DataRecord,
        data_type: DataType,
        **kwargs: Any,
    ) -> bool:
        """Store data in SQLite database.

        Args:
            data: Data to store
            data_type: Type of data being stored
            **kwargs: Additional storage parameters

        Returns:
            True if storage successful, False otherwise

        """
        try:
            # Normalize data to list of records
            records = self._normalize_data_input(data)

            if not records:
                return True  # Nothing to store

            with self._get_connection() as conn:
                cursor = conn.cursor()

                for record in records:
                    # Extract standard fields
                    timestamp = record.get("timestamp", datetime.now())
                    component_id = record.get("component_id", "")
                    metric_name = record.get("metric_name", "")
                    value = record.get("value", 0.0)
                    unit = record.get("unit", "")
                    quality_score = record.get("quality_score", 1.0)

                    # Store additional fields as JSON metadata
                    metadata_fields = {
                        k: v
                        for k, v in record.items()
                        if k
                        not in [
                            "timestamp",
                            "component_id",
                            "metric_name",
                            "value",
                            "unit",
                            "quality_score",
                        ]
                    }
                    metadata_json = (
                        json.dumps(metadata_fields) if metadata_fields else None
                    )

                    cursor.execute(
                        """
                        INSERT INTO mfc_data
                        (timestamp, data_type, component_id, metric_name, value, unit, quality_score, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            (
                                timestamp.isoformat()
                                if isinstance(timestamp, datetime)
                                else timestamp
                            ),
                            data_type.name,
                            component_id,
                            metric_name,
                            (
                                float(value)
                                if value is not None
                                and isinstance(value, int | float | np.number)
                                else None
                            ),
                            unit,
                            (
                                float(quality_score)
                                if isinstance(quality_score, int | float | np.number)
                                else 0.0
                            ),
                            metadata_json,
                        ),
                    )

                conn.commit()
                self.logger.info(
                    f"Stored {len(records)} records of type {data_type.name}",
                )
                return True

        except Exception as e:
            self.logger.exception(f"Data storage failed: {e!s}")
            return False

    def retrieve_data(
        self,
        data_type: DataType,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """Retrieve data from SQLite database.

        Args:
            data_type: Type of data to retrieve
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            **kwargs: Additional query parameters

        Returns:
            Query results as DataFrame

        """
        try:
            with self._get_connection() as conn:
                # Build query
                query = "SELECT * FROM mfc_data WHERE data_type = ?"
                params = [data_type.name]

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                # Add additional filters
                component_id = kwargs.get("component_id")
                if component_id:
                    query += " AND component_id = ?"
                    params.append(component_id)

                metric_name = kwargs.get("metric_name")
                if metric_name:
                    query += " AND metric_name = ?"
                    params.append(metric_name)

                # Add ordering
                query += " ORDER BY timestamp"

                # Add limit if specified
                limit = kwargs.get("limit")
                if limit:
                    query += f" LIMIT {int(limit)}"

                # Execute query
                df = pd.read_sql_query(query, conn, params=params)

                # Convert timestamp column to datetime
                if not df.empty and "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                self.logger.info(
                    f"Retrieved {len(df)} records of type {data_type.name}",
                )
                return df

        except Exception as e:
            self.logger.exception(f"Data retrieval failed: {e!s}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def delete_data(
        self,
        data_type: DataType,
        conditions: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Delete data from SQLite database.

        Args:
            data_type: Type of data to delete
            conditions: Deletion conditions
            **kwargs: Additional deletion parameters

        Returns:
            True if deletion successful, False otherwise

        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Build delete query
                query = "DELETE FROM mfc_data WHERE data_type = ?"
                params = [data_type.name]

                if conditions:
                    for key, value in conditions.items():
                        if key in ["timestamp", "component_id", "metric_name"]:
                            query += f" AND {key} = ?"
                            params.append(value)

                # Execute deletion
                cursor.execute(query, params)
                deleted_count = cursor.rowcount

                conn.commit()
                self.logger.info(
                    f"Deleted {deleted_count} records of type {data_type.name}",
                )
                return True

        except Exception as e:
            self.logger.exception(f"Data deletion failed: {e!s}")
            return False

    def _normalize_data_input(
        self,
        data: pd.DataFrame | list[DataRecord] | DataRecord,
    ) -> list[DataRecord]:
        """Normalize data input to list of records."""
        if isinstance(data, pd.DataFrame):
            return data.to_dict("records")
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        msg = f"Unsupported data type: {type(data)}"
        raise ValueError(msg)


class MFCDataManager:
    """Main data manager for MFC stability analysis."""

    def __init__(
        self,
        storage: DataStorage | None = None,
        schema_registry: dict[DataType, DataSchema] | None = None,
        quality_thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialize MFC data manager.

        Args:
            storage: Data storage implementation
            schema_registry: Registry of data schemas
            quality_thresholds: Quality assessment thresholds

        """
        self.storage = storage or SQLiteDataStorage("./data/mfc_stability")
        self.schema_registry = schema_registry or self._create_default_schemas()
        self.quality_thresholds = (
            quality_thresholds or self._create_default_thresholds()
        )
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the data manager."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_default_schemas(self) -> dict[DataType, DataSchema]:
        """Create default data schemas for MFC data types."""
        schemas = {}

        # Sensor reading schema
        schemas[DataType.SENSOR_READING] = DataSchema(
            schema_name="sensor_reading",
            schema_version="1.0",
            fields={
                "timestamp": {"type": datetime, "required": True},
                "component_id": {"type": str, "required": True},
                "metric_name": {"type": str, "required": True},
                "value": {"type": float, "required": True},
                "unit": {"type": str, "required": False},
                "quality_score": {
                    "type": float,
                    "required": False,
                    "constraints": {"min_value": 0.0, "max_value": 1.0},
                },
            },
            description="Schema for sensor readings and measurements",
        )

        # Performance metric schema
        schemas[DataType.PERFORMANCE_METRIC] = DataSchema(
            schema_name="performance_metric",
            schema_version="1.0",
            fields={
                "timestamp": {"type": datetime, "required": True},
                "metric_name": {"type": str, "required": True},
                "value": {"type": float, "required": True},
                "baseline_value": {"type": float, "required": False},
                "threshold_value": {"type": float, "required": False},
                "status": {"type": str, "required": False},
            },
            description="Schema for performance metrics and KPIs",
        )

        return schemas

    def _create_default_thresholds(self) -> dict[str, float]:
        """Create default quality assessment thresholds."""
        return {
            "completeness_threshold": 0.95,
            "accuracy_threshold": 0.90,
            "consistency_threshold": 0.85,
            "timeliness_threshold": 0.80,
            "outlier_threshold": 5.0,  # Percentage
            "missing_data_threshold": 10.0,  # Percentage
        }

    def store_data(
        self,
        data: pd.DataFrame | list[DataRecord] | DataRecord,
        data_type: DataType,
        validate: bool = True,
        **kwargs: Any,
    ) -> tuple[bool, list[str]]:
        """Store data with optional validation.

        Args:
            data: Data to store
            data_type: Type of data being stored
            validate: Whether to validate data before storage
            **kwargs: Additional storage parameters

        Returns:
            Tuple of (success, validation_errors)

        """
        validation_errors = []

        try:
            # Validate data if requested and schema exists
            if validate and data_type in self.schema_registry:
                schema = self.schema_registry[data_type]
                records = self._normalize_data_for_validation(data)

                for i, record in enumerate(records):
                    is_valid, errors = schema.validate_record(record)
                    if not is_valid:
                        validation_errors.extend(
                            [f"Record {i}: {error}" for error in errors],
                        )

                # Stop if validation failed and we have errors
                if validation_errors:
                    self.logger.warning(
                        f"Data validation failed with {len(validation_errors)} errors",
                    )
                    return False, validation_errors

            # Store data
            success = self.storage.store_data(data, data_type, **kwargs)

            if success:
                self.logger.info(f"Successfully stored data of type {data_type.name}")
            else:
                self.logger.error(f"Failed to store data of type {data_type.name}")

            return success, validation_errors

        except Exception as e:
            self.logger.exception(f"Data storage operation failed: {e!s}")
            return False, [str(e)]

    def retrieve_data(
        self,
        data_type: DataType,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """Retrieve data from storage.

        Args:
            data_type: Type of data to retrieve
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            **kwargs: Additional query parameters

        Returns:
            Retrieved data

        """
        try:
            result = self.storage.retrieve_data(
                data_type,
                start_time,
                end_time,
                **kwargs,
            )
            self.logger.info(f"Retrieved data of type {data_type.name}")
            return result

        except Exception as e:
            self.logger.exception(f"Data retrieval failed: {e!s}")
            return pd.DataFrame()

    def assess_data_quality(
        self,
        data: pd.DataFrame | DataType,
        time_range: tuple[datetime, datetime] | None = None,
        **kwargs: Any,
    ) -> DataQualityReport:
        """Assess quality of data.

        Args:
            data: Data to assess or data type to retrieve and assess
            time_range: Time range for assessment
            **kwargs: Additional assessment parameters

        Returns:
            Data quality assessment report

        """
        try:
            # Retrieve data if DataType provided
            if isinstance(data, DataType):
                start_time, end_time = time_range if time_range else (None, None)
                df = self.retrieve_data(data, start_time, end_time, **kwargs)
                if isinstance(df, pd.DataFrame):
                    data = df
                else:
                    return DataQualityReport(quality_level=DataQuality.INVALID)

            if not isinstance(data, pd.DataFrame):
                return DataQualityReport(quality_level=DataQuality.INVALID)

            # Initialize quality report
            report = DataQualityReport(
                total_records=len(data),
                data_source=kwargs.get("data_source", "unknown"),
            )

            if len(data) == 0:
                return report

            # Assess completeness
            report.missing_data_percentage = (
                data.isnull().sum().sum() / (len(data) * len(data.columns))
            ) * 100
            report.completeness_score = max(
                0.0,
                1.0 - report.missing_data_percentage / 100.0,
            )

            # Assess duplicates
            report.duplicate_records = len(data) - len(data.drop_duplicates())

            # Assess temporal quality if timestamp column exists
            if "timestamp" in data.columns:
                report = self._assess_temporal_quality(data, report)

            # Assess value quality for numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                report = self._assess_value_quality(data[numeric_columns], report)

            # Calculate overall scores
            report.accuracy_score = self._calculate_accuracy_score(data)
            report.consistency_score = self._calculate_consistency_score(data)
            report.timeliness_score = self._calculate_timeliness_score(data)

            # Determine valid/invalid records
            report.valid_records = len(data) - report.duplicate_records
            report.invalid_records = report.duplicate_records

            self.logger.info(
                f"Data quality assessment completed. Overall score: {report.overall_quality_score:.3f}",
            )
            return report

        except Exception as e:
            self.logger.exception(f"Data quality assessment failed: {e!s}")
            return DataQualityReport(quality_level=DataQuality.INVALID)

    def _assess_temporal_quality(
        self,
        data: pd.DataFrame,
        report: DataQualityReport,
    ) -> DataQualityReport:
        """Assess temporal aspects of data quality."""
        try:
            timestamps = pd.to_datetime(data["timestamp"])

            # Calculate temporal coverage
            if len(timestamps) > 1:
                report.temporal_coverage = timestamps.max() - timestamps.min()

                # Find temporal gaps (gaps > expected sampling interval)
                time_diffs = timestamps.diff().dropna()
                if len(time_diffs) > 0:
                    median_interval = time_diffs.median()
                    gap_threshold = median_interval * 3  # Gaps > 3x median interval

                    gaps = time_diffs[time_diffs > gap_threshold]
                    report.temporal_gaps = [
                        (timestamps.iloc[i - 1], timestamps.iloc[i]) for i in gaps.index
                    ]

                    # Sampling rate consistency
                    interval_cv = (
                        (time_diffs.std() / time_diffs.mean())
                        if time_diffs.mean() > 0
                        else 1.0
                    )
                    report.sampling_rate_consistency = max(0.0, 1.0 - interval_cv)

        except Exception as e:
            self.logger.warning(f"Temporal quality assessment failed: {e!s}")

        return report

    def _assess_value_quality(
        self,
        data: pd.DataFrame,
        report: DataQualityReport,
    ) -> DataQualityReport:
        """Assess value quality for numeric data."""
        try:
            # Outlier detection using IQR method
            outlier_counts = []

            for column in data.columns:
                if data[column].dtype in ["float64", "int64"]:
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = (
                        (data[column] < lower_bound) | (data[column] > upper_bound)
                    ).sum()
                    outlier_counts.append(outliers)

            if outlier_counts:
                report.outlier_count = sum(outlier_counts)
                report.outlier_percentage = (report.outlier_count / len(data)) * 100

        except Exception as e:
            self.logger.warning(f"Value quality assessment failed: {e!s}")

        return report

    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """Calculate accuracy score based on data characteristics."""
        try:
            # Use outlier percentage as proxy for accuracy
            outlier_penalty = min(
                0.5,
                len(data.select_dtypes(include=[np.number]).columns) * 0.1,
            )
            return max(0.0, 1.0 - outlier_penalty)
        except Exception:
            return 0.5

    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate consistency score based on data patterns."""
        try:
            # Check for consistent data types and patterns
            consistency_factors = []

            for column in data.columns:
                if data[column].dtype == "object":
                    # For string columns, check for consistent formatting
                    unique_ratio = len(data[column].unique()) / len(data[column])
                    consistency_factors.append(min(1.0, 1.0 - unique_ratio))
                # For numeric columns, check for consistent ranges
                elif data[column].std() > 0:
                    cv = data[column].std() / abs(data[column].mean())
                    consistency_factors.append(max(0.0, 1.0 - min(cv, 1.0)))
                else:
                    consistency_factors.append(1.0)

            return float(np.mean(consistency_factors)) if consistency_factors else 0.5

        except Exception:
            return 0.5

    def _calculate_timeliness_score(self, data: pd.DataFrame) -> float:
        """Calculate timeliness score based on data recency."""
        try:
            if "timestamp" in data.columns:
                timestamps = pd.to_datetime(data["timestamp"])
                latest_timestamp = timestamps.max()
                current_time = datetime.now()

                # Score based on how recent the latest data is
                time_diff = current_time - latest_timestamp
                hours_old = time_diff.total_seconds() / 3600

                # Full score if data is less than 1 hour old, decreasing score for older data
                return max(0.0, 1.0 - min(hours_old / 24.0, 1.0))  # 24 hours = 0 score
            return 1.0  # No timestamp info, assume timely

        except Exception:
            return 0.5

    def _normalize_data_for_validation(
        self,
        data: pd.DataFrame | list[DataRecord] | DataRecord,
    ) -> list[DataRecord]:
        """Normalize data for validation."""
        if isinstance(data, pd.DataFrame):
            return data.to_dict("records")
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        msg = f"Unsupported data type for validation: {type(data)}"
        raise ValueError(msg)

    def export_data(
        self,
        data_type: DataType,
        output_path: str | Path,
        format_type: StorageFormat = StorageFormat.CSV,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> bool:
        """Export data to external format.

        Args:
            data_type: Type of data to export
            output_path: Path for exported data
            format_type: Export format
            start_time: Start time for data export
            end_time: End time for data export
            **kwargs: Additional export parameters

        Returns:
            True if export successful, False otherwise

        """
        try:
            # Retrieve data
            data = self.retrieve_data(data_type, start_time, end_time, **kwargs)

            if isinstance(data, pd.DataFrame) and not data.empty:
                output_file = Path(output_path).with_suffix(format_type.file_extension)

                if format_type == StorageFormat.CSV:
                    data.to_csv(output_file, index=False)
                elif format_type == StorageFormat.JSON:
                    data.to_json(output_file, orient="records", date_format="iso")
                elif format_type == StorageFormat.PARQUET:
                    data.to_parquet(output_file)
                elif format_type == StorageFormat.PICKLE:
                    data.to_pickle(output_file)
                else:
                    msg = f"Unsupported export format: {format_type}"
                    raise ValueError(msg)

                self.logger.info(f"Data exported to {output_file}")
                return True
            self.logger.warning("No data to export")
            return False

        except Exception as e:
            self.logger.exception(f"Data export failed: {e!s}")
            return False


# Factory functions and utilities
def create_mfc_data_manager(
    storage_path: str = "./data/mfc_stability",
    storage_type: str = "sqlite",
    **kwargs: Any,
) -> MFCDataManager:
    """Create MFC data manager with specified storage."""
    if storage_type.lower() == "sqlite":
        storage = SQLiteDataStorage(storage_path, **kwargs)
    else:
        msg = f"Unsupported storage type: {storage_type}"
        raise ValueError(msg)

    return MFCDataManager(storage=storage)


def generate_sample_sensor_data(
    start_time: datetime,
    duration_hours: int = 24,
    sampling_interval_minutes: int = 5,
) -> pd.DataFrame:
    """Generate sample sensor data for testing."""
    np.random.seed(42)

    # Generate timestamps
    timestamps = pd.date_range(
        start=start_time,
        periods=duration_hours * 60 // sampling_interval_minutes,
        freq=f"{sampling_interval_minutes}min",
    )

    # Generate synthetic sensor data
    base_power = 20.0
    power_variation = 2.0 * np.sin(
        2 * np.pi * np.arange(len(timestamps)) / (24 * 60 / sampling_interval_minutes),
    )
    noise = np.random.normal(0, 0.5, len(timestamps))
    power_data = base_power + power_variation + noise

    # Create DataFrame
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "component_id": "MFC_001",
            "metric_name": "power_output",
            "value": power_data,
            "unit": "W",
            "quality_score": np.random.uniform(0.8, 1.0, len(timestamps)),
        },
    )


# Example usage and testing
def run_example_data_management() -> None:
    """Run example data management operations."""
    # Create data manager
    data_manager = create_mfc_data_manager()

    # Generate sample data
    start_time = datetime.now() - timedelta(hours=24)
    sample_data = generate_sample_sensor_data(start_time, duration_hours=24)

    # Store data
    success, errors = data_manager.store_data(
        sample_data,
        DataType.SENSOR_READING,
        validate=True,
    )

    if success:
        pass
    else:
        pass

    # Retrieve data
    retrieved_data = data_manager.retrieve_data(
        DataType.SENSOR_READING,
        start_time=start_time,
        end_time=datetime.now(),
    )

    # Assess data quality
    quality_report = data_manager.assess_data_quality(retrieved_data)

    if quality_report.recommendations:
        for _rec in quality_report.recommendations:
            pass

    # Export data
    export_success = data_manager.export_data(
        DataType.SENSOR_READING,
        "./data/exported_sensor_data",
        StorageFormat.CSV,
        start_time=start_time,
    )

    if export_success:
        pass
    else:
        pass


if __name__ == "__main__":
    run_example_data_management()
