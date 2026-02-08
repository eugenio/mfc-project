"""Tests for data_manager module - comprehensive coverage."""
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stability.data_manager import (
    BaseDataStorage,
    DataQuality,
    DataQualityReport,
    DataSchema,
    DataType,
    MFCDataManager,
    SQLiteDataStorage,
    StorageFormat,
    create_mfc_data_manager,
    generate_sample_sensor_data,
    run_example_data_management,
)


# ---------------------------------------------------------------------------
# DataType enum
# ---------------------------------------------------------------------------
class TestDataType:
    def test_str_representation(self):
        assert str(DataType.SENSOR_READING) == "sensor reading"
        assert str(DataType.PERFORMANCE_METRIC) == "performance metric"
        assert str(DataType.ALARM_EVENT) == "alarm event"

    def test_all_members(self):
        assert len(DataType) == 8


# ---------------------------------------------------------------------------
# DataQuality enum
# ---------------------------------------------------------------------------
class TestDataQuality:
    def test_numeric_value(self):
        assert DataQuality.EXCELLENT.numeric_value == 1.0
        assert DataQuality.GOOD.numeric_value == 0.8
        assert DataQuality.ACCEPTABLE.numeric_value == 0.6
        assert DataQuality.POOR.numeric_value == 0.4
        assert DataQuality.INVALID.numeric_value == 0.0


# ---------------------------------------------------------------------------
# StorageFormat enum
# ---------------------------------------------------------------------------
class TestStorageFormat:
    def test_file_extensions(self):
        assert StorageFormat.CSV.file_extension == ".csv"
        assert StorageFormat.JSON.file_extension == ".json"
        assert StorageFormat.PARQUET.file_extension == ".parquet"
        assert StorageFormat.HDF5.file_extension == ".h5"
        assert StorageFormat.SQLITE.file_extension == ".db"
        assert StorageFormat.PICKLE.file_extension == ".pkl"


# ---------------------------------------------------------------------------
# DataSchema
# ---------------------------------------------------------------------------
class TestDataSchema:
    @pytest.fixture
    def schema(self):
        return DataSchema(
            schema_name="test_schema",
            schema_version="1.0",
            fields={
                "name": {"type": str, "required": True},
                "value": {
                    "type": float,
                    "required": True,
                    "constraints": {"min_value": 0.0, "max_value": 100.0},
                },
                "status": {
                    "type": str,
                    "required": False,
                    "constraints": {
                        "allowed_values": ["active", "inactive"],
                        "min_length": 3,
                        "max_length": 10,
                    },
                },
                "ts": {"type": datetime, "required": False},
            },
            description="A test schema",
        )

    def test_validate_record_valid(self, schema):
        record = {"name": "sensor1", "value": 50.0}
        valid, errors = schema.validate_record(record)
        assert valid
        assert errors == []

    def test_validate_record_missing_required(self, schema):
        record = {"value": 50.0}
        valid, errors = schema.validate_record(record)
        assert not valid
        assert any("Missing required field: name" in e for e in errors)

    def test_validate_record_wrong_type(self, schema):
        record = {"name": 123, "value": 50.0}
        valid, errors = schema.validate_record(record)
        assert not valid
        assert any("incorrect type" in e for e in errors)

    def test_validate_record_constraint_min(self, schema):
        record = {"name": "sensor1", "value": -5.0}
        valid, errors = schema.validate_record(record)
        assert not valid
        assert any("below minimum" in e for e in errors)

    def test_validate_record_constraint_max(self, schema):
        record = {"name": "sensor1", "value": 200.0}
        valid, errors = schema.validate_record(record)
        assert not valid
        assert any("above maximum" in e for e in errors)

    def test_validate_record_constraint_allowed_values(self, schema):
        record = {"name": "sensor1", "value": 10.0, "status": "unknown"}
        valid, errors = schema.validate_record(record)
        assert not valid
        assert any("not in allowed values" in e for e in errors)

    def test_validate_record_constraint_min_length(self, schema):
        record = {"name": "sensor1", "value": 10.0, "status": "ab"}
        valid, errors = schema.validate_record(record)
        assert not valid

    def test_validate_record_constraint_max_length(self, schema):
        record = {"name": "sensor1", "value": 10.0, "status": "a_very_long_string"}
        valid, errors = schema.validate_record(record)
        assert not valid

    def test_compatible_type_int_as_float(self, schema):
        record = {"name": "sensor1", "value": 50}
        valid, errors = schema.validate_record(record)
        assert valid

    def test_compatible_type_numpy_as_float(self, schema):
        record = {"name": "sensor1", "value": np.float64(50.0)}
        valid, errors = schema.validate_record(record)
        assert valid

    def test_compatible_type_numpy_int(self):
        schema = DataSchema(
            schema_name="t", schema_version="1",
            fields={"val": {"type": int, "required": True}},
        )
        assert schema._is_compatible_type(np.int64(5), int)

    def test_compatible_type_datetime_from_timestamp(self):
        schema = DataSchema(
            schema_name="t", schema_version="1",
            fields={"ts": {"type": datetime, "required": True}},
        )
        assert schema._is_compatible_type(pd.Timestamp.now(), datetime)

    def test_incompatible_type_returns_false(self, schema):
        assert not schema._is_compatible_type([], float)

    def test_optional_field_absent(self, schema):
        record = {"name": "sensor1", "value": 50.0}
        valid, _ = schema.validate_record(record)
        assert valid


# ---------------------------------------------------------------------------
# DataQualityReport
# ---------------------------------------------------------------------------
class TestDataQualityReport:
    def test_default_report(self):
        report = DataQualityReport()
        assert report.overall_quality_score == 0.0
        assert report.quality_level == DataQuality.INVALID

    def test_excellent_quality(self):
        report = DataQualityReport(
            completeness_score=1.0,
            accuracy_score=1.0,
            consistency_score=1.0,
            timeliness_score=1.0,
        )
        assert report.quality_level == DataQuality.EXCELLENT

    def test_good_quality(self):
        report = DataQualityReport(
            completeness_score=0.85,
            accuracy_score=0.85,
            consistency_score=0.85,
            timeliness_score=0.85,
        )
        assert report.quality_level == DataQuality.GOOD

    def test_acceptable_quality(self):
        report = DataQualityReport(
            completeness_score=0.7,
            accuracy_score=0.7,
            consistency_score=0.7,
            timeliness_score=0.7,
        )
        assert report.quality_level == DataQuality.ACCEPTABLE

    def test_poor_quality(self):
        report = DataQualityReport(
            completeness_score=0.4,
            accuracy_score=0.4,
            consistency_score=0.4,
            timeliness_score=0.3,
        )
        assert report.quality_level == DataQuality.POOR

    def test_recommendations_generated(self):
        report = DataQualityReport(
            missing_data_percentage=15.0,
            outlier_percentage=10.0,
            duplicate_records=5,
            temporal_gaps=[(datetime.now(), datetime.now())],
            sampling_rate_consistency=0.5,
        )
        assert len(report.recommendations) > 0

    def test_to_dict(self):
        now = datetime.now()
        report = DataQualityReport(
            total_records=100,
            temporal_gaps=[(now, now + timedelta(hours=1))],
        )
        d = report.to_dict()
        assert d["total_records"] == 100
        assert "quality_level" in d
        assert "assessment_timestamp" in d
        assert isinstance(d["temporal_gaps"], list)


# ---------------------------------------------------------------------------
# SQLiteDataStorage
# ---------------------------------------------------------------------------
class TestSQLiteDataStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        return SQLiteDataStorage(str(tmp_path / "test_storage"))

    def test_initialization(self, storage):
        assert storage.db_path.exists()

    def test_store_single_record(self, storage):
        record = {
            "timestamp": datetime.now(),
            "component_id": "comp1",
            "metric_name": "power",
            "value": 25.0,
            "unit": "W",
            "quality_score": 0.95,
        }
        result = storage.store_data(record, DataType.SENSOR_READING)
        assert result is True

    def test_store_list_of_records(self, storage):
        records = [
            {"timestamp": datetime.now(), "value": 20.0},
            {"timestamp": datetime.now(), "value": 21.0},
        ]
        result = storage.store_data(records, DataType.SENSOR_READING)
        assert result is True

    def test_store_dataframe(self, storage):
        df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "component_id": ["c1"],
            "metric_name": ["power"],
            "value": [10.0],
            "unit": ["W"],
            "quality_score": [0.9],
        })
        result = storage.store_data(df, DataType.SENSOR_READING)
        assert result is True

    def test_store_empty_data(self, storage):
        result = storage.store_data([], DataType.SENSOR_READING)
        assert result is True

    def test_store_with_none_value(self, storage):
        record = {"timestamp": datetime.now(), "value": None, "quality_score": "bad"}
        result = storage.store_data(record, DataType.SENSOR_READING)
        assert result is True

    def test_store_with_metadata_fields(self, storage):
        record = {
            "timestamp": datetime.now(),
            "value": 10.0,
            "extra_field": "hello",
        }
        result = storage.store_data(record, DataType.SENSOR_READING)
        assert result is True

    def test_retrieve_data(self, storage):
        storage.store_data(
            {"timestamp": datetime.now(), "value": 10.0},
            DataType.SENSOR_READING,
        )
        result = storage.retrieve_data(DataType.SENSOR_READING)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_retrieve_with_filters(self, storage):
        now = datetime.now()
        storage.store_data(
            {"timestamp": now, "component_id": "c1", "metric_name": "power", "value": 10.0},
            DataType.SENSOR_READING,
        )
        result = storage.retrieve_data(
            DataType.SENSOR_READING,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
            component_id="c1",
            metric_name="power",
            limit=10,
        )
        assert isinstance(result, pd.DataFrame)

    def test_retrieve_empty(self, storage):
        result = storage.retrieve_data(DataType.ALARM_EVENT)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_delete_data(self, storage):
        storage.store_data(
            {"timestamp": datetime.now(), "value": 10.0, "component_id": "c1"},
            DataType.SENSOR_READING,
        )
        result = storage.delete_data(
            DataType.SENSOR_READING,
            conditions={"component_id": "c1"},
        )
        assert result is True

    def test_delete_data_no_conditions(self, storage):
        storage.store_data(
            {"timestamp": datetime.now(), "value": 10.0},
            DataType.SENSOR_READING,
        )
        result = storage.delete_data(DataType.SENSOR_READING)
        assert result is True

    def test_normalize_unsupported_type(self, storage):
        with pytest.raises(ValueError, match="Unsupported data type"):
            storage._normalize_data_input(42)

    def test_store_failure(self, storage):
        with patch.object(storage, "_get_connection", side_effect=Exception("fail")):
            result = storage.store_data({"value": 1}, DataType.SENSOR_READING)
            assert result is False

    def test_retrieve_failure(self, storage):
        with patch.object(storage, "_get_connection", side_effect=Exception("fail")):
            result = storage.retrieve_data(DataType.SENSOR_READING)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_delete_failure(self, storage):
        with patch.object(storage, "_get_connection", side_effect=Exception("fail")):
            result = storage.delete_data(DataType.SENSOR_READING)
            assert result is False

    def test_connection_rollback_on_error(self, storage):
        with pytest.raises(RuntimeError):
            with storage._get_connection() as conn:
                raise RuntimeError("test error")

    def test_store_non_datetime_timestamp(self, storage):
        record = {"timestamp": "2025-01-01T00:00:00", "value": 5.0}
        result = storage.store_data(record, DataType.SENSOR_READING)
        assert result is True


# ---------------------------------------------------------------------------
# BaseDataStorage
# ---------------------------------------------------------------------------
class TestBaseDataStorage:
    def test_create_backup(self, tmp_path):
        storage = SQLiteDataStorage(str(tmp_path / "test_storage"))
        backup_path = tmp_path / "backup"
        result = storage.create_backup(backup_path)
        assert result is True
        assert (backup_path / "data").exists()

    def test_create_backup_default_path(self, tmp_path):
        storage = SQLiteDataStorage(str(tmp_path / "test_storage"))
        result = storage.create_backup()
        assert result is True

    def test_create_backup_disabled(self, tmp_path):
        storage = SQLiteDataStorage(str(tmp_path / "test_storage"), backup_enabled=False)
        result = storage.create_backup()
        assert result is False

    def test_create_backup_failure(self, tmp_path):
        storage = SQLiteDataStorage(str(tmp_path / "test_storage"))
        with patch("shutil.copytree", side_effect=Exception("fail")):
            result = storage.create_backup(tmp_path / "backup_fail")
            assert result is False


# ---------------------------------------------------------------------------
# MFCDataManager
# ---------------------------------------------------------------------------
class TestMFCDataManager:
    @pytest.fixture
    def manager(self, tmp_path):
        storage = SQLiteDataStorage(str(tmp_path / "data"))
        return MFCDataManager(storage=storage)

    def test_default_creation(self, tmp_path):
        with patch("stability.data_manager.SQLiteDataStorage") as mock_cls:
            mock_cls.return_value = MagicMock()
            mgr = MFCDataManager()
            assert mgr.storage is not None

    def test_store_data_with_validation_pass(self, manager):
        record = {
            "timestamp": datetime.now(),
            "component_id": "c1",
            "metric_name": "power",
            "value": 25.0,
        }
        success, errors = manager.store_data(
            record, DataType.SENSOR_READING, validate=True,
        )
        assert success
        assert errors == []

    def test_store_data_validation_fail(self, manager):
        record = {"value": 25.0}  # missing required timestamp
        success, errors = manager.store_data(
            record, DataType.SENSOR_READING, validate=True,
        )
        assert not success
        assert len(errors) > 0

    def test_store_data_no_validation(self, manager):
        record = {"value": 25.0}
        success, errors = manager.store_data(
            record, DataType.SENSOR_READING, validate=False,
        )
        assert success

    def test_store_data_no_schema(self, manager):
        record = {"value": 25.0}
        success, errors = manager.store_data(
            record, DataType.ALARM_EVENT, validate=True,
        )
        assert success

    def test_store_data_exception(self, manager):
        with patch.object(manager.storage, "store_data", side_effect=Exception("fail")):
            success, errors = manager.store_data(
                {"value": 1}, DataType.SENSOR_READING, validate=False,
            )
            assert not success

    def test_store_data_storage_returns_false(self, manager):
        with patch.object(manager.storage, "store_data", return_value=False):
            success, errors = manager.store_data(
                {"value": 1}, DataType.ALARM_EVENT, validate=False,
            )
            assert not success

    def test_retrieve_data(self, manager):
        manager.store_data(
            {"timestamp": datetime.now(), "value": 10.0},
            DataType.SENSOR_READING, validate=False,
        )
        result = manager.retrieve_data(DataType.SENSOR_READING)
        assert isinstance(result, pd.DataFrame)

    def test_retrieve_data_exception(self, manager):
        with patch.object(manager.storage, "retrieve_data", side_effect=Exception("f")):
            result = manager.retrieve_data(DataType.SENSOR_READING)
            assert isinstance(result, pd.DataFrame)

    def test_assess_data_quality_with_dataframe(self, manager):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=50, freq="h"),
            "value": np.random.uniform(10, 30, 50),
        })
        report = manager.assess_data_quality(df)
        assert report.total_records == 50

    def test_assess_data_quality_with_datatype(self, manager):
        now = datetime.now()
        for i in range(10):
            manager.store_data(
                {"timestamp": now + timedelta(hours=i), "value": float(i + 10)},
                DataType.SENSOR_READING, validate=False,
            )
        report = manager.assess_data_quality(
            DataType.SENSOR_READING,
            time_range=(now - timedelta(hours=1), now + timedelta(hours=20)),
        )
        assert isinstance(report, DataQualityReport)

    def test_assess_data_quality_empty_df(self, manager):
        df = pd.DataFrame()
        report = manager.assess_data_quality(df)
        assert report.quality_level == DataQuality.INVALID

    def test_assess_data_quality_empty_rows(self, manager):
        df = pd.DataFrame({"timestamp": [], "value": []})
        report = manager.assess_data_quality(df)
        assert report.total_records == 0

    def test_assess_data_quality_exception(self, manager):
        with patch.object(manager, "retrieve_data", side_effect=Exception("fail")):
            report = manager.assess_data_quality(DataType.SENSOR_READING)
            assert report.quality_level == DataQuality.INVALID

    def test_assess_data_quality_non_df_result(self, manager):
        with patch.object(manager, "retrieve_data", return_value=[]):
            report = manager.assess_data_quality(DataType.SENSOR_READING)
            assert report.quality_level == DataQuality.INVALID

    def test_assess_data_quality_no_time_range(self, manager):
        with patch.object(manager, "retrieve_data", return_value=pd.DataFrame()):
            report = manager.assess_data_quality(DataType.SENSOR_READING)
            assert isinstance(report, DataQualityReport)

    def test_assess_with_duplicates(self, manager):
        df = pd.DataFrame({
            "timestamp": [datetime.now()] * 10,
            "value": [10.0] * 10,
        })
        report = manager.assess_data_quality(df)
        assert report.duplicate_records > 0

    def test_assess_with_numeric_columns(self, manager):
        df = pd.DataFrame({
            "value": np.random.normal(20, 2, 100),
            "other": np.random.normal(5, 10, 100),
        })
        report = manager.assess_data_quality(df)
        assert report.total_records == 100

    def test_calculate_consistency_score_object_column(self, manager):
        df = pd.DataFrame({
            "status": ["active"] * 50 + ["inactive"] * 50,
            "value": np.random.normal(10, 1, 100),
        })
        score = manager._calculate_consistency_score(df)
        assert 0.0 <= score <= 1.0

    def test_calculate_consistency_score_zero_std(self, manager):
        df = pd.DataFrame({"value": [5.0] * 10})
        score = manager._calculate_consistency_score(df)
        assert score == 1.0

    def test_calculate_consistency_score_exception(self, manager):
        score = manager._calculate_consistency_score(None)
        assert score == 0.5

    def test_calculate_accuracy_score(self, manager):
        df = pd.DataFrame({"value": np.random.normal(10, 1, 50)})
        score = manager._calculate_accuracy_score(df)
        assert 0.0 <= score <= 1.0

    def test_calculate_accuracy_score_exception(self, manager):
        score = manager._calculate_accuracy_score(None)
        assert score == 0.5

    def test_calculate_timeliness_with_timestamp(self, manager):
        df = pd.DataFrame({
            "timestamp": [datetime.now() - timedelta(minutes=5)],
            "value": [10],
        })
        score = manager._calculate_timeliness_score(df)
        assert score > 0.9

    def test_calculate_timeliness_old_data(self, manager):
        df = pd.DataFrame({
            "timestamp": [datetime.now() - timedelta(hours=25)],
            "value": [10],
        })
        score = manager._calculate_timeliness_score(df)
        assert score == 0.0

    def test_calculate_timeliness_no_timestamp(self, manager):
        df = pd.DataFrame({"value": [10]})
        score = manager._calculate_timeliness_score(df)
        assert score == 1.0

    def test_calculate_timeliness_exception(self, manager):
        score = manager._calculate_timeliness_score(None)
        assert score == 0.5

    def test_normalize_data_for_validation_dict(self, manager):
        result = manager._normalize_data_for_validation({"a": 1})
        assert result == [{"a": 1}]

    def test_normalize_data_for_validation_list(self, manager):
        result = manager._normalize_data_for_validation([{"a": 1}])
        assert result == [{"a": 1}]

    def test_normalize_data_for_validation_dataframe(self, manager):
        df = pd.DataFrame({"a": [1]})
        result = manager._normalize_data_for_validation(df)
        assert result == [{"a": 1}]

    def test_normalize_data_for_validation_unsupported(self, manager):
        with pytest.raises(ValueError):
            manager._normalize_data_for_validation(42)

    def test_export_csv(self, manager, tmp_path):
        manager.store_data(
            {"timestamp": datetime.now(), "value": 10.0},
            DataType.SENSOR_READING, validate=False,
        )
        output = tmp_path / "export"
        result = manager.export_data(DataType.SENSOR_READING, output, StorageFormat.CSV)
        assert result is True

    def test_export_json(self, manager, tmp_path):
        manager.store_data(
            {"timestamp": datetime.now(), "value": 10.0},
            DataType.SENSOR_READING, validate=False,
        )
        output = tmp_path / "export"
        result = manager.export_data(DataType.SENSOR_READING, output, StorageFormat.JSON)
        assert result is True

    def test_export_parquet(self, manager, tmp_path):
        manager.store_data(
            {"timestamp": datetime.now(), "value": 10.0},
            DataType.SENSOR_READING, validate=False,
        )
        output = tmp_path / "export"
        result = manager.export_data(DataType.SENSOR_READING, output, StorageFormat.PARQUET)
        assert result is True

    def test_export_pickle(self, manager, tmp_path):
        manager.store_data(
            {"timestamp": datetime.now(), "value": 10.0},
            DataType.SENSOR_READING, validate=False,
        )
        output = tmp_path / "export"
        result = manager.export_data(DataType.SENSOR_READING, output, StorageFormat.PICKLE)
        assert result is True

    def test_export_unsupported_format(self, manager, tmp_path):
        manager.store_data(
            {"timestamp": datetime.now(), "value": 10.0},
            DataType.SENSOR_READING, validate=False,
        )
        output = tmp_path / "export"
        result = manager.export_data(DataType.SENSOR_READING, output, StorageFormat.HDF5)
        assert result is False

    def test_export_empty_data(self, manager, tmp_path):
        output = tmp_path / "export"
        result = manager.export_data(DataType.ALARM_EVENT, output, StorageFormat.CSV)
        assert result is False

    def test_export_failure(self, manager, tmp_path):
        with patch.object(manager, "retrieve_data", side_effect=Exception("fail")):
            result = manager.export_data(
                DataType.SENSOR_READING, tmp_path / "out", StorageFormat.CSV,
            )
            assert result is False


# ---------------------------------------------------------------------------
# Factory & helpers
# ---------------------------------------------------------------------------
class TestFactoryAndHelpers:
    def test_create_mfc_data_manager_sqlite(self, tmp_path):
        mgr = create_mfc_data_manager(str(tmp_path / "data"), "sqlite")
        assert isinstance(mgr, MFCDataManager)

    def test_create_mfc_data_manager_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported storage type"):
            create_mfc_data_manager(storage_type="mongo")

    def test_generate_sample_sensor_data(self):
        start = datetime(2025, 1, 1)
        df = generate_sample_sensor_data(start, duration_hours=2, sampling_interval_minutes=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "timestamp" in df.columns

    def test_run_example_data_management(self, tmp_path):
        with patch("stability.data_manager.create_mfc_data_manager") as mock_create:
            mock_storage = MagicMock()
            mock_storage.store_data.return_value = True
            mock_storage.retrieve_data.return_value = pd.DataFrame({
                "timestamp": pd.date_range("2025-01-01", periods=10, freq="h"),
                "value": np.random.normal(20, 1, 10),
            })
            mock_mgr = MFCDataManager(storage=mock_storage)
            mock_create.return_value = mock_mgr
            run_example_data_management()
