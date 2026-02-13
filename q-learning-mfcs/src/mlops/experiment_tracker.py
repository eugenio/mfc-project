"""ExperimentTracker for TDD implementation."""

import json
import logging
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class ExperimentTrackerError(Exception):
    """Custom exception for ExperimentTracker-related errors."""
    pass


class MetricValue:
    """Represents a single metric value with timestamp and step information."""

    def __init__(self, value: int | float, timestamp: datetime | None = None, step: int | None = None):
        if not isinstance(value, int | float) or np.isnan(value):
            raise ExperimentTrackerError(f"Metric value must be numeric, got {type(value)}")
        self.value = float(value)
        self.timestamp = timestamp or datetime.now()
        self.step = step

    def to_dict(self) -> dict[str, Any]:
        return {
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'step': self.step
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MetricValue':
        return cls(
            value=data['value'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            step=data.get('step')
        )


class ArtifactInfo:
    """Information about stored artifacts."""

    def __init__(self, name: str, file_path: Path, artifact_type: str, size_bytes: int | None = None, created_at: datetime | None = None):
        self.name = name
        self.file_path = Path(file_path)
        self.artifact_type = artifact_type
        self.size_bytes = size_bytes or (self.file_path.stat().st_size if self.file_path.exists() else None)
        self.created_at = created_at or datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'file_path': str(self.file_path),
            'artifact_type': self.artifact_type,
            'size_bytes': self.size_bytes,
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ArtifactInfo':
        return cls(
            name=data['name'],
            file_path=Path(data['file_path']),
            artifact_type=data['artifact_type'],
            size_bytes=data.get('size_bytes'),
            created_at=datetime.fromisoformat(data['created_at'])
        )


class ExperimentRun:
    """Represents a single experiment run with all associated data."""

    def __init__(
        self,
        run_id: str,
        experiment_name: str,
        start_time: datetime,
        end_time: datetime | None = None,
        status: str = "running",
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, list[MetricValue]] | None = None,
        artifacts: dict[str, ArtifactInfo] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None
    ):
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.start_time = start_time
        self.end_time = end_time
        self.status = status
        self.hyperparameters = hyperparameters or {}
        self.metrics = metrics or {}
        self.artifacts = artifacts or {}
        self.tags = tags or []
        self.notes = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'hyperparameters': self.hyperparameters,
            'metrics': {
                name: [metric.to_dict() for metric in values]
                for name, values in self.metrics.items()
            },
            'artifacts': {
                name: artifact.to_dict()
                for name, artifact in self.artifacts.items()
            },
            'tags': self.tags,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ExperimentRun':
        return cls(
            run_id=data['run_id'],
            experiment_name=data['experiment_name'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
            status=data['status'],
            hyperparameters=data.get('hyperparameters', {}),
            metrics={
                name: [MetricValue.from_dict(metric_data) for metric_data in metrics_data]
                for name, metrics_data in data.get('metrics', {}).items()
            },
            artifacts={
                name: ArtifactInfo.from_dict(artifact_data)
                for name, artifact_data in data.get('artifacts', {}).items()
            },
            tags=data.get('tags', []),
            notes=data.get('notes')
        )


class ExperimentTracker:
    """Comprehensive experiment tracking system for machine learning experiments."""

    def __init__(self, base_path: str | Path = "experiments"):
        self.base_path = Path(base_path)
        self.experiments_path = self.base_path / "experiments"
        self.runs_path = self.base_path / "runs"
        self.artifacts_path = self.base_path / "artifacts"

        for path in [self.experiments_path, self.runs_path, self.artifacts_path]:
            path.mkdir(parents=True, exist_ok=True)

        self.current_run: ExperimentRun | None = None
        logger.info(f"ExperimentTracker initialized with base path: {self.base_path}")

    def create_experiment(self, name: str, description: str = "") -> None:
        experiment_file = self.experiments_path / f"{name}.json"
        if experiment_file.exists():
            logger.warning(f"Experiment '{name}' already exists")
            return

        experiment_data = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'runs': []
        }

        with open(experiment_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        logger.info(f"Created experiment: {name}")

    def list_experiments(self) -> dict[str, dict[str, Any]]:
        experiments = {}
        for experiment_file in self.experiments_path.glob("*.json"):
            try:
                with open(experiment_file) as f:
                    experiment_data = json.load(f)
                    experiments[experiment_data['name']] = {
                        'description': experiment_data['description'],
                        'created_at': experiment_data['created_at'],
                        'num_runs': len(experiment_data.get('runs', []))
                    }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load experiment {experiment_file}: {e}")
        return experiments

    def start_run(self, experiment_name: str, hyperparameters: dict[str, Any] | None = None,
                 tags: list[str] | None = None, notes: str | None = None) -> str:
        if self.current_run is not None:
            raise ExperimentTrackerError(
                f"Run {self.current_run.run_id} is already active. "
                "End the current run before starting a new one."
            )

        experiment_file = self.experiments_path / f"{experiment_name}.json"
        if not experiment_file.exists():
            raise ExperimentTrackerError(f"Experiment '{experiment_name}' does not exist")

        run_id = str(uuid.uuid4())
        self.current_run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            start_time=datetime.now(),
            hyperparameters=hyperparameters or {},
            tags=tags or [],
            notes=notes
        )
        logger.info(f"Started run {run_id} for experiment {experiment_name}")
        return run_id

    def log_metric(self, name: str, value: int | float, step: int | None = None) -> None:
        if self.current_run is None:
            raise ExperimentTrackerError("No active run. Start a run before logging metrics.")

        try:
            metric_value = MetricValue(value=value, step=step)
        except ExperimentTrackerError:
            raise ExperimentTrackerError(f"Invalid metric value: {value}") from None

        if name not in self.current_run.metrics:
            self.current_run.metrics[name] = []

        self.current_run.metrics[name].append(metric_value)
        logger.debug(f"Logged metric {name}={value} (step={step})")

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        if self.current_run is None:
            raise ExperimentTrackerError("No active run. Start a run before logging hyperparameters.")

        self.current_run.hyperparameters.update(hyperparameters)
        logger.debug(f"Logged hyperparameters: {list(hyperparameters.keys())}")

    def log_artifact(self, name: str, artifact: Any, artifact_type: str = "pickle") -> Path:
        if self.current_run is None:
            raise ExperimentTrackerError("No active run. Start a run before logging artifacts.")

        run_artifacts_path = self.artifacts_path / self.current_run.run_id
        run_artifacts_path.mkdir(exist_ok=True)

        artifact_file = run_artifacts_path / f"{name}.{artifact_type}"

        try:
            if artifact_type == "pickle":
                with open(artifact_file, 'wb') as f:
                    pickle.dump(artifact, f)
            elif artifact_type == "numpy":
                np.save(artifact_file.with_suffix('.npy'), artifact)
                artifact_file = artifact_file.with_suffix('.npy')
            elif artifact_type == "json":
                with open(artifact_file.with_suffix('.json'), 'w') as f:
                    json.dump(artifact, f, indent=2)
                artifact_file = artifact_file.with_suffix('.json')
            else:
                with open(artifact_file, 'wb') as f:
                    pickle.dump(artifact, f)
        except Exception as e:
            raise ExperimentTrackerError(f"Failed to save artifact '{name}': {e}") from e

        artifact_info = ArtifactInfo(name=name, file_path=artifact_file, artifact_type=artifact_type)
        self.current_run.artifacts[name] = artifact_info
        logger.debug(f"Logged artifact '{name}' ({artifact_type})")
        return artifact_file

    def log_model(self, name: str, model: Any) -> Path:
        return self.log_artifact(name, model, "model")

    def load_artifact(self, name: str) -> Any:
        if self.current_run is None:
            raise ExperimentTrackerError("No active run.")

        if name not in self.current_run.artifacts:
            raise ExperimentTrackerError(f"Artifact '{name}' not found in current run")

        artifact_info = self.current_run.artifacts[name]
        artifact_path = artifact_info.file_path

        if not artifact_path.exists():
            raise ExperimentTrackerError(f"Artifact file not found: {artifact_path}")

        try:
            if artifact_info.artifact_type == "numpy":
                return np.load(artifact_path)
            elif artifact_info.artifact_type == "json":
                with open(artifact_path) as f:
                    return json.load(f)
            else:
                with open(artifact_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            raise ExperimentTrackerError(f"Failed to load artifact '{name}': {e}") from e

    def end_run(self, status: str = "completed") -> None:
        if self.current_run is None:
            logger.warning("No active run to end")
            return

        self.current_run.end_time = datetime.now()
        self.current_run.status = status

        self._save_run(self.current_run)
        self._update_experiment_runs(self.current_run)

        logger.info(f"Ended run {self.current_run.run_id} with status: {status}")
        self.current_run = None

    def get_run(self, run_id: str) -> ExperimentRun | None:
        run_file = self.runs_path / f"{run_id}.json"
        if not run_file.exists():
            return None

        try:
            with open(run_file) as f:
                run_data = json.load(f)
                return ExperimentRun.from_dict(run_data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load run {run_id}: {e}")
            return None

    def compare_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        comparison_data = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                comparison_data.append({
                    'run_id': run.run_id,
                    'experiment_name': run.experiment_name,
                    'hyperparameters': run.hyperparameters,
                    'metrics': run.metrics,
                    'status': run.status,
                    'duration': ((run.end_time - run.start_time).total_seconds() if run.end_time else None),
                    'tags': run.tags
                })
        return comparison_data

    def get_best_run(self, experiment_name: str, metric_name: str, maximize: bool = True) -> ExperimentRun | None:
        runs = self._get_experiment_runs(experiment_name)
        if not runs:
            return None

        best_run = None
        best_value = None

        for run in runs:
            if metric_name in run.metrics and run.metrics[metric_name]:
                last_value = run.metrics[metric_name][-1].value
                if best_value is None or ((maximize and last_value > best_value) or (not maximize and last_value < best_value)):
                    best_value = last_value
                    best_run = run
        return best_run

    def search_runs(self, experiment_name: str, hyperparameters: dict[str, Any] | None = None,
                   tags: list[str] | None = None, status: str | None = None) -> list[ExperimentRun]:
        runs = self._get_experiment_runs(experiment_name)
        filtered_runs = []

        for run in runs:
            if hyperparameters:
                match = True
                for key, value in hyperparameters.items():
                    if key not in run.hyperparameters or run.hyperparameters[key] != value:
                        match = False
                        break
                if not match:
                    continue

            if tags:
                if not all(tag in run.tags for tag in tags):
                    continue

            if status and run.status != status:
                continue

            filtered_runs.append(run)
        return filtered_runs

    def export_experiment(self, experiment_name: str, output_path: Path, format: str = "json") -> None:
        runs = self._get_experiment_runs(experiment_name)

        if format.lower() == "json":
            export_data = {
                'experiment_name': experiment_name,
                'exported_at': datetime.now().isoformat(),
                'runs': [run.to_dict() for run in runs]
            }
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "csv":
            csv_data = []
            for run in runs:
                row = {
                    'run_id': run.run_id,
                    'experiment_name': run.experiment_name,
                    'start_time': run.start_time.isoformat(),
                    'end_time': run.end_time.isoformat() if run.end_time else None,
                    'status': run.status,
                    'duration_seconds': ((run.end_time - run.start_time).total_seconds() if run.end_time else None),
                    'tags': ','.join(run.tags)
                }
                for key, value in run.hyperparameters.items():
                    row[f'hyperparam_{key}'] = value
                for metric_name, metric_values in run.metrics.items():
                    if metric_values:
                        row[f'metric_{metric_name}'] = metric_values[-1].value
                csv_data.append(row)

            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
        else:
            raise ExperimentTrackerError(f"Unsupported export format: {format}")
        logger.info(f"Exported experiment '{experiment_name}' to {output_path}")

    def get_experiment_summary(self, experiment_name: str) -> dict[str, Any]:
        runs = self._get_experiment_runs(experiment_name)
        if not runs:
            return {'total_runs': 0}

        status_counts: dict[str, int] = {}
        for run in runs:
            status_counts[run.status] = status_counts.get(run.status, 0) + 1

        all_metrics: dict[str, list[float]] = {}
        for run in runs:
            for metric_name, metric_values in run.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                if metric_values:
                    all_metrics[metric_name].append(metric_values[-1].value)

        metrics_summary = {}
        for metric_name, values in all_metrics.items():
            if values:
                metrics_summary[metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        best_runs = {}
        for metric_name in all_metrics:
            best_run = self.get_best_run(experiment_name, metric_name, maximize=True)
            if best_run:
                best_runs[f'best_{metric_name}'] = {
                    'run_id': best_run.run_id,
                    'value': best_run.metrics[metric_name][-1].value
                }

        return {
            'total_runs': len(runs),
            'status_counts': status_counts,
            'completed_runs': status_counts.get('completed', 0),
            'failed_runs': status_counts.get('failed', 0),
            'metrics_summary': metrics_summary,
            **best_runs
        }

    def _save_run(self, run: ExperimentRun) -> None:
        run_file = self.runs_path / f"{run.run_id}.json"
        with open(run_file, 'w') as f:
            json.dump(run.to_dict(), f, indent=2)

    def _update_experiment_runs(self, run: ExperimentRun) -> None:
        experiment_file = self.experiments_path / f"{run.experiment_name}.json"
        if experiment_file.exists():
            with open(experiment_file) as f:
                experiment_data = json.load(f)
            if run.run_id not in experiment_data.get('runs', []):
                experiment_data.setdefault('runs', []).append(run.run_id)
            with open(experiment_file, 'w') as f:
                json.dump(experiment_data, f, indent=2)

    def _get_experiment_runs(self, experiment_name: str) -> list[ExperimentRun]:
        experiment_file = self.experiments_path / f"{experiment_name}.json"
        if not experiment_file.exists():
            return []

        try:
            with open(experiment_file) as f:
                experiment_data = json.load(f)
            runs = []
            for run_id in experiment_data.get('runs', []):
                run = self.get_run(run_id)
                if run:
                    runs.append(run)
            return runs
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load experiment {experiment_name}: {e}")
            return []
