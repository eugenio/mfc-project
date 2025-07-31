#!/usr/bin/env python3
"""
Maintenance Scheduler Module for MFC Systems

This module provides comprehensive maintenance scheduling and management for
Microbial Fuel Cell (MFC) systems, including predictive maintenance,
preventive maintenance scheduling, resource optimization, and maintenance
cost analysis.

Author: MFC Analysis Team
Created: 2025-07-31
Last Modified: 2025-07-31
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Protocol,
    TypeVar, DefaultDict
)
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
MaintenanceTime = Union[datetime, pd.Timestamp]
CostValue = Union[float, int, np.number]
Priority = Union[int, float]

# Generic types
T = TypeVar('T')
MaintenanceTaskType = TypeVar('MaintenanceTaskType', bound='MaintenanceTask')


class MaintenanceType(Enum):
    """Types of maintenance activities."""
    PREVENTIVE = auto()
    PREDICTIVE = auto()
    CORRECTIVE = auto()
    CONDITION_BASED = auto()
    EMERGENCY = auto()
    ROUTINE_INSPECTION = auto()
    CALIBRATION = auto()
    CLEANING = auto()
    REPLACEMENT = auto()
    UPGRADE = auto()

    def __str__(self) -> str:
        return self.name.lower().replace('_', ' ')


class MaintenancePriority(Enum):
    """Priority levels for maintenance tasks."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
    EMERGENCY = auto()

    @property
    def numeric_value(self) -> int:
        """Return numeric representation of priority level."""
        return {
            MaintenancePriority.LOW: 1,
            MaintenancePriority.MEDIUM: 2,
            MaintenancePriority.HIGH: 3,
            MaintenancePriority.CRITICAL: 4,
            MaintenancePriority.EMERGENCY: 5
        }[self]


class MaintenanceStatus(Enum):
    """Status of maintenance tasks."""
    SCHEDULED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    OVERDUE = auto()
    POSTPONED = auto()

    def __str__(self) -> str:
        return self.name.lower().replace('_', ' ')


class ResourceType(Enum):
    """Types of maintenance resources."""
    TECHNICIAN = auto()
    EQUIPMENT = auto()
    SPARE_PARTS = auto()
    CONSUMABLES = auto()
    TOOLS = auto()
    EXTERNAL_SERVICE = auto()

    def __str__(self) -> str:
        return self.name.lower().replace('_', ' ')


@dataclass(frozen=True)
class MaintenanceResource:
    """Represents a maintenance resource."""
    resource_id: str
    resource_type: ResourceType
    name: str
    availability_schedule: Dict[str, List[Tuple[datetime, datetime]]] = field(default_factory=dict)
    cost_per_hour: float = 0.0
    cost_per_use: float = 0.0
    capacity: int = 1
    current_utilization: float = 0.0
    location: str = ""
    qualifications: List[str] = field(default_factory=list)
    maintenance_requirements: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate resource data."""
        if self.cost_per_hour < 0:
            raise ValueError("Cost per hour must be non-negative")
        if self.cost_per_use < 0:
            raise ValueError("Cost per use must be non-negative")
        if not (0.0 <= self.current_utilization <= 1.0):
            raise ValueError("Current utilization must be between 0 and 1")

    def is_available(
        self,
        start_time: datetime,
        end_time: datetime,
        day_of_week: Optional[str] = None
    ) -> bool:
        """Check if resource is available during specified time period."""
        if day_of_week is None:
            day_of_week = start_time.strftime('%A').lower()

        if day_of_week not in self.availability_schedule:
            return True  # Available if no schedule specified

        schedule = self.availability_schedule[day_of_week]
        for available_start, available_end in schedule:
            if available_start <= start_time and end_time <= available_end:
                return True

        return False

    def calculate_cost(self, duration_hours: float, usage_count: int = 1) -> float:
        """Calculate cost for using this resource."""
        hourly_cost = self.cost_per_hour * duration_hours
        usage_cost = self.cost_per_use * usage_count
        return hourly_cost + usage_cost


@dataclass
class MaintenanceTask:
    """Represents a maintenance task."""
    task_id: str
    task_name: str
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    affected_component: str
    description: str = ""

    # Scheduling information
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    duration_estimate: timedelta = field(default=timedelta(hours=1))

    # Status and progress
    status: MaintenanceStatus = MaintenanceStatus.SCHEDULED
    progress_percentage: float = 0.0

    # Resource requirements
    required_resources: List[str] = field(default_factory=list)  # Resource IDs
    assigned_resources: List[str] = field(default_factory=list)  # Assigned resource IDs

    # Dependencies
    prerequisite_tasks: List[str] = field(default_factory=list)  # Task IDs
    dependent_tasks: List[str] = field(default_factory=list)  # Task IDs

    # Cost and impact
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    downtime_impact: float = 0.0  # Hours of system downtime
    reliability_impact: float = 0.0  # Impact on system reliability (0-1)

    # Recurrence
    recurrence_interval: Optional[timedelta] = None
    last_performed: Optional[datetime] = None
    next_due: Optional[datetime] = None

    # Documentation
    work_instructions: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)
    completion_notes: str = ""

    # Metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate task data and compute derived fields."""
        self._validate_task_data()
        self._update_next_due_date()

    def _validate_task_data(self) -> None:
        """Validate task data."""
        if not (0.0 <= self.progress_percentage <= 100.0):
            raise ValueError("Progress percentage must be between 0 and 100")

        if self.estimated_cost < 0:
            raise ValueError("Estimated cost must be non-negative")

        if self.downtime_impact < 0:
            raise ValueError("Downtime impact must be non-negative")

    def _update_next_due_date(self) -> None:
        """Update next due date based on recurrence interval."""
        if self.recurrence_interval and self.last_performed:
            self.next_due = self.last_performed + self.recurrence_interval
        elif self.recurrence_interval and not self.last_performed:
            # If no last performed date, schedule for now + interval
            self.next_due = datetime.now() + self.recurrence_interval

    def is_overdue(self, current_time: Optional[datetime] = None) -> bool:
        """Check if task is overdue."""
        if not self.next_due:
            return False

        current_time = current_time or datetime.now()
        return current_time > self.next_due and self.status != MaintenanceStatus.COMPLETED

    def mark_completed(
        self,
        completion_time: Optional[datetime] = None,
        actual_cost: Optional[float] = None,
        notes: str = ""
    ) -> None:
        """Mark task as completed."""
        completion_time = completion_time or datetime.now()

        self.status = MaintenanceStatus.COMPLETED
        self.actual_end = completion_time
        self.progress_percentage = 100.0
        self.completion_notes = notes
        self.updated_at = datetime.now()

        if actual_cost is not None:
            self.actual_cost = actual_cost

        # Update last performed and next due date
        self.last_performed = completion_time
        self._update_next_due_date()

    def estimate_effort(self) -> Dict[str, float]:
        """Estimate effort required for this task."""
        base_hours = self.duration_estimate.total_seconds() / 3600

        # Adjust based on priority and complexity
        priority_multiplier = {
            MaintenancePriority.LOW: 0.8,
            MaintenancePriority.MEDIUM: 1.0,
            MaintenancePriority.HIGH: 1.2,
            MaintenancePriority.CRITICAL: 1.5,
            MaintenancePriority.EMERGENCY: 2.0
        }

        complexity_multiplier = 1.0
        if self.maintenance_type == MaintenanceType.EMERGENCY:
            complexity_multiplier = 1.8
        elif self.maintenance_type == MaintenanceType.PREDICTIVE:
            complexity_multiplier = 1.3
        elif self.maintenance_type == MaintenanceType.ROUTINE_INSPECTION:
            complexity_multiplier = 0.6

        adjusted_hours = base_hours * priority_multiplier[self.priority] * complexity_multiplier

        return {
            'base_hours': base_hours,
            'adjusted_hours': adjusted_hours,
            'priority_factor': priority_multiplier[self.priority],
            'complexity_factor': complexity_multiplier
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'maintenance_type': self.maintenance_type.name,
            'priority': self.priority.name,
            'affected_component': self.affected_component,
            'description': self.description,
            'scheduled_start': self.scheduled_start.isoformat() if self.scheduled_start else None,
            'scheduled_end': self.scheduled_end.isoformat() if self.scheduled_end else None,
            'actual_start': self.actual_start.isoformat() if self.actual_start else None,
            'actual_end': self.actual_end.isoformat() if self.actual_end else None,
            'duration_estimate': self.duration_estimate.total_seconds(),
            'status': self.status.name,
            'progress_percentage': self.progress_percentage,
            'required_resources': self.required_resources,
            'assigned_resources': self.assigned_resources,
            'prerequisite_tasks': self.prerequisite_tasks,
            'dependent_tasks': self.dependent_tasks,
            'estimated_cost': self.estimated_cost,
            'actual_cost': self.actual_cost,
            'downtime_impact': self.downtime_impact,
            'reliability_impact': self.reliability_impact,
            'recurrence_interval': self.recurrence_interval.total_seconds() if self.recurrence_interval else None,
            'last_performed': self.last_performed.isoformat() if self.last_performed else None,
            'next_due': self.next_due.isoformat() if self.next_due else None,
            'work_instructions': self.work_instructions,
            'safety_requirements': self.safety_requirements,
            'completion_notes': self.completion_notes,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_overdue': self.is_overdue()
        }


@dataclass
class MaintenanceSchedule:
    """Represents a complete maintenance schedule."""
    schedule_id: str
    schedule_name: str
    tasks: List[MaintenanceTask] = field(default_factory=list)
    resources: List[MaintenanceResource] = field(default_factory=list)
    schedule_start: datetime = field(default_factory=datetime.now)
    schedule_end: Optional[datetime] = None
    optimization_objective: str = "cost_minimize"  # cost_minimize, downtime_minimize, reliability_maximize

    # Schedule metrics
    total_estimated_cost: float = 0.0
    total_estimated_downtime: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    schedule_efficiency: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Initialize schedule and compute metrics."""
        self._compute_schedule_metrics()

    def _compute_schedule_metrics(self) -> None:
        """Compute schedule-wide metrics."""
        if not self.tasks:
            return

        # Calculate total estimated cost
        self.total_estimated_cost = sum(task.estimated_cost for task in self.tasks)

        # Calculate total estimated downtime
        self.total_estimated_downtime = sum(task.downtime_impact for task in self.tasks)

        # Calculate resource utilization
        resource_hours: DefaultDict[str, float] = defaultdict(float)
        total_schedule_hours = (self.schedule_end - self.schedule_start).total_seconds() / 3600 if self.schedule_end else 168  # Default to 1 week

        for task in self.tasks:
            task_hours = task.duration_estimate.total_seconds() / 3600
            for resource_id in task.assigned_resources:
                resource_hours[resource_id] += task_hours

        self.resource_utilization = {
            resource_id: min(1.0, hours / total_schedule_hours)
            for resource_id, hours in resource_hours.items()
        }

    def add_task(self, task: MaintenanceTask) -> None:
        """Add a task to the schedule."""
        self.tasks.append(task)
        self.updated_at = datetime.now()
        self._compute_schedule_metrics()

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the schedule."""
        original_length = len(self.tasks)
        self.tasks = [task for task in self.tasks if task.task_id != task_id]

        if len(self.tasks) < original_length:
            self.updated_at = datetime.now()
            self._compute_schedule_metrics()
            return True
        return False

    def get_tasks_by_priority(self, priority: MaintenancePriority) -> List[MaintenanceTask]:
        """Get tasks with specified priority."""
        return [task for task in self.tasks if task.priority == priority]

    def get_overdue_tasks(self, current_time: Optional[datetime] = None) -> List[MaintenanceTask]:
        """Get overdue tasks."""
        return [task for task in self.tasks if task.is_overdue(current_time)]

    def get_tasks_in_timeframe(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[MaintenanceTask]:
        """Get tasks scheduled within specified timeframe."""
        return [
            task for task in self.tasks
            if task.scheduled_start and task.scheduled_end and
            task.scheduled_start >= start_time and task.scheduled_end <= end_time
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert schedule to dictionary format."""
        return {
            'schedule_id': self.schedule_id,
            'schedule_name': self.schedule_name,
            'tasks': [task.to_dict() for task in self.tasks],
            'resources': [
                {
                    'resource_id': resource.resource_id,
                    'resource_type': resource.resource_type.name,
                    'name': resource.name,
                    'cost_per_hour': resource.cost_per_hour,
                    'cost_per_use': resource.cost_per_use,
                    'capacity': resource.capacity,
                    'current_utilization': resource.current_utilization,
                    'location': resource.location
                }
                for resource in self.resources
            ],
            'schedule_start': self.schedule_start.isoformat(),
            'schedule_end': self.schedule_end.isoformat() if self.schedule_end else None,
            'optimization_objective': self.optimization_objective,
            'total_estimated_cost': self.total_estimated_cost,
            'total_estimated_downtime': self.total_estimated_downtime,
            'resource_utilization': self.resource_utilization,
            'schedule_efficiency': self.schedule_efficiency,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class MaintenanceScheduler(Protocol):
    """Protocol for maintenance scheduler implementations."""

    def create_schedule(
        self,
        tasks: List[MaintenanceTask],
        resources: Optional[List[MaintenanceResource]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> MaintenanceSchedule:
        """Create optimized maintenance schedule."""
        ...

    def optimize_schedule(
        self,
        schedule: MaintenanceSchedule,
        objective: str = "cost_minimize",
        **kwargs: Any
    ) -> MaintenanceSchedule:
        """Optimize existing maintenance schedule."""
        ...


class BaseMaintenanceScheduler(ABC):
    """Base class for maintenance schedulers."""

    def __init__(
        self,
        optimization_method: str = "greedy",
        default_buffer_time: timedelta = timedelta(hours=1),
        max_daily_tasks: int = 8
    ) -> None:
        """Initialize maintenance scheduler.
        
        Args:
            optimization_method: Method for schedule optimization
            default_buffer_time: Default buffer time between tasks
            max_daily_tasks: Maximum tasks per day per resource
        """
        self.optimization_method = optimization_method
        self.default_buffer_time = default_buffer_time
        self.max_daily_tasks = max_daily_tasks
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the scheduler."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def create_schedule(
        self,
        tasks: List[MaintenanceTask],
        resources: Optional[List[MaintenanceResource]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> MaintenanceSchedule:
        """Create optimized maintenance schedule."""
        pass

    @abstractmethod
    def optimize_schedule(
        self,
        schedule: MaintenanceSchedule,
        objective: str = "cost_minimize",
        **kwargs: Any
    ) -> MaintenanceSchedule:
        """Optimize existing maintenance schedule."""
        pass

    def validate_schedule(self, schedule: MaintenanceSchedule) -> Dict[str, Any]:
        """Validate schedule for conflicts and constraints."""
        validation_results: Dict[str, Any] = {
            'is_valid': True,
            'conflicts': [],
            'warnings': [],
            'resource_conflicts': [],
            'dependency_violations': []
        }

        try:
            # Check for task scheduling conflicts
            scheduled_tasks = [
                task for task in schedule.tasks
                if task.scheduled_start and task.scheduled_end
            ]

            for i, task1 in enumerate(scheduled_tasks):
                for task2 in scheduled_tasks[i+1:]:
                    # Check for resource conflicts
                    common_resources = set(task1.assigned_resources) & set(task2.assigned_resources)
                    if common_resources:
                        # Check time overlap (both tasks must have scheduled times)
                        if (task1.scheduled_start is not None and task1.scheduled_end is not None and
                            task2.scheduled_start is not None and task2.scheduled_end is not None and
                            task1.scheduled_start < task2.scheduled_end and
                            task2.scheduled_start < task1.scheduled_end):
                            conflict = {
                                'task1': task1.task_id,
                                'task2': task2.task_id,
                                'conflicting_resources': list(common_resources),
                                'time_overlap': True
                            }
                            validation_results['resource_conflicts'].append(conflict)
                            validation_results['is_valid'] = False

            # Check dependency constraints
            for task in schedule.tasks:
                for prereq_id in task.prerequisite_tasks:
                    prereq_task = next(
                        (t for t in schedule.tasks if t.task_id == prereq_id), None
                    )
                    if prereq_task and task.scheduled_start and prereq_task.scheduled_end:
                        if task.scheduled_start < prereq_task.scheduled_end:
                            violation = {
                                'task': task.task_id,
                                'prerequisite': prereq_id,
                                'violation_type': 'starts_before_prerequisite_ends'
                            }
                            validation_results['dependency_violations'].append(violation)
                            validation_results['is_valid'] = False

            # Check for overdue tasks
            overdue_tasks = schedule.get_overdue_tasks()
            if overdue_tasks:
                validation_results['warnings'].append(
                    f"{len(overdue_tasks)} tasks are overdue"
                )

        except Exception as e:
            self.logger.error(f"Schedule validation failed: {str(e)}")
            validation_results['is_valid'] = False
            validation_results['errors'] = [str(e)]

        return validation_results


class GreedyMaintenanceScheduler(BaseMaintenanceScheduler):
    """Greedy maintenance scheduler using priority-based heuristics."""

    def __init__(
        self,
        optimization_method: str = "greedy",
        default_buffer_time: timedelta = timedelta(hours=1),
        max_daily_tasks: int = 8,
        priority_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """Initialize greedy scheduler.
        
        Args:
            optimization_method: Method for schedule optimization
            default_buffer_time: Default buffer time between tasks
            max_daily_tasks: Maximum tasks per day per resource
            priority_weights: Custom weights for different priority factors
        """
        super().__init__(optimization_method, default_buffer_time, max_daily_tasks)

        self.priority_weights = priority_weights or {
            'urgency': 0.4,        # How urgent/overdue the task is
            'impact': 0.3,         # Impact on system reliability
            'cost_efficiency': 0.2, # Cost-effectiveness of the task
            'resource_availability': 0.1  # Availability of required resources
        }

    def create_schedule(
        self,
        tasks: List[MaintenanceTask],
        resources: Optional[List[MaintenanceResource]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> MaintenanceSchedule:
        """Create optimized maintenance schedule using greedy algorithm.
        
        Args:
            tasks: List of maintenance tasks to schedule
            resources: Available maintenance resources
            constraints: Scheduling constraints
            **kwargs: Additional scheduling parameters
            
        Returns:
            Optimized maintenance schedule
        """
        try:
            self.logger.info(f"Creating schedule for {len(tasks)} tasks using greedy algorithm")

            # Initialize schedule
            schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            schedule = MaintenanceSchedule(
                schedule_id=schedule_id,
                schedule_name="Greedy Optimized Schedule",
                resources=resources or [],
                optimization_objective=kwargs.get('objective', 'cost_minimize')
            )

            # Sort tasks by priority score
            prioritized_tasks = self._prioritize_tasks(tasks, resources)

            # Schedule tasks greedily
            current_time = kwargs.get('start_time', datetime.now())
            resource_schedules: DefaultDict[str, List[Tuple[datetime, datetime, str]]] = defaultdict(list)  # Track when each resource is busy

            for task in prioritized_tasks:
                # Find best time slot for this task
                best_start_time = self._find_best_time_slot(
                    task, current_time, resource_schedules, resources
                )

                if best_start_time:
                    # Schedule the task
                    task.scheduled_start = best_start_time
                    task.scheduled_end = best_start_time + task.duration_estimate

                    # Update resource schedules
                    for resource_id in task.required_resources:
                        resource_schedules[resource_id].append(
                            (task.scheduled_start, task.scheduled_end, task.task_id)
                        )

                    # Assign available resources
                    task.assigned_resources = task.required_resources.copy()

                else:
                    self.logger.warning(f"Could not schedule task {task.task_id}")

                schedule.add_task(task)

            # Set schedule end time
            if schedule.tasks:
                latest_end = max(
                    task.scheduled_end for task in schedule.tasks
                    if task.scheduled_end
                )
                schedule.schedule_end = latest_end

            self.logger.info(f"Schedule created with {len(schedule.tasks)} tasks")
            return schedule

        except Exception as e:
            self.logger.error(f"Schedule creation failed: {str(e)}")
            raise

    def optimize_schedule(
        self,
        schedule: MaintenanceSchedule,
        objective: str = "cost_minimize",
        **kwargs: Any
    ) -> MaintenanceSchedule:
        """Optimize existing maintenance schedule.
        
        Args:
            schedule: Existing maintenance schedule
            objective: Optimization objective
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized maintenance schedule
        """
        try:
            self.logger.info(f"Optimizing schedule with {len(schedule.tasks)} tasks")

            # Create a copy for optimization
            optimized_tasks = []
            for task in schedule.tasks:
                # Reset scheduling for re-optimization
                optimized_task = MaintenanceTask(
                    task_id=task.task_id,
                    task_name=task.task_name,
                    maintenance_type=task.maintenance_type,
                    priority=task.priority,
                    affected_component=task.affected_component,
                    description=task.description,
                    duration_estimate=task.duration_estimate,
                    required_resources=task.required_resources,
                    prerequisite_tasks=task.prerequisite_tasks,
                    dependent_tasks=task.dependent_tasks,
                    estimated_cost=task.estimated_cost,
                    downtime_impact=task.downtime_impact,
                    reliability_impact=task.reliability_impact,
                    recurrence_interval=task.recurrence_interval,
                    last_performed=task.last_performed,
                    work_instructions=task.work_instructions,
                    safety_requirements=task.safety_requirements
                )
                optimized_tasks.append(optimized_task)

            # Create new optimized schedule
            optimized_schedule = self.create_schedule(
                optimized_tasks,
                schedule.resources,
                objective=objective,
                **kwargs
            )

            # Update schedule metadata
            optimized_schedule.schedule_id = schedule.schedule_id + "_optimized"
            optimized_schedule.schedule_name = schedule.schedule_name + " (Optimized)"

            self.logger.info("Schedule optimization completed")
            return optimized_schedule

        except Exception as e:
            self.logger.error(f"Schedule optimization failed: {str(e)}")
            return schedule  # Return original schedule if optimization fails

    def _prioritize_tasks(
        self,
        tasks: List[MaintenanceTask],
        resources: Optional[List[MaintenanceResource]] = None
    ) -> List[MaintenanceTask]:
        """Prioritize tasks based on multiple criteria."""
        task_scores = []

        for task in tasks:
            score = self._calculate_priority_score(task, resources)
            task_scores.append((score, task))

        # Sort by score (descending) and then by priority level
        task_scores.sort(key=lambda x: (x[0], x[1].priority.numeric_value), reverse=True)

        return [task for score, task in task_scores]

    def _calculate_priority_score(
        self,
        task: MaintenanceTask,
        resources: Optional[List[MaintenanceResource]] = None
    ) -> float:
        """Calculate priority score for a task."""
        score = 0.0

        # Urgency factor (overdue tasks get higher scores)
        urgency_score = task.priority.numeric_value / 5.0  # Normalize to 0-1
        if task.is_overdue():
            urgency_score *= 2.0  # Double score for overdue tasks
        score += self.priority_weights['urgency'] * urgency_score

        # Impact factor (reliability and downtime impact)
        impact_score = (task.reliability_impact + task.downtime_impact / 24.0) / 2.0
        score += self.priority_weights['impact'] * impact_score

        # Cost efficiency factor (inverse of cost)
        if task.estimated_cost > 0:
            cost_efficiency = 1.0 / (1.0 + task.estimated_cost / 1000.0)  # Normalize
        else:
            cost_efficiency = 1.0
        score += self.priority_weights['cost_efficiency'] * cost_efficiency

        # Resource availability factor
        if resources:
            available_resources = sum(
                1 for resource in resources
                if resource.resource_id in task.required_resources and
                resource.current_utilization < 0.8
            )
            total_required = len(task.required_resources)
            availability_score = available_resources / max(total_required, 1)
        else:
            availability_score = 1.0
        score += self.priority_weights['resource_availability'] * availability_score

        return score

    def _find_best_time_slot(
        self,
        task: MaintenanceTask,
        earliest_start: datetime,
        resource_schedules: Dict[str, List[Tuple[datetime, datetime, str]]],
        resources: Optional[List[MaintenanceResource]] = None
    ) -> Optional[datetime]:
        """Find the best time slot for scheduling a task."""
        # Check if task has prerequisites
        latest_prereq_end = earliest_start

        # Start search from the latest constraint
        search_start = max(earliest_start, latest_prereq_end)

        # Look for available time slot up to 30 days ahead
        max_search_time = search_start + timedelta(days=30)
        current_search_time = search_start

        while current_search_time < max_search_time:
            # Check if all required resources are available
            if self._resources_available(
                task, current_search_time,
                current_search_time + task.duration_estimate,
                resource_schedules, resources
            ):
                return current_search_time

            # Try next hour
            current_search_time += timedelta(hours=1)

        # If no slot found, return None
        return None

    def _resources_available(
        self,
        task: MaintenanceTask,
        start_time: datetime,
        end_time: datetime,
        resource_schedules: Dict[str, List[Tuple[datetime, datetime, str]]],
        resources: Optional[List[MaintenanceResource]] = None
    ) -> bool:
        """Check if all required resources are available during time slot."""
        for resource_id in task.required_resources:
            # Check against existing schedule
            if resource_id in resource_schedules:
                for scheduled_start, scheduled_end, _ in resource_schedules[resource_id]:
                    # Check for time overlap
                    if start_time < scheduled_end and end_time > scheduled_start:
                        return False

            # Check resource availability schedule
            if resources:
                resource = next(
                    (r for r in resources if r.resource_id == resource_id), None
                )
                if resource and not resource.is_available(start_time, end_time):
                    return False

        return True


# Factory functions and utilities
def create_greedy_scheduler(
    buffer_time_hours: float = 1.0,
    max_daily_tasks: int = 8,
    **kwargs: Any
) -> GreedyMaintenanceScheduler:
    """Create a greedy maintenance scheduler."""
    return GreedyMaintenanceScheduler(
        default_buffer_time=timedelta(hours=buffer_time_hours),
        max_daily_tasks=max_daily_tasks,
        **kwargs
    )


def create_sample_maintenance_tasks() -> List[MaintenanceTask]:
    """Create sample maintenance tasks for testing."""
    tasks = []

    # Routine biofilm cleaning
    tasks.append(MaintenanceTask(
        task_id="TASK_001",
        task_name="Biofilm Cleaning",
        maintenance_type=MaintenanceType.PREVENTIVE,
        priority=MaintenancePriority.MEDIUM,
        affected_component="Biofilm",
        description="Clean biofilm from electrodes",
        duration_estimate=timedelta(hours=2),
        required_resources=["TECH_001", "EQUIPMENT_001"],
        estimated_cost=150.0,
        downtime_impact=2.0,
        reliability_impact=0.1,
        recurrence_interval=timedelta(days=7),
        work_instructions=["Turn off system", "Remove electrodes", "Clean with solution"],
        safety_requirements=["Wear protective gloves", "Ensure ventilation"]
    ))

    # Electrode inspection
    tasks.append(MaintenanceTask(
        task_id="TASK_002",
        task_name="Electrode Inspection",
        maintenance_type=MaintenanceType.ROUTINE_INSPECTION,
        priority=MaintenancePriority.LOW,
        affected_component="Electrodes",
        description="Visual inspection of electrode condition",
        duration_estimate=timedelta(hours=1),
        required_resources=["TECH_001"],
        estimated_cost=75.0,
        downtime_impact=0.5,
        reliability_impact=0.05,
        recurrence_interval=timedelta(days=14),
        work_instructions=["Visual inspection", "Check for corrosion", "Document findings"]
    ))

    # Emergency repair
    tasks.append(MaintenanceTask(
        task_id="TASK_003",
        task_name="Emergency System Repair",
        maintenance_type=MaintenanceType.EMERGENCY,
        priority=MaintenancePriority.EMERGENCY,
        affected_component="Control System",
        description="Fix critical control system failure",
        duration_estimate=timedelta(hours=4),
        required_resources=["TECH_002", "EQUIPMENT_002", "SPARE_PARTS_001"],
        estimated_cost=500.0,
        downtime_impact=4.0,
        reliability_impact=0.3,
        work_instructions=["Diagnose issue", "Replace faulty components", "Test system"],
        safety_requirements=["Lock-out tag-out procedure", "Use proper PPE"]
    ))

    return tasks


def create_sample_resources() -> List[MaintenanceResource]:
    """Create sample maintenance resources for testing."""
    resources = []

    # Technician 1
    resources.append(MaintenanceResource(
        resource_id="TECH_001",
        resource_type=ResourceType.TECHNICIAN,
        name="Senior Technician",
        availability_schedule={
            'monday': [(datetime.now().replace(hour=8, minute=0), datetime.now().replace(hour=17, minute=0))],
            'tuesday': [(datetime.now().replace(hour=8, minute=0), datetime.now().replace(hour=17, minute=0))],
            'wednesday': [(datetime.now().replace(hour=8, minute=0), datetime.now().replace(hour=17, minute=0))],
            'thursday': [(datetime.now().replace(hour=8, minute=0), datetime.now().replace(hour=17, minute=0))],
            'friday': [(datetime.now().replace(hour=8, minute=0), datetime.now().replace(hour=17, minute=0))]
        },
        cost_per_hour=50.0,
        qualifications=["MFC maintenance", "Electrical systems", "Safety certified"]
    ))

    # Emergency technician
    resources.append(MaintenanceResource(
        resource_id="TECH_002",
        resource_type=ResourceType.TECHNICIAN,
        name="Emergency Technician",
        cost_per_hour=75.0,
        qualifications=["Emergency response", "MFC systems", "Control systems"]
    ))

    # Cleaning equipment
    resources.append(MaintenanceResource(
        resource_id="EQUIPMENT_001",
        resource_type=ResourceType.EQUIPMENT,
        name="Cleaning Equipment",
        cost_per_use=25.0,
        capacity=2
    ))

    return resources


# Example usage and testing
def run_example_maintenance_scheduling() -> None:
    """Run example maintenance scheduling."""
    # Create sample tasks and resources
    tasks = create_sample_maintenance_tasks()
    resources = create_sample_resources()

    # Create scheduler
    scheduler = create_greedy_scheduler(buffer_time_hours=0.5, max_daily_tasks=6)

    # Create schedule
    schedule = scheduler.create_schedule(
        tasks=tasks,
        resources=resources,
        start_time=datetime.now()
    )

    print("Maintenance Schedule Results:")
    print(f"Schedule ID: {schedule.schedule_id}")
    print(f"Total Tasks: {len(schedule.tasks)}")
    print(f"Total Estimated Cost: ${schedule.total_estimated_cost:.2f}")
    print(f"Total Estimated Downtime: {schedule.total_estimated_downtime:.1f} hours")

    print("\nScheduled Tasks:")
    for task in schedule.tasks:
        if task.scheduled_start and task.scheduled_end:
            print(f"- {task.task_name} ({task.priority.name}): "
                  f"{task.scheduled_start.strftime('%Y-%m-%d %H:%M')} - "
                  f"{task.scheduled_end.strftime('%Y-%m-%d %H:%M')}")

    # Validate schedule
    validation = scheduler.validate_schedule(schedule)
    print("\nSchedule Validation:")
    print(f"Valid: {validation['is_valid']}")
    if validation['warnings']:
        print(f"Warnings: {len(validation['warnings'])}")
    if validation['conflicts']:
        print(f"Conflicts: {len(validation['conflicts'])}")

    # Test optimization
    optimized_schedule = scheduler.optimize_schedule(schedule, objective="cost_minimize")
    print("\nOptimized Schedule:")
    print(f"Original Cost: ${schedule.total_estimated_cost:.2f}")
    print(f"Optimized Cost: ${optimized_schedule.total_estimated_cost:.2f}")


if __name__ == "__main__":
    run_example_maintenance_scheduling()
