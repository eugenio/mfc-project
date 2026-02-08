"""Tests for maintenance_scheduler module - comprehensive coverage."""
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stability.maintenance_scheduler import (
    BaseMaintenanceScheduler,
    GreedyMaintenanceScheduler,
    MaintenancePriority,
    MaintenanceResource,
    MaintenanceSchedule,
    MaintenanceStatus,
    MaintenanceTask,
    MaintenanceType,
    ResourceType,
    create_greedy_scheduler,
    create_sample_maintenance_tasks,
    create_sample_resources,
    run_example_maintenance_scheduling,
)


class TestEnums:
    def test_maintenance_type_str(self):
        assert str(MaintenanceType.PREVENTIVE) == "preventive"
        assert str(MaintenanceType.CONDITION_BASED) == "condition based"

    def test_maintenance_priority_numeric(self):
        assert MaintenancePriority.LOW.numeric_value == 1
        assert MaintenancePriority.EMERGENCY.numeric_value == 5

    def test_maintenance_status_str(self):
        assert str(MaintenanceStatus.SCHEDULED) == "scheduled"
        assert str(MaintenanceStatus.IN_PROGRESS) == "in progress"

    def test_resource_type_str(self):
        assert str(ResourceType.TECHNICIAN) == "technician"
        assert str(ResourceType.SPARE_PARTS) == "spare parts"


class TestMaintenanceResource:
    def test_creation(self):
        r = MaintenanceResource(
            resource_id="R1", resource_type=ResourceType.TECHNICIAN,
            name="Tech 1", cost_per_hour=50.0,
        )
        assert r.resource_id == "R1"

    def test_invalid_cost_per_hour(self):
        with pytest.raises(ValueError):
            MaintenanceResource(
                resource_id="R1", resource_type=ResourceType.TECHNICIAN,
                name="T", cost_per_hour=-10.0,
            )

    def test_invalid_cost_per_use(self):
        with pytest.raises(ValueError):
            MaintenanceResource(
                resource_id="R1", resource_type=ResourceType.TECHNICIAN,
                name="T", cost_per_use=-5.0,
            )

    def test_invalid_utilization(self):
        with pytest.raises(ValueError):
            MaintenanceResource(
                resource_id="R1", resource_type=ResourceType.TECHNICIAN,
                name="T", current_utilization=1.5,
            )

    def test_is_available_no_schedule(self):
        r = MaintenanceResource(
            resource_id="R1", resource_type=ResourceType.TECHNICIAN, name="T",
        )
        now = datetime.now()
        assert r.is_available(now, now + timedelta(hours=1)) is True

    def test_is_available_in_schedule(self):
        now = datetime.now().replace(hour=10, minute=0)
        day = now.strftime("%A").lower()
        r = MaintenanceResource(
            resource_id="R1", resource_type=ResourceType.TECHNICIAN, name="T",
            availability_schedule={
                day: [(now.replace(hour=8), now.replace(hour=17))],
            },
        )
        assert r.is_available(now, now + timedelta(hours=1)) is True

    def test_is_not_available(self):
        now = datetime.now().replace(hour=20, minute=0)
        day = now.strftime("%A").lower()
        r = MaintenanceResource(
            resource_id="R1", resource_type=ResourceType.TECHNICIAN, name="T",
            availability_schedule={
                day: [(now.replace(hour=8), now.replace(hour=17))],
            },
        )
        assert r.is_available(now, now + timedelta(hours=1)) is False

    def test_is_available_explicit_day(self):
        r = MaintenanceResource(
            resource_id="R1", resource_type=ResourceType.TECHNICIAN, name="T",
        )
        now = datetime.now()
        assert r.is_available(now, now + timedelta(hours=1), day_of_week="monday") is True

    def test_calculate_cost(self):
        r = MaintenanceResource(
            resource_id="R1", resource_type=ResourceType.TECHNICIAN, name="T",
            cost_per_hour=50.0, cost_per_use=25.0,
        )
        assert r.calculate_cost(2.0, 3) == 175.0


class TestMaintenanceTask:
    @pytest.fixture
    def task(self):
        return MaintenanceTask(
            task_id="T1", task_name="Test Task",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.MEDIUM,
            affected_component="Biofilm", estimated_cost=100.0,
        )

    def test_creation(self, task):
        assert task.task_id == "T1"

    def test_invalid_progress(self):
        with pytest.raises(ValueError):
            MaintenanceTask(
                task_id="T", task_name="T",
                maintenance_type=MaintenanceType.PREVENTIVE,
                priority=MaintenancePriority.LOW,
                affected_component="X", progress_percentage=150.0,
            )

    def test_invalid_cost(self):
        with pytest.raises(ValueError):
            MaintenanceTask(
                task_id="T", task_name="T",
                maintenance_type=MaintenanceType.PREVENTIVE,
                priority=MaintenancePriority.LOW,
                affected_component="X", estimated_cost=-10.0,
            )

    def test_invalid_downtime(self):
        with pytest.raises(ValueError):
            MaintenanceTask(
                task_id="T", task_name="T",
                maintenance_type=MaintenanceType.PREVENTIVE,
                priority=MaintenancePriority.LOW,
                affected_component="X", downtime_impact=-1.0,
            )

    def test_next_due_with_recurrence(self):
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW,
            affected_component="X",
            recurrence_interval=timedelta(days=7),
            last_performed=datetime(2025, 1, 1),
        )
        assert t.next_due == datetime(2025, 1, 8)

    def test_next_due_no_last(self):
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW,
            affected_component="X",
            recurrence_interval=timedelta(days=7),
        )
        assert t.next_due is not None

    def test_is_overdue_no_due(self, task):
        assert task.is_overdue() is False

    def test_is_overdue_true(self):
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW,
            affected_component="X",
            recurrence_interval=timedelta(days=7),
            last_performed=datetime(2020, 1, 1),
        )
        assert t.is_overdue() is True

    def test_is_overdue_completed(self):
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW,
            affected_component="X",
            status=MaintenanceStatus.COMPLETED,
            recurrence_interval=timedelta(days=7),
            last_performed=datetime(2020, 1, 1),
        )
        assert t.is_overdue() is False

    def test_mark_completed(self, task):
        task.mark_completed(actual_cost=90.0, notes="done")
        assert task.status == MaintenanceStatus.COMPLETED
        assert task.actual_cost == 90.0

    def test_mark_completed_recurrence(self):
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW,
            affected_component="X",
            recurrence_interval=timedelta(days=7),
        )
        now = datetime.now()
        t.mark_completed(completion_time=now)
        assert t.next_due == now + timedelta(days=7)

    def test_estimate_effort_types(self):
        for mtype, expected in [
            (MaintenanceType.EMERGENCY, 1.8),
            (MaintenanceType.PREDICTIVE, 1.3),
            (MaintenanceType.ROUTINE_INSPECTION, 0.6),
            (MaintenanceType.CLEANING, 1.0),
        ]:
            t = MaintenanceTask(
                task_id="T", task_name="T",
                maintenance_type=mtype,
                priority=MaintenancePriority.MEDIUM,
                affected_component="X",
            )
            assert t.estimate_effort()["complexity_factor"] == expected

    def test_to_dict(self, task):
        d = task.to_dict()
        assert d["task_id"] == "T1"
        assert "is_overdue" in d

    def test_to_dict_with_dates(self):
        now = datetime.now()
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW,
            affected_component="X",
            scheduled_start=now, scheduled_end=now + timedelta(hours=1),
            actual_start=now, actual_end=now + timedelta(hours=1),
            recurrence_interval=timedelta(days=7), last_performed=now,
        )
        d = t.to_dict()
        assert d["scheduled_start"] is not None


class TestMaintenanceSchedule:
    @pytest.fixture
    def schedule(self):
        return MaintenanceSchedule(schedule_id="S1", schedule_name="Test")

    @pytest.fixture
    def task(self):
        return MaintenanceTask(
            task_id="T1", task_name="Test",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.MEDIUM,
            affected_component="X", estimated_cost=100.0,
            downtime_impact=2.0, duration_estimate=timedelta(hours=2),
            assigned_resources=["R1"],
        )

    def test_empty_schedule(self, schedule):
        assert schedule.total_estimated_cost == 0.0

    def test_add_task(self, schedule, task):
        schedule.add_task(task)
        assert len(schedule.tasks) == 1

    def test_remove_task(self, schedule, task):
        schedule.add_task(task)
        assert schedule.remove_task("T1") is True

    def test_remove_nonexistent(self, schedule):
        assert schedule.remove_task("MISSING") is False

    def test_get_tasks_by_priority(self, schedule, task):
        schedule.add_task(task)
        assert len(schedule.get_tasks_by_priority(MaintenancePriority.MEDIUM)) == 1

    def test_get_overdue_tasks(self, schedule):
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW, affected_component="X",
            recurrence_interval=timedelta(days=7),
            last_performed=datetime(2020, 1, 1),
        )
        schedule.add_task(t)
        assert len(schedule.get_overdue_tasks()) == 1

    def test_get_tasks_in_timeframe(self, schedule):
        now = datetime.now()
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW, affected_component="X",
            scheduled_start=now + timedelta(hours=1),
            scheduled_end=now + timedelta(hours=2),
        )
        schedule.add_task(t)
        assert len(schedule.get_tasks_in_timeframe(now, now + timedelta(hours=3))) == 1

    def test_schedule_with_end_time(self):
        s = MaintenanceSchedule(
            schedule_id="S", schedule_name="T",
            schedule_end=datetime.now() + timedelta(days=7),
        )
        assert s.schedule_end is not None

    def test_to_dict(self, schedule, task):
        schedule.add_task(task)
        schedule.resources = [MaintenanceResource(
            resource_id="R1", resource_type=ResourceType.TECHNICIAN,
            name="T", cost_per_hour=50.0,
        )]
        d = schedule.to_dict()
        assert d["schedule_id"] == "S1"


class TestGreedyScheduler:
    @pytest.fixture
    def scheduler(self):
        return GreedyMaintenanceScheduler(max_daily_tasks=8)

    @pytest.fixture
    def tasks(self):
        return [
            MaintenanceTask(
                task_id="T1", task_name="Clean",
                maintenance_type=MaintenanceType.CLEANING,
                priority=MaintenancePriority.MEDIUM,
                affected_component="Bio", estimated_cost=50.0,
                downtime_impact=1.0, reliability_impact=0.1,
            ),
            MaintenanceTask(
                task_id="T2", task_name="Repair",
                maintenance_type=MaintenanceType.EMERGENCY,
                priority=MaintenancePriority.EMERGENCY,
                affected_component="System", estimated_cost=500.0,
                downtime_impact=4.0, reliability_impact=0.5,
            ),
        ]

    def test_create_schedule(self, scheduler, tasks):
        schedule = scheduler.create_schedule(tasks)
        assert len(schedule.tasks) == 2

    def test_create_schedule_with_resources(self, scheduler):
        tasks = [MaintenanceTask(
            task_id="T1", task_name="X",
            maintenance_type=MaintenanceType.CLEANING,
            priority=MaintenancePriority.LOW, affected_component="X",
            required_resources=["R1"],
        )]
        resources = [MaintenanceResource(
            resource_id="R1", resource_type=ResourceType.TECHNICIAN,
            name="T", cost_per_hour=50.0,
        )]
        schedule = scheduler.create_schedule(tasks, resources)
        assert len(schedule.tasks) == 1

    def test_optimize_schedule(self, scheduler, tasks):
        schedule = scheduler.create_schedule(tasks)
        optimized = scheduler.optimize_schedule(schedule)
        assert "Optimized" in optimized.schedule_name

    def test_optimize_failure(self, scheduler, tasks):
        schedule = scheduler.create_schedule(tasks)
        with patch.object(scheduler, "create_schedule", side_effect=Exception("fail")):
            result = scheduler.optimize_schedule(schedule)
            assert result is schedule

    def test_validate_resource_conflict(self, scheduler):
        now = datetime.now()
        tasks = [
            MaintenanceTask(
                task_id="T1", task_name="A",
                maintenance_type=MaintenanceType.CLEANING,
                priority=MaintenancePriority.HIGH, affected_component="X",
                scheduled_start=now, scheduled_end=now + timedelta(hours=2),
                assigned_resources=["R1"],
            ),
            MaintenanceTask(
                task_id="T2", task_name="B",
                maintenance_type=MaintenanceType.CLEANING,
                priority=MaintenancePriority.HIGH, affected_component="X",
                scheduled_start=now + timedelta(hours=1),
                scheduled_end=now + timedelta(hours=3),
                assigned_resources=["R1"],
            ),
        ]
        schedule = MaintenanceSchedule(schedule_id="S1", schedule_name="T", tasks=tasks)
        result = scheduler.validate_schedule(schedule)
        assert not result["is_valid"]

    def test_validate_dependency_violation(self, scheduler):
        now = datetime.now()
        t1 = MaintenanceTask(
            task_id="T1", task_name="A",
            maintenance_type=MaintenanceType.CLEANING,
            priority=MaintenancePriority.HIGH, affected_component="X",
            scheduled_start=now, scheduled_end=now + timedelta(hours=3),
        )
        t2 = MaintenanceTask(
            task_id="T2", task_name="B",
            maintenance_type=MaintenanceType.CLEANING,
            priority=MaintenancePriority.HIGH, affected_component="X",
            scheduled_start=now + timedelta(hours=1),
            scheduled_end=now + timedelta(hours=2),
            prerequisite_tasks=["T1"],
        )
        schedule = MaintenanceSchedule(schedule_id="S", schedule_name="T", tasks=[t1, t2])
        result = scheduler.validate_schedule(schedule)
        assert not result["is_valid"]

    def test_validate_overdue_warning(self, scheduler):
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW, affected_component="X",
            recurrence_interval=timedelta(days=7),
            last_performed=datetime(2020, 1, 1),
        )
        schedule = MaintenanceSchedule(schedule_id="S", schedule_name="T", tasks=[t])
        result = scheduler.validate_schedule(schedule)
        assert len(result["warnings"]) > 0

    def test_priority_score(self, scheduler):
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.HIGH, affected_component="X",
            estimated_cost=100.0, reliability_impact=0.5, downtime_impact=4.0,
            recurrence_interval=timedelta(days=7),
            last_performed=datetime(2020, 1, 1),
        )
        score = scheduler._calculate_priority_score(t)
        assert score > 0

    def test_priority_score_with_resources(self, scheduler):
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.LOW, affected_component="X",
            required_resources=["R1"],
        )
        resources = [MaintenanceResource(
            resource_id="R1", resource_type=ResourceType.TECHNICIAN,
            name="T", current_utilization=0.5,
        )]
        score = scheduler._calculate_priority_score(t, resources)
        assert score > 0

    def test_resources_conflict(self, scheduler):
        now = datetime.now()
        t = MaintenanceTask(
            task_id="T", task_name="T",
            maintenance_type=MaintenanceType.CLEANING,
            priority=MaintenancePriority.LOW, affected_component="X",
            required_resources=["R1"],
        )
        schedules = {"R1": [(now, now + timedelta(hours=2), "X")]}
        assert not scheduler._resources_available(
            t, now + timedelta(hours=1), now + timedelta(hours=3), schedules,
        )


class TestFactoryHelpers:
    def test_create_greedy_scheduler(self):
        assert isinstance(create_greedy_scheduler(), GreedyMaintenanceScheduler)

    def test_create_sample_tasks(self):
        assert len(create_sample_maintenance_tasks()) == 3

    def test_create_sample_resources(self):
        assert len(create_sample_resources()) == 3

    def test_run_example(self):
        run_example_maintenance_scheduling()
