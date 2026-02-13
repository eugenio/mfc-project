"""
Predictive Maintenance Scheduling for MFC Long-term Stability

Intelligent maintenance scheduling system that uses degradation patterns,
reliability analysis, and operational constraints to optimize maintenance
timing and minimize system downtime.

Created: 2025-07-28
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import field
from enum import Enum
from datetime import datetime, timedelta
import json
import logging

from degradation_detector import DegradationDetector, DegradationPattern, DegradationSeverity
from reliability_analyzer import ReliabilityAnalyzer

class MaintenanceType(Enum):
    """Types of maintenance activities."""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    ROUTINE = "routine"


class MaintenancePriority(Enum):
    """Priority levels for maintenance tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ComponentStatus(Enum):
    """Component operational status."""
    HEALTHY = "healthy"
    DEGRADING = "degrading"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

class MaintenanceTask:
    """Data structure for a maintenance task."""
    task_id: str
    component: str
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    scheduled_date: datetime
    estimated_duration_hours: float
    description: str
    required_resources: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    cost_estimate: float = 0.0
    downtime_impact: float = 0.0  # Hours of system downtime
    safety_requirements: List[str] = field(default_factory=list)
    completion_criteria: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    notes: str = ""
    
class MaintenanceWindow:
    """Available maintenance window."""
    start_time: datetime
    end_time: datetime
    max_downtime_hours: float
    available_resources: List[str]
    restrictions: List[str] = field(default_factory=list)


class OptimizationResult:
    """Result of maintenance schedule optimization."""
    total_cost: float
    total_downtime: float
    scheduled_tasks: List[MaintenanceTask]
    unscheduled_tasks: List[MaintenanceTask]
    optimization_score: float
    constraints_violated: List[str]

class MaintenanceScheduler:
    """
    Intelligent predictive maintenance scheduling system.
    
    Integrates with degradation detection and reliability analysis to
    create optimal maintenance schedules that balance:
    - System reliability and availability
    - Maintenance costs and resources
    - Operational constraints
    - Safety requirements
    """
    
    def __init__(self,
                 planning_horizon_days: int = 365,
                 max_simultaneous_tasks: int = 3,
                 emergency_response_hours: float = 2.0,
                 cost_per_downtime_hour: float = 100.0):
        """
        Initialize the maintenance scheduler.
        
        Args:
            planning_horizon_days: How far ahead to schedule maintenance
            max_simultaneous_tasks: Maximum concurrent maintenance tasks
            emergency_response_hours: Maximum response time for emergencies
            cost_per_downtime_hour: Cost per hour of system downtime
        """
        self.planning_horizon_days = planning_horizon_days
        self.max_simultaneous_tasks = max_simultaneous_tasks
        self.emergency_response_hours = emergency_response_hours
        self.cost_per_downtime_hour = cost_per_downtime_hour
        
        # Component libraries
        self.degradation_detector = DegradationDetector()
        self.reliability_analyzer = ReliabilityAnalyzer()
        
        # Maintenance data
        self.scheduled_tasks: List[MaintenanceTask] = []
        self.completed_tasks: List[MaintenanceTask] = []
        self.maintenance_windows: List[MaintenanceWindow] = []
        
        # Component tracking
        self.component_status: Dict[str, ComponentStatus] = {}
        self.component_last_maintenance: Dict[str, datetime] = {}
        self.component_maintenance_intervals: Dict[str, float] = {}  # hours
        
        # Resource management
        self.available_resources: List[str] = []
        self.resource_costs: Dict[str, float] = {}
        
        # Maintenance templates
        self.maintenance_templates = self._initialize_maintenance_templates()
        
        # Optimization weights
        self.optimization_weights = {
            'reliability': 0.4,
            'cost': 0.3,
            'downtime': 0.2,
            'resource_utilization': 0.1
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _initialize_maintenance_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize maintenance task templates for different components."""
        return {
            'membrane': {
                'cleaning': {
                    'duration_hours': 4.0,
                    'cost': 50.0,
                    'downtime': 4.0,
                    'interval_hours': 168,  # Weekly
                    'resources': ['maintenance_tech', 'cleaning_solution'],
                    'description': 'Membrane cleaning and inspection'
                },
                'replacement': {
                    'duration_hours': 8.0,
                    'cost': 500.0,
                    'downtime': 8.0,
                    'interval_hours': 8760,  # Yearly
                    'resources': ['senior_tech', 'membrane_replacement'],
                    'description': 'Complete membrane replacement'
                }
            },
            'electrode': {
                'inspection': {
                    'duration_hours': 2.0,
                    'cost': 25.0,
                    'downtime': 2.0,
                    'interval_hours': 720,  # Monthly
                    'resources': ['maintenance_tech'],
                    'description': 'Electrode visual inspection and testing'
                },
                'refurbishment': {
                    'duration_hours': 6.0,
                    'cost': 200.0,
                    'downtime': 6.0,
                    'interval_hours': 4380,  # Semi-annually
                    'resources': ['senior_tech', 'electrode_materials'],
                    'description': 'Electrode surface treatment and refurbishment'
                }
            },
            'biofilm': {
                'refresh': {
                    'duration_hours': 12.0,
                    'cost': 100.0,
                    'downtime': 12.0,
                    'interval_hours': 2160,  # Quarterly
                    'resources': ['bio_tech', 'culture_medium'],
                    'description': 'Biofilm culture refresh and optimization'
                },
                'analysis': {
                    'duration_hours': 1.0,
                    'cost': 15.0,
                    'downtime': 0.0,
                    'interval_hours': 168,  # Weekly
                    'resources': ['bio_tech'],
                    'description': 'Biofilm health analysis and monitoring'
                }
            },
            'system': {
                'calibration': {
                    'duration_hours': 3.0,
                    'cost': 40.0,
                    'downtime': 3.0,
                    'interval_hours': 720,  # Monthly
                    'resources': ['calibration_tech', 'reference_standards'],
                    'description': 'System sensor calibration and verification'
                },
                'comprehensive_inspection': {
                    'duration_hours': 16.0,
                    'cost': 300.0,
                    'downtime': 16.0,
                    'interval_hours': 4380,  # Semi-annually
                    'resources': ['senior_tech', 'inspection_tools'],
                    'description': 'Comprehensive system inspection and testing'
                }
            }
        }
    
    def add_maintenance_window(self, window: MaintenanceWindow):
        """Add an available maintenance window."""
        self.maintenance_windows.append(window)
        self.maintenance_windows.sort(key=lambda w: w.start_time)
    
    def update_component_status(self, component: str, status: ComponentStatus):
        """Update the operational status of a component."""
        self.component_status[component] = status
        
        # Log status change
        self.logger.info(f"Component {component} status updated to {status.value}")
        
        # Check if emergency maintenance is needed
        if status in [ComponentStatus.CRITICAL, ComponentStatus.FAILED]:
            self._create_emergency_maintenance(component, status)
    
    def _create_emergency_maintenance(self, component: str, status: ComponentStatus):
        """Create emergency maintenance task for critical/failed components."""
        task_id = f"emergency_{component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine maintenance type based on status
        if status == ComponentStatus.FAILED:
            description = f"EMERGENCY: {component} failure - immediate repair required"
            priority = MaintenancePriority.EMERGENCY
        else:
            description = f"CRITICAL: {component} in critical state - urgent maintenance required"
            priority = MaintenancePriority.CRITICAL
        
        # Schedule for immediate execution
        scheduled_date = datetime.now() + timedelta(hours=self.emergency_response_hours)
        
        emergency_task = MaintenanceTask(
            task_id=task_id,
            component=component,
            maintenance_type=MaintenanceType.EMERGENCY,
            priority=priority,
            scheduled_date=scheduled_date,
            estimated_duration_hours=8.0,  # Default emergency duration
            description=description,
            required_resources=['emergency_tech', 'emergency_parts'],
            downtime_impact=8.0,
            safety_requirements=['lockout_tagout', 'safety_review'],
            completion_criteria=['component_functional', 'safety_verified']
        )
        
        self.scheduled_tasks.append(emergency_task)
        self.logger.warning(f"Emergency maintenance scheduled for {component}: {task_id}")
    
    def analyze_degradation_patterns(self, patterns: List[DegradationPattern]) -> List[MaintenanceTask]:
        """Analyze degradation patterns and create predictive maintenance tasks."""
        predictive_tasks = []
        
        for pattern in patterns:
            # Skip patterns with low confidence
            if pattern.confidence < 0.7:
                continue
            
            # Determine maintenance urgency based on severity
            if pattern.severity == DegradationSeverity.CRITICAL:
                priority = MaintenancePriority.CRITICAL
                lead_time_hours = 24  # 1 day
            elif pattern.severity == DegradationSeverity.HIGH:
                priority = MaintenancePriority.HIGH
                lead_time_hours = 168  # 1 week
            elif pattern.severity == DegradationSeverity.MODERATE:
                priority = MaintenancePriority.MEDIUM
                lead_time_hours = 720  # 1 month
            else:
                priority = MaintenancePriority.LOW
                lead_time_hours = 2160  # 3 months
            
            # Create maintenance task for each affected component
            for component in pattern.affected_components:
                task_id = f"predictive_{pattern.pattern_id}_{component}"
                scheduled_date = datetime.now() + timedelta(hours=lead_time_hours)
                
                # Get maintenance template
                template = self._get_maintenance_template(component, pattern.degradation_type.value)
                
                predictive_task = MaintenanceTask(
                    task_id=task_id,
                    component=component,
                    maintenance_type=MaintenanceType.PREDICTIVE,
                    priority=priority,
                    scheduled_date=scheduled_date,
                    estimated_duration_hours=template['duration_hours'],
                    description=f"Predictive maintenance for {pattern.degradation_type.value} in {component}",
                    required_resources=template['resources'],
                    cost_estimate=template['cost'],
                    downtime_impact=template['downtime'],
                    completion_criteria=template.get('completion_criteria', []),
                    related_patterns=[pattern.pattern_id]
                )
                
                predictive_tasks.append(predictive_task)
        
        return predictive_tasks
    
    def _get_maintenance_template(self, component: str, degradation_type: str) -> Dict[str, Any]:
        """Get appropriate maintenance template for component and degradation type."""
        component_templates = self.maintenance_templates.get(component, {})
        
        # Map degradation types to maintenance activities
        degradation_maintenance_map = {
            'membrane_fouling': 'cleaning',
            'electrode_corrosion': 'refurbishment',
            'biofilm_aging': 'refresh',
            'catalyst_deactivation': 'refurbishment',
            'structural_fatigue': 'inspection',
            'chemical_poisoning': 'cleaning',
            'thermal_damage': 'inspection',
            'mechanical_wear': 'refurbishment'
        }
        
        maintenance_type = degradation_maintenance_map.get(degradation_type, 'inspection')
        template = component_templates.get(maintenance_type)
        
        if not template:
            # Default template
            template = {
                'duration_hours': 4.0,
                'cost': 100.0,
                'downtime': 4.0,
                'resources': ['maintenance_tech'],
                'completion_criteria': ['component_inspected']
            }
        
        return template
    
    def generate_preventive_schedule(self) -> List[MaintenanceTask]:
        """Generate preventive maintenance schedule based on intervals."""
        preventive_tasks = []
        now = datetime.now()
        
        for component, interval_hours in self.component_maintenance_intervals.items():
            last_maintenance = self.component_last_maintenance.get(component, now - timedelta(days=365))
            
            # Calculate next maintenance date
            next_maintenance = last_maintenance + timedelta(hours=interval_hours)
            
            # If due within planning horizon
            if next_maintenance <= now + timedelta(days=self.planning_horizon_days):
                # Get appropriate maintenance template
                templates = self.maintenance_templates.get(component, {})
                
                for maintenance_name, template in templates.items():
                    if template.get('interval_hours', float('inf')) == interval_hours:
                        task_id = f"preventive_{component}_{maintenance_name}_{next_maintenance.strftime('%Y%m%d')}"
                        
                        preventive_task = MaintenanceTask(
                            task_id=task_id,
                            component=component,
                            maintenance_type=MaintenanceType.PREVENTIVE,
                            priority=MaintenancePriority.MEDIUM,
                            scheduled_date=next_maintenance,
                            estimated_duration_hours=template['duration_hours'],
                            description=template['description'],
                            required_resources=template['resources'],
                            cost_estimate=template['cost'],
                            downtime_impact=template['downtime']
                        )
                        
                        preventive_tasks.append(preventive_task)
                        break
        
        return preventive_tasks
    
    def optimize_schedule(self, 
                         tasks: List[MaintenanceTask],
                         constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimize maintenance schedule using multiple objectives.
        
        Args:
            tasks: List of maintenance tasks to schedule
            constraints: Additional scheduling constraints
            
        Returns:
            Optimization result with scheduled tasks and metrics
        """
        if not tasks:
            return OptimizationResult(
                total_cost=0.0,
                total_downtime=0.0,
                scheduled_tasks=[],
                unscheduled_tasks=[],
                optimization_score=1.0,
                constraints_violated=[]
            )
        
        # Sort tasks by priority and due date
        sorted_tasks = sorted(tasks, key=lambda t: (
            -self._priority_score(t.priority),
            t.scheduled_date
        ))
        
        scheduled_tasks = []
        unscheduled_tasks = []
        total_cost = 0.0
        total_downtime = 0.0
        constraints_violated = []
        
        # Track resource utilization
        resource_schedule = {}
        
        for task in sorted_tasks:
            # Check if task can be scheduled
            can_schedule, violation_reason = self._can_schedule_task(
                task, scheduled_tasks, resource_schedule, constraints
            )
            
            if can_schedule:
                # Find best maintenance window
                best_window = self._find_best_maintenance_window(task, scheduled_tasks)
                
                if best_window:
                    # Adjust task timing to fit window
                    task.scheduled_date = max(task.scheduled_date, best_window.start_time)
                    
                    # Check if task fits in window
                    task_end = task.scheduled_date + timedelta(hours=task.estimated_duration_hours)
                    
                    if task_end <= best_window.end_time:
                        scheduled_tasks.append(task)
                        total_cost += task.cost_estimate
                        total_downtime += task.downtime_impact
                        
                        # Update resource schedule
                        self._update_resource_schedule(task, resource_schedule)
                    else:
                        unscheduled_tasks.append(task)
                        constraints_violated.append(f"Task {task.task_id} exceeds maintenance window")
                else:
                    unscheduled_tasks.append(task)
                    constraints_violated.append(f"No suitable maintenance window for {task.task_id}")
            else:
                unscheduled_tasks.append(task)
                constraints_violated.append(violation_reason)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            scheduled_tasks, total_cost, total_downtime, constraints_violated
        )
        
        return OptimizationResult(
            total_cost=total_cost,
            total_downtime=total_downtime,
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks=unscheduled_tasks,
            optimization_score=optimization_score,
            constraints_violated=constraints_violated
        )
    
    def _priority_score(self, priority: MaintenancePriority) -> int:
        """Convert priority to numeric score for sorting."""
        priority_scores = {
            MaintenancePriority.EMERGENCY: 5,
            MaintenancePriority.CRITICAL: 4,
            MaintenancePriority.HIGH: 3,
            MaintenancePriority.MEDIUM: 2,
            MaintenancePriority.LOW: 1
        }
        return priority_scores.get(priority, 1)
    
    def _can_schedule_task(self, 
                          task: MaintenanceTask,
                          scheduled_tasks: List[MaintenanceTask],
                          resource_schedule: Dict[str, List[Tuple[datetime, datetime]]],
                          constraints: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check if a task can be scheduled given current constraints."""
        
        # Check resource availability
        for resource in task.required_resources:
            if resource in resource_schedule:
                task_start = task.scheduled_date
                task_end = task_start + timedelta(hours=task.estimated_duration_hours)
                
                # Check for conflicts with existing resource bookings
                for booking_start, booking_end in resource_schedule[resource]:
                    if not (task_end <= booking_start or task_start >= booking_end):
                        return False, f"Resource {resource} conflict"
        
        # Check maximum simultaneous tasks
        concurrent_count = 0
        task_start = task.scheduled_date
        task_end = task_start + timedelta(hours=task.estimated_duration_hours)
        
        for scheduled_task in scheduled_tasks:
            scheduled_start = scheduled_task.scheduled_date
            scheduled_end = scheduled_start + timedelta(hours=scheduled_task.estimated_duration_hours)
            
            # Check for time overlap
            if not (task_end <= scheduled_start or task_start >= scheduled_end):
                concurrent_count += 1
        
        if concurrent_count >= self.max_simultaneous_tasks:
            return False, f"Maximum concurrent tasks ({self.max_simultaneous_tasks}) exceeded"
        
        # Check custom constraints
        if constraints:
            blackout_periods = constraints.get('blackout_periods', [])
            for blackout_start, blackout_end in blackout_periods:
                if not (task_end <= blackout_start or task_start >= blackout_end):
                    return False, "Task conflicts with blackout period"
        
        return True, "No constraints violated"
    
    def _find_best_maintenance_window(self, 
                                    task: MaintenanceTask,
                                    scheduled_tasks: List[MaintenanceTask]) -> Optional[MaintenanceWindow]:
        """Find the best maintenance window for a task."""
        suitable_windows = []
        
        for window in self.maintenance_windows:
            # Check if window can accommodate task
            if (window.end_time - window.start_time).total_seconds() / 3600 >= task.estimated_duration_hours:
                # Check if task fits in downtime limit
                if task.downtime_impact <= window.max_downtime_hours:
                    # Check resource availability
                    if all(resource in window.available_resources for resource in task.required_resources):
                        suitable_windows.append(window)
        
        if not suitable_windows:
            return None
        
        # Choose window closest to desired schedule date
        best_window = min(suitable_windows, 
                         key=lambda w: abs((w.start_time - task.scheduled_date).total_seconds()))
        
        return best_window
    
    def _update_resource_schedule(self, 
                                task: MaintenanceTask,
                                resource_schedule: Dict[str, List[Tuple[datetime, datetime]]]):
        """Update resource schedule with new task."""
        task_start = task.scheduled_date
        task_end = task_start + timedelta(hours=task.estimated_duration_hours)
        
        for resource in task.required_resources:
            if resource not in resource_schedule:
                resource_schedule[resource] = []
            
            resource_schedule[resource].append((task_start, task_end))
            resource_schedule[resource].sort(key=lambda x: x[0])  # Sort by start time
    
    def _calculate_optimization_score(self, 
                                    scheduled_tasks: List[MaintenanceTask],
                                    total_cost: float,
                                    total_downtime: float,
                                    constraints_violated: List[str]) -> float:
        """Calculate optimization score (0-1, higher is better)."""
        if not scheduled_tasks:
            return 0.0
        
        # Normalize metrics
        avg_cost_per_task = total_cost / len(scheduled_tasks) if scheduled_tasks else 0
        avg_downtime_per_task = total_downtime / len(scheduled_tasks) if scheduled_tasks else 0
        
        # Cost score (lower cost is better)
        cost_score = max(0, 1 - (avg_cost_per_task / 500))  # Normalize to $500 baseline
        
        # Downtime score (lower downtime is better)
        downtime_score = max(0, 1 - (avg_downtime_per_task / 8))  # Normalize to 8 hours baseline
        
        # Constraint violation penalty
        violation_penalty = len(constraints_violated) * 0.1
        
        # Task completion score
        completion_score = len(scheduled_tasks) / (len(scheduled_tasks) + len(constraints_violated))
        
        # Weighted combination
        optimization_score = (
            self.optimization_weights['cost'] * cost_score +
            self.optimization_weights['downtime'] * downtime_score +
            self.optimization_weights['resource_utilization'] * completion_score
        ) - violation_penalty
        
        return max(0.0, min(1.0, optimization_score))
    
    def get_maintenance_dashboard(self) -> Dict[str, Any]:
        """Get maintenance dashboard data."""
        now = datetime.now()
        
        # Upcoming tasks (next 7 days)
        upcoming_tasks = [
            task for task in self.scheduled_tasks
            if now <= task.scheduled_date <= now + timedelta(days=7)
        ]
        
        # Overdue tasks
        overdue_tasks = [
            task for task in self.scheduled_tasks
            if task.scheduled_date < now and task.completed_at is None
        ]
        
        # Tasks by priority
        priority_counts = {}
        for task in self.scheduled_tasks:
            priority = task.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Component status summary
        status_counts = {}
        for status in self.component_status.values():
            status_name = status.value
            status_counts[status_name] = status_counts.get(status_name, 0) + 1
        
        # Cost and downtime projections
        next_month_tasks = [
            task for task in self.scheduled_tasks
            if now <= task.scheduled_date <= now + timedelta(days=30)
        ]
        
        projected_cost = sum(task.cost_estimate for task in next_month_tasks)
        projected_downtime = sum(task.downtime_impact for task in next_month_tasks)
        
        return {
            'current_time': now.isoformat(),
            'upcoming_tasks': len(upcoming_tasks),
            'overdue_tasks': len(overdue_tasks),
            'total_scheduled_tasks': len(self.scheduled_tasks),
            'completed_tasks': len(self.completed_tasks),
            'tasks_by_priority': priority_counts,
            'components_by_status': status_counts,
            'next_30_days': {
                'scheduled_tasks': len(next_month_tasks),
                'projected_cost': projected_cost,
                'projected_downtime_hours': projected_downtime
            },
            'maintenance_windows': len(self.maintenance_windows),
            'emergency_tasks': len([t for t in self.scheduled_tasks if t.maintenance_type == MaintenanceType.EMERGENCY])
        }
    
    def export_schedule(self, filepath: str, format: str = 'json'):
        """Export maintenance schedule to file."""
        schedule_data = {
            'generated_at': datetime.now().isoformat(),
            'planning_horizon_days': self.planning_horizon_days,
            'scheduled_tasks': [],
            'completed_tasks': [],
            'component_status': {k: v.value for k, v in self.component_status.items()},
            'maintenance_windows': [],
            'dashboard_summary': self.get_maintenance_dashboard()
        }
        
        # Convert tasks to serializable format
        for task in self.scheduled_tasks:
            task_dict = {
                'task_id': task.task_id,
                'component': task.component,
                'maintenance_type': task.maintenance_type.value,
                'priority': task.priority.value,
                'scheduled_date': task.scheduled_date.isoformat(),
                'estimated_duration_hours': task.estimated_duration_hours,
                'description': task.description,
                'required_resources': task.required_resources,
                'cost_estimate': task.cost_estimate,
                'downtime_impact': task.downtime_impact,
                'related_patterns': task.related_patterns,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None
            }
            schedule_data['scheduled_tasks'].append(task_dict)
        
        for task in self.completed_tasks:
            task_dict = {
                'task_id': task.task_id,
                'component': task.component,
                'maintenance_type': task.maintenance_type.value,
                'priority': task.priority.value,
                'scheduled_date': task.scheduled_date.isoformat(),
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'description': task.description,
                'cost_estimate': task.cost_estimate,
                'notes': task.notes
            }
            schedule_data['completed_tasks'].append(task_dict)
        
        # Convert maintenance windows
        for window in self.maintenance_windows:
            window_dict = {
                'start_time': window.start_time.isoformat(),
                'end_time': window.end_time.isoformat(),
                'max_downtime_hours': window.max_downtime_hours,
                'available_resources': window.available_resources,
                'restrictions': window.restrictions
            }
            schedule_data['maintenance_windows'].append(window_dict)
        
        # Export in specified format
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(schedule_data, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Export tasks as CSV
            df = pd.DataFrame([
                {
                    'task_id': task.task_id,
                    'component': task.component,
                    'type': task.maintenance_type.value,
                    'priority': task.priority.value,
                    'scheduled_date': task.scheduled_date,
                    'duration_hours': task.estimated_duration_hours,
                    'cost': task.cost_estimate,
                    'downtime': task.downtime_impact,
                    'description': task.description
                }
                for task in self.scheduled_tasks
            ])
            df.to_csv(filepath, index=False)
        
        self.logger.info(f"Maintenance schedule exported to {filepath}")
    
    def complete_task(self, task_id: str, notes: str = ""):
        """Mark a maintenance task as completed."""
        for i, task in enumerate(self.scheduled_tasks):
            if task.task_id == task_id:
                task.completed_at = datetime.now()
                task.notes = notes
                
                # Move to completed tasks
                completed_task = self.scheduled_tasks.pop(i)
                self.completed_tasks.append(completed_task)
                
                # Update component last maintenance date
                self.component_last_maintenance[task.component] = task.completed_at
                
                self.logger.info(f"Task {task_id} completed for component {task.component}")
                return True
        
        self.logger.warning(f"Task {task_id} not found in scheduled tasks")
        return False