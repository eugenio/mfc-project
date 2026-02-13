"""
Controller System Cost Analysis and Power Requirements

This module provides comprehensive cost analysis and power requirement calculations
for the complete controller system including hardware, software, development,
and operational costs.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .model_inference import InferenceSpecs, ModelFormat

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Cost categories for analysis"""
    HARDWARE = "hardware"
    SOFTWARE = "software"
    DEVELOPMENT = "development"
    LICENSING = "licensing"
    MAINTENANCE = "maintenance"
    OPERATIONAL = "operational"
    TRAINING = "training"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class PowerRequirement:
    """Power requirement specification"""
    component: str
    idle_power_w: float
    active_power_w: float
    peak_power_w: float
    duty_cycle_pct: float  # Percentage of time at active power
    efficiency: float = 1.0
    thermal_dissipation_w: float = 0.0


@dataclass
class CostItem:
    """Individual cost item"""
    item_name: str
    category: CostCategory
    initial_cost: float
    recurring_cost_per_year: float
    useful_life_years: float
    depreciation_rate: float = 0.0
    maintenance_factor: float = 0.05  # 5% of initial cost per year


@dataclass
class ControllerSystemSpecs:
    """Complete controller system specifications"""
    inference_engine_specs: InferenceSpecs
    electronics_power_w: float
    real_time_controller_overhead_w: float
    hal_power_w: float
    communication_power_w: float
    cooling_power_w: float

    # Development specifications
    development_person_months: float
    testing_person_months: float
    documentation_person_months: float
    certification_required: bool

    # Operational specifications
    expected_lifetime_years: float
    maintenance_interval_months: float
    software_update_frequency_months: float

    # Optional specifications with defaults
    redundancy_factor: float = 1.2  # 20% power margin


class ControllerCostAnalyzer:
    """Comprehensive cost analyzer for controller systems"""

    def __init__(self, system_specs: ControllerSystemSpecs):
        self.system_specs = system_specs
        self.cost_items = []
        self.power_requirements = []

        # Standard cost factors
        self.engineer_cost_per_month = 12000.0  # USD per month
        self.energy_cost_per_kwh = 0.15  # USD per kWh
        self.facility_cost_per_month = 2000.0  # USD per month

        # Initialize standard cost items
        self._initialize_standard_costs()
        self._initialize_power_requirements()

    def _initialize_standard_costs(self):
        """Initialize standard cost items for controller system"""

        # Hardware costs
        self.add_cost_item(CostItem(
            item_name="Inference Engine Hardware",
            category=CostCategory.HARDWARE,
            initial_cost=self.system_specs.inference_engine_specs.cost,
            recurring_cost_per_year=0.0,
            useful_life_years=5.0,
            maintenance_factor=0.02
        ))

        self.add_cost_item(CostItem(
            item_name="Control Electronics",
            category=CostCategory.HARDWARE,
            initial_cost=500.0,  # Estimated based on MCU + ADC + DAC + interfaces
            recurring_cost_per_year=0.0,
            useful_life_years=7.0,
            maintenance_factor=0.03
        ))

        self.add_cost_item(CostItem(
            item_name="Real-Time Controller Hardware",
            category=CostCategory.HARDWARE,
            initial_cost=1500.0,  # Dedicated real-time hardware
            recurring_cost_per_year=0.0,
            useful_life_years=10.0,
            maintenance_factor=0.02
        ))

        self.add_cost_item(CostItem(
            item_name="Hardware Abstraction Layer Components",
            category=CostCategory.HARDWARE,
            initial_cost=800.0,  # Interface boards, connectors, etc.
            recurring_cost_per_year=0.0,
            useful_life_years=8.0,
            maintenance_factor=0.04
        ))

        self.add_cost_item(CostItem(
            item_name="Communication Interfaces",
            category=CostCategory.HARDWARE,
            initial_cost=300.0,  # CAN, Ethernet, RS485 interfaces
            recurring_cost_per_year=0.0,
            useful_life_years=6.0,
            maintenance_factor=0.03
        ))

        self.add_cost_item(CostItem(
            item_name="Cooling System",
            category=CostCategory.HARDWARE,
            initial_cost=400.0,  # Fans, heat sinks, thermal management
            recurring_cost_per_year=50.0,  # Filter replacements
            useful_life_years=5.0,
            maintenance_factor=0.08
        ))

        # Software costs
        self.add_cost_item(CostItem(
            item_name="Real-Time Operating System",
            category=CostCategory.SOFTWARE,
            initial_cost=2000.0,  # Commercial RTOS license
            recurring_cost_per_year=400.0,  # Support and updates
            useful_life_years=float('inf'),  # Software doesn't depreciate
            maintenance_factor=0.0
        ))

        self.add_cost_item(CostItem(
            item_name="Development Tools",
            category=CostCategory.SOFTWARE,
            initial_cost=5000.0,  # IDE, debuggers, analyzers
            recurring_cost_per_year=1000.0,  # License renewals
            useful_life_years=float('inf'),
            maintenance_factor=0.0
        ))

        self.add_cost_item(CostItem(
            item_name="Machine Learning Framework",
            category=CostCategory.SOFTWARE,
            initial_cost=0.0,  # Open source (TensorFlow, PyTorch)
            recurring_cost_per_year=0.0,
            useful_life_years=float('inf'),
            maintenance_factor=0.0
        ))

        # Development costs
        development_cost = (self.system_specs.development_person_months *
                          self.engineer_cost_per_month)

        self.add_cost_item(CostItem(
            item_name="Software Development",
            category=CostCategory.DEVELOPMENT,
            initial_cost=development_cost,
            recurring_cost_per_year=0.0,
            useful_life_years=self.system_specs.expected_lifetime_years,
            maintenance_factor=0.0
        ))

        testing_cost = (self.system_specs.testing_person_months *
                       self.engineer_cost_per_month)

        self.add_cost_item(CostItem(
            item_name="Testing and Validation",
            category=CostCategory.DEVELOPMENT,
            initial_cost=testing_cost,
            recurring_cost_per_year=testing_cost * 0.2,  # Ongoing testing
            useful_life_years=self.system_specs.expected_lifetime_years,
            maintenance_factor=0.0
        ))

        documentation_cost = (self.system_specs.documentation_person_months *
                             self.engineer_cost_per_month)

        self.add_cost_item(CostItem(
            item_name="Documentation",
            category=CostCategory.DEVELOPMENT,
            initial_cost=documentation_cost,
            recurring_cost_per_year=documentation_cost * 0.1,  # Updates
            useful_life_years=self.system_specs.expected_lifetime_years,
            maintenance_factor=0.0
        ))

        # Certification costs (if required)
        if self.system_specs.certification_required:
            self.add_cost_item(CostItem(
                item_name="Safety Certification",
                category=CostCategory.LICENSING,
                initial_cost=25000.0,  # FDA, CE, IEC 61508 certification
                recurring_cost_per_year=5000.0,  # Maintenance of certification
                useful_life_years=5.0,  # Certification expires
                maintenance_factor=0.0
            ))

        # Infrastructure costs
        self.add_cost_item(CostItem(
            item_name="Development Infrastructure",
            category=CostCategory.INFRASTRUCTURE,
            initial_cost=10000.0,  # Test equipment, computers, servers
            recurring_cost_per_year=2000.0,  # Cloud services, utilities
            useful_life_years=5.0,
            maintenance_factor=0.05
        ))

        # Training costs
        self.add_cost_item(CostItem(
            item_name="Personnel Training",
            category=CostCategory.TRAINING,
            initial_cost=8000.0,  # Initial training for operators/maintainers
            recurring_cost_per_year=2000.0,  # Ongoing training
            useful_life_years=float('inf'),
            maintenance_factor=0.0
        ))

    def _initialize_power_requirements(self):
        """Initialize power requirements for all components"""

        # Inference engine power
        inference_power = self.system_specs.inference_engine_specs.power_consumption
        self.add_power_requirement(PowerRequirement(
            component="Model Inference Engine",
            idle_power_w=inference_power * 0.1,  # 10% idle power
            active_power_w=inference_power,
            peak_power_w=inference_power * 1.5,  # 50% peak overhead
            duty_cycle_pct=80.0,  # 80% active time
            efficiency=0.95,
            thermal_dissipation_w=inference_power * 0.15  # 15% heat
        ))

        # Control electronics power
        self.add_power_requirement(PowerRequirement(
            component="Control Electronics",
            idle_power_w=self.system_specs.electronics_power_w * 0.3,
            active_power_w=self.system_specs.electronics_power_w,
            peak_power_w=self.system_specs.electronics_power_w * 1.2,
            duty_cycle_pct=90.0,  # Always active
            efficiency=0.85,
            thermal_dissipation_w=self.system_specs.electronics_power_w * 0.20
        ))

        # Real-time controller overhead
        self.add_power_requirement(PowerRequirement(
            component="Real-Time Controller",
            idle_power_w=self.system_specs.real_time_controller_overhead_w * 0.2,
            active_power_w=self.system_specs.real_time_controller_overhead_w,
            peak_power_w=self.system_specs.real_time_controller_overhead_w * 1.3,
            duty_cycle_pct=95.0,  # Nearly always active
            efficiency=0.90,
            thermal_dissipation_w=self.system_specs.real_time_controller_overhead_w * 0.25
        ))

        # Hardware abstraction layer
        self.add_power_requirement(PowerRequirement(
            component="Hardware Abstraction Layer",
            idle_power_w=self.system_specs.hal_power_w * 0.4,
            active_power_w=self.system_specs.hal_power_w,
            peak_power_w=self.system_specs.hal_power_w * 1.1,
            duty_cycle_pct=70.0,  # Moderate activity
            efficiency=0.88,
            thermal_dissipation_w=self.system_specs.hal_power_w * 0.18
        ))

        # Communication interfaces
        self.add_power_requirement(PowerRequirement(
            component="Communication Interfaces",
            idle_power_w=self.system_specs.communication_power_w * 0.1,
            active_power_w=self.system_specs.communication_power_w,
            peak_power_w=self.system_specs.communication_power_w * 2.0,  # High peak for transmission
            duty_cycle_pct=30.0,  # Intermittent communication
            efficiency=0.80,
            thermal_dissipation_w=self.system_specs.communication_power_w * 0.30
        ))

        # Cooling system
        self.add_power_requirement(PowerRequirement(
            component="Cooling System",
            idle_power_w=self.system_specs.cooling_power_w * 0.5,
            active_power_w=self.system_specs.cooling_power_w,
            peak_power_w=self.system_specs.cooling_power_w * 1.0,  # No peak for fans
            duty_cycle_pct=60.0,  # Moderate cooling needs
            efficiency=0.75,  # Motor efficiency
            thermal_dissipation_w=self.system_specs.cooling_power_w * 0.25
        ))

    def add_cost_item(self, cost_item: CostItem):
        """Add a cost item to the analysis"""
        self.cost_items.append(cost_item)

    def add_power_requirement(self, power_req: PowerRequirement):
        """Add a power requirement to the analysis"""
        self.power_requirements.append(power_req)

    def calculate_total_power_requirements(self) -> dict[str, float]:
        """Calculate total power requirements for the system"""

        total_idle = 0.0
        total_active = 0.0
        total_peak = 0.0
        total_thermal = 0.0

        component_details = {}

        for req in self.power_requirements:
            # Calculate average power based on duty cycle
            avg_power = (req.idle_power_w * (100 - req.duty_cycle_pct) / 100 +
                        req.active_power_w * req.duty_cycle_pct / 100)

            # Account for efficiency
            input_power = avg_power / req.efficiency

            total_idle += req.idle_power_w / req.efficiency
            total_active += req.active_power_w / req.efficiency
            total_peak += req.peak_power_w / req.efficiency
            total_thermal += req.thermal_dissipation_w

            component_details[req.component] = {
                'idle_power_w': req.idle_power_w / req.efficiency,
                'active_power_w': req.active_power_w / req.efficiency,
                'peak_power_w': req.peak_power_w / req.efficiency,
                'average_power_w': input_power,
                'thermal_dissipation_w': req.thermal_dissipation_w,
                'duty_cycle_pct': req.duty_cycle_pct,
                'efficiency': req.efficiency
            }

        # Apply redundancy factor
        total_idle *= self.system_specs.redundancy_factor
        total_active *= self.system_specs.redundancy_factor
        total_peak *= self.system_specs.redundancy_factor
        total_thermal *= self.system_specs.redundancy_factor

        # Calculate average system power
        # Assume 20% idle, 70% active, 10% peak operation
        avg_system_power = (total_idle * 0.2 + total_active * 0.7 + total_peak * 0.1)

        return {
            'total_idle_power_w': total_idle,
            'total_active_power_w': total_active,
            'total_peak_power_w': total_peak,
            'average_system_power_w': avg_system_power,
            'total_thermal_dissipation_w': total_thermal,
            'redundancy_factor': self.system_specs.redundancy_factor,
            'component_breakdown': component_details
        }

    def calculate_cost_analysis(self, analysis_years: int = 10) -> dict[str, Any]:
        """Calculate comprehensive cost analysis over specified years"""

        total_initial_cost = 0.0
        total_recurring_cost_per_year = 0.0
        category_costs = {category: {'initial': 0.0, 'recurring': 0.0, 'total': 0.0}
                         for category in CostCategory}

        item_details = {}

        for item in self.cost_items:
            # Calculate amortized initial cost
            if item.useful_life_years == float('inf'):
                amortized_initial = 0.0  # Software doesn't depreciate
            else:
                amortized_initial = item.initial_cost / item.useful_life_years

            # Calculate maintenance cost
            maintenance_cost = item.initial_cost * item.maintenance_factor

            # Calculate total recurring cost
            total_recurring = item.recurring_cost_per_year + maintenance_cost

            # Calculate total cost over analysis period
            if item.useful_life_years == float('inf'):
                replacement_cost = 0.0
            else:
                replacements = analysis_years / item.useful_life_years
                replacement_cost = item.initial_cost * max(0, replacements - 1)

            total_item_cost = (item.initial_cost +
                             total_recurring * analysis_years +
                             replacement_cost)

            total_initial_cost += item.initial_cost
            total_recurring_cost_per_year += total_recurring

            # Update category totals
            category_costs[item.category]['initial'] += item.initial_cost
            category_costs[item.category]['recurring'] += total_recurring
            category_costs[item.category]['total'] += total_item_cost

            item_details[item.item_name] = {
                'category': item.category.value,
                'initial_cost': item.initial_cost,
                'recurring_cost_per_year': total_recurring,
                'maintenance_cost_per_year': maintenance_cost,
                'amortized_cost_per_year': amortized_initial,
                'total_cost_over_period': total_item_cost,
                'useful_life_years': item.useful_life_years
            }

        # Calculate operational costs
        power_analysis = self.calculate_total_power_requirements()
        annual_energy_cost = (power_analysis['average_system_power_w'] / 1000.0 *
                            8760 * self.energy_cost_per_kwh)  # kW * hours/year * $/kWh

        # Add operational costs
        operational_cost_per_year = annual_energy_cost + self.facility_cost_per_month * 12
        total_recurring_cost_per_year += operational_cost_per_year

        # Calculate total cost of ownership
        total_cost_of_ownership = (total_initial_cost +
                                 total_recurring_cost_per_year * analysis_years)

        # Calculate cost per unit metrics
        annual_operating_hours = 8760  # 24/7 operation
        cost_per_operating_hour = total_cost_of_ownership / (analysis_years * annual_operating_hours)

        return {
            'analysis_period_years': analysis_years,
            'total_initial_cost': total_initial_cost,
            'total_recurring_cost_per_year': total_recurring_cost_per_year,
            'annual_energy_cost': annual_energy_cost,
            'annual_facility_cost': self.facility_cost_per_month * 12,
            'total_cost_of_ownership': total_cost_of_ownership,
            'cost_per_operating_hour': cost_per_operating_hour,
            'category_breakdown': category_costs,
            'item_details': item_details,
            'power_requirements': power_analysis
        }

    def generate_cost_report(self, analysis_years: int = 10) -> str:
        """Generate a comprehensive cost analysis report"""

        analysis = self.calculate_cost_analysis(analysis_years)
        power_req = analysis['power_requirements']

        report = []
        report.append("MFC Controller System Cost Analysis Report")
        report.append("=" * 50)
        report.append(f"Analysis Period: {analysis_years} years")
        report.append(f"System Lifetime: {self.system_specs.expected_lifetime_years} years")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Initial Investment: ${analysis['total_initial_cost']:,.2f}")
        report.append(f"Annual Operating Cost: ${analysis['total_recurring_cost_per_year']:,.2f}")
        report.append(f"Total Cost of Ownership: ${analysis['total_cost_of_ownership']:,.2f}")
        report.append(f"Cost per Operating Hour: ${analysis['cost_per_operating_hour']:.4f}")
        report.append("")

        # Power Requirements
        report.append("POWER REQUIREMENTS")
        report.append("-" * 20)
        report.append(f"Peak Power Demand: {power_req['total_peak_power_w']:.1f} W")
        report.append(f"Active Power Consumption: {power_req['total_active_power_w']:.1f} W")
        report.append(f"Idle Power Consumption: {power_req['total_idle_power_w']:.1f} W")
        report.append(f"Average Power Consumption: {power_req['average_system_power_w']:.1f} W")
        report.append(f"Thermal Dissipation: {power_req['total_thermal_dissipation_w']:.1f} W")
        report.append(f"Annual Energy Consumption: {power_req['average_system_power_w'] * 8760 / 1000:.1f} kWh")
        report.append(f"Annual Energy Cost: ${analysis['annual_energy_cost']:,.2f}")
        report.append("")

        # Cost Breakdown by Category
        report.append("COST BREAKDOWN BY CATEGORY")
        report.append("-" * 30)
        for category, costs in analysis['category_breakdown'].items():
            if costs['total'] > 0:
                report.append(f"{category.value.title()}:")
                report.append(f"  Initial: ${costs['initial']:,.2f}")
                report.append(f"  Recurring: ${costs['recurring']:,.2f}/year")
                report.append(f"  Total ({analysis_years}y): ${costs['total']:,.2f}")
                report.append("")

        # Power Breakdown by Component
        report.append("POWER BREAKDOWN BY COMPONENT")
        report.append("-" * 32)
        for component, details in power_req['component_breakdown'].items():
            report.append(f"{component}:")
            report.append(f"  Average Power: {details['average_power_w']:.1f} W")
            report.append(f"  Peak Power: {details['peak_power_w']:.1f} W")
            report.append(f"  Duty Cycle: {details['duty_cycle_pct']:.1f}%")
            report.append(f"  Efficiency: {details['efficiency']:.1%}")
            report.append("")

        # Key Cost Items
        report.append("TOP COST ITEMS")
        report.append("-" * 15)
        sorted_items = sorted(analysis['item_details'].items(),
                            key=lambda x: x[1]['total_cost_over_period'],
                            reverse=True)

        for item_name, details in sorted_items[:10]:  # Top 10 items
            report.append(f"{item_name}:")
            report.append(f"  Total Cost: ${details['total_cost_over_period']:,.2f}")
            report.append(f"  Annual Cost: ${details['recurring_cost_per_year']:,.2f}")
            report.append("")

        # Recommendations
        report.append("COST OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 35)

        # Check for high-cost items
        high_cost_threshold = analysis['total_cost_of_ownership'] * 0.1
        high_cost_items = [name for name, details in analysis['item_details'].items()
                          if details['total_cost_over_period'] > high_cost_threshold]

        if high_cost_items:
            report.append("High-cost items requiring attention:")
            for item in high_cost_items:
                report.append(f"  - {item}")
            report.append("")

        # Power optimization recommendations
        high_power_threshold = power_req['average_system_power_w'] * 0.2
        high_power_components = [comp for comp, details in power_req['component_breakdown'].items()
                               if details['average_power_w'] > high_power_threshold]

        if high_power_components:
            report.append("High power consumption components:")
            for comp in high_power_components:
                report.append(f"  - {comp}")
            report.append("  Consider power optimization or efficiency improvements")
            report.append("")

        # General recommendations
        report.append("General recommendations:")
        report.append("  - Consider bulk purchasing for recurring items")
        report.append("  - Evaluate open-source alternatives for software licensing")
        report.append("  - Implement predictive maintenance to reduce failure costs")
        report.append("  - Monitor energy consumption and consider power management")
        report.append("")

        return "\n".join(report)

    def compare_configurations(self, other_analyzer: 'ControllerCostAnalyzer',
                             analysis_years: int = 10) -> dict[str, Any]:
        """Compare this configuration with another"""

        analysis_a = self.calculate_cost_analysis(analysis_years)
        analysis_b = other_analyzer.calculate_cost_analysis(analysis_years)

        comparison = {
            'configuration_a': {
                'total_cost': analysis_a['total_cost_of_ownership'],
                'initial_cost': analysis_a['total_initial_cost'],
                'annual_cost': analysis_a['total_recurring_cost_per_year'],
                'average_power_w': analysis_a['power_requirements']['average_system_power_w']
            },
            'configuration_b': {
                'total_cost': analysis_b['total_cost_of_ownership'],
                'initial_cost': analysis_b['total_initial_cost'],
                'annual_cost': analysis_b['total_recurring_cost_per_year'],
                'average_power_w': analysis_b['power_requirements']['average_system_power_w']
            },
            'differences': {
                'total_cost_diff': analysis_b['total_cost_of_ownership'] - analysis_a['total_cost_of_ownership'],
                'initial_cost_diff': analysis_b['total_initial_cost'] - analysis_a['total_initial_cost'],
                'annual_cost_diff': analysis_b['total_recurring_cost_per_year'] - analysis_a['total_recurring_cost_per_year'],
                'power_diff_w': (analysis_b['power_requirements']['average_system_power_w'] -
                               analysis_a['power_requirements']['average_system_power_w'])
            }
        }

        return comparison


def create_standard_controller_configurations() -> dict[str, ControllerCostAnalyzer]:
    """Create standard controller cost analysis configurations"""

    # High-performance configuration
    hp_inference_specs = InferenceSpecs(
        model_format=ModelFormat.NUMPY,  # Use actual enum value
        max_inference_time_ms=1.0,
        memory_limit_mb=512.0,
        cache_size=1000,
        batch_processing=True,
        quantization=True,
        optimization_level=2,
        power_consumption=5.0,
        cost=500.0,
        cpu_cores=4,
        ram_mb=1024.0,
        storage_mb=128.0,
        temperature_range=(-10, 70)
    )

    hp_specs = ControllerSystemSpecs(
        inference_engine_specs=hp_inference_specs,
        electronics_power_w=8.0,
        real_time_controller_overhead_w=3.0,
        hal_power_w=2.0,
        communication_power_w=4.0,
        cooling_power_w=15.0,
        redundancy_factor=1.2,
        development_person_months=18.0,
        testing_person_months=6.0,
        documentation_person_months=3.0,
        certification_required=True,
        expected_lifetime_years=10.0,
        maintenance_interval_months=6.0,
        software_update_frequency_months=12.0
    )

    # Low-cost configuration
    lc_inference_specs = InferenceSpecs(
        model_format=ModelFormat.JSON,  # Use actual enum value
        max_inference_time_ms=10.0,
        memory_limit_mb=64.0,
        cache_size=100,
        batch_processing=False,
        quantization=True,
        optimization_level=1,
        power_consumption=0.5,
        cost=100.0,
        cpu_cores=1,
        ram_mb=128.0,
        storage_mb=32.0,
        temperature_range=(-40, 85)
    )

    lc_specs = ControllerSystemSpecs(
        inference_engine_specs=lc_inference_specs,
        electronics_power_w=2.0,
        real_time_controller_overhead_w=1.0,
        hal_power_w=0.5,
        communication_power_w=1.0,
        cooling_power_w=3.0,
        redundancy_factor=1.1,
        development_person_months=12.0,
        testing_person_months=3.0,
        documentation_person_months=2.0,
        certification_required=False,
        expected_lifetime_years=7.0,
        maintenance_interval_months=12.0,
        software_update_frequency_months=24.0
    )

    configurations = {
        'high_performance': ControllerCostAnalyzer(hp_specs),
        'low_cost': ControllerCostAnalyzer(lc_specs)
    }

    return configurations


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create cost analyzers
    configurations = create_standard_controller_configurations()

    # Test high-performance configuration
    hp_analyzer = configurations['high_performance']

    print("MFC Controller System Cost Analysis")
    print("=" * 40)

    # Generate cost analysis
    analysis = hp_analyzer.calculate_cost_analysis(analysis_years=10)

    print("High-Performance Configuration:")
    print(f"Total Initial Cost: ${analysis['total_initial_cost']:,.2f}")
    print(f"Annual Operating Cost: ${analysis['total_recurring_cost_per_year']:,.2f}")
    print(f"10-Year Total Cost: ${analysis['total_cost_of_ownership']:,.2f}")
    print(f"Average Power: {analysis['power_requirements']['average_system_power_w']:.1f}W")
    print()

    # Test low-cost configuration
    lc_analyzer = configurations['low_cost']
    lc_analysis = lc_analyzer.calculate_cost_analysis(analysis_years=10)

    print("Low-Cost Configuration:")
    print(f"Total Initial Cost: ${lc_analysis['total_initial_cost']:,.2f}")
    print(f"Annual Operating Cost: ${lc_analysis['total_recurring_cost_per_year']:,.2f}")
    print(f"10-Year Total Cost: ${lc_analysis['total_cost_of_ownership']:,.2f}")
    print(f"Average Power: {lc_analysis['power_requirements']['average_system_power_w']:.1f}W")
    print()

    # Compare configurations
    comparison = hp_analyzer.compare_configurations(lc_analyzer)
    print("Configuration Comparison:")
    print(f"Cost Difference: ${comparison['differences']['total_cost_diff']:,.2f}")
    print(f"Power Difference: {comparison['differences']['power_diff_w']:.1f}W")
    print()

    # Generate detailed report
    print("Detailed Cost Report:")
    print("-" * 20)
    report = hp_analyzer.generate_cost_report(analysis_years=10)
    print(report[:1000] + "..." if len(report) > 1000 else report)
