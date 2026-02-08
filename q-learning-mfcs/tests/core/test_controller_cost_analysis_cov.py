"""Tests for controller_cost_analysis.py - 98%+ coverage target.

Covers ControllerCostAnalyzer, CostCategory, PowerRequirement, CostItem,
ControllerSystemSpecs, and helper functions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from controller_models.controller_cost_analysis import (
    ControllerCostAnalyzer,
    ControllerSystemSpecs,
    CostCategory,
    CostItem,
    PowerRequirement,
    create_standard_controller_configurations,
)
from controller_models.model_inference import InferenceSpecs, ModelFormat


@pytest.fixture
def inference_specs():
    return InferenceSpecs(
        model_format=ModelFormat.NUMPY,
        max_inference_time_ms=1.0,
        memory_limit_mb=256.0,
        cache_size=500,
        batch_processing=True,
        quantization=False,
        optimization_level=1,
        power_consumption=3.0,
        cost=300.0,
        cpu_cores=2,
        ram_mb=512.0,
        storage_mb=64.0,
        temperature_range=(-20, 60),
    )


@pytest.fixture
def system_specs(inference_specs):
    return ControllerSystemSpecs(
        inference_engine_specs=inference_specs,
        electronics_power_w=5.0,
        real_time_controller_overhead_w=2.0,
        hal_power_w=1.0,
        communication_power_w=2.0,
        cooling_power_w=8.0,
        development_person_months=12.0,
        testing_person_months=4.0,
        documentation_person_months=2.0,
        certification_required=True,
        expected_lifetime_years=8.0,
        maintenance_interval_months=6.0,
        software_update_frequency_months=12.0,
        redundancy_factor=1.15,
    )


@pytest.fixture
def system_specs_no_cert(inference_specs):
    return ControllerSystemSpecs(
        inference_engine_specs=inference_specs,
        electronics_power_w=5.0,
        real_time_controller_overhead_w=2.0,
        hal_power_w=1.0,
        communication_power_w=2.0,
        cooling_power_w=8.0,
        development_person_months=12.0,
        testing_person_months=4.0,
        documentation_person_months=2.0,
        certification_required=False,
        expected_lifetime_years=8.0,
        maintenance_interval_months=6.0,
        software_update_frequency_months=12.0,
    )


@pytest.fixture
def analyzer(system_specs):
    return ControllerCostAnalyzer(system_specs)


@pytest.fixture
def analyzer_no_cert(system_specs_no_cert):
    return ControllerCostAnalyzer(system_specs_no_cert)


class TestEnums:
    def test_cost_category_values(self):
        assert CostCategory.HARDWARE.value == "hardware"
        assert CostCategory.SOFTWARE.value == "software"
        assert CostCategory.DEVELOPMENT.value == "development"
        assert CostCategory.LICENSING.value == "licensing"
        assert CostCategory.MAINTENANCE.value == "maintenance"
        assert CostCategory.OPERATIONAL.value == "operational"
        assert CostCategory.TRAINING.value == "training"
        assert CostCategory.INFRASTRUCTURE.value == "infrastructure"


class TestDataclasses:
    def test_power_requirement(self):
        pr = PowerRequirement(
            component="Test", idle_power_w=1.0, active_power_w=5.0,
            peak_power_w=8.0, duty_cycle_pct=50.0,
        )
        assert pr.component == "Test"
        assert pr.efficiency == 1.0
        assert pr.thermal_dissipation_w == 0.0

    def test_power_requirement_with_opts(self):
        pr = PowerRequirement(
            component="Test", idle_power_w=1.0, active_power_w=5.0,
            peak_power_w=8.0, duty_cycle_pct=50.0,
            efficiency=0.9, thermal_dissipation_w=1.5,
        )
        assert pr.efficiency == 0.9
        assert pr.thermal_dissipation_w == 1.5

    def test_cost_item(self):
        ci = CostItem(
            item_name="Widget", category=CostCategory.HARDWARE,
            initial_cost=100.0, recurring_cost_per_year=10.0,
            useful_life_years=5.0,
        )
        assert ci.depreciation_rate == 0.0
        assert ci.maintenance_factor == 0.05

    def test_controller_system_specs(self, inference_specs):
        css = ControllerSystemSpecs(
            inference_engine_specs=inference_specs,
            electronics_power_w=5.0,
            real_time_controller_overhead_w=2.0,
            hal_power_w=1.0,
            communication_power_w=2.0,
            cooling_power_w=8.0,
            development_person_months=12.0,
            testing_person_months=4.0,
            documentation_person_months=2.0,
            certification_required=False,
            expected_lifetime_years=8.0,
            maintenance_interval_months=6.0,
            software_update_frequency_months=12.0,
        )
        assert css.redundancy_factor == 1.2


class TestControllerCostAnalyzerInit:
    def test_init_with_certification(self, analyzer):
        assert len(analyzer.cost_items) > 0
        assert len(analyzer.power_requirements) > 0
        cert_items = [
            i for i in analyzer.cost_items
            if i.item_name == "Safety Certification"
        ]
        assert len(cert_items) == 1

    def test_init_without_certification(self, analyzer_no_cert):
        cert_items = [
            i for i in analyzer_no_cert.cost_items
            if i.item_name == "Safety Certification"
        ]
        assert len(cert_items) == 0

    def test_standard_cost_factors(self, analyzer):
        assert analyzer.engineer_cost_per_month == 12000.0
        assert analyzer.energy_cost_per_kwh == 0.15
        assert analyzer.facility_cost_per_month == 2000.0


class TestAddMethods:
    def test_add_cost_item(self, analyzer):
        initial_count = len(analyzer.cost_items)
        analyzer.add_cost_item(CostItem(
            item_name="Custom", category=CostCategory.HARDWARE,
            initial_cost=50.0, recurring_cost_per_year=5.0,
            useful_life_years=3.0,
        ))
        assert len(analyzer.cost_items) == initial_count + 1

    def test_add_power_requirement(self, analyzer):
        initial_count = len(analyzer.power_requirements)
        analyzer.add_power_requirement(PowerRequirement(
            component="Custom", idle_power_w=0.5, active_power_w=2.0,
            peak_power_w=3.0, duty_cycle_pct=40.0,
        ))
        assert len(analyzer.power_requirements) == initial_count + 1


class TestPowerRequirements:
    def test_total_power_requirements(self, analyzer):
        result = analyzer.calculate_total_power_requirements()
        assert "total_idle_power_w" in result
        assert "total_active_power_w" in result
        assert "total_peak_power_w" in result
        assert "average_system_power_w" in result
        assert "total_thermal_dissipation_w" in result
        assert "redundancy_factor" in result
        assert "component_breakdown" in result
        assert result["total_idle_power_w"] > 0
        assert result["total_active_power_w"] > 0

    def test_component_breakdown(self, analyzer):
        result = analyzer.calculate_total_power_requirements()
        breakdown = result["component_breakdown"]
        assert len(breakdown) >= 6
        for comp, details in breakdown.items():
            assert "idle_power_w" in details
            assert "average_power_w" in details
            assert "efficiency" in details


class TestCostAnalysis:
    def test_cost_analysis_basic(self, analyzer):
        result = analyzer.calculate_cost_analysis(analysis_years=5)
        assert result["analysis_period_years"] == 5
        assert result["total_initial_cost"] > 0
        assert result["total_cost_of_ownership"] > 0
        assert result["cost_per_operating_hour"] > 0

    def test_cost_analysis_10_years(self, analyzer):
        result = analyzer.calculate_cost_analysis(analysis_years=10)
        assert result["analysis_period_years"] == 10

    def test_cost_analysis_category_breakdown(self, analyzer):
        result = analyzer.calculate_cost_analysis()
        categories = result["category_breakdown"]
        assert CostCategory.HARDWARE in categories
        assert CostCategory.SOFTWARE in categories

    def test_cost_analysis_item_details(self, analyzer):
        result = analyzer.calculate_cost_analysis()
        details = result["item_details"]
        assert len(details) > 0
        for item_name, item_detail in details.items():
            assert "category" in item_detail
            assert "initial_cost" in item_detail
            assert "total_cost_over_period" in item_detail

    def test_cost_analysis_infinite_life_items(self, analyzer):
        result = analyzer.calculate_cost_analysis()
        details = result["item_details"]
        rtos = details.get("Real-Time Operating System")
        if rtos:
            assert rtos["useful_life_years"] == float("inf")

    def test_cost_analysis_energy_cost(self, analyzer):
        result = analyzer.calculate_cost_analysis()
        assert result["annual_energy_cost"] > 0
        assert result["annual_facility_cost"] > 0


class TestCostReport:
    def test_generate_report(self, analyzer):
        report = analyzer.generate_cost_report(analysis_years=5)
        assert isinstance(report, str)
        assert "MFC Controller System Cost Analysis Report" in report
        assert "EXECUTIVE SUMMARY" in report
        assert "POWER REQUIREMENTS" in report
        assert "COST BREAKDOWN BY CATEGORY" in report
        assert "POWER BREAKDOWN BY COMPONENT" in report
        assert "TOP COST ITEMS" in report
        assert "COST OPTIMIZATION RECOMMENDATIONS" in report

    def test_report_contains_numbers(self, analyzer):
        report = analyzer.generate_cost_report()
        assert "$" in report
        assert "W" in report


class TestCompareConfigurations:
    def test_compare(self, system_specs, system_specs_no_cert):
        analyzer_a = ControllerCostAnalyzer(system_specs)
        analyzer_b = ControllerCostAnalyzer(system_specs_no_cert)
        result = analyzer_a.compare_configurations(analyzer_b, analysis_years=5)
        assert "configuration_a" in result
        assert "configuration_b" in result
        assert "differences" in result
        assert "total_cost_diff" in result["differences"]
        assert "power_diff_w" in result["differences"]


class TestCreateStandardConfigurations:
    def test_creates_two_configs(self):
        configs = create_standard_controller_configurations()
        assert "high_performance" in configs
        assert "low_cost" in configs

    def test_configs_are_analyzers(self):
        configs = create_standard_controller_configurations()
        assert isinstance(configs["high_performance"], ControllerCostAnalyzer)
        assert isinstance(configs["low_cost"], ControllerCostAnalyzer)

    def test_hp_has_certification(self):
        configs = create_standard_controller_configurations()
        hp = configs["high_performance"]
        cert_items = [
            i for i in hp.cost_items
            if i.item_name == "Safety Certification"
        ]
        assert len(cert_items) == 1

    def test_lc_no_certification(self):
        configs = create_standard_controller_configurations()
        lc = configs["low_cost"]
        cert_items = [
            i for i in lc.cost_items
            if i.item_name == "Safety Certification"
        ]
        assert len(cert_items) == 0
