"""Phase 2 Enhanced MFC Control System Demonstration.

Demonstrates the integrated Phase 2 enhancements:
- Advanced sensor fusion with predictive capabilities
- Predictive biofilm health monitoring with risk assessment
- Adaptive control algorithms with health-aware Q-learning
- Machine learning optimization with continuous learning

Created: 2025-07-31
Last Modified: 2025-07-31
"""

import logging

import numpy as np

# Import Phase 2 components
from ml_optimization import OptimizationStrategy, create_ml_optimized_controller
from sensing_models.eis_model import EISMeasurement, EISModel
from sensing_models.qcm_model import QCMMeasurement, QCMModel
from sensing_models.sensor_fusion import BacterialSpecies

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_sample_measurements(
    time_hours: float,
    base_thickness: float = 15.0,
) -> tuple:
    """Create sample EIS and QCM measurements for demonstration."""
    # Add some realistic noise and drift
    noise_factor = 0.1
    drift = 0.02 * time_hours

    # EIS measurement
    eis_thickness = (
        base_thickness + drift + np.random.normal(0, noise_factor * base_thickness)
    )
    eis_impedance = 1500 - eis_thickness * 100 + np.random.normal(0, 50)

    eis_measurement = EISMeasurement(
        frequency=1000.0,
        impedance_magnitude=eis_impedance,
        impedance_phase=-15.0 + np.random.normal(0, 2),
        real_impedance=eis_impedance * np.cos(-15.0 * np.pi / 180),
        imaginary_impedance=eis_impedance * np.sin(-15.0 * np.pi / 180),
        timestamp=time_hours,
        temperature=25.0 + 273.15 + np.random.normal(0, 1),  # Convert to Kelvin
    )

    # QCM measurement
    qcm_thickness = (
        base_thickness
        + drift * 0.8
        + np.random.normal(0, noise_factor * base_thickness * 0.8)
    )
    qcm_frequency_shift = -qcm_thickness * 25 + np.random.normal(0, 10)

    qcm_measurement = QCMMeasurement(
        frequency=5_000_000.0,
        frequency_shift=qcm_frequency_shift,
        dissipation=1e-6 * (1 + qcm_thickness / 100),
        quality_factor=8000 + np.random.normal(0, 500),
        timestamp=time_hours,
        temperature=25.0 + 273.15 + np.random.normal(0, 1),  # Convert to Kelvin
    )

    # Create EIS and QCM models for property extraction
    eis_model = EISModel(BacterialSpecies.MIXED)
    qcm_model = QCMModel()

    # Extract properties
    eis_properties = eis_model.get_biofilm_properties([eis_measurement])
    qcm_properties = qcm_model.estimate_biofilm_properties(qcm_measurement)

    return eis_measurement, qcm_measurement, eis_properties, qcm_properties


def demonstrate_phase2_enhancements() -> None:
    """Demonstrate all Phase 2 enhancements in action."""
    # 1. Initialize ML-optimized controller

    controller = create_ml_optimized_controller(
        species=BacterialSpecies.MIXED,
        optimization_strategy=OptimizationStrategy.BAYESIAN,
    )

    # 2. Simulate control sequence

    simulation_hours = 10.0
    time_step = 0.5  # 30 minute steps
    control_results = []

    base_thickness = 10.0  # Starting thickness

    for step in range(int(simulation_hours / time_step)):
        current_time = step * time_step

        # Generate measurements with some biofilm growth
        growing_thickness = base_thickness + current_time * 0.5  # 0.5 μm/hour growth
        eis_measurement, qcm_measurement, eis_props, qcm_props = (
            create_sample_measurements(current_time, growing_thickness)
        )

        # Execute control step
        result = controller.control_step_with_learning(
            eis_measurement,
            qcm_measurement,
            eis_props,
            qcm_props,
            current_time,
        )

        control_results.append(result)

        # Display key metrics
        result["system_health_score"]
        result["system_state"].fused_measurement.fusion_confidence
        result["system_state"].fused_measurement.sensor_agreement

        # Show prediction if available
        if result.get("prediction"):
            result["prediction"].predicted_values[0]
            result["prediction"].prediction_accuracy

        # Show alerts
        if result["health_alerts"]:
            for _alert in result["health_alerts"][:2]:  # Show first 2 alerts
                pass

        # Show ML insights
        if "ml_insights" in result:
            result["ml_insights"]["learning_status"]

            if result["ml_insights"]["optimization_recommendations"]:
                for _rec in result["ml_insights"]["optimization_recommendations"][:1]:
                    pass

        # Show optimization events
        if "optimization_result" in result:
            result["optimization_result"]

    # 3. Final Analysis

    # Get comprehensive status
    controller.get_ml_status_report()
    controller.base_controller.get_comprehensive_status()

    # 4. Feature Importance Analysis

    if control_results and "feature_importance" in control_results[-1]:
        feature_importance = control_results[-1]["feature_importance"]

        if feature_importance:
            for _i, _feature in enumerate(feature_importance[:5]):
                pass
        else:
            pass

    # 5. Performance Summary

    if len(control_results) > 1:
        initial_health = control_results[0]["system_health_score"]
        final_health = control_results[-1]["system_health_score"]
        final_health - initial_health

        # Calculate average metrics
        np.mean(
            [
                r["system_state"].fused_measurement.fusion_confidence
                for r in control_results
            ],
        )
        np.mean(
            [
                r["system_state"].fused_measurement.sensor_agreement
                for r in control_results
            ],
        )

        # Count anomalies and interventions
        sum(len(r["system_state"].anomalies) for r in control_results)
        sum(
            1
            for r in control_results
            if r["control_decision"].intervention_type is not None
        )


def demonstrate_individual_components() -> None:
    """Demonstrate individual Phase 2 components."""
    # Advanced Sensor Fusion

    from sensing_models.advanced_sensor_fusion import create_advanced_sensor_fusion

    fusion_system = create_advanced_sensor_fusion()

    # Create sample measurements
    eis_measurement, qcm_measurement, eis_props, qcm_props = create_sample_measurements(
        1.0,
        15.0,
    )

    # Fusion with prediction
    fused, prediction, anomalies = fusion_system.fuse_measurements_with_prediction(
        eis_measurement,
        qcm_measurement,
        eis_props,
        qcm_props,
        1.0,
        predict_steps=5,
    )

    if prediction:
        pass

    if anomalies:
        for _anomaly in anomalies:
            pass

    # Predictive Health Monitoring

    from biofilm_health_monitor import create_predictive_health_monitor

    health_monitor = create_predictive_health_monitor(BacterialSpecies.MIXED)

    # Health assessment
    health_metrics = health_monitor.assess_health(fused)
    health_alerts = health_monitor.generate_alerts(health_metrics, anomalies)
    recommendations = health_monitor.generate_intervention_recommendations(
        health_metrics,
    )

    if health_alerts:
        for _alert in health_alerts[:2]:
            pass

    if recommendations:
        for _rec in recommendations[:2]:
            pass

    # ML Optimization

    from ml_optimization import FeatureEngineer

    # Feature engineering
    feature_engineer = FeatureEngineer()

    # Create sample system state
    from adaptive_mfc_controller import AdaptationMode, ControlStrategy, SystemState

    sample_state = SystemState(
        fused_measurement=fused,
        prediction=prediction,
        anomalies=anomalies,
        health_metrics=health_metrics,
        health_alerts=health_alerts,
        flow_rate=15.0,
        inlet_concentration=10.0,
        outlet_concentration=8.0,
        current_density=0.5,
        power_output=0.12,
        current_strategy=ControlStrategy.BALANCED,
        adaptation_mode=AdaptationMode.MODERATE,
        intervention_active=False,
    )

    performance_metrics = {
        "power_efficiency": 0.8,
        "biofilm_health_score": health_metrics.overall_health_score,
        "sensor_reliability": fused.fusion_confidence,
        "system_stability": 0.7,
        "control_confidence": 0.8,
    }

    # Extract features
    features = feature_engineer.extract_features(sample_state, performance_metrics)

    # Show some key features
    key_features = dict(list(features.items())[:5])
    for _name, _value in key_features.items():
        pass


if __name__ == "__main__":
    """Run Phase 2 demonstration."""

    try:
        # Main demonstration
        demonstrate_phase2_enhancements()

        # Individual component demonstrations
        demonstrate_individual_components()

    except Exception as e:
        logger.exception(f"Demonstration failed: {e}")
        raise
