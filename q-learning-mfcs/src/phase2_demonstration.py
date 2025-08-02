"""
Phase 2 Enhanced MFC Control System Demonstration

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_measurements(time_hours: float, base_thickness: float = 15.0) -> tuple:
    """Create sample EIS and QCM measurements for demonstration."""

    # Add some realistic noise and drift
    noise_factor = 0.1
    drift = 0.02 * time_hours

    # EIS measurement
    eis_thickness = base_thickness + drift + np.random.normal(0, noise_factor * base_thickness)
    eis_impedance = 1500 - eis_thickness * 100 + np.random.normal(0, 50)

    eis_measurement = EISMeasurement(
        frequency=1000.0,
        impedance_magnitude=eis_impedance,
        impedance_phase=-15.0 + np.random.normal(0, 2),
        real_impedance=eis_impedance * np.cos(-15.0 * np.pi / 180),
        imaginary_impedance=eis_impedance * np.sin(-15.0 * np.pi / 180),
        timestamp=time_hours,
        temperature=25.0 + 273.15 + np.random.normal(0, 1)  # Convert to Kelvin
    )

    # QCM measurement
    qcm_thickness = base_thickness + drift * 0.8 + np.random.normal(0, noise_factor * base_thickness * 0.8)
    qcm_frequency_shift = -qcm_thickness * 25 + np.random.normal(0, 10)

    qcm_measurement = QCMMeasurement(
        frequency=5_000_000.0,
        frequency_shift=qcm_frequency_shift,
        dissipation=1e-6 * (1 + qcm_thickness / 100),
        quality_factor=8000 + np.random.normal(0, 500),
        timestamp=time_hours,
        temperature=25.0 + 273.15 + np.random.normal(0, 1)  # Convert to Kelvin
    )

    # Create EIS and QCM models for property extraction
    eis_model = EISModel(BacterialSpecies.MIXED)
    qcm_model = QCMModel()

    # Extract properties
    eis_properties = eis_model.get_biofilm_properties([eis_measurement])
    qcm_properties = qcm_model.estimate_biofilm_properties(qcm_measurement)

    return eis_measurement, qcm_measurement, eis_properties, qcm_properties


def demonstrate_phase2_enhancements():
    """Demonstrate all Phase 2 enhancements in action."""

    print("🚀 Phase 2 Enhanced MFC Control System Demonstration")
    print("=" * 60)

    # 1. Initialize ML-optimized controller
    print("\n1️⃣  Initializing ML-Optimized MFC Controller...")

    controller = create_ml_optimized_controller(
        species=BacterialSpecies.MIXED,
        optimization_strategy=OptimizationStrategy.BAYESIAN
    )

    print(f"   ✅ Controller initialized for {BacterialSpecies.MIXED.value}")
    print(f"   ✅ Optimization strategy: {OptimizationStrategy.BAYESIAN.value}")

    # 2. Simulate control sequence
    print("\n2️⃣  Running Control Simulation...")

    simulation_hours = 10.0
    time_step = 0.5  # 30 minute steps
    control_results = []

    base_thickness = 10.0  # Starting thickness

    for step in range(int(simulation_hours / time_step)):
        current_time = step * time_step

        print(f"\n   🕐 Step {step + 1}: t = {current_time:.1f} hours")

        # Generate measurements with some biofilm growth
        growing_thickness = base_thickness + current_time * 0.5  # 0.5 μm/hour growth
        eis_measurement, qcm_measurement, eis_props, qcm_props = create_sample_measurements(
            current_time, growing_thickness
        )

        # Execute control step
        result = controller.control_step_with_learning(
            eis_measurement, qcm_measurement, eis_props, qcm_props, current_time
        )

        control_results.append(result)

        # Display key metrics
        health_score = result['system_health_score']
        fusion_confidence = result['system_state'].fused_measurement.fusion_confidence
        sensor_agreement = result['system_state'].fused_measurement.sensor_agreement

        print(f"      📊 Health Score: {health_score:.3f}")
        print(f"      🔗 Fusion Confidence: {fusion_confidence:.3f}")
        print(f"      🤝 Sensor Agreement: {sensor_agreement:.3f}")

        # Show prediction if available
        if result.get('prediction'):
            pred_thickness = result['prediction'].predicted_values[0]
            pred_confidence = result['prediction'].prediction_accuracy
            print(f"      🔮 Predicted Thickness (24h): {pred_thickness:.1f} μm (confidence: {pred_confidence:.3f})")

        # Show alerts
        if result['health_alerts']:
            print(f"      ⚠️  Health Alerts: {len(result['health_alerts'])}")
            for alert in result['health_alerts'][:2]:  # Show first 2 alerts
                print(f"         • {alert.severity.upper()}: {alert.message}")

        # Show ML insights
        if 'ml_insights' in result:
            ml_status = result['ml_insights']['learning_status']
            print(f"      🧠 ML Data Points: {ml_status['data_points_collected']}")

            if result['ml_insights']['optimization_recommendations']:
                print(f"      💡 Recommendations: {len(result['ml_insights']['optimization_recommendations'])}")
                for rec in result['ml_insights']['optimization_recommendations'][:1]:
                    print(f"         • {rec}")

        # Show optimization events
        if 'optimization_result' in result:
            opt_result = result['optimization_result']
            print(f"      🎯 Optimization: {opt_result.performance_improvement:.3f} improvement")

    # 3. Final Analysis
    print("\n3️⃣  Final System Analysis...")

    # Get comprehensive status
    final_status = controller.get_ml_status_report()
    base_status = controller.base_controller.get_comprehensive_status()

    print("\n   📈 Learning Progress:")
    print(f"      • Total data points collected: {final_status['learning_progress']['total_data_points']}")
    print(f"      • Optimizations performed: {final_status['learning_progress']['optimizations_performed']}")
    print(f"      • Features generated: {final_status['feature_engineering']['features_generated']}")

    print("\n   🎯 System Performance:")
    print(f"      • Last optimization score: {final_status['performance_metrics']['last_optimization_score']:.3f}")
    print(f"      • Average performance: {final_status['performance_metrics']['average_performance']:.3f}")

    print("\n   🏥 System Health:")
    print(f"      • Overall health: {base_status['system_health']['overall_score']:.3f}")
    print(f"      • Health status: {base_status['system_health']['status']}")
    print(f"      • Health trend: {base_status['system_health']['trend']}")

    print("\n   🔧 Control Statistics:")
    print(f"      • Control strategy: {base_status['control_strategy']}")
    print(f"      • Active alerts: {base_status['active_alerts']}")
    print(f"      • Strategy changes (24h): {base_status['strategy_changes_24h']}")
    print(f"      • Interventions (24h): {base_status['interventions_24h']}")

    # 4. Feature Importance Analysis
    print("\n4️⃣  Feature Importance Analysis...")

    if control_results and 'feature_importance' in control_results[-1]:
        feature_importance = control_results[-1]['feature_importance']

        if feature_importance:
            print("\n   🔍 Top Features for Control Decisions:")
            for i, feature in enumerate(feature_importance[:5]):
                print(f"      {i+1}. {feature.feature_name} ({feature.feature_type.value})")
                print(f"         Importance: {feature.importance_score:.3f}")
                print(f"         Description: {feature.description}")
        else:
            print("   ℹ️  Feature importance analysis requires more data")

    # 5. Performance Summary
    print("\n5️⃣  Performance Summary...")

    if len(control_results) > 1:
        initial_health = control_results[0]['system_health_score']
        final_health = control_results[-1]['system_health_score']
        health_improvement = final_health - initial_health

        print("\n   📊 Health Evolution:")
        print(f"      • Initial health: {initial_health:.3f}")
        print(f"      • Final health: {final_health:.3f}")
        print(f"      • Net improvement: {health_improvement:+.3f}")

        # Calculate average metrics
        avg_fusion_confidence = np.mean([r['system_state'].fused_measurement.fusion_confidence
                                       for r in control_results])
        avg_sensor_agreement = np.mean([r['system_state'].fused_measurement.sensor_agreement
                                      for r in control_results])

        print("\n   🔗 Sensor Performance:")
        print(f"      • Average fusion confidence: {avg_fusion_confidence:.3f}")
        print(f"      • Average sensor agreement: {avg_sensor_agreement:.3f}")

        # Count anomalies and interventions
        total_anomalies = sum(len(r['system_state'].anomalies) for r in control_results)
        total_interventions = sum(1 for r in control_results
                                if r['control_decision'].intervention_type is not None)

        print("\n   ⚠️  System Events:")
        print(f"      • Total anomalies detected: {total_anomalies}")
        print(f"      • Total interventions: {total_interventions}")

    print("\n" + "=" * 60)
    print("🎉 Phase 2 Demonstration Complete!")
    print("\nKey achievements demonstrated:")
    print("  ✅ Advanced sensor fusion with predictive capabilities")
    print("  ✅ Predictive biofilm health monitoring with risk assessment")
    print("  ✅ Adaptive control algorithms with health-aware Q-learning")
    print("  ✅ Machine learning optimization with continuous learning")
    print("  ✅ Real-time anomaly detection and intervention strategies")
    print("  ✅ Multi-objective optimization balancing power and health")


def demonstrate_individual_components():
    """Demonstrate individual Phase 2 components."""

    print("\n🔬 Individual Component Demonstrations")
    print("=" * 60)

    # Advanced Sensor Fusion
    print("\n1️⃣  Advanced Sensor Fusion with Predictions...")

    from sensing_models.advanced_sensor_fusion import create_advanced_sensor_fusion
    fusion_system = create_advanced_sensor_fusion()

    # Create sample measurements
    eis_measurement, qcm_measurement, eis_props, qcm_props = create_sample_measurements(1.0, 15.0)

    # Fusion with prediction
    fused, prediction, anomalies = fusion_system.fuse_measurements_with_prediction(
        eis_measurement, qcm_measurement, eis_props, qcm_props, 1.0, predict_steps=5
    )

    print(f"   ✅ Fused thickness: {fused.thickness_um:.2f} ± {fused.thickness_uncertainty:.2f} μm")
    print(f"   ✅ Sensor agreement: {fused.sensor_agreement:.3f}")
    print(f"   ✅ Fusion confidence: {fused.fusion_confidence:.3f}")

    if prediction:
        print(f"   🔮 Predicted thickness (5 steps): {prediction.predicted_values[0]:.2f} μm")
        print(f"   🔮 Prediction accuracy: {prediction.prediction_accuracy:.3f}")

    if anomalies:
        print(f"   ⚠️  Anomalies detected: {len(anomalies)}")
        for anomaly in anomalies:
            print(f"      • {anomaly.anomaly_type}: {anomaly.message}")

    # Predictive Health Monitoring
    print("\n2️⃣  Predictive Biofilm Health Monitoring...")

    from biofilm_health_monitor import create_predictive_health_monitor
    health_monitor = create_predictive_health_monitor(BacterialSpecies.MIXED)

    # Health assessment
    health_metrics = health_monitor.assess_health(fused)
    health_alerts = health_monitor.generate_alerts(health_metrics, anomalies)
    recommendations = health_monitor.generate_intervention_recommendations(health_metrics)

    print(f"   ✅ Overall health: {health_metrics.overall_health_score:.3f}")
    print(f"   ✅ Health status: {health_metrics.health_status.value}")
    print(f"   ✅ Health trend: {health_metrics.health_trend.value}")
    print(f"   ✅ Predicted health (24h): {health_metrics.predicted_health_24h:.3f}")

    print("   🚨 Risk Assessment:")
    print(f"      • Fouling risk: {health_metrics.fouling_risk:.3f}")
    print(f"      • Detachment risk: {health_metrics.detachment_risk:.3f}")
    print(f"      • Stagnation risk: {health_metrics.stagnation_risk:.3f}")

    if health_alerts:
        print(f"   ⚠️  Health alerts: {len(health_alerts)}")
        for alert in health_alerts[:2]:
            print(f"      • {alert.severity}: {alert.message}")

    if recommendations:
        print(f"   💡 Intervention recommendations: {len(recommendations)}")
        for rec in recommendations[:2]:
            print(f"      • {rec.intervention_type}: {rec.description}")
            print(f"        Urgency: {rec.urgency}, Success prob: {rec.success_probability:.2f}")

    # ML Optimization
    print("\n3️⃣  Machine Learning Optimization...")

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
        intervention_active=False
    )

    performance_metrics = {
        'power_efficiency': 0.8,
        'biofilm_health_score': health_metrics.overall_health_score,
        'sensor_reliability': fused.fusion_confidence,
        'system_stability': 0.7,
        'control_confidence': 0.8
    }

    # Extract features
    features = feature_engineer.extract_features(sample_state, performance_metrics)

    print(f"   ✅ Generated features: {len(features)}")
    print("   ✅ Feature categories: raw, statistical, temporal, health, derived")

    # Show some key features
    key_features = dict(list(features.items())[:5])
    print("   📊 Sample features:")
    for name, value in key_features.items():
        print(f"      • {name}: {value:.3f}")

    print("\n" + "=" * 60)
    print("🎯 Individual Component Demonstrations Complete!")


if __name__ == "__main__":
    """Run Phase 2 demonstration."""

    try:
        # Main demonstration
        demonstrate_phase2_enhancements()

        # Individual component demonstrations
        demonstrate_individual_components()

        print("\n🏆 Phase 2 Enhanced MFC Control System demonstration completed successfully!")
        print("\nThe system now features:")
        print("  • Predictive sensor fusion with multi-step forecasting")
        print("  • Health-aware control with risk assessment")
        print("  • Adaptive algorithms with continuous learning")
        print("  • ML-based parameter optimization")
        print("  • Real-time anomaly detection and intervention")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise
