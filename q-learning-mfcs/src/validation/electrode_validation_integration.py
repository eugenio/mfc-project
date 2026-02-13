"""
Electrode Validation Integration

This module integrates the Phase 5 literature validation framework
with the existing electrode configuration system, providing real-time
parameter validation for electrode properties.

Created: 2025-08-01
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Import existing electrode system
from ..config.electrode_config import (
    ElectrodeConfiguration,
    ElectrodeGeometry,
    ElectrodeMaterial,
)
from .citation_manager import CitationManager, CitationReport
from .literature_database import LiteratureDatabase, ValidationQuery, ValidationResult
from .quality_assessor import QualityAssessor


@dataclass
class ElectrodeValidationResult:
    """Container for electrode parameter validation results."""

    parameter_name: str
    model_value: float
    units: str
    validation_result: ValidationResult
    quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ElectrodeValidationReport:
    """Comprehensive validation report for electrode configuration."""

    electrode_name: str
    material: ElectrodeMaterial
    geometry: ElectrodeGeometry
    validation_results: List[ElectrodeValidationResult] = field(default_factory=list)
    overall_validation_score: float = 0.0
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    citation_report: Optional[CitationReport] = None
class ElectrodeValidationIntegrator:
    """
    Integration layer between electrode configuration and validation framework.
    
    Provides real-time validation of electrode parameters against literature
    and generates comprehensive validation reports.
    """

    def __init__(self, db_path: str = "data/literature.db", cache_dir: str = "data/pubmed_cache"):
        """Initialize validation integrator."""

        # Initialize validation components
        self.lit_db = LiteratureDatabase(db_path=db_path, cache_dir=cache_dir)
        self.quality_assessor = QualityAssessor()
        self.citation_manager = CitationManager()

        # Define parameter mappings
        self.parameter_mappings = self._define_parameter_mappings()

        # Validation thresholds
        self.validation_thresholds = {
            'critical': 0.8,  # Above this = well validated
            'acceptable': 0.6,  # Above this = acceptable
            'warning': 0.4,   # Below this = needs review
        }

    def _define_parameter_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Define mappings between electrode parameters and literature search terms."""

        return {
            'specific_conductance': {
                'search_terms': ['electrode conductivity', 'electrical conductance', 'resistivity'],
                'units': 'S/m',
                'typical_range': (1e-4, 1e6),
                'organisms': ['general'],
                'conditions': ['aqueous', 'room temperature']
            },
            'contact_resistance': {
                'search_terms': ['electrode resistance', 'contact resistance', 'interface resistance'],
                'units': 'Ω·cm²',
                'typical_range': (1e-6, 1e-2),
                'organisms': ['general'],
                'conditions': ['electrode-electrolyte interface']
            },
            'surface_charge_density': {
                'search_terms': ['surface charge', 'electrode potential', 'zeta potential'],
                'units': 'C/m²',
                'typical_range': (-1.0, 1.0),
                'organisms': ['general'],
                'conditions': ['neutral pH', 'aqueous solution']
            },
            'hydrophobicity_angle': {
                'search_terms': ['contact angle', 'hydrophobicity', 'wettability'],
                'units': 'degrees',
                'typical_range': (0, 180),
                'organisms': ['general'],
                'conditions': ['water contact angle']
            },
            'biofilm_adhesion_coefficient': {
                'search_terms': ['biofilm adhesion', 'microbial attachment', 'cell attachment'],
                'units': 'dimensionless',
                'typical_range': (0.1, 10.0),
                'organisms': ['Geobacter', 'Shewanella', 'mixed culture'],
                'conditions': ['anaerobic', 'aqueous medium']
            },
            'attachment_energy': {
                'search_terms': ['attachment energy', 'adhesion energy', 'binding energy'],
                'units': 'kJ/mol',
                'typical_range': (5, 100),
                'organisms': ['Geobacter', 'Shewanella', 'mixed culture'],
                'conditions': ['microbial attachment', 'electrode surface']
            },
            'specific_surface_area': {
                'search_terms': ['specific surface area', 'BET surface area', 'porosity'],
                'units': 'm²/m³',
                'typical_range': (100, 100000),
                'organisms': ['general'],
                'conditions': ['porous electrode', 'carbon materials']
            },
            'porosity': {
                'search_terms': ['porosity', 'void fraction', 'porous structure'],
                'units': 'dimensionless',
                'typical_range': (0.1, 0.9),
                'organisms': ['general'],
                'conditions': ['porous electrode', 'carbon materials']
            }
        }

    def validate_electrode_parameter(self, parameter_name: str, value: float,
                                   material: ElectrodeMaterial,
                                   organism: str = "mixed culture") -> ElectrodeValidationResult:
        """
        Validate a single electrode parameter against literature.
        
        Args:
            parameter_name: Name of the parameter to validate
            value: Parameter value to validate
            material: Electrode material context
            organism: Target organism for validation
            
        Returns:
            ElectrodeValidationResult with validation analysis
        """

        if parameter_name not in self.parameter_mappings:
            return ElectrodeValidationResult(
                parameter_name=parameter_name,
                model_value=value,
                units="unknown",
                validation_result=ValidationResult(
                    query=ValidationQuery(parameter_name, value, "unknown"),
                    validation_status="ERROR",
                    confidence_score=0.0,
                    recommendations=["Parameter not recognized for validation"]
                ),
                recommendations=["Parameter not in validation database"]
            )

        param_info = self.parameter_mappings[parameter_name]

        # Create validation query
        search_terms = param_info['search_terms'].copy()

        # Add material-specific terms
        material_terms = self._get_material_search_terms(material)
        search_terms.extend(material_terms)

        conditions = param_info['conditions'].copy()
        if organism != "general":
            conditions.append(organism)

        query = ValidationQuery(
            parameter_name=parameter_name,
            parameter_value=value,
            units=param_info['units'],
            organism=organism,
            experimental_conditions=conditions
        )

        # Perform validation
        validation_result = self.lit_db.validate_parameter(query)

        # Assess quality of literature matches
        quality_score = 0.0
        if validation_result.literature_matches:
            quality_scores = self.quality_assessor.batch_assess_quality(validation_result.literature_matches)
            if quality_scores:
                quality_score = np.mean([q.overall_score for q in quality_scores])

        # Generate specific recommendations
        recommendations = list(validation_result.recommendations)

        # Add range-based recommendations
        typical_range = param_info['typical_range']
        if value < typical_range[0]:
            recommendations.append(f"Value {value} below typical range {typical_range}")
        elif value > typical_range[1]:
            recommendations.append(f"Value {value} above typical range {typical_range}")

        # Add material-specific recommendations
        material_recs = self._get_material_recommendations(parameter_name, value, material)
        recommendations.extend(material_recs)

        return ElectrodeValidationResult(
            parameter_name=parameter_name,
            model_value=value,
            units=param_info['units'],
            validation_result=validation_result,
            quality_score=quality_score,
            recommendations=recommendations
        )

    def validate_electrode_configuration(self, config: ElectrodeConfiguration,
                                       organism: str = "mixed culture") -> ElectrodeValidationReport:
        """
        Validate complete electrode configuration.
        
        Args:
            config: ElectrodeConfiguration to validate
            organism: Target organism for validation
            
        Returns:
            ElectrodeValidationReport with comprehensive analysis
        """

        validation_results = []
        critical_issues = []
        overall_recommendations = []

        # Extract material properties for validation
        material_props = config.material_properties

        # Define parameters to validate
        parameters_to_validate = [
            ('specific_conductance', material_props.specific_conductance),
            ('contact_resistance', material_props.contact_resistance),
            ('surface_charge_density', material_props.surface_charge_density),
            ('hydrophobicity_angle', material_props.hydrophobicity_angle),
            ('biofilm_adhesion_coefficient', material_props.biofilm_adhesion_coefficient),
            ('attachment_energy', material_props.attachment_energy)
        ]

        # Validate porous material properties if applicable
        if material_props.specific_surface_area is not None:
            parameters_to_validate.append(('specific_surface_area', material_props.specific_surface_area))
        if material_props.porosity is not None:
            parameters_to_validate.append(('porosity', material_props.porosity))

        # Validate each parameter
        for param_name, param_value in parameters_to_validate:
            if param_value is not None:
                result = self.validate_electrode_parameter(
                    param_name, param_value, config.material, organism
                )
                validation_results.append(result)

                # Check for critical issues
                if result.validation_result.validation_status == "NEEDS_REVIEW":
                    critical_issues.append(f"{param_name}: {result.validation_result.recommendations[0] if result.validation_result.recommendations else 'Needs review'}")

        # Calculate overall validation score
        if validation_results:
            confidence_scores = [r.validation_result.confidence_score for r in validation_results]
            quality_scores = [r.quality_score for r in validation_results if r.quality_score > 0]

            overall_score = np.mean(confidence_scores) * 0.7  # Weight confidence more
            if quality_scores:
                overall_score += np.mean(quality_scores) * 0.3  # Add quality component
        else:
            overall_score = 0.0

        # Generate overall recommendations
        if overall_score < self.validation_thresholds['warning']:
            overall_recommendations.append("Consider using well-documented electrode materials")
            overall_recommendations.append("Verify parameters with additional literature sources")
        elif overall_score < self.validation_thresholds['acceptable']:
            overall_recommendations.append("Some parameters need additional validation")
        else:
            overall_recommendations.append("Electrode configuration well-supported by literature")

        # Generate citation report
        all_articles = []
        for result in validation_results:
            all_articles.extend(result.validation_result.literature_matches)

        citation_report = None
        if all_articles:
            citation_report = self.citation_manager.create_parameter_report(
                parameter_name=f"{config.material.value}_electrode",
                parameter_value=len(validation_results),
                units="parameters",
                validation_status="MULTI_PARAMETER",
                articles=all_articles
            )

        return ElectrodeValidationReport(
            electrode_name=f"{config.material.value}_{config.geometry.geometry_type.value}",
            material=config.material,
            geometry=config.geometry.geometry_type,
            validation_results=validation_results,
            overall_validation_score=overall_score,
            critical_issues=critical_issues,
            recommendations=overall_recommendations,
            citation_report=citation_report
        )

    def _get_material_search_terms(self, material: ElectrodeMaterial) -> List[str]:
        """Get material-specific search terms."""

        material_terms = {
            ElectrodeMaterial.GRAPHITE_PLATE: ['graphite', 'graphite plate', 'carbon electrode'],
            ElectrodeMaterial.GRAPHITE_ROD: ['graphite rod', 'graphite', 'carbon electrode'],
            ElectrodeMaterial.CARBON_FELT: ['carbon felt', 'porous carbon', 'carbon fiber'],
            ElectrodeMaterial.CARBON_CLOTH: ['carbon cloth', 'carbon fabric', 'carbon fiber'],
            ElectrodeMaterial.CARBON_PAPER: ['carbon paper', 'gas diffusion layer', 'carbon fiber'],
            ElectrodeMaterial.STAINLESS_STEEL: ['stainless steel', 'steel electrode', 'metal electrode'],
            ElectrodeMaterial.PLATINUM: ['platinum', 'Pt electrode', 'noble metal'],
            ElectrodeMaterial.GOLD: ['gold', 'Au electrode', 'noble metal']
        }

        return material_terms.get(material, ['electrode material'])

    def _get_material_recommendations(self, parameter_name: str, value: float,
                                    material: ElectrodeMaterial) -> List[str]:
        """Get material-specific parameter recommendations."""

        recommendations = []

        # Conductance recommendations by material
        if parameter_name == 'specific_conductance':
            if material in [ElectrodeMaterial.CARBON_FELT, ElectrodeMaterial.CARBON_CLOTH]:
                if value < 1000:  # S/m
                    recommendations.append("Low conductance for carbon fiber materials - consider pre-treatment")
            elif material == ElectrodeMaterial.GRAPHITE_PLATE:
                if value < 10000:  # S/m
                    recommendations.append("Low conductance for graphite - verify material quality")

        # Surface area recommendations
        elif parameter_name == 'specific_surface_area':
            if material in [ElectrodeMaterial.CARBON_FELT, ElectrodeMaterial.CARBON_CLOTH]:
                if value < 1000:  # m²/m³
                    recommendations.append("Low specific surface area for porous carbon materials")
            elif material == ElectrodeMaterial.GRAPHITE_PLATE:
                if value is not None and value > 100:
                    recommendations.append("High specific surface area unusual for graphite plates")

        # Biofilm adhesion recommendations
        elif parameter_name == 'biofilm_adhesion_coefficient':
            if material == ElectrodeMaterial.CARBON_FELT and value < 2.0:
                recommendations.append("Carbon felt typically shows high biofilm adhesion")
            elif material == ElectrodeMaterial.STAINLESS_STEEL and value > 0.5:
                recommendations.append("Stainless steel typically shows lower biofilm adhesion")

        return recommendations

    def generate_validation_summary(self, reports: List[ElectrodeValidationReport]) -> Dict[str, Any]:
        """Generate summary of multiple electrode validations."""

        if not reports:
            return {'error': 'No validation reports provided'}

        summary = {
            'total_electrodes': len(reports),
            'well_validated': 0,
            'acceptable': 0,
            'needs_review': 0,
            'average_score': 0.0,
            'common_issues': {},
            'common_recommendations': {},
            'material_distribution': {},
            'geometry_distribution': {}
        }

        scores = []
        all_issues = []
        all_recommendations = []

        for report in reports:
            score = report.overall_validation_score
            scores.append(score)

            # Categorize validation status
            if score >= self.validation_thresholds['critical']:
                summary['well_validated'] += 1
            elif score >= self.validation_thresholds['acceptable']:
                summary['acceptable'] += 1
            else:
                summary['needs_review'] += 1

            # Collect issues and recommendations
            all_issues.extend(report.critical_issues)
            all_recommendations.extend(report.recommendations)

            # Count materials and geometries
            material = report.material.value
            geometry = report.geometry.value
            summary['material_distribution'][material] = summary['material_distribution'].get(material, 0) + 1
            summary['geometry_distribution'][geometry] = summary['geometry_distribution'].get(geometry, 0) + 1

        # Calculate statistics
        summary['average_score'] = np.mean(scores) if scores else 0.0

        # Find common issues
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        summary['common_issues'] = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])

        # Find common recommendations
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        summary['common_recommendations'] = dict(sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5])

        return summary

    def export_validation_report(self, report: ElectrodeValidationReport,
                               filepath: str, format: str = "markdown"):
        """Export validation report to file."""

        if format.lower() == "markdown":
            content = self._generate_markdown_report(report)
        elif format.lower() == "json":
            import json
            content = json.dumps(self._report_to_dict(report), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"📊 Validation report exported to: {filepath}")

    def _generate_markdown_report(self, report: ElectrodeValidationReport) -> str:
        """Generate markdown validation report."""

        md = "# Electrode Validation Report\n\n"
        md += f"**Electrode:** {report.electrode_name}\n"
        md += f"**Material:** {report.material.value}\n"
        md += f"**Geometry:** {report.geometry.value}\n"
        md += f"**Overall Score:** {report.overall_validation_score:.3f}\n\n"

        # Validation status
        if report.overall_validation_score >= self.validation_thresholds['critical']:
            status = "✅ WELL VALIDATED"
        elif report.overall_validation_score >= self.validation_thresholds['acceptable']:
            status = "⚠️ ACCEPTABLE"
        else:
            status = "❌ NEEDS REVIEW"

        md += f"**Status:** {status}\n\n"

        # Critical issues
        if report.critical_issues:
            md += "## ⚠️ Critical Issues\n\n"
            for issue in report.critical_issues:
                md += f"- {issue}\n"
            md += "\n"

        # Parameter validation results
        md += "## Parameter Validation Results\n\n"

        for result in report.validation_results:
            md += f"### {result.parameter_name}\n\n"
            md += f"- **Value:** {result.model_value} {result.units}\n"
            md += f"- **Status:** {result.validation_result.validation_status}\n"
            md += f"- **Confidence:** {result.validation_result.confidence_score:.3f}\n"
            md += f"- **Quality Score:** {result.quality_score:.3f}\n"
            md += f"- **Literature Matches:** {len(result.validation_result.literature_matches)}\n"

            if result.recommendations:
                md += "- **Recommendations:**\n"
                for rec in result.recommendations:
                    md += f"  - {rec}\n"

            md += "\n"

        # Overall recommendations
        if report.recommendations:
            md += "## 💡 Recommendations\n\n"
            for rec in report.recommendations:
                md += f"- {rec}\n"
            md += "\n"

        # Citations
        if report.citation_report and report.citation_report.citations:
            md += "## 📚 Literature References\n\n"
            formatted_citations = self.citation_manager.format_citations(
                report.citation_report.citations, "apa"
            )
            for i, citation in enumerate(formatted_citations, 1):
                md += f"{i}. {citation}\n\n"

        return md

    def _report_to_dict(self, report: ElectrodeValidationReport) -> Dict[str, Any]:
        """Convert validation report to dictionary."""

        return {
            'electrode_name': report.electrode_name,
            'material': report.material.value,
            'geometry': report.geometry.value,
            'overall_validation_score': report.overall_validation_score,
            'critical_issues': report.critical_issues,
            'recommendations': report.recommendations,
            'validation_results': [
                {
                    'parameter_name': r.parameter_name,
                    'model_value': r.model_value,
                    'units': r.units,
                    'validation_status': r.validation_result.validation_status,
                    'confidence_score': r.validation_result.confidence_score,
                    'quality_score': r.quality_score,
                    'literature_matches': len(r.validation_result.literature_matches),
                    'recommendations': r.recommendations
                }
                for r in report.validation_results
            ],
            'citation_count': len(report.citation_report.citations) if report.citation_report else 0
        }
if __name__ == "__main__":
    # Example usage
    print("🔬 Electrode Validation Integration Test")
    print("=" * 50)

    # Initialize integrator
    integrator = ElectrodeValidationIntegrator()

    # Test parameter validation
    print("Testing parameter validation...")

    result = integrator.validate_electrode_parameter(
        parameter_name="specific_conductance",
        value=25000.0,  # S/m - typical for graphite
        material=ElectrodeMaterial.GRAPHITE_PLATE,
        organism="Shewanella oneidensis"
    )

    print("📊 Validation Result:")
    print(f"  Parameter: {result.parameter_name}")
    print(f"  Value: {result.model_value} {result.units}")
    print(f"  Status: {result.validation_result.validation_status}")
    print(f"  Confidence: {result.validation_result.confidence_score:.3f}")
    print(f"  Quality: {result.quality_score:.3f}")

    if result.recommendations:
        print("  Recommendations:")
        for rec in result.recommendations[:3]:  # Show first 3
            print(f"    - {rec}")

    print("\n✅ Electrode validation integration test completed!")
