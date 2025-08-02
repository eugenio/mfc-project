"""
Literature Database for MFC Parameter Validation and Citations

This module provides literature references and validation ranges for MFC parameters
used in scientific research. All parameters are backed by peer-reviewed publications
to ensure scientific rigor and reproducibility.

Created: 2025-07-31
Last Modified: 2025-07-31
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ParameterCategory(Enum):
    """Categories of MFC parameters."""
    ELECTROCHEMICAL = "electrochemical"
    BIOLOGICAL = "biological"
    QLEARNING = "qlearning"
    SUBSTRATE = "substrate"
    BIOFILM = "biofilm"
    SENSOR = "sensor"


@dataclass
class LiteratureReference:
    """Container for literature reference information."""

    authors: str
    title: str
    journal: str
    year: int
    volume: str
    pages: str
    doi: str | None = None

    def format_citation(self, style: str = "apa") -> str:
        """Format citation in specified style."""
        if style.lower() == "apa":
            return f"{self.authors} ({self.year}). {self.title}. {self.journal}, {self.volume}, {self.pages}."
        elif style.lower() == "bibtex":
            return f"@article{{{self.authors.split(',')[0].strip().replace(' ', '').replace('.', '')}{self.year},\n" \
                   f"  author = {{{self.authors}}},\n" \
                   f"  title = {{{self.title}}},\n" \
                   f"  journal = {{{self.journal}}},\n" \
                   f"  year = {{{self.year}}},\n" \
                   f"  volume = {{{self.volume}}},\n" \
                   f"  pages = {{{self.pages}}}" + \
                   (f",\n  doi = {{{self.doi}}}" if self.doi else "") + "\n}"
        else:
            return f"{self.authors} ({self.year}). {self.title}. {self.journal}, {self.volume}, {self.pages}."


@dataclass
class ParameterInfo:
    """Information about a validated MFC parameter."""

    name: str
    symbol: str
    description: str
    unit: str
    typical_value: float
    min_value: float
    max_value: float
    recommended_range: tuple[float, float]
    category: ParameterCategory
    references: list[LiteratureReference]
    notes: str | None = None

    def is_within_recommended_range(self, value: float) -> bool:
        """Check if value is within recommended range."""
        return self.recommended_range[0] <= value <= self.recommended_range[1]

    def is_valid_value(self, value: float) -> bool:
        """Check if value is within absolute valid range."""
        return self.min_value <= value <= self.max_value

    def get_validation_status(self, value: float) -> str:
        """Get validation status for a parameter value."""
        if not self.is_valid_value(value):
            return "invalid"
        elif not self.is_within_recommended_range(value):
            return "caution"
        else:
            return "valid"


class LiteratureDatabase:
    """Database of literature-validated MFC parameters."""

    def __init__(self):
        """Initialize literature database with parameter information."""
        self.parameters = self._initialize_parameter_database()

    def _initialize_parameter_database(self) -> dict[str, ParameterInfo]:
        """Initialize the parameter database with literature values."""

        # Literature references
        logan_2006 = LiteratureReference(
            authors="Logan, B.E., Hamelers, B., Rozendal, R., Schröder, U., Keller, J., Freguia, S., Aelterman, P., Verstraete, W., Rabaey, K.",
            title="Microbial fuel cells: methodology and technology",
            journal="Environmental Science & Technology",
            year=2006,
            volume="40",
            pages="5181-5192",
            doi="10.1021/es0605016"
        )

        kim_2007 = LiteratureReference(
            authors="Kim, H.J., Park, H.S., Hyun, M.S., Chang, I.S., Kim, M., Kim, B.H.",
            title="A mediator-less microbial fuel cell using a metal reducing bacterium, Shewanella putrefaciens",
            journal="Enzyme and Microbial Technology",
            year=2002,
            volume="30",
            pages="145-152",
            doi="10.1016/S0141-0229(01)00478-1"
        )

        bond_2002 = LiteratureReference(
            authors="Bond, D.R., Lovley, D.R.",
            title="Electricity production by Geobacter sulfurreducens attached to electrodes",
            journal="Applied and Environmental Microbiology",
            year=2003,
            volume="69",
            pages="1548-1555",
            doi="10.1128/AEM.69.3.1548-1555.2003"
        )

        torres_2010 = LiteratureReference(
            authors="Torres, C.I., Marcus, A.K., Lee, H.S., Parameswaran, P., Krajmalnik-Brown, R., Rittmann, B.E.",
            title="A kinetic perspective on extracellular electron transfer by anode-respiring bacteria",
            journal="FEMS Microbiology Reviews",
            year=2010,
            volume="34",
            pages="3-17",
            doi="10.1111/j.1574-6976.2009.00191.x"
        )

        sutton_2018 = LiteratureReference(
            authors="Sutton, R.S., Barto, A.G.",
            title="Reinforcement Learning: An Introduction",
            journal="MIT Press",
            year=2018,
            volume="2nd Edition",
            pages="1-526"
        )

        parameters = {}

        # Electrochemical Parameters
        parameters["anode_potential"] = ParameterInfo(
            name="Anode Potential",
            symbol="E_an",
            description="Standard anode potential for MFC operation",
            unit="V vs SHE",
            typical_value=-0.3,
            min_value=-0.6,
            max_value=0.0,
            recommended_range=(-0.4, -0.2),
            category=ParameterCategory.ELECTROCHEMICAL,
            references=[logan_2006, bond_2002],
            notes="Standard hydrogen electrode reference"
        )

        parameters["cathode_potential"] = ParameterInfo(
            name="Cathode Potential",
            symbol="E_cat",
            description="Standard cathode potential for oxygen reduction",
            unit="V vs SHE",
            typical_value=0.8,
            min_value=0.4,
            max_value=1.2,
            recommended_range=(0.6, 1.0),
            category=ParameterCategory.ELECTROCHEMICAL,
            references=[logan_2006],
            notes="Oxygen reduction reaction potential"
        )

        parameters["internal_resistance"] = ParameterInfo(
            name="Internal Resistance",
            symbol="R_int",
            description="Total internal resistance of MFC system",
            unit="Ω",
            typical_value=100.0,
            min_value=10.0,
            max_value=1000.0,
            recommended_range=(50.0, 300.0),
            category=ParameterCategory.ELECTROCHEMICAL,
            references=[logan_2006, kim_2007],
            notes="Includes ohmic, activation, and concentration losses"
        )

        parameters["electrode_area"] = ParameterInfo(
            name="Electrode Area",
            symbol="A_e",
            description="Surface area of electrode in contact with biofilm",
            unit="m²",
            typical_value=0.001,
            min_value=1e-6,
            max_value=1.0,
            recommended_range=(1e-4, 0.01),
            category=ParameterCategory.ELECTROCHEMICAL,
            references=[logan_2006, bond_2002],
            notes="Geometric surface area, not accounting for roughness"
        )

        # Biological Parameters
        parameters["max_current_density"] = ParameterInfo(
            name="Maximum Current Density",
            symbol="j_max",
            description="Maximum achievable current density for bacterial species",
            unit="mA/cm²",
            typical_value=0.39,
            min_value=0.001,
            max_value=5.0,
            recommended_range=(0.01, 2.0),
            category=ParameterCategory.BIOLOGICAL,
            references=[bond_2002, torres_2010],
            notes="Species-dependent: G. sulfurreducens ~0.39, S. oneidensis ~0.034"
        )

        parameters["biofilm_conductivity"] = ParameterInfo(
            name="Biofilm Conductivity",
            symbol="σ_bf",
            description="Electrical conductivity of bacterial biofilm",
            unit="S/m",
            typical_value=1e-4,
            min_value=1e-6,
            max_value=1e-2,
            recommended_range=(1e-5, 1e-3),
            category=ParameterCategory.BIOLOGICAL,
            references=[torres_2010, bond_2002],
            notes="Varies with biofilm density and species composition"
        )

        parameters["growth_rate"] = ParameterInfo(
            name="Maximum Specific Growth Rate",
            symbol="μ_max",
            description="Maximum specific growth rate of electroactive bacteria",
            unit="h⁻¹",
            typical_value=0.15,
            min_value=0.01,
            max_value=1.0,
            recommended_range=(0.05, 0.5),
            category=ParameterCategory.BIOLOGICAL,
            references=[torres_2010, logan_2006],
            notes="Temperature and substrate dependent"
        )

        parameters["half_saturation"] = ParameterInfo(
            name="Half-Saturation Constant",
            symbol="K_s",
            description="Monod half-saturation constant for substrate utilization",
            unit="mM",
            typical_value=0.5,
            min_value=0.01,
            max_value=10.0,
            recommended_range=(0.1, 2.0),
            category=ParameterCategory.BIOLOGICAL,
            references=[torres_2010, logan_2006],
            notes="Substrate-specific: acetate ~0.5 mM, lactate ~1.0 mM"
        )

        # Substrate Parameters
        parameters["substrate_concentration"] = ParameterInfo(
            name="Substrate Concentration",
            symbol="S",
            description="Initial substrate concentration in feed medium",
            unit="mM",
            typical_value=25.0,
            min_value=1.0,
            max_value=100.0,
            recommended_range=(10.0, 50.0),
            category=ParameterCategory.SUBSTRATE,
            references=[logan_2006, torres_2010],
            notes="Avoid inhibitory concentrations >100 mM"
        )

        parameters["flow_rate"] = ParameterInfo(
            name="Flow Rate",
            symbol="Q",
            description="Volumetric flow rate of feed medium",
            unit="mL/h",
            typical_value=15.0,
            min_value=1.0,
            max_value=100.0,
            recommended_range=(5.0, 50.0),
            category=ParameterCategory.SUBSTRATE,
            references=[logan_2006],
            notes="Affects residence time and mass transfer"
        )

        # Q-Learning Parameters
        parameters["learning_rate"] = ParameterInfo(
            name="Learning Rate",
            symbol="α",
            description="Q-learning algorithm learning rate parameter",
            unit="dimensionless",
            typical_value=0.1,
            min_value=0.001,
            max_value=1.0,
            recommended_range=(0.05, 0.2),
            category=ParameterCategory.QLEARNING,
            references=[sutton_2018],
            notes="Balance between stability and adaptation speed"
        )

        parameters["discount_factor"] = ParameterInfo(
            name="Discount Factor",
            symbol="γ",
            description="Q-learning discount factor for future rewards",
            unit="dimensionless",
            typical_value=0.95,
            min_value=0.0,
            max_value=1.0,
            recommended_range=(0.8, 0.99),
            category=ParameterCategory.QLEARNING,
            references=[sutton_2018],
            notes="Higher values emphasize long-term rewards"
        )

        parameters["exploration_rate"] = ParameterInfo(
            name="Exploration Rate",
            symbol="ε",
            description="Epsilon-greedy exploration rate for Q-learning",
            unit="dimensionless",
            typical_value=0.3,
            min_value=0.0,
            max_value=1.0,
            recommended_range=(0.1, 0.5),
            category=ParameterCategory.QLEARNING,
            references=[sutton_2018],
            notes="Should decay over time during training"
        )

        # Biofilm Parameters
        parameters["biofilm_thickness"] = ParameterInfo(
            name="Biofilm Thickness",
            symbol="L_bf",
            description="Thickness of bacterial biofilm on electrode",
            unit="μm",
            typical_value=30.0,
            min_value=1.0,
            max_value=200.0,
            recommended_range=(10.0, 100.0),
            category=ParameterCategory.BIOFILM,
            references=[bond_2002, torres_2010],
            notes="Optimal thickness balances conductivity and mass transfer"
        )

        parameters["biofilm_density"] = ParameterInfo(
            name="Biofilm Density",
            symbol="ρ_bf",
            description="Wet density of bacterial biofilm",
            unit="kg/m³",
            typical_value=80.0,
            min_value=20.0,
            max_value=200.0,
            recommended_range=(50.0, 150.0),
            category=ParameterCategory.BIOFILM,
            references=[logan_2006, torres_2010],
            notes="Typical range for hydrated bacterial biofilms"
        )

        return parameters

    def get_parameter(self, name: str) -> ParameterInfo | None:
        """Get parameter information by name."""
        return self.parameters.get(name)

    def get_parameters_by_category(self, category: ParameterCategory) -> list[ParameterInfo]:
        """Get all parameters in a specific category."""
        return [param for param in self.parameters.values() if param.category == category]

    def validate_parameter_value(self, name: str, value: float) -> dict[str, Any]:
        """
        Validate a parameter value against literature recommendations.

        Args:
            name: Parameter name
            value: Parameter value to validate

        Returns:
            Dictionary with validation results
        """
        param = self.get_parameter(name)
        if not param:
            return {
                "status": "unknown",
                "message": f"Parameter '{name}' not found in database",
                "recommendations": []
            }

        status = param.get_validation_status(value)

        validation_result: dict[str, Any] = {
            "status": status,
            "parameter": param,
            "value": value,
            "unit": param.unit,
            "recommendations": []
        }

        if status == "invalid":
            validation_result["message"] = f"Value {value} {param.unit} is outside valid range ({param.min_value}-{param.max_value} {param.unit})"
            recommendations = validation_result["recommendations"]
            if isinstance(recommendations, list):
                recommendations.append(f"Use values between {param.min_value} and {param.max_value} {param.unit}")
        elif status == "caution":
            validation_result["message"] = f"Value {value} {param.unit} is outside recommended range ({param.recommended_range[0]}-{param.recommended_range[1]} {param.unit})"
            recommendations = validation_result["recommendations"]
            if isinstance(recommendations, list):
                recommendations.append(f"Consider using values between {param.recommended_range[0]} and {param.recommended_range[1]} {param.unit}")
        else:
            validation_result["message"] = f"Value {value} {param.unit} is within recommended range"

        return validation_result

    def get_all_categories(self) -> list[ParameterCategory]:
        """Get all available parameter categories."""
        return list(ParameterCategory)

    def search_parameters(self, query: str) -> list[ParameterInfo]:
        """Search parameters by name, symbol, or description."""
        query_lower = query.lower()
        results = []

        for param in self.parameters.values():
            if (query_lower in param.name.lower() or
                query_lower in param.symbol.lower() or
                query_lower in param.description.lower()):
                results.append(param)

        return results

    def get_citation_list(self, format_style: str = "apa") -> list[str]:
        """Get formatted citation list for all references."""
        all_refs = set()
        for param in self.parameters.values():
            for ref in param.references:
                all_refs.add(ref.format_citation(format_style))

        return sorted(list(all_refs))


# Global instance for easy access
LITERATURE_DB = LiteratureDatabase()
