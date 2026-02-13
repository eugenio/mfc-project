"""Scientific Unit Converter for MFC Parameters.

This module provides unit conversion functionality for MFC parameters,
supporting common electrochemical, biological, and engineering units.

User Story 1.1.1: Scientific unit validation and conversion
Created: 2025-07-31
Last Modified: 2025-07-31
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnitDefinition:
    """Definition of a scientific unit with conversion factors."""

    name: str
    symbol: str
    dimension: str  # e.g., 'current_density', 'concentration', 'length'
    si_conversion_factor: float  # Factor to convert to SI base unit
    si_unit: str  # SI base unit
    common_aliases: list[str] | None = None

    def __post_init__(self):
        if self.common_aliases is None:
            self.common_aliases = []


class UnitConverter:
    """Handles unit conversion for MFC parameters."""

    def __init__(self) -> None:
        """Initialize unit converter with common MFC units."""
        self.units = self._define_units()
        self.dimension_map = self._create_dimension_map()

    def _define_units(self) -> dict[str, UnitDefinition]:
        """Define all supported units and their conversion factors."""
        return {
            # Current Density Units
            "A/m²": UnitDefinition(
                "amperes per square meter",
                "A/m²",
                "current_density",
                1.0,
                "A/m²",
            ),
            "mA/cm²": UnitDefinition(
                "milliamperes per square centimeter",
                "mA/cm²",
                "current_density",
                10.0,
                "A/m²",
                ["mA/cm2"],
            ),
            "A/cm²": UnitDefinition(
                "amperes per square centimeter",
                "A/cm²",
                "current_density",
                10000.0,
                "A/m²",
                ["A/cm2"],
            ),
            "μA/cm²": UnitDefinition(
                "microamperes per square centimeter",
                "μA/cm²",
                "current_density",
                0.01,
                "A/m²",
                ["uA/cm2", "µA/cm²"],
            ),
            # Concentration Units
            "mM": UnitDefinition(
                "millimolar",
                "mM",
                "concentration",
                1.0,
                "mol/m³",
                ["mmol/L"],
            ),
            "M": UnitDefinition(
                "molar",
                "M",
                "concentration",
                1000.0,
                "mol/m³",
                ["mol/L"],
            ),
            "μM": UnitDefinition(
                "micromolar",
                "μM",
                "concentration",
                0.001,
                "mol/m³",
                ["uM", "µM", "umol/L"],
            ),
            "g/L": UnitDefinition(
                "grams per liter",
                "g/L",
                "mass_concentration",
                1.0,
                "g/L",
                ["g/l"],
            ),
            "mg/L": UnitDefinition(
                "milligrams per liter",
                "mg/L",
                "mass_concentration",
                0.001,
                "g/L",
                ["mg/l"],
            ),
            "kg/m³": UnitDefinition(
                "kilograms per cubic meter",
                "kg/m³",
                "mass_concentration",
                1.0,
                "g/L",
                ["kg/m3"],
            ),
            # Length Units
            "m": UnitDefinition("meters", "m", "length", 1.0, "m"),
            "cm": UnitDefinition("centimeters", "cm", "length", 0.01, "m"),
            "mm": UnitDefinition("millimeters", "mm", "length", 0.001, "m"),
            "μm": UnitDefinition(
                "micrometers",
                "μm",
                "length",
                1e-6,
                "m",
                ["um", "µm", "micron"],
            ),
            "nm": UnitDefinition("nanometers", "nm", "length", 1e-9, "m"),
            # Area Units
            "m²": UnitDefinition(
                "square meters",
                "m²",
                "area",
                1.0,
                "m²",
                ["m2", "sq m"],
            ),
            "cm²": UnitDefinition(
                "square centimeters",
                "cm²",
                "area",
                1e-4,
                "m²",
                ["cm2", "sq cm"],
            ),
            "mm²": UnitDefinition(
                "square millimeters",
                "mm²",
                "area",
                1e-6,
                "m²",
                ["mm2", "sq mm"],
            ),
            # Flow Rate Units
            "mL/h": UnitDefinition(
                "milliliters per hour",
                "mL/h",
                "flow_rate",
                2.778e-10,
                "m³/s",
                ["ml/h", "mL/hr"],
            ),
            "mL/min": UnitDefinition(
                "milliliters per minute",
                "mL/min",
                "flow_rate",
                1.667e-8,
                "m³/s",
                ["ml/min"],
            ),
            "L/h": UnitDefinition(
                "liters per hour",
                "L/h",
                "flow_rate",
                2.778e-7,
                "m³/s",
                ["l/h", "L/hr"],
            ),
            "L/min": UnitDefinition(
                "liters per minute",
                "L/min",
                "flow_rate",
                1.667e-5,
                "m³/s",
                ["l/min"],
            ),
            "m³/s": UnitDefinition(
                "cubic meters per second",
                "m³/s",
                "flow_rate",
                1.0,
                "m³/s",
                ["m3/s"],
            ),
            # Voltage Units
            "V": UnitDefinition("volts", "V", "voltage", 1.0, "V", ["volt"]),
            "mV": UnitDefinition(
                "millivolts",
                "mV",
                "voltage",
                0.001,
                "V",
                ["millivolt"],
            ),
            "V vs SHE": UnitDefinition(
                "volts vs standard hydrogen electrode",
                "V vs SHE",
                "voltage",
                1.0,
                "V",
                ["V SHE"],
            ),
            "V vs Ag/AgCl": UnitDefinition(
                "volts vs silver/silver chloride",
                "V vs Ag/AgCl",
                "voltage",
                1.197,
                "V",
                ["V AgAgCl"],
            ),
            # Time Units
            "s": UnitDefinition("seconds", "s", "time", 1.0, "s", ["sec", "second"]),
            "min": UnitDefinition("minutes", "min", "time", 60.0, "s", ["minute"]),
            "h": UnitDefinition("hours", "h", "time", 3600.0, "s", ["hr", "hour"]),
            "d": UnitDefinition("days", "d", "time", 86400.0, "s", ["day"]),
            # Rate Units
            "h⁻¹": UnitDefinition(
                "per hour",
                "h⁻¹",
                "rate",
                2.778e-4,
                "s⁻¹",
                ["1/h", "/h", "h^-1"],
            ),
            "s⁻¹": UnitDefinition(
                "per second",
                "s⁻¹",
                "rate",
                1.0,
                "s⁻¹",
                ["1/s", "/s", "s^-1"],
            ),
            "d⁻¹": UnitDefinition(
                "per day",
                "d⁻¹",
                "rate",
                1.157e-5,
                "s⁻¹",
                ["1/d", "/d", "d^-1"],
            ),
            # Conductivity Units
            "S/m": UnitDefinition(
                "siemens per meter",
                "S/m",
                "conductivity",
                1.0,
                "S/m",
                ["siemens/m"],
            ),
            "mS/cm": UnitDefinition(
                "millisiemens per centimeter",
                "mS/cm",
                "conductivity",
                0.1,
                "S/m",
                ["mS/cm"],
            ),
            "μS/cm": UnitDefinition(
                "microsiemens per centimeter",
                "μS/cm",
                "conductivity",
                0.0001,
                "S/m",
                ["uS/cm", "µS/cm"],
            ),
            # Power Density Units
            "W/m²": UnitDefinition(
                "watts per square meter",
                "W/m²",
                "power_density",
                1.0,
                "W/m²",
                ["W/m2"],
            ),
            "mW/m²": UnitDefinition(
                "milliwatts per square meter",
                "mW/m²",
                "power_density",
                0.001,
                "W/m²",
                ["mW/m2"],
            ),
            "W/cm²": UnitDefinition(
                "watts per square centimeter",
                "W/cm²",
                "power_density",
                10000.0,
                "W/m²",
                ["W/cm2"],
            ),
            # Dimensionless
            "dimensionless": UnitDefinition(
                "dimensionless",
                "dimensionless",
                "dimensionless",
                1.0,
                "dimensionless",
                [""],
            ),
            "%": UnitDefinition(
                "percent",
                "%",
                "dimensionless",
                0.01,
                "dimensionless",
                ["percent"],
            ),
        }

    def _create_dimension_map(self) -> dict[str, list[str]]:
        """Create mapping from dimensions to units."""
        dimension_map: dict[str, list[str]] = {}

        for unit_symbol, unit_def in self.units.items():
            if unit_def.dimension not in dimension_map:
                dimension_map[unit_def.dimension] = []
            dimension_map[unit_def.dimension].append(unit_symbol)

        return dimension_map

    def normalize_unit(self, unit_str: str) -> str | None:
        """Normalize unit string to standard format.

        Args:
            unit_str: Unit string (may have variations)

        Returns:
            Normalized unit string or None if not recognized

        """
        # Direct match
        if unit_str in self.units:
            return unit_str

        # Check aliases
        for std_unit, unit_def in self.units.items():
            if unit_str in unit_def.common_aliases:
                return std_unit

        # Try case-insensitive match
        unit_lower = unit_str.lower()
        for std_unit, unit_def in self.units.items():
            if std_unit.lower() == unit_lower:
                return std_unit
            for alias in unit_def.common_aliases:
                if alias.lower() == unit_lower:
                    return std_unit

        return None

    def convert(self, value: float, from_unit: str, to_unit: str) -> float | None:
        """Convert value from one unit to another.

        Args:
            value: Numeric value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value or None if conversion not possible

        """
        # Normalize units
        from_unit_norm = self.normalize_unit(from_unit)
        to_unit_norm = self.normalize_unit(to_unit)

        if not from_unit_norm or not to_unit_norm:
            return None

        from_def = self.units[from_unit_norm]
        to_def = self.units[to_unit_norm]

        # Check dimension compatibility
        if from_def.dimension != to_def.dimension:
            return None

        # Convert through SI base unit
        si_value = value * from_def.si_conversion_factor
        return si_value / to_def.si_conversion_factor

    def get_compatible_units(self, unit: str) -> list[str]:
        """Get list of units compatible for conversion with given unit.

        Args:
            unit: Reference unit

        Returns:
            List of compatible unit symbols

        """
        unit_norm = self.normalize_unit(unit)
        if not unit_norm:
            return []

        dimension = self.units[unit_norm].dimension
        return self.dimension_map.get(dimension, [])

    def validate_unit(
        self,
        unit: str,
        expected_dimension: str | None = None,
    ) -> tuple[bool, str | None]:
        """Validate unit string and optionally check dimension.

        Args:
            unit: Unit string to validate
            expected_dimension: Expected dimension (optional)

        Returns:
            Tuple of (is_valid, normalized_unit)

        """
        normalized = self.normalize_unit(unit)
        if not normalized:
            return False, None

        if expected_dimension:
            actual_dimension = self.units[normalized].dimension
            if actual_dimension != expected_dimension:
                return False, None

        return True, normalized

    def format_value_with_unit(
        self,
        value: float,
        unit: str,
        precision: int = 3,
    ) -> str:
        """Format value with unit for display.

        Args:
            value: Numeric value
            unit: Unit string
            precision: Number of significant figures

        Returns:
            Formatted string

        """
        # Format number based on magnitude
        if abs(value) >= 1000 or (0 < abs(value) < 0.01):
            formatted = f"{value:.{precision}e}"
        else:
            formatted = f"{value:.{precision}g}"

        return f"{formatted} {unit}"

    def suggest_unit_for_dimension(
        self,
        dimension: str,
        prefer_common: bool = True,
    ) -> str | None:
        """Suggest appropriate unit for given dimension.

        Args:
            dimension: Dimension type
            prefer_common: Prefer commonly used units

        Returns:
            Suggested unit or None

        """
        units = self.dimension_map.get(dimension, [])
        if not units:
            return None

        if prefer_common:
            # Define common preferences
            common_preferences = {
                "current_density": "mA/cm²",
                "concentration": "mM",
                "mass_concentration": "g/L",
                "length": "mm",
                "area": "cm²",
                "flow_rate": "mL/h",
                "voltage": "V",
                "conductivity": "S/m",
                "power_density": "mW/m²",
            }

            if dimension in common_preferences:
                return common_preferences[dimension]

        # Return first available unit
        return units[0] if units else None


# Global instance for convenience
UNIT_CONVERTER = UnitConverter()
