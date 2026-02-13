#!/usr/bin/env python3
"""Membrane Configuration UI Component.

Provides comprehensive membrane configuration interface including:
- Material selection with literature-based properties
- Operating conditions specification
- Resistance calculations and visualization
- Membrane property customization
- Configuration summary and analysis

Created: 2025-08-03
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config.membrane_config import (
    MEMBRANE_PROPERTIES_DATABASE,
    MembraneConfiguration,
    MembraneMaterial,
    MembraneProperties,
    create_membrane_config,
)


class MembraneConfigurationUI:
    """UI component for membrane configuration."""

    def __init__(self):
        """Initialize membrane configuration UI component."""
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize membrane configuration in session state."""
        if 'membrane_config' not in st.session_state:
            st.session_state.membrane_config = None
        if 'membrane_calculations' not in st.session_state:
            st.session_state.membrane_calculations = {}

    def render_material_selector(self) -> MembraneMaterial:
        """Render membrane material selection interface."""
        st.markdown("### 🧬 Membrane Material Selection")

        # Material options with descriptions
        material_options = {
            "Nafion 117": MembraneMaterial.NAFION_117,
            "Nafion 112": MembraneMaterial.NAFION_112,
            "Nafion 115": MembraneMaterial.NAFION_115,
            "Ultrex CMI-7000": MembraneMaterial.ULTREX_CMI_7000,
            "Fumasep FKE": MembraneMaterial.FUMASEP_FKE,
            "Fumasep FAA": MembraneMaterial.FUMASEP_FAA,
            "Cellulose Acetate": MembraneMaterial.CELLULOSE_ACETATE,
            "Bipolar Membrane": MembraneMaterial.BIPOLAR_MEMBRANE,
            "Ceramic Separator": MembraneMaterial.CERAMIC_SEPARATOR,
            "J-Cloth": MembraneMaterial.J_CLOTH,
            "Custom Material": MembraneMaterial.CUSTOM
        }

        selected_material_name = st.selectbox(
            "Membrane Material",
            options=list(material_options.keys()),
            help="Select membrane material based on your experimental setup",
            key="membrane_material_select"
        )

        selected_material = material_options[selected_material_name]

        # Show material properties
        if selected_material != MembraneMaterial.CUSTOM and selected_material in MEMBRANE_PROPERTIES_DATABASE:
            props = MEMBRANE_PROPERTIES_DATABASE[selected_material]
            self._display_material_properties(props, selected_material_name)
        elif selected_material != MembraneMaterial.CUSTOM:
            st.warning(f"Properties for {selected_material_name} not yet defined in database.")

        return selected_material

    def _display_material_properties(self, props: MembraneProperties, material_name: str):
        """Display membrane material properties in an expandable section."""
        with st.expander(f"📋 {material_name} Properties", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Proton Conductivity", f"{props.proton_conductivity:.3f} S/cm")
                st.metric("Ion Exchange Capacity", f"{props.ion_exchange_capacity:.1f} meq/g")
                st.metric("Permselectivity", f"{props.permselectivity:.3f}")
                st.metric("Thickness", f"{props.thickness:.0f} μm")
                st.metric("Water Uptake", f"{props.water_uptake:.0f}%")

            with col2:
                st.metric("Area Resistance", f"{props.area_resistance:.1f} Ω·cm²")
                st.metric("Oxygen Permeability", f"{props.oxygen_permeability:.2e} cm²/s")
                st.metric("Substrate Permeability", f"{props.substrate_permeability:.2e} cm²/s")
                if props.cost_per_m2:
                    st.metric("Cost", f"${props.cost_per_m2:.0f}/m²")
                st.metric("Expected Lifetime", f"{props.expected_lifetime:.0f} hours")

            st.caption(f"📚 Reference: {props.reference}")

    def render_area_input(self) -> float:
        """Render membrane area input interface."""
        st.markdown("### 📐 Membrane Area Configuration")

        col1, col2 = st.columns(2)

        with col1:
            area_cm2 = st.number_input(
                "Membrane Area (cm²)",
                min_value=0.1,
                max_value=10000.0,
                value=25.0,
                step=0.1,
                help="Active membrane area for ion transport",
                key="membrane_area_cm2"
            )

        with col2:
            area_m2 = area_cm2 / 10000
            st.metric("Area (m²)", f"{area_m2:.6f}")

        return area_m2

    def render_operating_conditions(self) -> tuple[float, float, float]:
        """Render operating conditions input interface."""
        st.markdown("### 🌡️ Operating Conditions")

        col1, col2, col3 = st.columns(3)

        with col1:
            temperature = st.number_input(
                "Operating Temperature (°C)",
                min_value=15.0,
                max_value=80.0,
                value=25.0,
                step=0.5,
                help="Operating temperature affects conductivity and lifetime",
                key="membrane_temperature"
            )

        with col2:
            ph_anode = st.number_input(
                "Anode pH",
                min_value=1.0,
                max_value=14.0,
                value=7.0,
                step=0.1,
                help="pH at anode compartment",
                key="membrane_ph_anode"
            )

        with col3:
            ph_cathode = st.number_input(
                "Cathode pH",
                min_value=1.0,
                max_value=14.0,
                value=7.0,
                step=0.1,
                help="pH at cathode compartment",
                key="membrane_ph_cathode"
            )

        # Display pH gradient warning if significant
        ph_gradient = abs(ph_anode - ph_cathode)
        if ph_gradient > 2.0:
            st.warning(f"Large pH gradient ({ph_gradient:.1f}) may affect membrane performance and lifetime.")

        return temperature, ph_anode, ph_cathode

    def render_custom_membrane_properties(self) -> MembraneProperties:
        """Render custom membrane properties input."""
        st.markdown("### ⚙️ Custom Membrane Properties")

        col1, col2 = st.columns(2)

        with col1:
            proton_conductivity = st.number_input(
                "Proton Conductivity (S/cm)",
                min_value=0.001,
                max_value=1.0,
                value=0.05,
                step=0.001,
                format="%.3f",
                key="custom_proton_conductivity",
                help="Proton conductivity at 25°C"
            )

            ion_exchange_capacity = st.number_input(
                "Ion Exchange Capacity (meq/g)",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                key="custom_iec",
                help="Ion exchange capacity"
            )

            permselectivity = st.slider(
                "Permselectivity",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
                key="custom_permselectivity",
                help="Cation selectivity (1.0 = perfect selectivity)"
            )

            thickness = st.number_input(
                "Thickness (μm)",
                min_value=10.0,
                max_value=2000.0,
                value=150.0,
                step=10.0,
                key="custom_thickness",
                help="Membrane thickness"
            )

            water_uptake = st.number_input(
                "Water Uptake (%)",
                min_value=5.0,
                max_value=200.0,
                value=30.0,
                step=5.0,
                key="custom_water_uptake",
                help="Water uptake percentage"
            )

        with col2:
            density = st.number_input(
                "Density (g/cm³)",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                format="%.1f",
                key="custom_density",
                help="Dry membrane density"
            )

            area_resistance = st.number_input(
                "Area Resistance (Ω·cm²)",
                min_value=0.1,
                max_value=50.0,
                value=2.0,
                step=0.1,
                format="%.1f",
                key="custom_area_resistance",
                help="Area-specific resistance"
            )

            oxygen_permeability = st.number_input(
                "Oxygen Permeability (cm²/s)",
                min_value=1e-15,
                max_value=1e-8,
                value=1e-12,
                step=1e-13,
                format="%.2e",
                key="custom_oxygen_perm",
                help="Oxygen crossover coefficient"
            )

            substrate_permeability = st.number_input(
                "Substrate Permeability (cm²/s)",
                min_value=1e-16,
                max_value=1e-9,
                value=1e-14,
                step=1e-15,
                format="%.2e",
                key="custom_substrate_perm",
                help="Substrate crossover coefficient"
            )

            expected_lifetime = st.number_input(
                "Expected Lifetime (hours)",
                min_value=100.0,
                max_value=50000.0,
                value=2000.0,
                step=100.0,
                key="custom_lifetime",
                help="Expected operational lifetime"
            )

        # Optional properties
        col1, col2 = st.columns(2)

        with col1:
            tensile_strength = st.number_input(
                "Tensile Strength (MPa)",
                min_value=1.0,
                max_value=100.0,
                value=40.0,
                step=1.0,
                key="custom_tensile_strength",
                help="Mechanical strength (optional)"
            )

            max_operating_temp = st.number_input(
                "Max Operating Temperature (°C)",
                min_value=40.0,
                max_value=120.0,
                value=60.0,
                step=5.0,
                key="custom_max_temp",
                help="Maximum safe operating temperature"
            )

        with col2:
            cost_per_m2 = st.number_input(
                "Cost ($/m²)",
                min_value=1.0,
                max_value=2000.0,
                value=200.0,
                step=10.0,
                key="custom_cost",
                help="Material cost per square meter (optional)"
            )

        reference = st.text_input(
            "Reference/Notes",
            value="Custom user specification",
            key="custom_reference",
            help="Literature reference or notes about the material"
        )

        return MembraneProperties(
            proton_conductivity=proton_conductivity,
            ion_exchange_capacity=ion_exchange_capacity,
            permselectivity=permselectivity,
            thickness=thickness,
            water_uptake=water_uptake,
            density=density,
            area_resistance=area_resistance,
            oxygen_permeability=oxygen_permeability,
            substrate_permeability=substrate_permeability,
            tensile_strength=tensile_strength,
            max_operating_temp=max_operating_temp,
            cost_per_m2=cost_per_m2,
            expected_lifetime=expected_lifetime,
            reference=reference
        )

    def calculate_and_display_performance(self, config: MembraneConfiguration):
        """Calculate and display membrane performance metrics."""
        st.markdown("### 📊 Membrane Performance Analysis")

        # Performance calculations
        resistance = config.calculate_resistance()

        # Test with different current densities
        current_densities = [10, 50, 100, 200, 500]  # A/m²

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Membrane Resistance",
                f"{resistance:.3f} Ω",
                help="Total membrane resistance"
            )

        with col2:
            conductance = 1 / resistance if resistance > 0 else 0
            st.metric(
                "Conductance",
                f"{conductance:.3f} S",
                help="Membrane conductance (1/resistance)"
            )

        with col3:
            area_conductivity = config.properties.proton_conductivity * config.area * 100  # Convert to S
            st.metric(
                "Area Conductivity",
                f"{area_conductivity:.4f} S",
                help="Conductivity accounting for membrane area"
            )

        with col4:
            ph_gradient = abs(config.ph_anode - config.ph_cathode)
            st.metric(
                "pH Gradient",
                f"{ph_gradient:.1f}",
                help="pH difference across membrane"
            )

        # Proton flux calculations
        st.markdown("#### Proton Flux at Different Current Densities")

        flux_data = []
        for cd in current_densities:
            proton_flux = config.calculate_proton_flux(cd)
            lifetime_factor = config.estimate_lifetime_factor(cd)
            estimated_lifetime = config.properties.expected_lifetime * lifetime_factor

            flux_data.append({
                'Current Density (A/m²)': cd,
                'Proton Flux (mol/m²/s)': f"{proton_flux:.6f}",
                'Lifetime Factor': f"{lifetime_factor:.3f}",
                'Est. Lifetime (hours)': f"{estimated_lifetime:.0f}"
            })

        df = pd.DataFrame(flux_data)
        st.dataframe(df, use_container_width=True)

        # Store calculations
        st.session_state.membrane_calculations = {
            'resistance_ohm': resistance,
            'conductance_s': conductance,
            'area_conductivity_s': area_conductivity,
            'ph_gradient': ph_gradient,
            'area_m2': config.area,
            'area_cm2': config.area * 10000
        }

    def render_membrane_visualization(self, config: MembraneConfiguration):
        """Render membrane property visualization."""
        st.markdown("### 📈 Membrane Property Visualization")

        # Create visualization tabs
        transport_tab, lifetime_tab, comparison_tab = st.tabs(["🔄 Transport", "⏱️ Lifetime", "📊 Comparison"])

        with transport_tab:
            # Transport properties radar chart
            fig = go.Figure()

            # Normalize properties for radar chart (0-1 scale)
            properties = {
                'Conductivity': min(config.properties.proton_conductivity / 0.1, 1.0),
                'Selectivity': config.properties.permselectivity,
                'Low Resistance': min(5.0 / config.properties.area_resistance, 1.0),
                'Low O₂ Permeability': min(1e-11 / config.properties.oxygen_permeability, 1.0),
                'Low Substrate Crossover': min(1e-13 / config.properties.substrate_permeability, 1.0)
            }

            fig.add_trace(go.Scatterpolar(
                r=list(properties.values()),
                theta=list(properties.keys()),
                fill='toself',
                name='Membrane Performance'
            ))

            fig.update_layout(
                polar={
                    'radialaxis': {
                        'visible': True,
                        'range': [0, 1]
                    }},
                showlegend=False,
                title="Membrane Transport Properties",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with lifetime_tab:
            # Lifetime vs current density
            current_densities = range(10, 501, 10)
            lifetime_factors = [config.estimate_lifetime_factor(cd) for cd in current_densities]
            estimated_lifetimes = [config.properties.expected_lifetime * lf for lf in lifetime_factors]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=list(current_densities),
                y=estimated_lifetimes,
                mode='lines+markers',
                name='Estimated Lifetime',
                line={'color': 'red', 'width': 2}
            ))

            fig.update_layout(
                title='Membrane Lifetime vs Current Density',
                xaxis_title='Current Density (A/m²)',
                yaxis_title='Estimated Lifetime (hours)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Current operating point
            if 'target_current_density' not in st.session_state:
                st.session_state.target_current_density = 100.0

            target_cd = st.slider(
                "Target Current Density (A/m²)",
                min_value=10.0,
                max_value=500.0,
                value=st.session_state.target_current_density,
                step=10.0,
                key="target_current_density"
            )

            target_lifetime = config.properties.expected_lifetime * config.estimate_lifetime_factor(target_cd)
            st.info(f"At {target_cd} A/m², estimated lifetime: {target_lifetime:.0f} hours ({target_lifetime/24:.1f} days)")

        with comparison_tab:
            # Compare with common membrane materials
            if config.material != MembraneMaterial.CUSTOM:
                st.write("Comparison with other common membranes:")

                comparison_materials = [
                    MembraneMaterial.NAFION_117,
                    MembraneMaterial.ULTREX_CMI_7000,
                    MembraneMaterial.J_CLOTH
                ]

                comparison_data = []
                for material in comparison_materials:
                    if material in MEMBRANE_PROPERTIES_DATABASE:
                        props = MEMBRANE_PROPERTIES_DATABASE[material]
                        comparison_data.append({
                            'Material': material.value.replace('_', ' ').title(),
                            'Conductivity (S/cm)': props.proton_conductivity,
                            'Resistance (Ω·cm²)': props.area_resistance,
                            'Selectivity': props.permselectivity,
                            'Cost ($/m²)': props.cost_per_m2 or 0,
                            'Lifetime (hours)': props.expected_lifetime
                        })

                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)

    def render_configuration_summary(self, config: MembraneConfiguration):
        """Render a summary of the membrane configuration."""
        st.markdown("### 📋 Membrane Configuration Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Material:** {config.material.value.replace('_', ' ').title()}")
            st.write(f"**Area:** {config.area * 10000:.2f} cm² ({config.area:.6f} m²)")
            st.write(f"**Thickness:** {config.properties.thickness:.0f} μm")
            st.write(f"**Operating Temperature:** {config.operating_temperature:.1f}°C")
            st.write(f"**pH Gradient:** {abs(config.ph_anode - config.ph_cathode):.1f}")

        with col2:
            st.write(f"**Resistance:** {config.calculate_resistance():.3f} Ω")
            st.write(f"**Conductivity:** {config.properties.proton_conductivity:.3f} S/cm")
            st.write(f"**Selectivity:** {config.properties.permselectivity:.3f}")
            st.write(f"**Expected Lifetime:** {config.properties.expected_lifetime:.0f} hours")
            if config.properties.cost_per_m2:
                total_cost = config.properties.cost_per_m2 * config.area
                st.write(f"**Material Cost:** ${total_cost:.2f}")

    def render_full_membrane_configuration(self) -> MembraneConfiguration | None:
        """Render complete membrane configuration interface."""
        st.markdown("## 🧬 Membrane Configuration")
        st.markdown("Configure membrane material, area, and operating conditions for accurate MFC modeling.")

        # Material selection
        membrane_material = self.render_material_selector()

        # Area input
        membrane_area = self.render_area_input()

        # Operating conditions
        temperature, ph_anode, ph_cathode = self.render_operating_conditions()

        # Custom material properties if needed
        membrane_props = None
        if membrane_material == MembraneMaterial.CUSTOM:
            membrane_props = self.render_custom_membrane_properties()

        # Create configuration
        try:
            membrane_config = create_membrane_config(
                material=membrane_material,
                area=membrane_area,
                custom_properties=membrane_props
            )

            # Set operating conditions
            membrane_config.operating_temperature = temperature
            membrane_config.ph_anode = ph_anode
            membrane_config.ph_cathode = ph_cathode

            st.session_state.membrane_config = membrane_config

            # Display calculations and visualizations
            self.calculate_and_display_performance(membrane_config)
            self.render_membrane_visualization(membrane_config)
            self.render_configuration_summary(membrane_config)

            return membrane_config

        except ValueError as e:
            st.error(f"Configuration error: {e}")
            return None


# Export main class
__all__ = ['MembraneConfigurationUI']

