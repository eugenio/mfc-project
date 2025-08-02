#!/usr/bin/env python3
"""
Electrode Configuration UI Component

Provides comprehensive electrode configuration interface including:
- Material selection with literature-based properties
- Geometry specification with real-time area calculations
- Surface area visualization and validation
- Material property customization

Created: 2025-08-01
"""


import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config.electrode_config import (
    MATERIAL_PROPERTIES_DATABASE,
    ElectrodeConfiguration,
    ElectrodeGeometry,
    ElectrodeMaterial,
    MaterialProperties,
    create_electrode_config,
)


class ElectrodeConfigurationUI:
    """UI component for electrode configuration."""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize electrode configuration in session state."""
        if 'anode_config' not in st.session_state:
            st.session_state.anode_config = None
        if 'cathode_config' not in st.session_state:
            st.session_state.cathode_config = None
        if 'electrode_calculations' not in st.session_state:
            st.session_state.electrode_calculations = {}

    def render_material_selector(self, electrode_type: str = "anode") -> ElectrodeMaterial:
        """Render material selection interface."""
        st.markdown(f"### 🧪 {electrode_type.title()} Material Selection")

        # Material options with descriptions
        material_options = {
            "Graphite Plate": ElectrodeMaterial.GRAPHITE_PLATE,
            "Graphite Rod": ElectrodeMaterial.GRAPHITE_ROD,
            "Carbon Felt": ElectrodeMaterial.CARBON_FELT,
            "Carbon Cloth": ElectrodeMaterial.CARBON_CLOTH,
            "Carbon Paper": ElectrodeMaterial.CARBON_PAPER,
            "Stainless Steel": ElectrodeMaterial.STAINLESS_STEEL,
            "Platinum": ElectrodeMaterial.PLATINUM,
            "Custom Material": ElectrodeMaterial.CUSTOM
        }

        selected_material_name = st.selectbox(
            f"{electrode_type.title()} Material",
            options=list(material_options.keys()),
            help="Select electrode material based on your experimental setup",
            key=f"{electrode_type}_material_select"
        )

        selected_material = material_options[selected_material_name]

        # Show material properties
        if selected_material != ElectrodeMaterial.CUSTOM:
            props = MATERIAL_PROPERTIES_DATABASE[selected_material]
            self._display_material_properties(props, selected_material_name)

        return selected_material

    def _display_material_properties(self, props: MaterialProperties, material_name: str):
        """Display material properties in an expandable section."""
        with st.expander(f"📋 {material_name} Properties", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Conductivity", f"{props.specific_conductance:,.0f} S/m")
                st.metric("Contact Resistance", f"{props.contact_resistance:.2f} Ω·cm²")
                st.metric("Hydrophobicity", f"{props.hydrophobicity_angle:.0f}°")
                st.metric("Surface Roughness", f"{props.surface_roughness:.1f}×")

            with col2:
                st.metric("Biofilm Adhesion", f"{props.biofilm_adhesion_coefficient:.1f}×")
                st.metric("Attachment Energy", f"{props.attachment_energy:.1f} kJ/mol")
                if props.specific_surface_area:
                    st.metric("Specific Surface Area", f"{props.specific_surface_area:,.0f} m²/m³")
                if props.porosity:
                    st.metric("Porosity", f"{props.porosity:.1%}")

            st.caption(f"📚 Reference: {props.reference}")

    def render_geometry_selector(self, electrode_type: str = "anode") -> tuple[ElectrodeGeometry, dict[str, float]]:
        """Render geometry selection and dimension input interface."""
        st.markdown(f"### 📐 {electrode_type.title()} Geometry Configuration")

        # Geometry options
        geometry_options = {
            "Rectangular Plate": ElectrodeGeometry.RECTANGULAR_PLATE,
            "Cylindrical Rod": ElectrodeGeometry.CYLINDRICAL_ROD,
            "Cylindrical Tube": ElectrodeGeometry.CYLINDRICAL_TUBE,
            "Spherical": ElectrodeGeometry.SPHERICAL,
            "Custom": ElectrodeGeometry.CUSTOM
        }

        selected_geometry_name = st.selectbox(
            f"{electrode_type.title()} Geometry",
            options=list(geometry_options.keys()),
            help="Select electrode geometry type",
            key=f"{electrode_type}_geometry_select"
        )

        selected_geometry = geometry_options[selected_geometry_name]

        # Dimension inputs based on geometry
        dimensions = self._render_dimension_inputs(selected_geometry, electrode_type)

        return selected_geometry, dimensions

    def _render_dimension_inputs(self, geometry: ElectrodeGeometry, electrode_type: str) -> dict[str, float]:
        """Render dimension input fields based on geometry type."""
        dimensions = {}

        col1, col2 = st.columns(2)

        if geometry == ElectrodeGeometry.RECTANGULAR_PLATE:
            with col1:
                dimensions['length'] = st.number_input(
                    "Length (cm)", min_value=0.1, max_value=50.0, value=5.0, step=0.1,
                    key=f"{electrode_type}_length", help="Electrode length in cm"
                ) / 100  # Convert to meters

                dimensions['width'] = st.number_input(
                    "Width (cm)", min_value=0.1, max_value=50.0, value=5.0, step=0.1,
                    key=f"{electrode_type}_width", help="Electrode width in cm"
                ) / 100  # Convert to meters

            with col2:
                dimensions['thickness'] = st.number_input(
                    "Thickness (mm)", min_value=0.1, max_value=20.0, value=5.0, step=0.1,
                    key=f"{electrode_type}_thickness", help="Electrode thickness in mm"
                ) / 1000  # Convert to meters

        elif geometry == ElectrodeGeometry.CYLINDRICAL_ROD:
            with col1:
                dimensions['diameter'] = st.number_input(
                    "Diameter (mm)", min_value=0.1, max_value=50.0, value=10.0, step=0.1,
                    key=f"{electrode_type}_diameter", help="Rod diameter in mm"
                ) / 1000  # Convert to meters

            with col2:
                dimensions['length'] = st.number_input(
                    "Length (cm)", min_value=0.1, max_value=50.0, value=10.0, step=0.1,
                    key=f"{electrode_type}_rod_length", help="Rod length in cm"
                ) / 100  # Convert to meters

        elif geometry == ElectrodeGeometry.CYLINDRICAL_TUBE:
            with col1:
                dimensions['diameter'] = st.number_input(
                    "Outer Diameter (mm)", min_value=1.0, max_value=50.0, value=15.0, step=0.1,
                    key=f"{electrode_type}_tube_diameter", help="Outer diameter in mm"
                ) / 1000  # Convert to meters

                dimensions['thickness'] = st.number_input(
                    "Wall Thickness (mm)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    key=f"{electrode_type}_wall_thickness", help="Wall thickness in mm"
                ) / 1000  # Convert to meters

            with col2:
                dimensions['length'] = st.number_input(
                    "Tube Length (cm)", min_value=0.1, max_value=50.0, value=10.0, step=0.1,
                    key=f"{electrode_type}_tube_length", help="Tube length in cm"
                ) / 100  # Convert to meters

        elif geometry == ElectrodeGeometry.SPHERICAL:
            with col1:
                dimensions['diameter'] = st.number_input(
                    "Diameter (mm)", min_value=1.0, max_value=100.0, value=20.0, step=0.1,
                    key=f"{electrode_type}_sphere_diameter", help="Sphere diameter in mm"
                ) / 1000  # Convert to meters

        elif geometry == ElectrodeGeometry.CUSTOM:
            with col1:
                dimensions['projected_area'] = st.number_input(
                    "Projected Area (cm²)", min_value=0.01, max_value=1000.0, value=25.0, step=0.01,
                    key=f"{electrode_type}_projected_area", help="Manually specified projected area"
                ) / 10000  # Convert to m²

            with col2:
                dimensions['total_surface_area'] = st.number_input(
                    "Total Surface Area (cm²)", min_value=0.01, max_value=10000.0, value=50.0, step=0.01,
                    key=f"{electrode_type}_total_area", help="Manually specified total surface area"
                ) / 10000  # Convert to m²

        return dimensions

    def render_custom_material_properties(self, electrode_type: str = "anode") -> MaterialProperties:
        """Render custom material properties input."""
        st.markdown(f"### ⚙️ Custom {electrode_type.title()} Material Properties")

        col1, col2 = st.columns(2)

        with col1:
            conductance = st.number_input(
                "Specific Conductance (S/m)", min_value=0.1, max_value=10000000.0,
                value=1000.0, step=100.0, format="%.1f",
                key=f"{electrode_type}_custom_conductance",
                help="Electrical conductivity of the material"
            )

            contact_resistance = st.number_input(
                "Contact Resistance (Ω·cm²)", min_value=0.001, max_value=100.0,
                value=0.5, step=0.01, format="%.3f",
                key=f"{electrode_type}_custom_resistance",
                help="Electrode-electrolyte interface resistance"
            )

            surface_charge = st.number_input(
                "Surface Charge Density (C/m²)", min_value=-1.0, max_value=1.0,
                value=0.0, step=0.01, format="%.3f",
                key=f"{electrode_type}_custom_charge",
                help="Surface charge at neutral pH"
            )

            hydrophobicity = st.slider(
                "Hydrophobicity Angle (°)", min_value=0, max_value=180, value=75,
                key=f"{electrode_type}_custom_hydrophobicity",
                help="Water contact angle (0° = hydrophilic, 180° = hydrophobic)"
            )

        with col2:
            surface_roughness = st.number_input(
                "Surface Roughness Factor", min_value=1.0, max_value=50.0,
                value=1.0, step=0.1, format="%.1f",
                key=f"{electrode_type}_custom_roughness",
                help="Surface area multiplier relative to smooth surface"
            )

            adhesion_coeff = st.number_input(
                "Biofilm Adhesion Coefficient", min_value=0.1, max_value=10.0,
                value=1.0, step=0.1, format="%.1f",
                key=f"{electrode_type}_custom_adhesion",
                help="Biofilm adhesion relative to graphite (1.0 = same as graphite)"
            )

            attachment_energy = st.number_input(
                "Attachment Energy (kJ/mol)", min_value=-50.0, max_value=10.0,
                value=-10.0, step=0.5, format="%.1f",
                key=f"{electrode_type}_custom_energy",
                help="Microbial attachment energy (negative = favorable)"
            )

            # Porous material properties
            is_porous = st.checkbox(
                "Porous Material",
                key=f"{electrode_type}_custom_porous",
                help="Check if material has significant internal surface area"
            )

        specific_surface_area = None
        porosity = None

        if is_porous:
            specific_surface_area = st.number_input(
                "Specific Surface Area (m²/m³)", min_value=100.0, max_value=10000.0,
                value=1000.0, step=100.0, format="%.0f",
                key=f"{electrode_type}_custom_specific_area",
                help="Internal surface area per unit volume"
            )

            porosity = st.slider(
                "Porosity", min_value=0.1, max_value=0.99, value=0.8,
                key=f"{electrode_type}_custom_porosity",
                help="Void fraction (0.8 = 80% void space)"
            )

        reference = st.text_input(
            "Reference/Notes",
            value="Custom user specification",
            key=f"{electrode_type}_custom_reference",
            help="Literature reference or notes about the material"
        )

        return MaterialProperties(
            specific_conductance=conductance,
            contact_resistance=contact_resistance,
            surface_charge_density=surface_charge,
            hydrophobicity_angle=hydrophobicity,
            surface_roughness=surface_roughness,
            biofilm_adhesion_coefficient=adhesion_coeff,
            attachment_energy=attachment_energy,
            specific_surface_area=specific_surface_area,
            porosity=porosity,
            reference=reference
        )

    def calculate_and_display_areas(self, config: ElectrodeConfiguration, electrode_type: str):
        """Calculate and display electrode surface areas with visualization."""
        st.markdown(f"### 📊 {electrode_type.title()} Surface Area Analysis")

        try:
            projected_area = config.geometry.calculate_projected_area()
            total_geometric_area = config.geometry.calculate_total_surface_area()
            effective_area = config.calculate_effective_surface_area()
            biofilm_capacity = config.calculate_biofilm_capacity()
            charge_transfer_coeff = config.calculate_charge_transfer_coefficient()

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Projected Area",
                    f"{projected_area * 10000:.2f} cm²",
                    help="Cross-sectional or footprint area"
                )

            with col2:
                st.metric(
                    "Geometric Area",
                    f"{total_geometric_area * 10000:.2f} cm²",
                    help="Total geometric surface area"
                )

            with col3:
                st.metric(
                    "Effective Area",
                    f"{effective_area * 10000:.2f} cm²",
                    help="Surface area available for microbial colonization"
                )

            with col4:
                enhancement_factor = effective_area / projected_area
                st.metric(
                    "Area Enhancement",
                    f"{enhancement_factor:.1f}×",
                    help="Effective area relative to projected area"
                )

            # Additional metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Biofilm Capacity",
                    f"{biofilm_capacity * 1e9:.1f} μL",
                    help="Maximum biofilm volume capacity"
                )

            with col2:
                st.metric(
                    "Charge Transfer Coeff.",
                    f"{charge_transfer_coeff:.3f}",
                    help="Relative charge transfer efficiency"
                )

            with col3:
                volume = config.geometry.calculate_volume()
                st.metric(
                    "Electrode Volume",
                    f"{volume * 1e6:.2f} cm³",
                    help="Total electrode material volume"
                )

            # Store calculations for comparison
            st.session_state.electrode_calculations[electrode_type] = {
                'projected_area_cm2': projected_area * 10000,
                'geometric_area_cm2': total_geometric_area * 10000,
                'effective_area_cm2': effective_area * 10000,
                'enhancement_factor': enhancement_factor,
                'biofilm_capacity_ul': biofilm_capacity * 1e9,
                'charge_transfer_coeff': charge_transfer_coeff,
                'volume_cm3': volume * 1e6
            }

        except ValueError as e:
            st.error(f"Error calculating areas: {e}")
            st.info("Please check that all required dimensions are specified.")

    def render_electrode_comparison(self):
        """Render comparison between anode and cathode configurations."""
        if len(st.session_state.electrode_calculations) < 2:
            return

        st.markdown("### ⚖️ Electrode Comparison")

        # Create comparison DataFrame
        comparison_data = []
        for electrode_type, calc in st.session_state.electrode_calculations.items():
            comparison_data.append({
                'Electrode': electrode_type.title(),
                'Projected Area (cm²)': calc['projected_area_cm2'],
                'Effective Area (cm²)': calc['effective_area_cm2'],
                'Enhancement Factor': calc['enhancement_factor'],
                'Biofilm Capacity (μL)': calc['biofilm_capacity_ul'],
                'Charge Transfer Coeff.': calc['charge_transfer_coeff']
            })

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)

        # Visualization
        fig = go.Figure()

        electrodes = df['Electrode'].tolist()
        projected_areas = df['Projected Area (cm²)'].tolist()
        effective_areas = df['Effective Area (cm²)'].tolist()

        fig.add_trace(go.Bar(
            name='Projected Area',
            x=electrodes,
            y=projected_areas,
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            name='Effective Area',
            x=electrodes,
            y=effective_areas,
            marker_color='darkblue'
        ))

        fig.update_layout(
            title='Electrode Surface Area Comparison',
            xaxis_title='Electrode Type',
            yaxis_title='Surface Area (cm²)',
            barmode='group',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_configuration_summary(self, config: ElectrodeConfiguration, electrode_type: str):
        """Render a summary of the electrode configuration."""
        st.markdown(f"### 📋 {electrode_type.title()} Configuration Summary")

        summary = config.get_configuration_summary()

        # Display in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Material:** {summary['material'].replace('_', ' ').title()}")
            st.write(f"**Geometry:** {summary['geometry'].replace('_', ' ').title()}")
            st.write(f"**Projected Area:** {summary['projected_area_cm2']:.2f} cm²")
            st.write(f"**Effective Area:** {summary['effective_area_cm2']:.2f} cm²")

        with col2:
            st.write(f"**Biofilm Capacity:** {summary['biofilm_capacity_ul']:.1f} μL")
            st.write(f"**Charge Transfer:** {summary['charge_transfer_coeff']:.3f}")
            st.write(f"**Conductance:** {summary['specific_conductance_S_per_m']:,.0f} S/m")
            st.write(f"**Hydrophobicity:** {summary['hydrophobicity_angle_deg']:.0f}°")

    def render_full_electrode_configuration(self) -> tuple[ElectrodeConfiguration | None, ElectrodeConfiguration | None]:
        """Render complete electrode configuration interface."""
        st.markdown("## ⚡ Electrode Configuration")
        st.markdown("Configure electrode materials, geometry, and properties for accurate MFC modeling.")

        # Tabs for anode and cathode
        anode_tab, cathode_tab, comparison_tab = st.tabs(["🔋 Anode", "🔋 Cathode", "📊 Comparison"])

        anode_config = None
        cathode_config = None

        with anode_tab:
            st.markdown("Configure the anode (negative electrode) where microbial oxidation occurs.")

            # Material selection
            anode_material = self.render_material_selector("anode")

            # Geometry selection
            anode_geometry, anode_dimensions = self.render_geometry_selector("anode")

            # Custom material properties if needed
            anode_props = None
            if anode_material == ElectrodeMaterial.CUSTOM:
                anode_props = self.render_custom_material_properties("anode")

            # Create configuration
            try:
                anode_config = create_electrode_config(
                    material=anode_material,
                    geometry_type=anode_geometry,
                    dimensions=anode_dimensions,
                    custom_properties=anode_props
                )

                st.session_state.anode_config = anode_config

                # Display calculations
                self.calculate_and_display_areas(anode_config, "anode")
                self.render_configuration_summary(anode_config, "anode")

            except ValueError as e:
                st.error(f"Configuration error: {e}")

        with cathode_tab:
            st.markdown("Configure the cathode (positive electrode) where reduction occurs.")

            # Option to use same as anode
            use_same_as_anode = st.checkbox(
                "Use same configuration as anode",
                help="Use identical material and geometry for cathode"
            )

            if not use_same_as_anode:
                # Material selection
                cathode_material = self.render_material_selector("cathode")

                # Geometry selection
                cathode_geometry, cathode_dimensions = self.render_geometry_selector("cathode")

                # Custom material properties if needed
                cathode_props = None
                if cathode_material == ElectrodeMaterial.CUSTOM:
                    cathode_props = self.render_custom_material_properties("cathode")

                # Create configuration
                try:
                    cathode_config = create_electrode_config(
                        material=cathode_material,
                        geometry_type=cathode_geometry,
                        dimensions=cathode_dimensions,
                        custom_properties=cathode_props
                    )

                    st.session_state.cathode_config = cathode_config

                    # Display calculations
                    self.calculate_and_display_areas(cathode_config, "cathode")
                    self.render_configuration_summary(cathode_config, "cathode")

                except ValueError as e:
                    st.error(f"Configuration error: {e}")
            else:
                if anode_config:
                    cathode_config = anode_config
                    st.session_state.cathode_config = cathode_config
                    st.success("Cathode configuration set to match anode.")

        with comparison_tab:
            st.markdown("Compare anode and cathode configurations.")
            self.render_electrode_comparison()

        return anode_config, cathode_config


# Export main class
__all__ = ['ElectrodeConfigurationUI']
