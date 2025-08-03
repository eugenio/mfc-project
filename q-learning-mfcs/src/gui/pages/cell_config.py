#!/usr/bin/env python3
"""
Cell Configuration Page for Enhanced MFC Platform

Provides comprehensive cell geometry configuration including:
- Simple geometric cell shapes with parameter validation
- 3D model file upload capability
- Dimensional constraint validation
- Cell volume and electrode spacing calculations

Created: 2025-08-02
"""

import math

import streamlit as st
from gui.pages.cell_config_helpers import (
    render_3d_model_upload,
    render_cell_calculations,
    render_validation_analysis,
)
from gui.scientific_widgets import ParameterSpec


def render_cell_configuration_page():
    """Render the cell configuration page."""

    # Page header
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.title("🏗️ MFC Cell Configuration")
        st.caption("Design and validate cell geometry for optimal performance")

    with col2:
        st.info("⚗️ Cell Design")

    with col3:
        st.metric("Configurations", "6", "Available")

    # Main interface
    render_cell_configuration_interface()


# Cell Parameter Specifications
CELL_PARAMETERS = {
    "volume": ParameterSpec(
        name="Cell Volume",
        unit="mL",
        min_value=1.0,
        max_value=10000.0,
        typical_range=(50.0, 1000.0),
        literature_refs="Logan, B.E. (2008). Microbial Fuel Cells: Methodology and Technology",
        description="Total internal volume of the MFC cell"
    ),

    "electrode_spacing": ParameterSpec(
        name="Electrode Spacing",
        unit="cm",
        min_value=0.5,
        max_value=50.0,
        typical_range=(2.0, 10.0),
        literature_refs="Du, Z. et al. (2007). Environ. Sci. Technol.",
        description="Distance between anode and cathode"
    ),

    "membrane_area": ParameterSpec(
        name="Membrane Area",
        unit="cm²",
        min_value=1.0,
        max_value=1000.0,
        typical_range=(10.0, 100.0),
        literature_refs="Cheng, S. et al. (2006). Environ. Sci. Technol.",
        description="Effective area of proton exchange membrane"
    ),

    "flow_rate": ParameterSpec(
        name="Flow Rate",
        unit="mL/min",
        min_value=0.1,
        max_value=100.0,
        typical_range=(1.0, 20.0),
        literature_refs="Liu, H. et al. (2005). Environ. Sci. Technol.",
        description="Liquid flow rate through the cell"
    )
}


def render_cell_configuration_interface():
    """Render the main cell configuration interface."""

    # Configuration tabs
    tab1, tab2, tab3 = st.tabs([
        "📐 Simple Geometries",
        "🎯 3D Model Upload",
        "🔬 Validation & Analysis"
    ])

    with tab1:
        render_simple_geometries()

    with tab2:
        render_3d_model_upload()

    with tab3:
        render_validation_analysis()


def render_simple_geometries():
    """Render simple geometry configuration with parameter validation."""

    st.subheader("📐 Cell Geometry Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cell Shape Selection")

        cell_shape = st.selectbox(
            "Cell Type:",
            ["Rectangular Chamber", "Cylindrical Reactor", "H-Type Cell",
             "Tubular MFC", "Microbial Electrolysis Cell", "Custom Geometry"],
            help="Select the basic cell geometry type"
        )

        # Dynamic geometry parameters based on cell shape
        if cell_shape == "Rectangular Chamber":
            render_rectangular_cell_parameters()
        elif cell_shape == "Cylindrical Reactor":
            render_cylindrical_cell_parameters()
        elif cell_shape == "H-Type Cell":
            render_h_type_cell_parameters()
        elif cell_shape == "Tubular MFC":
            render_tubular_cell_parameters()
        elif cell_shape == "Microbial Electrolysis Cell":
            render_mec_cell_parameters()
        else:
            st.info("Custom geometry parameters coming soon!")

    with col2:
        st.markdown("#### Real-time Calculations")

        # Calculate and display properties
        if 'cell_config' in st.session_state:
            render_cell_calculations()
        else:
            st.info("Configure cell geometry to see calculations")


def render_rectangular_cell_parameters():
    """Render parameters for rectangular chamber cell."""

    st.markdown("##### Rectangular Chamber Parameters")

    length = st.number_input(
        "Length (cm)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Internal length of the rectangular chamber"
    )

    width = st.number_input(
        "Width (cm)",
        min_value=1.0,
        max_value=100.0,
        value=8.0,
        step=0.5,
        help="Internal width of the rectangular chamber"
    )

    height = st.number_input(
        "Height (cm)",
        min_value=1.0,
        max_value=50.0,
        value=6.0,
        step=0.5,
        help="Internal height of the rectangular chamber"
    )

    electrode_spacing = st.number_input(
        "Electrode Spacing (cm)",
        min_value=0.5,
        max_value=min(length, width, height) * 0.8,
        value=min(5.0, min(length, width, height) * 0.5),
        step=0.1,
        help="Distance between anode and cathode"
    )

    # Store configuration
    st.session_state.cell_config = {
        'type': 'rectangular',
        'length': length,
        'width': width,
        'height': height,
        'electrode_spacing': electrode_spacing,
        'volume': length * width * height,
        'electrode_area': length * height
    }


def render_cylindrical_cell_parameters():
    """Render parameters for cylindrical reactor cell."""

    st.markdown("##### Cylindrical Reactor Parameters")

    diameter = st.number_input(
        "Internal Diameter (cm)",
        min_value=1.0,
        max_value=50.0,
        value=8.0,
        step=0.5,
        help="Internal diameter of the cylindrical reactor"
    )

    height = st.number_input(
        "Height (cm)",
        min_value=1.0,
        max_value=100.0,
        value=12.0,
        step=0.5,
        help="Internal height of the cylindrical reactor"
    )

    electrode_spacing = st.number_input(
        "Electrode Spacing (cm)",
        min_value=0.5,
        max_value=diameter * 0.8,
        value=min(4.0, diameter * 0.5),
        step=0.1,
        help="Distance between cylindrical electrodes"
    )

    # Store configuration
    volume = math.pi * (diameter / 2) ** 2 * height
    electrode_area = math.pi * diameter * height

    st.session_state.cell_config = {
        'type': 'cylindrical',
        'diameter': diameter,
        'height': height,
        'electrode_spacing': electrode_spacing,
        'volume': volume,
        'electrode_area': electrode_area
    }


def render_h_type_cell_parameters():
    """Render parameters for H-type cell."""

    st.markdown("##### H-Type Cell Parameters")

    chamber_volume = st.number_input(
        "Each Chamber Volume (mL)",
        min_value=10.0,
        max_value=2000.0,
        value=250.0,
        step=10.0,
        help="Volume of each chamber (anode and cathode sides)"
    )

    membrane_area = st.number_input(
        "Membrane Area (cm²)",
        min_value=1.0,
        max_value=100.0,
        value=12.5,
        step=0.5,
        help="Effective area of the proton exchange membrane"
    )

    bridge_length = st.number_input(
        "Salt Bridge Length (cm)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Length of the connecting salt bridge"
    )

    # Store configuration
    st.session_state.cell_config = {
        'type': 'h_type',
        'chamber_volume': chamber_volume,
        'membrane_area': membrane_area,
        'bridge_length': bridge_length,
        'total_volume': chamber_volume * 2,
        'electrode_spacing': bridge_length
    }


def render_tubular_cell_parameters():
    """Render parameters for tubular MFC cell."""

    st.markdown("##### Tubular MFC Parameters")

    outer_diameter = st.number_input(
        "Outer Diameter (cm)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Outer diameter of the tubular MFC"
    )

    inner_diameter = st.number_input(
        "Inner Diameter (cm)",
        min_value=0.5,
        max_value=outer_diameter * 0.8,
        value=min(3.0, outer_diameter * 0.6),
        step=0.5,
        help="Inner diameter of the tubular MFC"
    )

    length = st.number_input(
        "Length (cm)",
        min_value=5.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        help="Length of the tubular MFC"
    )

    # Store configuration
    volume = math.pi * ((outer_diameter / 2) ** 2 - (inner_diameter / 2) ** 2) * length
    electrode_area = math.pi * inner_diameter * length

    st.session_state.cell_config = {
        'type': 'tubular',
        'outer_diameter': outer_diameter,
        'inner_diameter': inner_diameter,
        'length': length,
        'volume': volume,
        'electrode_area': electrode_area,
        'electrode_spacing': (outer_diameter - inner_diameter) / 2
    }


def render_mec_cell_parameters():
    """Render parameters for microbial electrolysis cell."""

    st.markdown("##### Microbial Electrolysis Cell Parameters")

    reactor_volume = st.number_input(
        "Reactor Volume (mL)",
        min_value=50.0,
        max_value=5000.0,
        value=500.0,
        step=50.0,
        help="Total volume of the MEC reactor"
    )

    cathode_area = st.number_input(
        "Cathode Area (cm²)",
        min_value=5.0,
        max_value=200.0,
        value=25.0,
        step=1.0,
        help="Surface area of the hydrogen-producing cathode"
    )

    anode_area = st.number_input(
        "Anode Area (cm²)",
        min_value=5.0,
        max_value=200.0,
        value=30.0,
        step=1.0,
        help="Surface area of the biofilm-hosting anode"
    )

    voltage_input = st.number_input(
        "Applied Voltage (V)",
        min_value=0.2,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="External voltage applied to drive electrolysis"
    )

    # Store configuration
    st.session_state.cell_config = {
        'type': 'mec',
        'reactor_volume': reactor_volume,
        'cathode_area': cathode_area,
        'anode_area': anode_area,
        'applied_voltage': voltage_input,
        'volume': reactor_volume,
        'electrode_area': max(cathode_area, anode_area),
        'electrode_spacing': (reactor_volume / max(cathode_area, anode_area)) ** 0.5
    }


def render_membrane_configuration():
    """
    Render membrane configuration interface for MFC cell design.

    This function integrates membrane selection, properties display,
    custom membrane input, resistance calculations, and performance metrics
    following the same pattern as other cell configuration functions.
    """
    st.subheader("🧬 Membrane Configuration")
    st.markdown("Configure membrane material and properties for ion transport modeling.")

    # Import membrane configuration components
    try:
        from gui.membrane_configuration_ui import MembraneConfigurationUI

        # Initialize membrane UI component
        membrane_ui = MembraneConfigurationUI()

        # Create two main columns for layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Material Selection & Properties")

            # Material selector
            membrane_material = membrane_ui.render_material_selector()

            # Area input
            membrane_area = membrane_ui.render_area_input()

        with col2:
            st.markdown("#### Operating Conditions")

            # Operating conditions
            temperature, ph_anode, ph_cathode = membrane_ui.render_operating_conditions()

        # Custom properties section (full width if custom material)
        if hasattr(membrane_ui, 'render_custom_membrane_properties'):
            from config.membrane_config import MembraneMaterial
            if membrane_material == MembraneMaterial.CUSTOM:
                st.markdown("---")
                membrane_props = membrane_ui.render_custom_membrane_properties()
            else:
                membrane_props = None
        else:
            membrane_props = None

        # Create and display membrane configuration
        try:
            from config.membrane_config import create_membrane_config

            membrane_config = create_membrane_config(
                material=membrane_material,
                area=membrane_area,
                custom_properties=membrane_props
            )

            # Set operating conditions
            membrane_config.operating_temperature = temperature
            membrane_config.ph_anode = ph_anode
            membrane_config.ph_cathode = ph_cathode

            # Store in session state for integration with cell config
            st.session_state.membrane_config = membrane_config

            # Display separator
            st.markdown("---")

            # Performance analysis section
            st.markdown("#### Performance Analysis")

            # Calculate key metrics
            resistance = membrane_config.calculate_resistance()

            # Display key metrics in columns
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric(
                    "Membrane Resistance",
                    f"{resistance:.3f} Ω",
                    help="Total membrane resistance based on area and material properties"
                )

            with metric_col2:
                conductance = 1 / resistance if resistance > 0 else 0
                st.metric(
                    "Conductance",
                    f"{conductance:.3f} S",
                    help="Membrane conductance (inverse of resistance)"
                )

            with metric_col3:
                ph_gradient = abs(ph_anode - ph_cathode)
                st.metric(
                    "pH Gradient",
                    f"{ph_gradient:.1f}",
                    help="pH difference across membrane affecting performance"
                )

            with metric_col4:
                # Calculate proton flux at typical MFC current density (100 A/m²)
                typical_current_density = 100.0  # A/m²
                proton_flux = membrane_config.calculate_proton_flux(typical_current_density)
                st.metric(
                    "Proton Flux",
                    f"{proton_flux:.2e} mol/m²/s",
                    help=f"Proton flux at {typical_current_density} A/m²"
                )

            # Membrane properties summary table
            st.markdown("##### Membrane Properties Summary")

            properties_data = {
                "Property": [
                    "Proton Conductivity",
                    "Ion Exchange Capacity",
                    "Permselectivity",
                    "Thickness",
                    "Area Resistance",
                    "Expected Lifetime"
                ],
                "Value": [
                    f"{membrane_config.properties.proton_conductivity:.3f} S/cm",
                    f"{membrane_config.properties.ion_exchange_capacity:.1f} meq/g",
                    f"{membrane_config.properties.permselectivity:.3f}",
                    f"{membrane_config.properties.thickness:.0f} μm",
                    f"{membrane_config.properties.area_resistance:.1f} Ω·cm²",
                    f"{membrane_config.properties.expected_lifetime:.0f} hours"
                ],
                "Reference": [
                    membrane_config.properties.reference,
                    "Material specification",
                    "Literature value",
                    "Standard thickness",
                    "Measured in 0.5M NaCl",
                    "Estimated operational lifetime"
                ]
            }

            import pandas as pd
            df_properties = pd.DataFrame(properties_data)
            st.dataframe(df_properties, use_container_width=True, hide_index=True)

            # Integration with cell configuration
            if 'cell_config' in st.session_state:
                st.markdown("##### Integration with Cell Geometry")

                cell_config = st.session_state.cell_config

                # Check if membrane area is compatible with cell geometry
                cell_volume_ml = cell_config.get('volume', 0)
                membrane_area_cm2 = membrane_area * 10000

                integration_col1, integration_col2 = st.columns(2)

                with integration_col1:
                    st.write(f"**Cell Volume:** {cell_volume_ml:.1f} mL")
                    st.write(f"**Membrane Area:** {membrane_area_cm2:.1f} cm²")

                    # Calculate area-to-volume ratio
                    if cell_volume_ml > 0:
                        area_to_volume = membrane_area_cm2 / cell_volume_ml
                        st.write(f"**Area-to-Volume Ratio:** {area_to_volume:.2f} cm²/mL")

                with integration_col2:
                    # Provide recommendations
                    if cell_volume_ml > 0:
                        recommended_area = cell_volume_ml * 0.1  # Typical ratio
                        area_ratio = membrane_area_cm2 / recommended_area if recommended_area > 0 else 0

                        if area_ratio < 0.5:
                            st.warning("⚠️ Membrane area may be too small for this cell volume")
                        elif area_ratio > 2.0:
                            st.warning("⚠️ Membrane area may be oversized for this cell volume")
                        else:
                            st.success("✅ Membrane area is appropriate for cell volume")

            # Performance visualization option
            if st.checkbox("Show Advanced Membrane Analysis", value=False):
                membrane_ui.render_membrane_visualization(membrane_config)

        except ValueError as e:
            st.error(f"Membrane configuration error: {e}")
        except ImportError as e:
            st.error(f"Import error: {e}")
            st.warning("Please ensure membrane configuration modules are properly installed.")

    except ImportError:
        st.error("Membrane configuration UI not available. Please install required modules.")

        # Fallback basic membrane configuration
        st.markdown("#### Basic Membrane Configuration")

        basic_material = st.selectbox(
            "Membrane Type",
            ["Nafion 117", "Nafion 112", "Ultrex CMI-7000", "J-Cloth", "Custom"],
            help="Select membrane material for basic configuration"
        )

        basic_area = st.number_input(
            "Membrane Area (cm²)",
            min_value=0.1,
            max_value=1000.0,
            value=25.0,
            step=0.1,
            help="Active membrane area"
        )

        basic_resistance = st.number_input(
            "Area Resistance (Ω·cm²)",
            min_value=0.1,
            max_value=50.0,
            value=2.0,
            step=0.1,
            help="Membrane area-specific resistance"
        )

        # Store basic configuration
        st.session_state.membrane_config = {
            'material': basic_material,
            'area_cm2': basic_area,
            'area_m2': basic_area / 10000,
            'resistance_ohm_cm2': basic_resistance,
            'total_resistance_ohm': basic_resistance / basic_area
        }

        st.info(f"Basic membrane configuration: {basic_material}, {basic_area} cm², {basic_resistance / basic_area:.3f} Ω total resistance")




