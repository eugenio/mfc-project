"""
Literature-Referenced Parameter Input Component

This module provides scientific parameter input forms with literature citations,
real-time validation, and visual indicators for parameter ranges based on
peer-reviewed research.

User Story 1.1.1: Literature-Referenced Parameter Input
Created: 2025-07-31
Last Modified: 2025-07-31
"""

import json
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import literature database
from config.literature_database import LITERATURE_DB, ParameterCategory, ParameterInfo

# Import parameter bridge for integration
from config.parameter_bridge import PARAMETER_BRIDGE

# Import existing config
from config.qlearning_config import QLearningConfig

# Import real-time validator
from config.real_time_validator import REAL_TIME_VALIDATOR, ValidationLevel

# Import unit converter
from config.unit_converter import UNIT_CONVERTER


class ParameterInputComponent:
    """Component for literature-referenced parameter input and validation."""

    def __init__(self):
        """Initialize parameter input component."""
        self.literature_db = LITERATURE_DB
        self.current_config = None

        # Initialize session state for parameter tracking
        if 'parameter_values' not in st.session_state:
            st.session_state.parameter_values = {}
        if 'validation_results' not in st.session_state:
            st.session_state.validation_results = {}
        if 'parameter_citations' not in st.session_state:
            st.session_state.parameter_citations = {}
        if 'research_objective' not in st.session_state:
            st.session_state.research_objective = None
        if 'show_performance_metrics' not in st.session_state:
            st.session_state.show_performance_metrics = False

    def render_parameter_input_form(self) -> dict[str, Any]:
        """
        Render the main parameter input form with literature validation.

        Returns:
            Dictionary of validated parameter values and metadata
        """
        st.header("🔬 Scientific Parameter Configuration")
        st.markdown("""
        Configure MFC parameters with literature-validated ranges and citations.
        All parameters are backed by peer-reviewed research for scientific rigor.
        """)

        # Research objective selector
        col1, col2 = st.columns([2, 1])
        with col1:
            research_objectives = ["None"] + REAL_TIME_VALIDATOR.get_research_objectives()
            selected_objective = st.selectbox(
                "🎯 Research Objective (Optional)",
                options=research_objectives,
                help="Select your research objective to get targeted parameter suggestions"
            )
            st.session_state.research_objective = selected_objective if selected_objective != "None" else None

        with col2:
            if st.button("📊 Show Performance Metrics"):
                st.session_state.show_performance_metrics = not st.session_state.show_performance_metrics

        # Show research objective info
        if st.session_state.research_objective:
            obj_info = REAL_TIME_VALIDATOR.get_research_objective_info(st.session_state.research_objective)
            if obj_info:
                st.info(f"🎯 **{obj_info.name}**: {obj_info.description}")
                st.caption(f"📝 {obj_info.scientific_context}")

        # Show performance metrics if requested
        if st.session_state.show_performance_metrics:
            self._render_performance_metrics()

        # Category selector
        categories = self.literature_db.get_all_categories()
        selected_categories = st.multiselect(
            "Select Parameter Categories",
            options=[cat.value for cat in categories],
            default=["electrochemical", "biological", "qlearning"],
            help="Choose which parameter categories to configure"
        )

        # Parameter input tabs
        if selected_categories:
            tabs = st.tabs([cat.title() for cat in selected_categories])

            for i, category_name in enumerate(selected_categories):
                with tabs[i]:
                    category = ParameterCategory(category_name)
                    self._render_category_parameters(category)

        # Configuration summary and validation
        st.subheader("📋 Configuration Summary")
        if st.session_state.parameter_values:
            self._render_parameter_summary()

            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Export Configuration"):
                    self._export_configuration()
            with col2:
                if st.button("Validate All Parameters"):
                    self._validate_all_parameters()
            with col3:
                if st.button("Generate Citations"):
                    self._show_citations()

            # Configuration integration section
            self.render_config_integration_section()

        return {
            'parameter_values': st.session_state.parameter_values,
            'validation_results': st.session_state.validation_results,
            'citations': st.session_state.parameter_citations,
            'config': self._create_validated_config()
        }

    def _render_category_parameters(self, category: ParameterCategory):
        """Render parameter inputs for a specific category."""
        parameters = self.literature_db.get_parameters_by_category(category)

        if not parameters:
            st.info(f"No parameters available for {category.value} category")
            return

        st.markdown(f"### {category.value.title()} Parameters")

        for param in parameters:
            self._render_parameter_input(param)

    def _render_parameter_input(self, param: ParameterInfo):
        """Render input widget for a single parameter with validation and unit conversion."""
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            # Parameter input with help text
            help_text = f"{param.description}\n\nUnit: {param.unit}\nRecommended range: {param.recommended_range[0]} - {param.recommended_range[1]} {param.unit}"

            current_value = st.session_state.parameter_values.get(param.name, param.typical_value)

            # Use appropriate input widget based on parameter range
            if param.max_value - param.min_value > 1000:
                # Use number input for large ranges
                value = st.number_input(
                    f"{param.name} ({param.symbol})",
                    min_value=float(param.min_value),
                    max_value=float(param.max_value),
                    value=float(current_value),
                    help=help_text,
                    key=f"input_{param.name}"
                )
            else:
                # Use slider for smaller ranges
                value = st.slider(
                    f"{param.name} ({param.symbol})",
                    min_value=float(param.min_value),
                    max_value=float(param.max_value),
                    value=float(current_value),
                    help=help_text,
                    key=f"slider_{param.name}"
                )

            # Store parameter value
            st.session_state.parameter_values[param.name] = value

        with col2:
            # Unit conversion options
            compatible_units = UNIT_CONVERTER.get_compatible_units(param.unit)
            if len(compatible_units) > 1:
                selected_unit = st.selectbox(
                    "Unit",
                    options=compatible_units,
                    index=compatible_units.index(param.unit) if param.unit in compatible_units else 0,
                    key=f"unit_{param.name}",
                    label_visibility="collapsed"
                )

                # Convert value if different unit selected
                if selected_unit != param.unit:
                    converted_value = UNIT_CONVERTER.convert(value, param.unit, selected_unit)
                    if converted_value is not None:
                        st.caption(f"= {UNIT_CONVERTER.format_value_with_unit(converted_value, selected_unit)}")
            else:
                st.caption(param.unit)

        with col3:
            # Enhanced real-time validation
            validation_result = REAL_TIME_VALIDATOR.validate_parameter_realtime(
                param.name,
                value,
                st.session_state.research_objective
            )
            self._render_enhanced_validation_indicator(validation_result)

        # Show literature references
        with st.expander(f"📚 Literature References for {param.name}"):
            self._render_parameter_references(param)

        # Store enhanced validation result
        st.session_state.validation_results[param.name] = {
            'status': validation_result.level.value,
            'message': validation_result.message,
            'scientific_reasoning': validation_result.scientific_reasoning,
            'confidence_score': validation_result.confidence_score,
            'uncertainty_bounds': validation_result.uncertainty_bounds,
            'response_time_ms': validation_result.response_time_ms,
            'recommendations': validation_result.recommendations,
            'warnings': validation_result.warnings
        }

    def _render_validation_indicator(self, validation: dict[str, Any]):
        """Render visual validation indicator."""
        status = validation['status']

        if status == "valid":
            st.success("✅ Valid")
            st.caption(validation['message'])
        elif status == "caution":
            st.warning("⚠️ Caution")
            st.caption(validation['message'])
            for rec in validation['recommendations']:
                st.caption(f"💡 {rec}")
        elif status == "invalid":
            st.error("❌ Invalid")
            st.caption(validation['message'])
            for rec in validation['recommendations']:
                st.caption(f"🔧 {rec}")
        else:
            st.info("❓ Unknown")
            st.caption(validation['message'])

    def _render_parameter_references(self, param: ParameterInfo):
        """Render literature references for a parameter."""
        for i, ref in enumerate(param.references):
            st.markdown(f"**{i+1}.** {ref.format_citation('apa')}")
            if ref.doi:
                st.markdown(f"   DOI: [{ref.doi}](https://doi.org/{ref.doi})")

        if param.notes:
            st.markdown(f"**Notes:** {param.notes}")

    def _render_parameter_summary(self):
        """Render summary of current parameter configuration."""
        # Create summary dataframe
        summary_data = []
        for name, value in st.session_state.parameter_values.items():
            param = self.literature_db.get_parameter(name)
            validation = st.session_state.validation_results.get(name, {})

            if param:
                summary_data.append({
                    'Parameter': param.name,
                    'Symbol': param.symbol,
                    'Value': f"{value} {param.unit}",
                    'Status': validation.get('status', 'unknown').title(),
                    'Category': param.category.value.title()
                })

        if summary_data:
            df = pd.DataFrame(summary_data)

            # Color-code status
            def color_status(val):
                if val == 'Valid':
                    return 'background-color: #d4edda'
                elif val == 'Caution':
                    return 'background-color: #fff3cd'
                elif val == 'Invalid':
                    return 'background-color: #f8d7da'
                else:
                    return ''

            styled_df = df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)

            # Validation summary
            valid_count = sum(1 for v in st.session_state.validation_results.values() if v.get('status') == 'valid')
            caution_count = sum(1 for v in st.session_state.validation_results.values() if v.get('status') == 'caution')
            invalid_count = sum(1 for v in st.session_state.validation_results.values() if v.get('status') == 'invalid')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Valid Parameters", valid_count)
            with col2:
                st.metric("⚠️ Caution Parameters", caution_count)
            with col3:
                st.metric("❌ Invalid Parameters", invalid_count)

    def _validate_all_parameters(self):
        """Validate all current parameter values."""
        validation_results = {}

        for name, value in st.session_state.parameter_values.items():
            validation = self.literature_db.validate_parameter_value(name, value)
            validation_results[name] = validation

        st.session_state.validation_results = validation_results

        # Show validation summary
        valid_count = sum(1 for v in validation_results.values() if v.get('status') == 'valid')
        total_count = len(validation_results)

        if valid_count == total_count:
            st.success(f"🎉 All {total_count} parameters are valid!")
        else:
            st.warning(f"⚠️ {valid_count}/{total_count} parameters are valid. Please review parameters with caution or invalid status.")

    def _export_configuration(self):
        """Export parameter configuration with citations."""
        export_data = {
            'configuration_metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '1.0',
                'total_parameters': len(st.session_state.parameter_values)
            },
            'parameters': {},
            'literature_references': [],
            'validation_summary': {
                'valid': 0,
                'caution': 0,
                'invalid': 0
            }
        }

        # Collect parameter data with citations
        all_references = set()
        for name, value in st.session_state.parameter_values.items():
            param = self.literature_db.get_parameter(name)
            validation = st.session_state.validation_results.get(name, {})

            if param:
                export_data['parameters'][name] = {
                    'name': param.name,
                    'symbol': param.symbol,
                    'value': value,
                    'unit': param.unit,
                    'category': param.category.value,
                    'validation_status': validation.get('status', 'unknown'),
                    'description': param.description,
                    'recommended_range': param.recommended_range,
                    'references': [ref.format_citation('apa') for ref in param.references]
                }

                # Collect unique references
                for ref in param.references:
                    all_references.add(ref.format_citation('apa'))

                # Update validation summary
                status = validation.get('status', 'unknown')
                if status in export_data['validation_summary']:
                    export_data['validation_summary'][status] += 1

        export_data['literature_references'] = sorted(all_references)

        # Display export options
        st.subheader("📄 Export Configuration")

        export_format = st.selectbox(
            "Export Format",
            ["JSON", "CSV", "BibTeX Citations Only"],
            help="Choose export format for parameter configuration"
        )

        if export_format == "JSON":
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON Configuration",
                data=json_str,
                file_name=f"mfc_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.code(json_str, language="json")

        elif export_format == "CSV":
            # Create CSV data
            csv_data = []
            for name, param_data in export_data['parameters'].items():
                csv_data.append({
                    'Parameter': param_data['name'],
                    'Symbol': param_data['symbol'],
                    'Value': param_data['value'],
                    'Unit': param_data['unit'],
                    'Category': param_data['category'],
                    'Status': param_data['validation_status'],
                    'Description': param_data['description']
                })

            df = pd.DataFrame(csv_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV Configuration",
                data=csv,
                file_name=f"mfc_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.dataframe(df)

        elif export_format == "BibTeX Citations Only":
            # Generate BibTeX citations
            bibtex_entries = []
            for name, value in st.session_state.parameter_values.items():
                param = self.literature_db.get_parameter(name)
                if param:
                    for ref in param.references:
                        bibtex_entries.append(ref.format_citation('bibtex'))

            bibtex_str = '\n\n'.join(set(bibtex_entries))  # Remove duplicates
            st.download_button(
                label="Download BibTeX Citations",
                data=bibtex_str,
                file_name=f"mfc_literature_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bib",
                mime="text/plain"
            )
            st.code(bibtex_str, language="bibtex")

    def _show_citations(self):
        """Display formatted citations for all used parameters."""
        st.subheader("📚 Literature Citations")

        # Collect all unique references
        all_references = set()
        used_references = {}

        for name, _value in st.session_state.parameter_values.items():
            param = self.literature_db.get_parameter(name)
            if param:
                for ref in param.references:
                    ref_key = f"{ref.authors}_{ref.year}"
                    if ref_key not in used_references:
                        used_references[ref_key] = ref
                    all_references.add(ref_key)

        # Display citation formats
        citation_format = st.selectbox(
            "Citation Format",
            ["APA", "BibTeX"],
            help="Choose citation format style"
        )

        if citation_format == "APA":
            st.markdown("### APA Format")
            for i, (ref_key, ref) in enumerate(sorted(used_references.items()), 1):
                st.markdown(f"{i}. {ref.format_citation('apa')}")

        elif citation_format == "BibTeX":
            st.markdown("### BibTeX Format")
            for ref_key, ref in sorted(used_references.items()):
                st.code(ref.format_citation('bibtex'), language="bibtex")

        # Citation statistics
        st.markdown("### Citation Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total References", len(used_references))
        with col2:
            # Calculate parameter coverage
            total_params = len(st.session_state.parameter_values)
            st.metric("Parameters with Citations", total_params)

    def create_parameter_range_visualization(self, parameter_name: str) -> go.Figure:
        """Create visualization of parameter ranges and current value."""
        param = self.literature_db.get_parameter(parameter_name)
        if not param:
            return None

        current_value = st.session_state.parameter_values.get(parameter_name, param.typical_value)

        fig = go.Figure()

        # Add range bars
        fig.add_trace(go.Scatter(
            x=[param.min_value, param.max_value],
            y=['Valid Range', 'Valid Range'],
            mode='lines+markers',
            line={'color': 'red', 'width': 8},
            name='Valid Range',
            hovertemplate=f'Valid Range: {param.min_value} - {param.max_value} {param.unit}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=[param.recommended_range[0], param.recommended_range[1]],
            y=['Recommended', 'Recommended'],
            mode='lines+markers',
            line={'color': 'green', 'width': 8},
            name='Recommended Range',
            hovertemplate=f'Recommended: {param.recommended_range[0]} - {param.recommended_range[1]} {param.unit}<extra></extra>'
        ))

        # Add current value
        validation = self.literature_db.validate_parameter_value(parameter_name, current_value)
        color = {'valid': 'green', 'caution': 'orange', 'invalid': 'red'}.get(validation['status'], 'blue')

        fig.add_trace(go.Scatter(
            x=[current_value],
            y=['Current Value'],
            mode='markers',
            marker={'color': color, 'size': 15, 'symbol': 'diamond'},
            name='Current Value',
            hovertemplate=f'Current: {current_value} {param.unit}<br>Status: {validation["status"]}<extra></extra>'
        ))

        fig.update_layout(
            title=f'{param.name} ({param.symbol}) - Parameter Ranges',
            xaxis_title=f'Value ({param.unit})',
            yaxis_title='Range Type',
            showlegend=True,
            height=300
        )

        return fig

    def _render_enhanced_validation_indicator(self, validation_result):
        """Render enhanced validation indicator with scientific context."""
        level = validation_result.level

        # Color-coded status with enhanced styling
        if level == ValidationLevel.VALID:
            st.success("✅ Valid")
            st.caption(f"🔬 {validation_result.scientific_reasoning[:50]}...")
            if validation_result.confidence_score >= 0.9:
                st.caption(f"📊 High confidence ({validation_result.confidence_score:.1%})")
            # Show uncertainty bounds for high confidence values
            if validation_result.confidence_score >= 0.8:
                lower, upper = validation_result.uncertainty_bounds
                uncertainty_range = upper - lower
                st.caption(f"📏 Uncertainty: ±{uncertainty_range/2:.3f}")
        elif level == ValidationLevel.CAUTION:
            st.warning("⚠️ Caution")
            st.caption(f"🔬 {validation_result.scientific_reasoning[:50]}...")
            st.caption(f"📊 Confidence: {validation_result.confidence_score:.1%}")
            # Show wider uncertainty bounds for caution
            lower, upper = validation_result.uncertainty_bounds
            uncertainty_range = upper - lower
            st.caption(f"📏 Uncertainty: ±{uncertainty_range/2:.3f}")
        elif level == ValidationLevel.INVALID:
            st.error("❌ Invalid")
            st.caption(f"🔬 {validation_result.scientific_reasoning[:50]}...")
            st.caption(f"📊 Low confidence ({validation_result.confidence_score:.1%})")
            # Show high uncertainty for invalid values
            lower, upper = validation_result.uncertainty_bounds
            uncertainty_range = upper - lower
            st.caption(f"📏 High uncertainty: ±{uncertainty_range/2:.3f}")
        else:
            st.info("❓ Unknown")
            st.caption("Parameter not in database")

        # Response time indicator (if < 200ms, show green indicator)
        if validation_result.response_time_ms < 200:
            st.caption(f"⚡ {validation_result.response_time_ms:.1f}ms")
        else:
            st.caption(f"🐌 {validation_result.response_time_ms:.1f}ms")

        # Show warnings if any
        if validation_result.warnings:
            for warning in validation_result.warnings[:2]:  # Show max 2 warnings
                st.caption(f"🚨 {warning[:30]}...")

        # Show enhanced recommendations in expander
        if validation_result.recommendations:
            with st.expander("💡 Recommendations", expanded=False):
                for rec in validation_result.recommendations:
                    st.markdown(f"- {rec}")

                # Show suggested ranges
                if validation_result.suggested_ranges:
                    st.markdown("**📊 Suggested Ranges:**")
                    for i, (min_val, max_val) in enumerate(validation_result.suggested_ranges):
                        range_name = ["Recommended", "Research Target", "Typical"][i] if i < 3 else f"Range {i+1}"
                        st.markdown(f"- {range_name}: {min_val:.3f} - {max_val:.3f}")

    def _render_performance_metrics(self):
        """Render real-time validation performance metrics."""
        metrics = REAL_TIME_VALIDATOR.get_performance_metrics()

        st.subheader("⚡ Real-Time Validation Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Avg Response Time",
                f"{metrics['avg_response_time_ms']:.1f}ms",
                delta="Target: <200ms"
            )

        with col2:
            st.metric(
                "Cache Hit Rate",
                f"{metrics['cache_hit_rate']:.1%}",
                delta=f"{metrics['cache_hit_rate'] - 0.8:.1%}" if metrics['cache_hit_rate'] > 0.8 else None
            )

        with col3:
            fast_validations = metrics.get('fast_validations', 0)
            total_validations = max(metrics.get('total_validations', 0), 1)
            fast_rate = fast_validations / total_validations
            st.metric(
                "Fast Validations",
                f"{fast_rate:.1%}",
                delta="<200ms target"
            )

        with col4:
            instant_validations = metrics.get('instant_validations', 0)
            instant_rate = instant_validations / total_validations
            st.metric(
                "Instant Validations",
                f"{instant_rate:.1%}",
                delta="<50ms optimal"
            )

        # Performance status
        if metrics['avg_response_time_ms'] < 200:
            st.success("🚀 Validation performance is excellent!")
        elif metrics['avg_response_time_ms'] < 500:
            st.warning("⚠️ Validation performance is acceptable")
        else:
            st.error("🐌 Validation performance needs improvement")

    def _create_validated_config(self) -> QLearningConfig | None:
        """Create a validated QLearningConfig from current parameter values."""
        if not st.session_state.parameter_values:
            return None

        try:
            # Use parameter bridge to create validated config
            config, validation_results = PARAMETER_BRIDGE.create_literature_validated_config(
                st.session_state.parameter_values
            )

            # Update validation results
            for param_name, result in validation_results.items():
                if param_name in st.session_state.validation_results:
                    st.session_state.validation_results[param_name].update(result)

            return config
        except Exception as e:
            st.error(f"Error creating configuration: {str(e)}")
            return None

    def render_config_integration_section(self):
        """Render section for integrating with existing Q-learning configuration."""
        st.subheader("🔧 Configuration Integration")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Create Q-Learning Config"):
                config = self._create_validated_config()
                if config:
                    st.success("✅ Q-Learning configuration created successfully!")

                    # Display configuration summary
                    st.json({
                        'learning_rate': config.learning_rate,
                        'discount_factor': config.discount_factor,
                        'epsilon': config.epsilon,
                        'anode_area_per_cell': config.anode_area_per_cell,
                        'substrate_target_concentration': config.substrate_target_concentration,
                        'optimal_biofilm_thickness': config.optimal_biofilm_thickness
                    })

        with col2:
            if st.button("Suggest Improvements"):
                config = self._create_validated_config()
                if config:
                    suggestions = PARAMETER_BRIDGE.suggest_parameter_improvements(config)

                    if suggestions:
                        st.warning(f"Found {len(suggestions)} improvement suggestions:")
                        for suggestion in suggestions:
                            st.markdown(f"**{suggestion['parameter']}**")
                            st.markdown(f"- Current: {suggestion['current_value']} ({suggestion['current_status']})")
                            st.markdown(f"- {suggestion['suggestion']}")
                    else:
                        st.success("✅ All parameters are within recommended ranges!")


def render_parameter_input_interface():
    """Main function to render the parameter input interface."""
    component = ParameterInputComponent()
    return component.render_parameter_input_form()


if __name__ == "__main__":
    # For testing the component
    st.title("Parameter Input Component Test")
    result = render_parameter_input_interface()
    st.json(result)
