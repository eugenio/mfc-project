#!/usr/bin/env python3
"""Community Metabolic Modeling for MFC Systems.

This module implements multi-organism community modeling for MFC systems,
enabling simulation of syntrophic interactions and metabolic cross-feeding.

Created: 2025-08-01
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Import our COBRApy integration
from .cobra_integration import COBRA_AVAILABLE, COBRAModelWrapper

logger = logging.getLogger(__name__)

if COBRA_AVAILABLE:
    pass


@dataclass
class OrganismAbundance:
    """Container for organism abundance in community."""

    organism_id: str
    initial_abundance: float  # Relative abundance (0-1)
    current_abundance: float
    growth_rate: float = 0.0

    def update_abundance(self, dt: float) -> None:
        """Update abundance based on growth rate."""
        self.current_abundance *= np.exp(self.growth_rate * dt)


@dataclass
class CommunityInteraction:
    """Defines metabolic interaction between organisms."""

    producer_id: str
    consumer_id: str
    metabolite_id: str
    interaction_type: str  # 'cross-feeding', 'competition', 'syntrophy'
    strength: float = 1.0  # Interaction strength factor


@dataclass
class CommunityState:
    """Current state of the microbial community."""

    time: float
    abundances: dict[str, float]  # organism_id: abundance
    metabolite_concentrations: dict[str, float]  # metabolite_id: concentration
    community_growth_rate: float
    electron_production_rate: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        data = {
            "time": self.time,
            "community_growth": self.community_growth_rate,
            "electron_production": self.electron_production_rate,
        }

        # Add abundances
        for org_id, abundance in self.abundances.items():
            data[f"{org_id}_abundance"] = abundance

        # Add key metabolites
        for met_id, conc in self.metabolite_concentrations.items():
            if any(key in met_id for key in ["lactate", "acetate", "formate"]):
                data[f"{met_id}_mM"] = conc

        return pd.DataFrame([data])


class MFCCommunityModel:
    """Multi-organism community model for MFC systems.

    Implements dynamic community modeling with metabolic interactions,
    electron transfer, and spatial considerations.
    """

    def __init__(self, electrode_area: float = 0.01) -> None:  # m²
        self.organisms: dict[
            str,
            COBRAModelWrapper,
        ] = {}  # organism_id: COBRAModelWrapper
        self.abundances: dict[
            str,
            OrganismAbundance,
        ] = {}  # organism_id: OrganismAbundance
        self.interactions: list[
            CommunityInteraction
        ] = []  # List of CommunityInteraction
        self.shared_metabolites: dict[str, float] = {}  # metabolite_id: concentration
        self.electrode_area = electrode_area

        # Community parameters
        self.max_total_biomass = 100.0  # g/m²
        self.diffusion_rate = 0.1  # h⁻¹

        # Simulation state
        self.time = 0.0
        self.history: list[CommunityState] = []

        # MFC-specific parameters
        self.electrode_mediators = {
            "riboflavin": 0.01,  # mM
            "flavin_mononucleotide": 0.005,
            "phenazine": 0.001,
        }

    def add_organism(
        self,
        organism_id: str,
        model: COBRAModelWrapper,
        initial_abundance: float = 0.1,
    ) -> None:
        """Add organism to community.

        Args:
            organism_id: Unique identifier for organism
            model: COBRAModelWrapper with loaded GSM
            initial_abundance: Initial relative abundance (0-1)

        """
        self.organisms[organism_id] = model
        self.abundances[organism_id] = OrganismAbundance(
            organism_id=organism_id,
            initial_abundance=initial_abundance,
            current_abundance=initial_abundance,
        )

        logger.info(
            f"Added {organism_id} to community (abundance: {initial_abundance})",
        )

    def add_interaction(
        self,
        producer: str,
        consumer: str,
        metabolite: str,
        interaction_type: str = "cross-feeding",
        strength: float = 1.0,
    ) -> None:
        """Add metabolic interaction between organisms."""
        interaction = CommunityInteraction(
            producer_id=producer,
            consumer_id=consumer,
            metabolite_id=metabolite,
            interaction_type=interaction_type,
            strength=strength,
        )
        self.interactions.append(interaction)

        logger.info(
            f"Added {interaction_type} interaction: {producer} -> {consumer} ({metabolite})",
        )

    def setup_mfc_community(self) -> None:
        """Set up typical MFC community with electroactive bacteria.

        Creates a community with:
        - Shewanella oneidensis (electroactive, flavin-mediated)
        - Geobacter sulfurreducens (electroactive, direct transfer)
        - Support organisms for substrate processing
        """
        # This is a simplified setup - in practice, load actual models
        logger.info("Setting up MFC community structure")

        # Define key cross-feeding interactions
        self.add_interaction("fermenter", "shewanella", "lactate", "cross-feeding")
        self.add_interaction("fermenter", "geobacter", "acetate", "cross-feeding")
        self.add_interaction(
            "shewanella",
            "community",
            "riboflavin",
            "electron-shuttle",
        )

        # Set shared metabolite pool
        self.shared_metabolites = {
            "lactate": 20.0,  # mM
            "acetate": 5.0,
            "formate": 2.0,
            "succinate": 1.0,
            "ethanol": 3.0,
            "co2": 10.0,
            "riboflavin": 0.01,
            "oxygen": 0.1,  # Microaerobic
        }

    def simulate_step(self, dt: float = 0.1) -> CommunityState:
        """Simulate one time step of community dynamics.

        Args:
            dt: Time step in hours

        Returns:
            CommunityState object

        """
        # Store current abundances
        current_abundances = {
            org_id: ab.current_abundance for org_id, ab in self.abundances.items()
        }

        # Initialize flux storage
        community_fluxes = {}
        growth_rates = {}

        # Run FBA for each organism based on current conditions
        for org_id, model in self.organisms.items():
            if org_id not in current_abundances or current_abundances[org_id] < 1e-6:
                continue

            # Set media conditions based on shared metabolite pool
            media = self._prepare_media_for_organism(org_id)
            model.set_media_conditions(media)

            # Run FBA
            try:
                fba_result = model.optimize()

                if fba_result.status == "optimal":
                    # Store growth rate
                    growth_rate = fba_result.objective_value
                    growth_rates[org_id] = growth_rate

                    # Scale fluxes by abundance
                    abundance = current_abundances[org_id]
                    scaled_fluxes = {
                        rxn: flux * abundance for rxn, flux in fba_result.fluxes.items()
                    }
                    community_fluxes[org_id] = scaled_fluxes
                else:
                    growth_rates[org_id] = 0.0

            except Exception as e:
                logger.warning(f"FBA failed for {org_id}: {e}")
                growth_rates[org_id] = 0.0

        # Update metabolite pool based on fluxes
        self._update_metabolite_pool(community_fluxes, dt)

        # Apply interactions
        self._apply_interactions(growth_rates)

        # Update abundances
        total_biomass = 0.0
        for org_id, abundance_obj in self.abundances.items():
            if org_id in growth_rates:
                abundance_obj.growth_rate = growth_rates[org_id]
                abundance_obj.update_abundance(dt)
                total_biomass += abundance_obj.current_abundance

        # Normalize if exceeding carrying capacity
        if total_biomass > 1.0:
            for abundance_obj in self.abundances.values():
                abundance_obj.current_abundance /= total_biomass

        # Calculate community metrics
        community_growth = (
            float(np.mean(list(growth_rates.values()))) if growth_rates else 0.0
        )
        electron_production = self._calculate_electron_production(community_fluxes)

        # Update time
        self.time += dt

        # Create state object
        state = CommunityState(
            time=self.time,
            abundances={
                org_id: ab.current_abundance for org_id, ab in self.abundances.items()
            },
            metabolite_concentrations=self.shared_metabolites.copy(),
            community_growth_rate=community_growth,
            electron_production_rate=electron_production,
        )

        # Store in history
        self.history.append(state)

        return state

    def _prepare_media_for_organism(self, organism_id: str) -> dict[str, float]:
        """Prepare media conditions for specific organism."""
        # Base media from shared pool
        media = {}

        # Convert concentrations to uptake rates
        # Negative values indicate uptake
        for met_id, conc in self.shared_metabolites.items():
            if conc > 0.1:  # Only if sufficient concentration
                # Simple Michaelis-Menten kinetics
                uptake_rate = -10.0 * conc / (5.0 + conc)  # Max 10 mmol/gDW/h
                media[met_id] = uptake_rate

        # Organism-specific adjustments
        if "shewanella" in organism_id.lower():
            # Shewanella prefers lactate
            if "lactate" in media:
                media["lactate"] *= 1.5
        elif "geobacter" in organism_id.lower():
            # Geobacter prefers acetate
            if "acetate" in media:
                media["acetate"] *= 1.5

        return media

    def _update_metabolite_pool(
        self,
        community_fluxes: dict[str, dict[str, float]],
        dt: float,
    ) -> None:
        """Update shared metabolite pool based on community fluxes."""
        # Sum exchange fluxes across community
        net_exchange = {}

        for fluxes in community_fluxes.values():
            for rxn_id, flux in fluxes.items():
                # Look for exchange reactions
                if rxn_id.startswith("EX_"):
                    met_id = rxn_id.replace("EX_", "").replace("_e", "")
                    if met_id not in net_exchange:
                        net_exchange[met_id] = 0.0
                    net_exchange[met_id] += flux

        # Update concentrations
        for met_id, net_flux in net_exchange.items():
            if met_id in self.shared_metabolites:
                # Simple mass balance
                self.shared_metabolites[met_id] += net_flux * dt
                # Keep non-negative
                self.shared_metabolites[met_id] = max(
                    0.0,
                    self.shared_metabolites[met_id],
                )

        # Add diffusion/dilution
        for met_id in self.shared_metabolites:
            self.shared_metabolites[met_id] *= 1 - self.diffusion_rate * dt

    def _apply_interactions(self, growth_rates: dict[str, float]) -> None:
        """Apply community interactions to growth rates."""
        for interaction in self.interactions:
            if interaction.producer_id not in growth_rates:
                continue

            producer_growth = growth_rates[interaction.producer_id]

            if interaction.interaction_type == "cross-feeding":
                # Positive effect on consumer
                if interaction.consumer_id in growth_rates:
                    bonus = producer_growth * interaction.strength * 0.1
                    growth_rates[interaction.consumer_id] += bonus

            elif interaction.interaction_type == "competition":
                # Negative effect on consumer
                if interaction.consumer_id in growth_rates:
                    penalty = producer_growth * interaction.strength * 0.1
                    growth_rates[interaction.consumer_id] -= penalty
                    growth_rates[interaction.consumer_id] = max(
                        0,
                        growth_rates[interaction.consumer_id],
                    )

    def _calculate_electron_production(
        self,
        community_fluxes: dict[str, dict[str, float]],
    ) -> float:
        """Calculate total electron production rate."""
        electron_rate = 0.0

        # Look for electron transfer reactions
        electron_reactions = ["cytochrome", "quinone", "flavin", "riboflavin_export"]

        for fluxes in community_fluxes.values():
            for rxn_id, flux in fluxes.items():
                if any(keyword in rxn_id.lower() for keyword in electron_reactions):
                    electron_rate += abs(flux)

        # Add mediator-based transfer
        for mediator in self.electrode_mediators:
            if mediator in self.shared_metabolites:
                # Mediator cycling rate
                cycling_rate = self.shared_metabolites[mediator] * 10.0  # h⁻¹
                electron_rate += cycling_rate

        return electron_rate

    def get_dominant_organism(self) -> str:
        """Get the currently dominant organism."""
        if not self.abundances:
            return "None"

        max_abundance = 0.0
        dominant = "None"

        for org_id, abundance_obj in self.abundances.items():
            if abundance_obj.current_abundance > max_abundance:
                max_abundance = abundance_obj.current_abundance
                dominant = org_id

        return dominant

    def get_diversity_index(self) -> float:
        """Calculate Shannon diversity index."""
        abundances = [ab.current_abundance for ab in self.abundances.values()]
        total = sum(abundances)

        if total == 0:
            return 0.0

        # Normalize
        proportions = [a / total for a in abundances]

        # Shannon index
        diversity = 0.0
        for p in proportions:
            if p > 0:
                diversity -= p * np.log(p)

        return diversity

    def simulate_succession(self, duration: float, dt: float = 0.1) -> pd.DataFrame:
        """Simulate community succession over time.

        Args:
            duration: Total simulation time (hours)
            dt: Time step

        Returns:
            DataFrame with time series data

        """
        steps = int(duration / dt)

        for _ in range(steps):
            self.simulate_step(dt)

        # Convert history to DataFrame
        if self.history:
            return pd.concat(
                [state.to_dataframe() for state in self.history],
                ignore_index=True,
            )
        return pd.DataFrame()


# Integration with electrode physics
class CommunityElectrodeIntegration:
    """Integrates community model with electrode physics."""

    def __init__(self, community_model: MFCCommunityModel) -> None:
        self.community = community_model
        self.biofilm_density = 50.0  # kg/m³
        self.biofilm_thickness = 100e-6  # 100 μm

    def calculate_current_density(self) -> float:
        """Calculate current density from community metabolism."""
        # Get electron production rate
        if self.community.history:
            electron_rate = self.community.history[-1].electron_production_rate
        else:
            electron_rate = 0.0

        # Convert to current density
        # Assume 1 mol electrons = 96485 C (Faraday constant)
        faraday = 96485  # C/mol

        # electron_rate is in mmol/gDW/h
        # Convert to A/m²
        biomass_density = self.biofilm_density * self.biofilm_thickness  # kg/m²

        return (electron_rate * biomass_density * faraday) / (3600 * 1000)

    def get_optimization_objectives(self) -> dict[str, float]:
        """Get optimization objectives from community state."""
        objectives = {
            "maximize_current_density": self.calculate_current_density(),
            "maximize_community_stability": self.community.get_diversity_index(),
            "maximize_electron_production": 0.0,
            "minimize_substrate_waste": 0.0,
        }

        if self.community.history:
            state = self.community.history[-1]
            objectives["maximize_electron_production"] = state.electron_production_rate

            # Calculate substrate utilization
            if "lactate" in state.metabolite_concentrations:
                substrate_remaining = state.metabolite_concentrations["lactate"]
                objectives["minimize_substrate_waste"] = 1.0 / (
                    1.0 + substrate_remaining
                )

        return objectives


# Example usage
if __name__ == "__main__":
    # Create community model
    community = MFCCommunityModel(electrode_area=0.01)  # 100 cm²

    # Set up MFC community
    community.setup_mfc_community()

    # Add some test organisms (in practice, load real models)

    # Simulate community dynamics
    results = community.simulate_succession(duration=24.0, dt=0.5)

    if not results.empty:
        # Test electrode integration
        integrator = CommunityElectrodeIntegration(community)
        current = integrator.calculate_current_density()

        # Get optimization objectives
        objectives = integrator.get_optimization_objectives()
        for _obj, _value in objectives.items():
            pass
