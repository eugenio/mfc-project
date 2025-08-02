#!/usr/bin/env python3
"""
COBRApy Integration for MFC GSM Models

This module provides integration with COBRApy (COnstraint-Based Reconstruction and Analysis)
for working with genome-scale metabolic models in the MFC optimization framework.

Uses existing COBRApy API for model loading, manipulation, and analysis.

Created: 2025-08-01
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import COBRApy - will be available when gsm environment is activated
try:
    import cobra
    from cobra.flux_analysis import flux_variability_analysis
    from cobra.flux_analysis.parsimonious import pfba
    from cobra.util.solver import linear_reaction_coefficients
    COBRA_AVAILABLE = True
except ImportError:
    COBRA_AVAILABLE = False
    logger.warning("COBRApy not available. Install with: pixi install -e gsm-research")

# Import Mackinac for ModelSEED integration
try:
    import mackinac
    MACKINAC_AVAILABLE = True
except ImportError:
    MACKINAC_AVAILABLE = False
    logger.warning("Mackinac not available for ModelSEED integration")


@dataclass
class FBAResult:
    """Container for Flux Balance Analysis results."""

    objective_value: float
    fluxes: Dict[str, float]
    shadow_prices: Dict[str, float]
    reduced_costs: Dict[str, float]
    status: str

    def get_active_reactions(self, threshold: float = 1e-6) -> List[str]:
        """Get reactions with non-zero flux."""
        return [rxn_id for rxn_id, flux in self.fluxes.items()
                if abs(flux) > threshold]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fluxes to pandas DataFrame."""
        return pd.DataFrame({
            'reaction_id': list(self.fluxes.keys()),
            'flux': list(self.fluxes.values())
        })


@dataclass
class FVAResult:
    """Container for Flux Variability Analysis results."""

    minimum_fluxes: Dict[str, float]
    maximum_fluxes: Dict[str, float]

    def get_variable_reactions(self, threshold: float = 1e-6) -> List[str]:
        """Get reactions with variable flux ranges."""
        variable = []
        for rxn_id in self.minimum_fluxes:
            if abs(self.maximum_fluxes[rxn_id] - self.minimum_fluxes[rxn_id]) > threshold:
                variable.append(rxn_id)
        return variable

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'reaction_id': list(self.minimum_fluxes.keys()),
            'minimum': list(self.minimum_fluxes.values()),
            'maximum': list(self.maximum_fluxes.values()),
            'range': [self.maximum_fluxes[r] - self.minimum_fluxes[r]
                     for r in self.minimum_fluxes]
        })


class COBRAModelWrapper:
    """
    Wrapper for COBRApy models with MFC-specific functionality.
    
    Provides unified interface for loading models from various sources
    and performing constraint-based analysis.
    """

    def __init__(self, model: Optional[Any] = None):
        if not COBRA_AVAILABLE:
            raise ImportError("COBRApy is required. Install with: pixi install -e gsm-research")

        self.model = model
        self.original_bounds: Dict[str, Dict[str, float]] = {}  # Store original reaction bounds
        self._cache_original_bounds()

    def _cache_original_bounds(self):
        """Cache original reaction bounds."""
        if self.model:
            for reaction in self.model.reactions:
                self.original_bounds[reaction.id] = {
                    'lower': reaction.lower_bound,
                    'upper': reaction.upper_bound
                }

    @classmethod
    def from_bigg(cls, model_id: str, cache_dir: str = "data/bigg_models") -> 'COBRAModelWrapper':
        """
        Load model from BiGG database.
        
        Args:
            model_id: BiGG model ID (e.g., 'iJO1366', 'iML1515')
            cache_dir: Directory to cache downloaded models
            
        Returns:
            COBRAModelWrapper instance
        """
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        model_file = cache_path / f"{model_id}.json"

        # Try to load from cache first
        if model_file.exists():
            logger.info(f"Loading {model_id} from cache")
            model = cobra.io.load_json_model(str(model_file))
        else:
            # Download from BiGG
            logger.info(f"Downloading {model_id} from BiGG database")
            try:
                # Use cobra's built-in BiGG loading (if available in newer versions)
                # Otherwise fall back to manual download
                import requests
                url = f"http://bigg.ucsd.edu/static/models/{model_id}.json"
                response = requests.get(url)
                response.raise_for_status()

                with open(model_file, 'w') as f:
                    f.write(response.text)

                model = cobra.io.load_json_model(str(model_file))

            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise

        return cls(model)

    @classmethod
    def from_sbml(cls, sbml_path: str) -> 'COBRAModelWrapper':
        """Load model from SBML file."""
        model = cobra.io.read_sbml_model(sbml_path)
        return cls(model)

    @classmethod
    def from_modelseed(cls, model_id: str) -> 'COBRAModelWrapper':
        """
        Load model from ModelSEED using Mackinac.
        
        Args:
            model_id: ModelSEED model ID
            
        Returns:
            COBRAModelWrapper instance
        """
        if not MACKINAC_AVAILABLE:
            raise ImportError("Mackinac is required for ModelSEED integration")

        # Use Mackinac to create model from ModelSEED
        logger.info(f"Creating model from ModelSEED: {model_id}")
        model = mackinac.create_cobra_model_from_modelseed(model_id)

        return cls(model)

    def optimize(self, objective: Optional[Union[str, Dict[str, float]]] = None) -> FBAResult:
        """
        Perform Flux Balance Analysis (FBA).
        
        Args:
            objective: Reaction ID or dict of {reaction_id: coefficient}
                      If None, uses model's current objective
            
        Returns:
            FBAResult object with optimization results
        """
        if not self.model:
            raise ValueError("No model loaded")

        # Set objective if provided
        if objective:
            if isinstance(objective, str):
                self.model.objective = objective
            else:
                self.model.objective = linear_reaction_coefficients(objective)

        # Run FBA
        solution = self.model.optimize()

        # Extract results
        fluxes = solution.fluxes.to_dict() if hasattr(solution.fluxes, 'to_dict') else dict(solution.fluxes)
        shadow_prices = solution.shadow_prices.to_dict() if hasattr(solution.shadow_prices, 'to_dict') else {}
        reduced_costs = solution.reduced_costs.to_dict() if hasattr(solution.reduced_costs, 'to_dict') else {}

        return FBAResult(
            objective_value=solution.objective_value,
            fluxes=fluxes,
            shadow_prices=shadow_prices,
            reduced_costs=reduced_costs,
            status=solution.status
        )

    def flux_variability_analysis(self, reactions: Optional[List[str]] = None,
                                 fraction_of_optimum: float = 0.9) -> FVAResult:
        """
        Perform Flux Variability Analysis (FVA).
        
        Args:
            reactions: List of reaction IDs to analyze (None = all)
            fraction_of_optimum: Fraction of optimal objective to maintain
            
        Returns:
            FVAResult object
        """
        if not self.model:
            raise ValueError("No model loaded")

        # Run FVA
        fva_result = flux_variability_analysis(
            self.model,
            reaction_list=reactions,
            fraction_of_optimum=fraction_of_optimum
        )

        # Extract results
        minimum_fluxes = fva_result['minimum'].to_dict()
        maximum_fluxes = fva_result['maximum'].to_dict()

        return FVAResult(
            minimum_fluxes=minimum_fluxes,
            maximum_fluxes=maximum_fluxes
        )

    def parsimonious_fba(self) -> FBAResult:
        """
        Perform parsimonious FBA (minimize total flux).
        
        Returns:
            FBAResult object
        """
        if not self.model:
            raise ValueError("No model loaded")

        solution = pfba(self.model)

        # Convert to FBAResult format
        fluxes = solution.fluxes.to_dict() if hasattr(solution.fluxes, 'to_dict') else dict(solution.fluxes)

        return FBAResult(
            objective_value=solution.objective_value,
            fluxes=fluxes,
            shadow_prices={},
            reduced_costs={},
            status=solution.status
        )

    def set_media_conditions(self, media: Dict[str, float]):
        """
        Set media conditions by adjusting exchange reaction bounds.
        
        Args:
            media: Dict of {metabolite_id: max_uptake_rate}
                  Negative values indicate uptake
        """
        if not self.model:
            raise ValueError("No model loaded")

        # Find exchange reactions
        for reaction in self.model.exchanges:
            # Check if any metabolite in this reaction is in media
            for metabolite in reaction.metabolites:
                met_id = metabolite.id.replace('_e', '')  # Remove compartment suffix

                if met_id in media:
                    # Set uptake rate (negative lower bound)
                    reaction.lower_bound = media[met_id]
                    logger.info(f"Set {reaction.id} lower bound to {media[met_id]}")

    def set_oxygen_availability(self, oxygen_uptake: float):
        """
        Set oxygen availability.
        
        Args:
            oxygen_uptake: Maximum oxygen uptake rate (negative for uptake)
        """
        if not self.model:
            raise ValueError("No model loaded")

        # Find oxygen exchange reaction
        o2_reactions = [r for r in self.model.exchanges
                       if 'o2' in r.id.lower() or 'oxygen' in r.id.lower()]

        if o2_reactions:
            o2_reactions[0].lower_bound = oxygen_uptake
            logger.info(f"Set oxygen uptake to {oxygen_uptake}")
        else:
            logger.warning("No oxygen exchange reaction found")

    def knock_out_genes(self, gene_ids: List[str]) -> FBAResult:
        """
        Perform gene knockout analysis.
        
        Args:
            gene_ids: List of gene IDs to knock out
            
        Returns:
            FBAResult after knockouts
        """
        if not self.model:
            raise ValueError("No model loaded")

        # Create a copy to avoid modifying original
        ko_model = self.model.copy()

        # Knock out genes
        for gene_id in gene_ids:
            if gene_id in ko_model.genes:
                cobra.manipulation.knock_out_model_genes(ko_model, [gene_id])
            else:
                logger.warning(f"Gene {gene_id} not found in model")

        # Run FBA on knockout model
        solution = ko_model.optimize()

        fluxes = solution.fluxes.to_dict() if hasattr(solution.fluxes, 'to_dict') else dict(solution.fluxes)

        return FBAResult(
            objective_value=solution.objective_value,
            fluxes=fluxes,
            shadow_prices={},
            reduced_costs={},
            status=solution.status
        )

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded model."""
        if not self.model:
            return {}

        return {
            'num_reactions': len(self.model.reactions),
            'num_metabolites': len(self.model.metabolites),
            'num_genes': len(self.model.genes),
            'num_compartments': len(self.model.compartments),
            'objective': str(self.model.objective),
            'solver': self.model.solver.interface.__name__ if self.model.solver else 'Unknown'
        }

    def find_electron_transfer_reactions(self) -> List[str]:
        """Find reactions involved in electron transfer."""
        if not self.model:
            return []

        # Keywords indicating electron transfer
        et_keywords = ['cytochrome', 'quinone', 'nadh', 'fadh', 'ferredoxin',
                      'flavin', 'heme', 'iron-sulfur', 'electron']

        et_reactions = []
        for reaction in self.model.reactions:
            reaction_str = (reaction.id + reaction.name + reaction.subsystem).lower()
            if any(keyword in reaction_str for keyword in et_keywords):
                et_reactions.append(reaction.id)

        return et_reactions

    def get_exchange_reactions(self) -> Dict[str, Dict[str, float]]:
        """Get all exchange reactions and their bounds."""
        if not self.model:
            return {}

        exchanges = {}
        for reaction in self.model.exchanges:
            exchanges[reaction.id] = {
                'name': reaction.name,
                'lower_bound': reaction.lower_bound,
                'upper_bound': reaction.upper_bound,
                'metabolites': [m.id for m in reaction.metabolites]
            }

        return exchanges


# Example usage
if __name__ == "__main__":
    print("🧬 COBRApy Integration Example")
    print("=" * 50)

    if not COBRA_AVAILABLE:
        print("❌ COBRApy not available. Install with: pixi install -e gsm-research")
    else:
        # Example: Load E. coli model from BiGG
        print("\n📥 Loading E. coli model from BiGG...")
        try:
            wrapper = COBRAModelWrapper.from_bigg('e_coli_core')

            # Get model statistics
            stats = wrapper.get_model_statistics()
            print("\n📊 Model statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Run FBA
            print("\n🧮 Running FBA...")
            fba_result = wrapper.optimize()
            print(f"  Objective value: {fba_result.objective_value:.4f}")
            print(f"  Status: {fba_result.status}")
            print(f"  Active reactions: {len(fba_result.get_active_reactions())}")

            # Run FVA on subset of reactions
            print("\n📈 Running FVA on exchange reactions...")
            if wrapper.model:
                exchanges = list(wrapper.model.exchanges)[:5]  # First 5 exchanges
                exchange_ids = [r.id for r in exchanges]

                fva_result = wrapper.flux_variability_analysis(exchange_ids)
                print(f"  Variable reactions: {len(fva_result.get_variable_reactions())}")

            # Find electron transfer reactions
            print("\n⚡ Finding electron transfer reactions...")
            et_reactions = wrapper.find_electron_transfer_reactions()
            print(f"  Found {len(et_reactions)} electron transfer reactions")
            for rxn in et_reactions[:5]:  # Show first 5
                print(f"    - {rxn}")

            print("\n✅ COBRApy integration example completed!")

        except Exception as e:
            print(f"❌ Error: {e}")
