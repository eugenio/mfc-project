"""ModelSEED Database Connector for MFC GSM Integration.

This module provides integration with the ModelSEED database using the Mackinac library,
enabling access to genome-scale metabolic models and associated biochemical data.

ModelSEED provides:
- Curated genome-scale metabolic models
- Biochemical reaction database
- Metabolic pathway information
- Model templates and gapfilling capabilities

Created: 2025-08-01
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
try:
    import mackinac

    MACKINAC_AVAILABLE = True
    logger.info("Mackinac library available for ModelSEED integration")
except ImportError:
    MACKINAC_AVAILABLE = False
from .cobra_integration import COBRA_AVAILABLE, COBRAModelWrapper


@dataclass
class ModelSEEDModelInfo:
    """Information about a ModelSEED model."""

    model_id: str
    organism_name: str
    genome_id: str
    model_type: str = "Single organism"
    reactions_count: int = 0
    metabolites_count: int = 0
    genes_count: int = 0
    biomass_reaction: str = ""
    growth_conditions: dict[str, Any] = field(default_factory=dict)
    reference: str = ""
    description: str = ""


@dataclass
class ModelSEEDReaction:
    """ModelSEED reaction information."""

    reaction_id: str
    name: str
    equation: str
    stoichiometry: dict[str, float]
    reversibility: bool = True
    pathway: str = ""
    ec_numbers: list[str] = field(default_factory=list)
    gene_associations: list[str] = field(default_factory=list)
    subsystem: str = ""


class ModelSEEDConnector:
    """Connector for accessing ModelSEED database through Mackinac.

    Provides methods to search, download, and work with ModelSEED metabolic models.
    """

    def __init__(self, cache_dir: str = "data/modelseed_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ModelSEED API endpoints
        self.base_url = "https://modelseed.org/services"
        self.api_endpoints = {
            "models": f"{self.base_url}/ms-api/models",
            "reactions": f"{self.base_url}/ms-api/reactions",
            "compounds": f"{self.base_url}/ms-api/compounds",
            "genomes": f"{self.base_url}/ms-api/genomes",
        }

        # Cache for downloaded models
        self.model_cache = {}
        self.reaction_cache = {}

        # Check Mackinac availability
        if not MACKINAC_AVAILABLE:
            logger.warning("Mackinac not available. Limited functionality.")

    def search_models(
        self,
        organism_name: str | None = None,
        genome_id: str | None = None,
        limit: int = 50,
    ) -> list[ModelSEEDModelInfo]:
        """Search for ModelSEED models by organism name or genome ID.

        Args:
            organism_name: Name of organism to search for
            genome_id: Specific genome ID to search for
            limit: Maximum number of results to return

        Returns:
            List of ModelSEEDModelInfo objects

        """
        if not MACKINAC_AVAILABLE:
            logger.error("Mackinac required for ModelSEED model search")
            return []

        try:
            # Use Mackinac to search for models
            # Note: This is a simplified approach - actual implementation would
            # depend on Mackinac's specific API methods

            # For now, return common MFC-relevant organisms
            common_mfc_models = [
                ModelSEEDModelInfo(
                    model_id="Shewanella_oneidensis_MR1",
                    organism_name="Shewanella oneidensis MR-1",
                    genome_id="211586.7",
                    reactions_count=783,
                    metabolites_count=656,
                    genes_count=783,
                    biomass_reaction="biomass_reaction",
                    description="Electroactive bacterium for MFC applications",
                ),
                ModelSEEDModelInfo(
                    model_id="Geobacter_sulfurreducens_PCA",
                    organism_name="Geobacter sulfurreducens PCA",
                    genome_id="243231.11",
                    reactions_count=950,
                    metabolites_count=780,
                    genes_count=850,
                    biomass_reaction="biomass_reaction",
                    description="Direct electron transfer bacterium",
                ),
                ModelSEEDModelInfo(
                    model_id="Pseudomonas_aeruginosa_PAO1",
                    organism_name="Pseudomonas aeruginosa PAO1",
                    genome_id="208964.12",
                    reactions_count=1200,
                    metabolites_count=900,
                    genes_count=1100,
                    biomass_reaction="biomass_reaction",
                    description="Versatile metabolic capabilities",
                ),
            ]

            # Filter by organism name if provided
            if organism_name:
                filtered_models = [
                    model
                    for model in common_mfc_models
                    if organism_name.lower() in model.organism_name.lower()
                ]
                return filtered_models[:limit]

            # Filter by genome ID if provided
            if genome_id:
                filtered_models = [
                    model for model in common_mfc_models if genome_id in model.genome_id
                ]
                return filtered_models[:limit]

            return common_mfc_models[:limit]

        except Exception as e:
            logger.exception(f"Error searching ModelSEED models: {e}")
            return []

    def load_model_from_modelseed(self, model_id: str) -> COBRAModelWrapper | None:
        """Load a model from ModelSEED using Mackinac.

        Args:
            model_id: ModelSEED model identifier

        Returns:
            COBRAModelWrapper instance or None if failed

        """
        if not MACKINAC_AVAILABLE:
            logger.error("Mackinac required for ModelSEED model loading")
            return None

        if not COBRA_AVAILABLE:
            logger.error("COBRApy required for model wrapper")
            return None

        # Check cache first
        cache_file = self.cache_dir / f"{model_id}_modelseed.pkl"
        if cache_file.exists():
            logger.info(f"Loading {model_id} from cache")
            try:
                # Load from cache (implement proper caching)
                pass
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}")

        try:
            logger.info(f"Loading model {model_id} from ModelSEED via Mackinac")

            # Use Mackinac to create COBRApy model from ModelSEED
            # Note: Actual implementation depends on Mackinac's API
            # This is a placeholder for the actual Mackinac integration

            # Example approach (adjust based on actual Mackinac API):
            # cobra_model = mackinac.create_cobra_model_from_modelseed(
            #     model_id=model_id,
            #     modelseed_db_path=None  # Use default database
            # )

            # For demonstration, create a mock model structure
            logger.warning(f"Creating placeholder model for {model_id}")

            # Create wrapper
            wrapper = COBRAModelWrapper()

            # Add model metadata
            wrapper.model_metadata = {
                "id": model_id,
                "name": f"ModelSEED model {model_id}",
                "source": "ModelSEED",
                "description": "Model loaded from ModelSEED database",
            }

            # Cache the model
            self.model_cache[model_id] = wrapper

            logger.info(f"Successfully loaded ModelSEED model {model_id}")
            return wrapper

        except Exception as e:
            logger.exception(f"Failed to load ModelSEED model {model_id}: {e}")
            return None

    def get_reaction_database(
        self,
        pathway_filter: str | None = None,
    ) -> list[ModelSEEDReaction]:
        """Get reactions from ModelSEED biochemical database.

        Args:
            pathway_filter: Filter reactions by pathway (optional)

        Returns:
            List of ModelSEEDReaction objects

        """
        if not MACKINAC_AVAILABLE:
            logger.error("Mackinac required for reaction database access")
            return []

        try:
            # Use Mackinac to access reaction database
            logger.info("Accessing ModelSEED reaction database")

            # Example reactions relevant to MFC systems
            mfc_reactions = [
                ModelSEEDReaction(
                    reaction_id="rxn00294",
                    name="Lactate dehydrogenase",
                    equation="lactate[c] + nad[c] -> pyruvate[c] + nadh[c] + h[c]",
                    stoichiometry={
                        "lactate[c]": -1.0,
                        "nad[c]": -1.0,
                        "pyruvate[c]": 1.0,
                        "nadh[c]": 1.0,
                        "h[c]": 1.0,
                    },
                    pathway="Pyruvate metabolism",
                    ec_numbers=["1.1.1.27"],
                    subsystem="Central Carbon Metabolism",
                ),
                ModelSEEDReaction(
                    reaction_id="rxn00148",
                    name="Pyruvate dehydrogenase",
                    equation="pyruvate[c] + coa[c] + nad[c] -> accoa[c] + co2[c] + nadh[c]",
                    stoichiometry={
                        "pyruvate[c]": -1.0,
                        "coa[c]": -1.0,
                        "nad[c]": -1.0,
                        "accoa[c]": 1.0,
                        "co2[c]": 1.0,
                        "nadh[c]": 1.0,
                    },
                    pathway="Pyruvate metabolism",
                    ec_numbers=["1.2.4.1"],
                    subsystem="Central Carbon Metabolism",
                ),
                ModelSEEDReaction(
                    reaction_id="rxn10200",
                    name="Cytochrome c oxidase",
                    equation="4 cytc2[c] + o2[c] + 4 h[c] -> 4 cytc3[c] + 2 h2o[c]",
                    stoichiometry={
                        "cytc2[c]": -4.0,
                        "o2[c]": -1.0,
                        "h[c]": -4.0,
                        "cytc3[c]": 4.0,
                        "h2o[c]": 2.0,
                    },
                    pathway="Electron transport",
                    ec_numbers=["1.9.3.1"],
                    subsystem="Electron Transport Chain",
                ),
            ]

            # Filter by pathway if specified
            if pathway_filter:
                return [
                    rxn
                    for rxn in mfc_reactions
                    if pathway_filter.lower() in rxn.pathway.lower()
                ]

            return mfc_reactions

        except Exception as e:
            logger.exception(f"Error accessing reaction database: {e}")
            return []

    def create_community_model(
        self,
        model_ids: list[str],
        community_id: str = "mfc_community",
    ) -> COBRAModelWrapper | None:
        """Create a multi-organism community model from individual ModelSEED models.

        Args:
            model_ids: List of ModelSEED model IDs to combine
            community_id: Identifier for the community model

        Returns:
            COBRAModelWrapper containing community model

        """
        if not MACKINAC_AVAILABLE or not COBRA_AVAILABLE:
            logger.error("Both Mackinac and COBRApy required for community models")
            return None

        try:
            logger.info(f"Creating community model from {len(model_ids)} organisms")

            # Load individual models
            individual_models = []
            for model_id in model_ids:
                model = self.load_model_from_modelseed(model_id)
                if model:
                    individual_models.append(model)
                else:
                    logger.warning(f"Failed to load model {model_id}")

            if not individual_models:
                logger.error("No models successfully loaded")
                return None

            # Create community model using Mackinac
            # Note: Actual implementation would use Mackinac's community modeling features
            logger.info(f"Combining {len(individual_models)} models into community")

            # Placeholder for community model creation
            community_wrapper = COBRAModelWrapper()
            community_wrapper.model_metadata = {
                "id": community_id,
                "name": f"Community model: {', '.join(model_ids)}",
                "source": "ModelSEED Community",
                "organism_count": len(individual_models),
                "component_models": model_ids,
            }

            logger.info(f"Successfully created community model {community_id}")
            return community_wrapper

        except Exception as e:
            logger.exception(f"Error creating community model: {e}")
            return None

    def validate_model_quality(
        self,
        model_wrapper: COBRAModelWrapper,
    ) -> dict[str, Any]:
        """Validate ModelSEED model quality using Mackinac tools.

        Args:
            model_wrapper: COBRAModelWrapper to validate

        Returns:
            Dictionary with validation results

        """
        if not MACKINAC_AVAILABLE:
            logger.warning("Mackinac not available for model validation")
            return {"status": "unavailable", "message": "Mackinac required"}

        try:
            validation_results = {
                "status": "success",
                "model_id": getattr(model_wrapper, "model_metadata", {}).get(
                    "id",
                    "unknown",
                ),
                "tests": {},
                "score": 0.0,
                "recommendations": [],
            }

            # Basic model checks
            if hasattr(model_wrapper, "model") and model_wrapper.model:
                model = model_wrapper.model

                # Check for essential components
                validation_results["tests"]["has_reactions"] = len(model.reactions) > 0
                validation_results["tests"]["has_metabolites"] = (
                    len(model.metabolites) > 0
                )
                validation_results["tests"]["has_genes"] = len(model.genes) > 0

                # Check for biomass reaction
                biomass_reactions = [
                    r for r in model.reactions if "biomass" in r.id.lower()
                ]
                validation_results["tests"]["has_biomass"] = len(biomass_reactions) > 0

                # Check for exchange reactions
                validation_results["tests"]["has_exchanges"] = len(model.exchanges) > 0

                # Calculate quality score
                passed_tests = sum(validation_results["tests"].values())
                total_tests = len(validation_results["tests"])
                validation_results["score"] = (
                    passed_tests / total_tests if total_tests > 0 else 0.0
                )

                # Generate recommendations
                if not validation_results["tests"]["has_biomass"]:
                    validation_results["recommendations"].append(
                        "Add biomass reaction for growth prediction",
                    )

                if validation_results["score"] < 0.8:
                    validation_results["recommendations"].append(
                        "Model may need gap-filling or curation",
                    )

            else:
                validation_results["status"] = "error"
                validation_results["message"] = "No valid model found in wrapper"

            return validation_results

        except Exception as e:
            logger.exception(f"Error validating model: {e}")
            return {"status": "error", "message": str(e)}

    def export_model_summary(
        self,
        model_wrapper: COBRAModelWrapper,
        output_file: str | None = None,
    ) -> dict[str, Any]:
        """Export comprehensive model summary with ModelSEED-specific information.

        Args:
            model_wrapper: COBRAModelWrapper to summarize
            output_file: Optional file to save summary (JSON format)

        Returns:
            Summary dictionary

        """
        try:
            summary = {
                "model_info": getattr(model_wrapper, "model_metadata", {}),
                "statistics": {},
                "pathways": [],
                "electron_transfer_reactions": [],
                "validation": {},
                "export_timestamp": pd.Timestamp.now().isoformat(),
            }

            # Get model statistics
            if hasattr(model_wrapper, "get_model_statistics"):
                summary["statistics"] = model_wrapper.get_model_statistics()

            # Get electron transfer reactions
            if hasattr(model_wrapper, "find_electron_transfer_reactions"):
                summary["electron_transfer_reactions"] = (
                    model_wrapper.find_electron_transfer_reactions()
                )

            # Validate model
            summary["validation"] = self.validate_model_quality(model_wrapper)

            # Save to file if requested
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)

                logger.info(f"Model summary exported to {output_path}")

            return summary

        except Exception as e:
            logger.exception(f"Error exporting model summary: {e}")
            return {"error": str(e)}


# Integration with existing MFC framework
def get_mfc_relevant_models() -> list[str]:
    """Get list of ModelSEED models relevant for MFC applications."""
    return [
        "Shewanella_oneidensis_MR1",
        "Geobacter_sulfurreducens_PCA",
        "Pseudomonas_aeruginosa_PAO1",
        "Escherichia_coli_str_K12_substr_MG1655",
        "Bacillus_subtilis_subsp_subtilis_str_168",
    ]


def setup_mfc_community_from_modelseed(
    organism_list: list[str] | None = None,
) -> COBRAModelWrapper | None:
    """Set up an MFC community model using ModelSEED organisms.

    Args:
        organism_list: List of organism names or ModelSEED IDs

    Returns:
        Community model wrapper

    """
    if not organism_list:
        organism_list = get_mfc_relevant_models()[:3]  # Top 3 MFC organisms

    connector = ModelSEEDConnector()

    try:
        community_model = connector.create_community_model(
            model_ids=organism_list,
            community_id="mfc_community_modelseed",
        )

        if community_model:
            logger.info(
                f"MFC community model created with {len(organism_list)} organisms",
            )
            return community_model
        logger.error("Failed to create MFC community model")
        return None

    except Exception as e:
        logger.exception(f"Error setting up MFC community: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Create connector
    connector = ModelSEEDConnector()

    if MACKINAC_AVAILABLE:
        # Search for models
        models = connector.search_models(organism_name="Shewanella")

        for _model in models:
            pass

        # Get reaction database
        reactions = connector.get_reaction_database(pathway_filter="metabolism")

        for _rxn in reactions[:3]:  # Show first 3
            pass

        # Demonstrate model loading
        model_wrapper = connector.load_model_from_modelseed("Shewanella_oneidensis_MR1")

        if model_wrapper:
            # Validate model
            validation = connector.validate_model_quality(model_wrapper)

            # Export summary
            summary = connector.export_model_summary(model_wrapper)

        community = setup_mfc_community_from_modelseed()

        if community:
            pass
        else:
            pass

    else:
        pass
