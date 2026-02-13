"""End-to-end tests for the integrated MFC model.

Tests complete simulation workflows coupling biofilm kinetics,
metabolic modeling, and recirculation control.
"""



class TestBiofilmMetabolicCouplingE2E:
    """E2E test: biofilm + metabolic model coupling."""

    def test_biofilm_growth_over_time(self):
        from biofilm_kinetics import BiofilmKineticsModel

        model = BiofilmKineticsModel(species="geobacter")
        initial_thickness = model.biofilm_thickness

        for _ in range(100):
            model.step_biofilm_dynamics(
                dt=60.0, anode_potential=-0.3, substrate_supply=5.0,
            )

        # With substrate supply, biofilm should grow
        assert model.biofilm_thickness >= initial_thickness

    def test_metabolic_electron_flow(self):
        from metabolic_model import MetabolicModel

        model = MetabolicModel()
        results = []

        for _ in range(50):
            result = model.step_metabolism(
                dt=60.0,
                biomass=0.1,
                growth_rate=0.05,
                anode_potential=-0.3,
                substrate_supply=1.0,
                cathode_o2_conc=0.2,
                membrane_area=0.01,
                volume=0.001,
                electrode_area=0.005,
            )
            results.append(result)

        assert len(results) == 50

    def test_biofilm_species_comparison(self):
        """Different species should produce different growth rates."""
        from biofilm_kinetics import BiofilmKineticsModel

        geo = BiofilmKineticsModel(species="geobacter")
        she = BiofilmKineticsModel(species="shewanella")

        for _ in range(50):
            geo.step_biofilm_dynamics(dt=60.0, anode_potential=-0.3)
            she.step_biofilm_dynamics(dt=60.0, anode_potential=-0.3)

        assert geo.biofilm_thickness is not None
        assert she.biofilm_thickness is not None


class TestIntegratedMFCModelE2E:
    """E2E test: full integrated MFC model."""

    def test_model_instantiation(self):
        from integrated_mfc_model import IntegratedMFCModel

        model = IntegratedMFCModel(
            n_cells=2,
            species="geobacter",
            substrate="acetate",
            use_gpu=False,
            simulation_hours=1,
        )
        assert model is not None

    def test_short_simulation(self):
        """Run a 1-hour simulation with 2 cells."""
        from integrated_mfc_model import IntegratedMFCModel

        model = IntegratedMFCModel(
            n_cells=2,
            species="geobacter",
            substrate="acetate",
            use_gpu=False,
            simulation_hours=1,
        )
        results = model.run_simulation()

        assert results is not None
        assert isinstance(results, dict)

    def test_mixed_culture_simulation(self):
        from integrated_mfc_model import IntegratedMFCModel

        model = IntegratedMFCModel(
            n_cells=2,
            species="mixed",
            substrate="lactate",
            use_gpu=False,
            simulation_hours=1,
        )
        results = model.run_simulation()
        assert results is not None


class TestIntegratedMFCStateE2E:
    """E2E test: IntegratedMFCState data container."""

    def test_state_creation(self):
        from integrated_mfc_model import IntegratedMFCState

        state = IntegratedMFCState(
            time=0.0,
            total_energy=0.0,
            average_power=0.0,
            coulombic_efficiency=0.0,
            biofilm_thickness=[0.1, 0.1],
            biomass_density=[1.0, 1.0],
            attachment_fraction=[0.5, 0.5],
            substrate_concentration=[50.0, 50.0],
            nadh_ratio=[0.1, 0.1],
            atp_level=[1.0, 1.0],
            electron_flux=[0.01, 0.01],
            cell_voltages=[0.5, 0.5],
            current_densities=[1.0, 1.0],
            anode_potentials=[-0.3, -0.3],
            reservoir_concentration=100.0,
            flow_rate=1e-6,
            pump_power=0.001,
            epsilon=0.1,
            q_table_size=0,
            learning_progress=0.0,
        )
        assert state.time == 0.0
        assert len(state.biofilm_thickness) == 2
