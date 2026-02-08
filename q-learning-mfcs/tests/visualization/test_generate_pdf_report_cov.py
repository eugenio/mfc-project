"""Tests for generate_pdf_report.py."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def mock_pdf():
    """Create a mock PdfPages object."""
    pdf = MagicMock()
    pdf.savefig = MagicMock()
    pdf.infodict.return_value = {}
    return pdf


@pytest.fixture
def _mock_report_path(tmp_path):
    with patch(
        "generate_pdf_report.get_report_path",
        side_effect=lambda f: str(tmp_path / f),
    ):
        yield tmp_path


class TestCreateCoverPage:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_cover_page
        create_cover_page(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestCreateExecutiveSummary:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_executive_summary
        create_executive_summary(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestCreateTechnicalOverview:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_technical_overview
        create_technical_overview(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestCreateSimulationResults:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_simulation_results
        create_simulation_results(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestCreateEnergyAnalysis:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_energy_analysis
        create_energy_analysis(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestCreateConclusionsFutureWork:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_conclusions_future_work
        create_conclusions_future_work(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestCreateAppendix:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_appendix
        create_appendix(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestEnhancedCoverPage:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_enhanced_cover_page
        create_enhanced_cover_page(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestSystemArchitecturePage:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_system_architecture_page
        create_system_architecture_page(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestEnhancedSimulationResults:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_enhanced_simulation_results
        create_enhanced_simulation_results(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestEnhancedEnergyAnalysis:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_enhanced_energy_analysis
        create_enhanced_energy_analysis(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestEnhancedConclusions:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_enhanced_conclusions
        create_enhanced_conclusions(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestEnhancedAppendix:
    def test_runs(self, mock_pdf):
        from generate_pdf_report import create_enhanced_appendix
        create_enhanced_appendix(mock_pdf)
        mock_pdf.savefig.assert_called_once()


class TestGenerateComprehensivePdfReport:
    def test_basic_report(self, _mock_report_path):
        from generate_pdf_report import generate_comprehensive_pdf_report
        path = generate_comprehensive_pdf_report(enhanced=False)
        assert os.path.exists(path)

    def test_enhanced_report(self, _mock_report_path):
        from generate_pdf_report import generate_comprehensive_pdf_report
        path = generate_comprehensive_pdf_report(enhanced=True)
        assert os.path.exists(path)


class TestMain:
    def test_main_basic(self, _mock_report_path):
        from generate_pdf_report import main
        with patch("sys.argv", ["prog"]):
            main()

    def test_main_enhanced(self, _mock_report_path):
        from generate_pdf_report import main
        with patch("sys.argv", ["prog", "--enhanced"]):
            main()

    def test_main_exception(self, _mock_report_path):
        from generate_pdf_report import main
        with patch("sys.argv", ["prog"]), \
             patch(
                 "generate_pdf_report.generate_comprehensive_pdf_report",
                 side_effect=RuntimeError("test"),
             ):
            main()
