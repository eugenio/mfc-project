"""Coverage boost tests for literature_database.py."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.literature_database import (
    LITERATURE_DB,
    LiteratureDatabase,
    LiteratureReference,
    ParameterCategory,
    ParameterInfo,
)


class TestLiteratureReference:
    def test_format_citation_apa(self):
        ref = LiteratureReference(
            authors="Smith, J.",
            title="Test Paper",
            journal="Test Journal",
            year=2020,
            volume="10",
            pages="1-10",
            doi="10.1234/test",
        )
        citation = ref.format_citation("apa")
        assert "Smith, J." in citation
        assert "(2020)" in citation

    def test_format_citation_bibtex(self):
        ref = LiteratureReference(
            authors="Smith, J.",
            title="Test Paper",
            journal="Test Journal",
            year=2020,
            volume="10",
            pages="1-10",
            doi="10.1234/test",
        )
        citation = ref.format_citation("bibtex")
        assert "@article" in citation
        assert "doi = {10.1234/test}" in citation

    def test_format_citation_bibtex_no_doi(self):
        ref = LiteratureReference(
            authors="Smith, J.",
            title="Test Paper",
            journal="Test Journal",
            year=2020,
            volume="10",
            pages="1-10",
        )
        citation = ref.format_citation("bibtex")
        assert "doi" not in citation

    def test_format_citation_unknown_style(self):
        ref = LiteratureReference(
            authors="Smith, J.",
            title="Test Paper",
            journal="Test Journal",
            year=2020,
            volume="10",
            pages="1-10",
        )
        citation = ref.format_citation("chicago")
        assert "Smith, J." in citation


class TestParameterInfo:
    def test_is_within_recommended_range(self):
        param = ParameterInfo(
            name="Test",
            symbol="T",
            description="Test param",
            unit="V",
            typical_value=0.5,
            min_value=0.0,
            max_value=1.0,
            recommended_range=(0.3, 0.7),
            category=ParameterCategory.ELECTROCHEMICAL,
            references=[],
        )
        assert param.is_within_recommended_range(0.5) is True
        assert param.is_within_recommended_range(0.1) is False

    def test_is_valid_value(self):
        param = ParameterInfo(
            name="Test",
            symbol="T",
            description="Test param",
            unit="V",
            typical_value=0.5,
            min_value=0.0,
            max_value=1.0,
            recommended_range=(0.3, 0.7),
            category=ParameterCategory.ELECTROCHEMICAL,
            references=[],
        )
        assert param.is_valid_value(0.5) is True
        assert param.is_valid_value(1.5) is False

    def test_get_validation_status_valid(self):
        param = ParameterInfo(
            name="Test",
            symbol="T",
            description="d",
            unit="V",
            typical_value=0.5,
            min_value=0.0,
            max_value=1.0,
            recommended_range=(0.3, 0.7),
            category=ParameterCategory.ELECTROCHEMICAL,
            references=[],
        )
        assert param.get_validation_status(0.5) == "valid"

    def test_get_validation_status_caution(self):
        param = ParameterInfo(
            name="Test",
            symbol="T",
            description="d",
            unit="V",
            typical_value=0.5,
            min_value=0.0,
            max_value=1.0,
            recommended_range=(0.3, 0.7),
            category=ParameterCategory.ELECTROCHEMICAL,
            references=[],
        )
        assert param.get_validation_status(0.1) == "caution"

    def test_get_validation_status_invalid(self):
        param = ParameterInfo(
            name="Test",
            symbol="T",
            description="d",
            unit="V",
            typical_value=0.5,
            min_value=0.0,
            max_value=1.0,
            recommended_range=(0.3, 0.7),
            category=ParameterCategory.ELECTROCHEMICAL,
            references=[],
        )
        assert param.get_validation_status(1.5) == "invalid"


class TestLiteratureDatabase:
    def test_validate_parameter_value_unknown(self):
        db = LiteratureDatabase()
        result = db.validate_parameter_value("nonexistent_param", 1.0)
        assert result["status"] == "unknown"

    def test_validate_parameter_value_invalid(self):
        db = LiteratureDatabase()
        result = db.validate_parameter_value("anode_potential", 5.0)
        assert result["status"] == "invalid"
        assert len(result["recommendations"]) > 0

    def test_validate_parameter_value_caution(self):
        db = LiteratureDatabase()
        result = db.validate_parameter_value("anode_potential", -0.55)
        assert result["status"] == "caution"
        assert len(result["recommendations"]) > 0

    def test_validate_parameter_value_valid(self):
        db = LiteratureDatabase()
        result = db.validate_parameter_value("anode_potential", -0.3)
        assert result["status"] == "valid"

    def test_search_parameters(self):
        db = LiteratureDatabase()
        results = db.search_parameters("growth")
        assert len(results) > 0

    def test_search_parameters_by_symbol(self):
        db = LiteratureDatabase()
        results = db.search_parameters("E_an")
        assert len(results) > 0

    def test_get_citation_list(self):
        db = LiteratureDatabase()
        citations = db.get_citation_list("apa")
        assert len(citations) > 0

    def test_get_citation_list_bibtex(self):
        db = LiteratureDatabase()
        citations = db.get_citation_list("bibtex")
        assert len(citations) > 0

    def test_get_all_categories(self):
        db = LiteratureDatabase()
        categories = db.get_all_categories()
        assert ParameterCategory.ELECTROCHEMICAL in categories
        assert ParameterCategory.BIOLOGICAL in categories

    def test_get_parameters_by_category(self):
        db = LiteratureDatabase()
        params = db.get_parameters_by_category(ParameterCategory.QLEARNING)
        assert len(params) >= 3

    def test_global_instance(self):
        assert LITERATURE_DB is not None
        assert LITERATURE_DB.get_parameter("anode_potential") is not None
