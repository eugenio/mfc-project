"""Tests for analyze_pdf_comments.py - coverage target 98%+."""
import sys
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture
def mock_fitz():
    """Create a mock fitz (PyMuPDF) module."""
    mock_mod = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"title": "Test PDF", "author": "Test Author"}
    mock_doc.__len__ = lambda self: 2

    mock_page = MagicMock()
    mock_page.get_text.return_value = "Sample text content for testing."

    mock_annot = MagicMock()
    mock_annot.info = {
        "type": "Text",
        "content": "This is a comment",
        "title": "Author Name",
        "subject": "Review",
        "creationDate": "D:20250101",
        "modDate": "D:20250102",
    }
    mock_annot.rect = [100.0, 200.0, 300.0, 400.0]

    mock_page.annots.return_value = [mock_annot]
    mock_doc.__getitem__ = lambda self, idx: mock_page

    mock_mod.open.return_value = mock_doc
    return mock_mod, mock_doc, mock_page, mock_annot


class TestAnalyzePdfAnnotations:
    def test_basic_analysis(self, mock_fitz):
        mock_mod, mock_doc, mock_page, mock_annot = mock_fitz
        with patch.dict("sys.modules", {"fitz": mock_mod}):
            from analyze_pdf_comments import analyze_pdf_annotations
            result = analyze_pdf_annotations("/tmp/test.pdf")
            assert result["page_count"] == 2
            assert result["summary"]["total_annotations"] == 2
            assert result["summary"]["has_comments"] is True

    def test_no_annotations(self, mock_fitz):
        mock_mod, mock_doc, mock_page, _ = mock_fitz
        mock_page.annots.return_value = []
        with patch.dict("sys.modules", {"fitz": mock_mod}):
            from analyze_pdf_comments import analyze_pdf_annotations
            result = analyze_pdf_annotations("/tmp/test.pdf")
            assert result["summary"]["total_annotations"] == 0

    def test_non_comment_annotation(self, mock_fitz):
        mock_mod, mock_doc, mock_page, mock_annot = mock_fitz
        mock_annot.info = {
            "type": "Highlight",
            "content": "",
            "title": "",
            "subject": "",
            "creationDate": "",
            "modDate": "",
        }
        mock_page.annots.return_value = [mock_annot]
        with patch.dict("sys.modules", {"fitz": mock_mod}):
            from analyze_pdf_comments import analyze_pdf_annotations
            result = analyze_pdf_annotations("/tmp/test.pdf")
            assert result["summary"]["has_comments"] is False

    def test_long_text_content(self, mock_fitz):
        mock_mod, mock_doc, mock_page, _ = mock_fitz
        mock_page.get_text.return_value = "A" * 300
        mock_page.annots.return_value = []
        with patch.dict("sys.modules", {"fitz": mock_mod}):
            from analyze_pdf_comments import analyze_pdf_annotations
            result = analyze_pdf_annotations("/tmp/test.pdf")
            assert result["text_content"][0]["text_preview"].endswith("...")

    def test_error_handling(self):
        mock_mod = MagicMock()
        mock_mod.open.side_effect = Exception("Cannot open PDF")
        with patch.dict("sys.modules", {"fitz": mock_mod}):
            from analyze_pdf_comments import analyze_pdf_annotations
            result = analyze_pdf_annotations("/tmp/bad.pdf")
            assert "error" in result

    def test_freetype_annotation(self, mock_fitz):
        mock_mod, mock_doc, mock_page, mock_annot = mock_fitz
        mock_annot.info = {
            "type": "FreeText",
            "content": "",
            "title": "",
            "subject": "",
            "creationDate": "",
            "modDate": "",
        }
        mock_page.annots.return_value = [mock_annot]
        with patch.dict("sys.modules", {"fitz": mock_mod}):
            from analyze_pdf_comments import analyze_pdf_annotations
            result = analyze_pdf_annotations("/tmp/test.pdf")
            assert result["summary"]["has_comments"] is True


class TestPrintAnalysisReport:
    def test_report_with_error(self):
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            from analyze_pdf_comments import print_analysis_report
            print_analysis_report({"error": "some error"})

    def test_report_with_annotations(self):
        analysis = {
            "metadata": {"title": "Test", "author": "Author", "empty_key": ""},
            "summary": {
                "total_annotations": 1,
                "annotation_types": {"Text": 1},
                "has_comments": True,
            },
            "annotations": [
                {
                    "page": 1,
                    "type": "Text",
                    "content": "A comment",
                    "author": "Author",
                    "subject": "Review",
                    "creation_date": "2025-01-01",
                },
            ],
            "text_content": [
                {"page": 1, "text_length": 100, "text_preview": "Hello world"},
            ],
        }
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            from analyze_pdf_comments import print_analysis_report
            print_analysis_report(analysis)

    def test_report_no_annotations(self):
        analysis = {
            "metadata": {},
            "summary": {
                "total_annotations": 0,
                "annotation_types": {},
                "has_comments": False,
            },
            "annotations": [],
            "text_content": [
                {"page": 1, "text_length": 0, "text_preview": ""},
            ],
        }
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            from analyze_pdf_comments import print_analysis_report
            print_analysis_report(analysis)

    def test_report_annotation_missing_fields(self):
        analysis = {
            "metadata": {"title": "T"},
            "summary": {
                "total_annotations": 1,
                "annotation_types": {"Text": 1},
                "has_comments": True,
            },
            "annotations": [
                {
                    "page": 1,
                    "type": "Text",
                    "content": "",
                    "author": "",
                    "subject": "",
                    "creation_date": "",
                },
            ],
            "text_content": [],
        }
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            from analyze_pdf_comments import print_analysis_report
            print_analysis_report(analysis)


class TestMain:
    def test_no_args(self):
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            with patch("sys.argv", ["analyze_pdf_comments.py"]):
                from analyze_pdf_comments import main
                with pytest.raises(SystemExit):
                    main()

    def test_file_not_found(self, tmp_path):
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            with patch("sys.argv", ["analyze_pdf_comments.py", str(tmp_path / "nonexistent.pdf")]):
                from analyze_pdf_comments import main
                with pytest.raises(SystemExit):
                    main()

    def test_successful_analysis(self, tmp_path, mock_fitz):
        mock_mod, _, _, _ = mock_fitz
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf")
        json_out = tmp_path / "test_analysis.json"

        with patch.dict("sys.modules", {"fitz": mock_mod}):
            with patch("sys.argv", ["analyze_pdf_comments.py", str(pdf_file)]):
                with patch("analyze_pdf_comments.get_simulation_data_path", return_value=str(json_out)):
                    from analyze_pdf_comments import main
                    main()
