#!/usr/bin/env python3
"""PDF Comment and Annotation Analyzer using PyMuPDF
Extracts and displays comments, annotations, and metadata from PDF files.
"""

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from path_config import get_simulation_data_path


def analyze_pdf_annotations(pdf_path):
    """Analyze PDF file for annotations, comments, and metadata.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        dict: Analysis results containing annotations, metadata, and text content

    """
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)

        analysis = {
            "file_path": pdf_path,
            "metadata": doc.metadata,
            "page_count": len(doc),
            "annotations": [],
            "text_content": [],
            "summary": {
                "total_annotations": 0,
                "annotation_types": {},
                "has_comments": False,
            },
        }

        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_annotations = []

            # Extract text content
            text_content = page.get_text()
            analysis["text_content"].append(
                {
                    "page": page_num + 1,
                    "text_length": len(text_content),
                    "text_preview": (
                        text_content[:200] + "..."
                        if len(text_content) > 200
                        else text_content
                    ),
                },
            )

            # Get annotations from the page
            annotations = page.annots()

            for annot in annotations:
                annot_dict = annot.info

                # Get annotation details
                annotation_data = {
                    "page": page_num + 1,
                    "type": annot_dict.get("type", "Unknown"),
                    "content": annot_dict.get("content", ""),
                    "author": annot_dict.get("title", ""),
                    "subject": annot_dict.get("subject", ""),
                    "creation_date": annot_dict.get("creationDate", ""),
                    "modification_date": annot_dict.get("modDate", ""),
                    "rect": list(annot.rect),  # Bounding rectangle
                }

                # Check if it's a comment/text annotation
                if annotation_data["content"] or annotation_data["type"] in [
                    "Text",
                    "FreeText",
                    "Note",
                ]:
                    analysis["summary"]["has_comments"] = True

                page_annotations.append(annotation_data)

                # Count annotation types
                annot_type = annotation_data["type"]
                analysis["summary"]["annotation_types"][annot_type] = (
                    analysis["summary"]["annotation_types"].get(annot_type, 0) + 1
                )

            if page_annotations:
                analysis["annotations"].extend(page_annotations)

        analysis["summary"]["total_annotations"] = len(analysis["annotations"])

        doc.close()
        return analysis

    except Exception as e:
        return {"error": f"Error analyzing PDF: {e!s}"}


def print_analysis_report(analysis) -> None:
    """Print a formatted analysis report."""
    if "error" in analysis:
        return

    # Metadata
    metadata = analysis["metadata"]
    for value in metadata.values():
        if value:
            pass

    # Summary
    summary = analysis["summary"]

    if summary["annotation_types"]:
        for _annot_type, _count in summary["annotation_types"].items():
            pass

    # Detailed annotations
    if analysis["annotations"]:
        for _i, annot in enumerate(analysis["annotations"], 1):
            if annot["author"]:
                pass
            if annot["subject"]:
                pass
            if annot["content"]:
                pass
            if annot["creation_date"]:
                pass
    else:
        pass

    # Text content summary
    for page_info in analysis["text_content"]:
        if page_info["text_preview"]:
            pass


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        sys.exit(1)

    # Analyze the PDF
    analysis = analyze_pdf_annotations(pdf_path)

    # Print the report
    print_analysis_report(analysis)

    # Optionally save to JSON
    json_output = get_simulation_data_path(Path(pdf_path).stem + "_analysis.json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
