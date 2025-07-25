#!/usr/bin/env python3
"""
PDF Comment and Annotation Analyzer using PyMuPDF
Extracts and displays comments, annotations, and metadata from PDF files.
"""

import fitz  # PyMuPDF
import sys
import json
from pathlib import Path
from path_config import get_simulation_data_path


def analyze_pdf_annotations(pdf_path):
    """
    Analyze PDF file for annotations, comments, and metadata.
    
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
                "has_comments": False
            }
        }
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_annotations = []
            
            # Extract text content
            text_content = page.get_text()
            analysis["text_content"].append({
                "page": page_num + 1,
                "text_length": len(text_content),
                "text_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
            })
            
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
                if annotation_data["content"] or annotation_data["type"] in ["Text", "FreeText", "Note"]:
                    analysis["summary"]["has_comments"] = True
                
                page_annotations.append(annotation_data)
                
                # Count annotation types
                annot_type = annotation_data["type"]
                analysis["summary"]["annotation_types"][annot_type] = \
                    analysis["summary"]["annotation_types"].get(annot_type, 0) + 1
            
            if page_annotations:
                analysis["annotations"].extend(page_annotations)
        
        analysis["summary"]["total_annotations"] = len(analysis["annotations"])
        
        doc.close()
        return analysis
        
    except Exception as e:
        return {"error": f"Error analyzing PDF: {str(e)}"}


def print_analysis_report(analysis):
    """Print a formatted analysis report."""
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    print("=" * 70)
    print("📄 PDF ANALYSIS REPORT")
    print("=" * 70)
    print(f"File: {analysis['file_path']}")
    print(f"Pages: {analysis['page_count']}")
    print()
    
    # Metadata
    print("📋 METADATA:")
    metadata = analysis['metadata']
    for key, value in metadata.items():
        if value:
            print(f"  {key}: {value}")
    print()
    
    # Summary
    summary = analysis['summary']
    print("📊 ANNOTATION SUMMARY:")
    print(f"  Total annotations: {summary['total_annotations']}")
    print(f"  Has comments: {'✅ Yes' if summary['has_comments'] else '❌ No'}")
    
    if summary['annotation_types']:
        print("  Annotation types:")
        for annot_type, count in summary['annotation_types'].items():
            print(f"    - {annot_type}: {count}")
    print()
    
    # Detailed annotations
    if analysis['annotations']:
        print("💬 ANNOTATIONS DETAILS:")
        for i, annot in enumerate(analysis['annotations'], 1):
            print(f"  [{i}] Page {annot['page']} - {annot['type']}")
            if annot['author']:
                print(f"      Author: {annot['author']}")
            if annot['subject']:
                print(f"      Subject: {annot['subject']}")
            if annot['content']:
                print(f"      Content: {annot['content']}")
            if annot['creation_date']:
                print(f"      Created: {annot['creation_date']}")
            print()
    else:
        print("📝 No annotations found in this PDF.")
        print()
    
    # Text content summary
    print("📖 TEXT CONTENT SUMMARY:")
    for page_info in analysis['text_content']:
        print(f"  Page {page_info['page']}: {page_info['text_length']} characters")
        if page_info['text_preview']:
            print(f"    Preview: {page_info['text_preview']}")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_pdf_comments.py <pdf_file_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"❌ Error: File '{pdf_path}' not found.")
        sys.exit(1)
    
    print(f"🔍 Analyzing PDF: {pdf_path}")
    print()
    
    # Analyze the PDF
    analysis = analyze_pdf_annotations(pdf_path)
    
    # Print the report
    print_analysis_report(analysis)
    
    # Optionally save to JSON
    json_output = get_simulation_data_path(Path(pdf_path).stem + '_analysis.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Detailed analysis saved to: {json_output}")


if __name__ == "__main__":
    main()