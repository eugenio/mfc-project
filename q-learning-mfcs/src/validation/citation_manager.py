"""
Citation Management and Reporting System

This module provides comprehensive citation management for literature
references used in MFC parameter validation and reporting.

Created: 2025-08-01
"""
import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .literature_validation import LiteratureReference
from .pubmed_connector import PubMedArticle
from .quality_assessor import QualityScore


@dataclass
class Citation:
    """Container for formatted citation information."""

    pmid: str = ""
    authors: list[str] = field(default_factory=list)
    title: str = ""
    journal: str = ""
    year: int = 0
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""

    def to_apa_format(self) -> str:
        """Format citation in APA style."""
        if not self.authors or not self.title:
            return "Citation information incomplete"

        # Format authors
        if len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) <= 3:
            author_str = ", ".join(self.authors[:-1]) + f", & {self.authors[-1]}"
        else:
            author_str = f"{self.authors[0]}, et al."

        # Format year
        year_str = f"({self.year})" if self.year > 0 else "(n.d.)"

        # Format title
        title_str = f"{self.title.rstrip('.')}."

        # Format journal with volume/issue
        journal_str = f"{self.journal}"
        if self.volume:
            journal_str += f", {self.volume}"
            if self.issue:
                journal_str += f"({self.issue})"
        if self.pages:
            journal_str += f", {self.pages}"
        journal_str += "."

        # Format DOI
        doi_str = f" https://doi.org/{self.doi}" if self.doi else ""

        return f"{author_str} {year_str}. {title_str} {journal_str}{doi_str}"

    def to_vancouver_format(self) -> str:
        """Format citation in Vancouver style."""
        if not self.authors or not self.title:
            return "Citation information incomplete"

        # Format authors (surname first)
        author_list = []
        for author in self.authors[:6]:  # Vancouver limits to first 6 authors
            if "," in author:
                # Already in "Surname, Initials" format
                author_list.append(author)
            else:
                # Convert "First Last" to "Last F"
                parts = author.split()
                if len(parts) >= 2:
                    surname = parts[-1]
                    initials = "".join([p[0] for p in parts[:-1]])
                    author_list.append(f"{surname} {initials}")
                else:
                    author_list.append(author)

        if len(self.authors) > 6:
            author_str = ", ".join(author_list) + ", et al."
        else:
            author_str = ", ".join(author_list)

        # Format title (no quotes, sentence case)
        title_str = self.title.rstrip('.')

        # Format journal
        journal_str = self.journal

        # Format date and volume/issue
        date_str = str(self.year) if self.year > 0 else ""

        vol_str = ""
        if self.volume:
            vol_str = f";{self.volume}"
            if self.issue:
                vol_str += f"({self.issue})"
        if self.pages:
            vol_str += f":{self.pages}"

        # Format DOI
        doi_str = f". doi:{self.doi}" if self.doi else ""

        return f"{author_str}. {title_str}. {journal_str}. {date_str}{vol_str}{doi_str}."

    def to_bibtex_format(self, cite_key: str = None) -> str:
        """Format citation in BibTeX format."""
        if not cite_key:
            # Generate cite key from first author and year
            if self.authors and self.year:
                first_author = self.authors[0].split()[-1].lower()  # Last name
                cite_key = f"{first_author}{self.year}"
            else:
                cite_key = f"ref{self.pmid}" if self.pmid else "unknown"

        # Clean author names for BibTeX
        author_list = []
        for author in self.authors:
            if "," in author:
                author_list.append(author)
            else:
                parts = author.split()
                if len(parts) >= 2:
                    surname = parts[-1]
                    given = " ".join(parts[:-1])
                    author_list.append(f"{surname}, {given}")
                else:
                    author_list.append(author)

        authors_str = " and ".join(author_list)

        # Build BibTeX entry
        bibtex = f"@article{{{cite_key},\n"
        bibtex += f"  author = {{{authors_str}}},\n"
        bibtex += f"  title = {{{self.title}}},\n"
        bibtex += f"  journal = {{{self.journal}}},\n"

        if self.year > 0:
            bibtex += f"  year = {{{self.year}}},\n"
        if self.volume:
            bibtex += f"  volume = {{{self.volume}}},\n"
        if self.issue:
            bibtex += f"  number = {{{self.issue}}},\n"
        if self.pages:
            bibtex += f"  pages = {{{self.pages}}},\n"
        if self.doi:
            bibtex += f"  doi = {{{self.doi}}},\n"
        if self.pmid:
            bibtex += f"  pmid = {{{self.pmid}}},\n"

        bibtex += "}"

        return bibtex
@dataclass
class CitationReport:
    """Container for citation report information."""

    parameter_name: str
    parameter_value: float
    units: str
    validation_status: str
    citations: list[Citation] = field(default_factory=list)
    quality_scores: list[QualityScore] = field(default_factory=list)
    report_metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
class CitationManager:
    """
    Comprehensive citation management system.

    Handles citation formatting, bibliography generation, and
    reporting for literature validation in MFC systems.
    """

    def __init__(self):
        """Initialize citation manager."""
        self.citations = {}  # pmid -> Citation
        self.reports = []

    def create_citation_from_article(self, article: PubMedArticle) -> Citation:
        """Create Citation object from PubMedArticle."""

        # Extract publication year
        year = 0
        if article.publication_date:
            year_match = re.search(r'\b(19|20)\d{2}\b', article.publication_date)
            if year_match:
                year = int(year_match.group())

        # Create URL from PMID or DOI
        url = ""
        if article.doi:
            url = f"https://doi.org/{article.doi}"
        elif article.pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"

        citation = Citation(
            pmid=article.pmid,
            authors=article.authors,
            title=article.title,
            journal=article.journal,
            year=year,
            doi=article.doi,
            url=url
        )

        # Store citation
        if article.pmid:
            self.citations[article.pmid] = citation

        return citation

    def create_citation_from_reference(self, reference: LiteratureReference) -> Citation:
        """Create Citation object from LiteratureReference."""

        # Parse authors
        authors = []
        if reference.authors:
            # Handle various author formats
            if ";" in reference.authors:
                authors = [a.strip() for a in reference.authors.split(";")]
            elif "," in reference.authors and not reference.authors.count(",") == 1:
                # Multiple authors separated by commas
                authors = [a.strip() for a in reference.authors.split(",")]
            else:
                authors = [reference.authors.strip()]

        # Parse page range
        pages = reference.page_range

        citation = Citation(
            authors=authors,
            title=reference.title,
            journal=reference.journal,
            year=reference.year,
            pages=pages,
            doi=reference.doi
        )

        # Generate unique key for storage
        key = f"ref_{reference.year}_{hash(reference.title) % 10000}"
        self.citations[key] = citation

        return citation

    def format_citations(self, citations: list[Citation], style: str = "apa") -> list[str]:
        """Format list of citations in specified style."""

        formatted = []

        for citation in citations:
            if style.lower() == "apa":
                formatted.append(citation.to_apa_format())
            elif style.lower() == "vancouver":
                formatted.append(citation.to_vancouver_format())
            elif style.lower() == "bibtex":
                formatted.append(citation.to_bibtex_format())
            else:
                warnings.warn(f"Unknown citation style: {style}. Using APA.", stacklevel=2)
                formatted.append(citation.to_apa_format())

        return formatted

    def generate_bibliography(self, pmids: list[str] = None, style: str = "apa") -> str:
        """Generate bibliography for specified citations."""

        if pmids is None:
            # Use all citations
            citations_to_format = list(self.citations.values())
        else:
            citations_to_format = [self.citations[pmid] for pmid in pmids if pmid in self.citations]

        if not citations_to_format:
            return "No citations available for bibliography."

        # Sort citations by first author and year
        citations_to_format.sort(key=lambda c: (c.authors[0] if c.authors else "", c.year))

        # Format citations
        formatted_citations = self.format_citations(citations_to_format, style)

        # Generate bibliography
        bibliography = f"# References ({style.upper()} Style)\n\n"

        for i, citation in enumerate(formatted_citations, 1):
            bibliography += f"{i}. {citation}\n\n"

        return bibliography

    def create_parameter_report(self, parameter_name: str, parameter_value: float,
                              units: str, validation_status: str,
                              articles: list[PubMedArticle] = None,
                              references: list[LiteratureReference] = None,
                              quality_scores: list[QualityScore] = None) -> CitationReport:
        """Create comprehensive citation report for parameter validation."""

        citations = []

        # Add citations from articles
        if articles:
            for article in articles:
                citation = self.create_citation_from_article(article)
                citations.append(citation)

        # Add citations from references
        if references:
            for reference in references:
                citation = self.create_citation_from_reference(reference)
                citations.append(citation)

        # Generate report metadata
        metadata = {
            'total_citations': len(citations),
            'high_quality_citations': 0,
            'recent_citations': 0,
            'doi_coverage': 0,
            'journal_diversity': 0
        }

        # Analyze citations
        if citations:
            # Count high-quality citations (if quality scores provided)
            if quality_scores:
                high_quality = sum(1 for score in quality_scores if score.overall_score >= 0.8)
                metadata['high_quality_citations'] = high_quality

            # Count recent citations (last 10 years)
            current_year = datetime.now().year
            recent = sum(1 for c in citations if c.year >= current_year - 10)
            metadata['recent_citations'] = recent

            # DOI coverage
            with_doi = sum(1 for c in citations if c.doi)
            metadata['doi_coverage'] = with_doi / len(citations)

            # Journal diversity
            journals = {c.journal for c in citations if c.journal}
            metadata['journal_diversity'] = len(journals)

        report = CitationReport(
            parameter_name=parameter_name,
            parameter_value=parameter_value,
            units=units,
            validation_status=validation_status,
            citations=citations,
            quality_scores=quality_scores or [],
            report_metadata=metadata
        )

        self.reports.append(report)
        return report

    def generate_validation_report(self, reports: list[CitationReport] = None,
                                 output_format: str = "markdown") -> str:
        """Generate comprehensive validation report with citations."""

        if reports is None:
            reports = self.reports

        if not reports:
            return "No citation reports available."

        if output_format.lower() == "markdown":
            return self._generate_markdown_report(reports)
        elif output_format.lower() == "html":
            return self._generate_html_report(reports)
        elif output_format.lower() == "json":
            return self._generate_json_report(reports)
        else:
            warnings.warn(f"Unknown output format: {output_format}. Using markdown.", stacklevel=2)
            return self._generate_markdown_report(reports)

    def _generate_markdown_report(self, reports: list[CitationReport]) -> str:
        """Generate markdown validation report."""

        report = "# MFC Parameter Validation Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Summary statistics
        total_params = len(reports)
        validated = sum(1 for r in reports if r.validation_status == "VALIDATED")
        total_citations = sum(len(r.citations) for r in reports)

        report += "## Summary\n\n"
        report += f"- **Total Parameters Validated:** {total_params}\n"
        report += f"- **Successfully Validated:** {validated} ({validated/total_params*100:.1f}%)\n"
        report += f"- **Total Literature Citations:** {total_citations}\n"
        report += f"- **Average Citations per Parameter:** {total_citations/total_params:.1f}\n\n"

        # Parameter details
        report += "## Parameter Validation Details\n\n"

        for i, param_report in enumerate(reports, 1):
            report += f"### {i}. {param_report.parameter_name}\n\n"
            report += f"- **Value:** {param_report.parameter_value} {param_report.units}\n"
            report += f"- **Status:** {param_report.validation_status}\n"
            report += f"- **Citations:** {len(param_report.citations)}\n"

            # Quality metrics
            if param_report.quality_scores:
                avg_quality = sum(q.overall_score for q in param_report.quality_scores) / len(param_report.quality_scores)
                report += f"- **Average Quality Score:** {avg_quality:.2f}\n"

            report += "\n"

            # Citations
            if param_report.citations:
                report += "**Literature Support:**\n\n"
                formatted_citations = self.format_citations(param_report.citations, "apa")

                for j, citation in enumerate(formatted_citations, 1):
                    report += f"{j}. {citation}\n\n"

            report += "---\n\n"

        # Bibliography
        all_citations = []
        for param_report in reports:
            all_citations.extend(param_report.citations)

        # Remove duplicates based on PMID or title
        unique_citations = {}
        for citation in all_citations:
            key = citation.pmid if citation.pmid else citation.title
            if key and key not in unique_citations:
                unique_citations[key] = citation

        if unique_citations:
            report += "## Complete Bibliography\n\n"
            sorted_citations = sorted(unique_citations.values(),
                                    key=lambda c: (c.authors[0] if c.authors else "", c.year))
            formatted_bib = self.format_citations(sorted_citations, "apa")

            for i, citation in enumerate(formatted_bib, 1):
                report += f"{i}. {citation}\n\n"

        return report

    def _generate_html_report(self, reports: list[CitationReport]) -> str:
        """Generate HTML validation report."""

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MFC Parameter Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #2c3e50; }
                .summary { background: #f8f9fa; padding: 20px; border-left: 4px solid #007bff; }
                .parameter { margin: 20px 0; padding: 15px; border: 1px solid #dee2e6; }
                .validated { border-left: 4px solid #28a745; }
                .needs-review { border-left: 4px solid #ffc107; }
                .citation { margin: 10px 0; padding: 10px; background: #f8f9fa; }
                .quality-high { color: #28a745; }
                .quality-medium { color: #ffc107; }
                .quality-low { color: #dc3545; }
            </style>
        </head>
        <body>
        """

        html += "<h1>MFC Parameter Validation Report</h1>\n"
        html += f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n"

        # Summary
        total_params = len(reports)
        validated = sum(1 for r in reports if r.validation_status == "VALIDATED")
        total_citations = sum(len(r.citations) for r in reports)

        html += '<div class="summary">\n'
        html += '<h2>Summary</h2>\n'
        html += '<ul>\n'
        html += f'<li><strong>Total Parameters:</strong> {total_params}</li>\n'
        html += f'<li><strong>Validated:</strong> {validated} ({validated/total_params*100:.1f}%)</li>\n'
        html += f'<li><strong>Total Citations:</strong> {total_citations}</li>\n'
        html += '</ul>\n'
        html += '</div>\n'

        # Parameters
        html += '<h2>Parameter Details</h2>\n'

        for param_report in reports:
            status_class = param_report.validation_status.lower().replace("_", "-")
            html += f'<div class="parameter {status_class}">\n'
            html += f'<h3>{param_report.parameter_name}</h3>\n'
            html += f'<p><strong>Value:</strong> {param_report.parameter_value} {param_report.units}</p>\n'
            html += f'<p><strong>Status:</strong> {param_report.validation_status}</p>\n'

            if param_report.citations:
                html += '<h4>Citations:</h4>\n'
                formatted_citations = self.format_citations(param_report.citations, "apa")

                for citation in formatted_citations:
                    html += f'<div class="citation">{citation}</div>\n'

            html += '</div>\n'

        html += '</body></html>'

        return html

    def _generate_json_report(self, reports: list[CitationReport]) -> str:
        """Generate JSON validation report."""

        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_parameters': len(reports),
                'total_citations': sum(len(r.citations) for r in reports)
            },
            'parameters': []
        }

        for param_report in reports:
            param_data = {
                'name': param_report.parameter_name,
                'value': param_report.parameter_value,
                'units': param_report.units,
                'validation_status': param_report.validation_status,
                'citations': [asdict(c) for c in param_report.citations],
                'quality_scores': [asdict(q) for q in param_report.quality_scores],
                'metadata': param_report.report_metadata
            }
            report_data['parameters'].append(param_data)

        return json.dumps(report_data, indent=2, default=str)

    def export_bibtex_file(self, filepath: str, pmids: list[str] = None):
        """Export citations to BibTeX file."""

        if pmids is None:
            citations_to_export = list(self.citations.values())
        else:
            citations_to_export = [self.citations[pmid] for pmid in pmids if pmid in self.citations]

        if not citations_to_export:
            warnings.warn("No citations to export.", stacklevel=2)
            return

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("% BibTeX entries for MFC parameter validation\n")
            f.write(f"% Generated: {datetime.now().isoformat()}\n\n")

            for citation in citations_to_export:
                f.write(citation.to_bibtex_format() + "\n\n")

        print(f"📚 BibTeX file exported to: {filepath}")

    def get_citation_statistics(self) -> dict[str, Any]:
        """Get comprehensive citation statistics."""

        if not self.citations:
            return {'total_citations': 0, 'error': 'No citations available'}

        citations = list(self.citations.values())
        current_year = datetime.now().year

        stats = {
            'total_citations': len(citations),
            'unique_journals': len({c.journal for c in citations if c.journal}),
            'unique_authors': len({author for c in citations for author in c.authors}),
            'doi_coverage': sum(1 for c in citations if c.doi) / len(citations),
            'recent_citations': sum(1 for c in citations if c.year >= current_year - 5),
            'year_distribution': {},
            'journal_distribution': {},
            'author_distribution': {}
        }

        # Year distribution
        for citation in citations:
            if citation.year > 0:
                stats['year_distribution'][citation.year] = stats['year_distribution'].get(citation.year, 0) + 1

        # Journal distribution (top 10)
        journal_counts = {}
        for citation in citations:
            if citation.journal:
                journal_counts[citation.journal] = journal_counts.get(citation.journal, 0) + 1
        stats['journal_distribution'] = dict(sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        # Author distribution (top 10 first authors)
        author_counts = {}
        for citation in citations:
            if citation.authors:
                first_author = citation.authors[0]
                author_counts[first_author] = author_counts.get(first_author, 0) + 1
        stats['author_distribution'] = dict(sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        return stats
if __name__ == "__main__":
    # Example usage
    print("📚 Citation Management System Test")
    print("=" * 50)

    # Initialize citation manager
    manager = CitationManager()

    # Create sample citation
    sample_article = PubMedArticle(
        pmid="12345",
        title="Microbial fuel cell optimization using advanced electrode materials",
        authors=["Smith, John A.", "Johnson, Mary B.", "Brown, David C."],
        journal="Energy & Environmental Science",
        publication_date="2023-05-15",
        doi="10.1039/test123"
    )

    citation = manager.create_citation_from_article(sample_article)

    print("📄 Sample Citation Formats:")
    print(f"\nAPA: {citation.to_apa_format()}")
    print(f"\nVancouver: {citation.to_vancouver_format()}")
    print(f"\nBibTeX:\n{citation.to_bibtex_format()}")

    # Create parameter report
    report = manager.create_parameter_report(
        parameter_name="max_current_density",
        parameter_value=2.5,
        units="A/m²",
        validation_status="VALIDATED",
        articles=[sample_article]
    )

    print("\n📊 Parameter Report:")
    print(f"  Parameter: {report.parameter_name}")
    print(f"  Citations: {len(report.citations)}")
    print(f"  Status: {report.validation_status}")

    # Generate bibliography
    bibliography = manager.generate_bibliography(style="apa")
    print(f"\n📚 Bibliography:\n{bibliography}")

    # Get statistics
    stats = manager.get_citation_statistics()
    print("📈 Citation Statistics:")
    for key, value in stats.items():
        if key not in ['year_distribution', 'journal_distribution', 'author_distribution']:
            print(f"  {key}: {value}")

    print("\n✅ Citation management system test completed!")
