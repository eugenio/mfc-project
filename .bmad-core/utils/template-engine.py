#!/usr/bin/env python3
"""
Template Engine for Documentation Agent

Handles template application, content migration, and standardization
for the MFC project documentation system.
Created: 2025-07-31
Integration: BMAD Documentation Agent
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class DocumentType(Enum):
    """Supported document types for template application."""
    TECHNICAL_SPEC = "technical-spec"
    API_DOC = "api-doc"
    USER_GUIDE = "user-guide"
    ARCHITECTURE = "architecture"
    UNKNOWN = "unknown"

@dataclass
class DocumentMetadata:
    """Document metadata structure."""
    title: str
    type: str
    created_at: str
    authors: list[str]
    last_modified_at: str | None = None
    version: str | None = None
    reviewers: list[str] | None = None
    tags: list[str] | None = None
    status: str | None = None
    related_docs: list[str] | None = None

@dataclass
class ContentSection:
    """Represents a content section for migration."""
    name: str
    content: str
    level: int
    subsections: list['ContentSection'] = None

    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []

class DocumentationTemplateEngine:
    """Template engine for documentation standardization."""

    def __init__(self, project_root: str = "/home/uge/mfc-project"):
        """Initialize template engine with project configuration."""
        self.project_root = Path(project_root)
        self.templates_dir = self.project_root / ".bmad-core" / "templates"
        self.standards_file = self.project_root / ".bmad-core" / "data" / "doc-standards.yaml"

        self.logger = self._setup_logging()
        self.standards = self._load_standards()
        self.templates = self._load_templates()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for template engine operations."""
        logger = logging.getLogger('doc-template-engine')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_standards(self) -> dict[str, Any]:
        """Load documentation standards configuration."""
        try:
            with open(self.standards_file) as f:
                standards = yaml.safe_load(f)
            self.logger.info("Loaded documentation standards configuration")
            return standards
        except Exception as e:
            self.logger.error(f"Failed to load standards configuration: {e}")
            return {}

    def _load_templates(self) -> dict[str, str]:
        """Load all available templates."""
        templates = {}

        try:
            for template_file in self.templates_dir.glob("*-template.md"):
                template_name = template_file.stem.replace("-template", "")
                with open(template_file) as f:
                    templates[template_name] = f.read()

                self.logger.info(f"Loaded template: {template_name}")

        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}")

        return templates

    def detect_document_type(self, file_path: str) -> DocumentType:
        """Detect document type from content and file path."""
        try:
            with open(file_path) as f:
                content = f.read()

            # Check for explicit type in metadata
            metadata = self.extract_metadata(content)
            if metadata and metadata.get('type'):
                try:
                    return DocumentType(metadata['type'])
                except ValueError:
                    pass

            # Heuristic detection based on content and file name
            content_lower = content.lower()
            file_name_lower = Path(file_path).name.lower()

            # API documentation indicators
            if any(indicator in content_lower for indicator in
                   ['api', 'endpoint', 'rest', 'http', 'json', 'curl']):
                return DocumentType.API_DOC

            # Technical specification indicators
            if any(indicator in content_lower for indicator in
                   ['specification', 'implementation', 'algorithm', 'model']):
                return DocumentType.TECHNICAL_SPEC

            # User guide indicators
            if any(indicator in content_lower for indicator in
                   ['guide', 'tutorial', 'getting started', 'how to']):
                return DocumentType.USER_GUIDE

            # Architecture documentation indicators
            if any(indicator in content_lower for indicator in
                   ['architecture', 'system design', 'component', 'flow']):
                return DocumentType.ARCHITECTURE

            # File name based detection
            if 'api' in file_name_lower:
                return DocumentType.API_DOC
            elif 'spec' in file_name_lower:
                return DocumentType.TECHNICAL_SPEC
            elif 'guide' in file_name_lower:
                return DocumentType.USER_GUIDE
            elif 'arch' in file_name_lower:
                return DocumentType.ARCHITECTURE

            return DocumentType.UNKNOWN

        except Exception as e:
            self.logger.error(f"Failed to detect document type for {file_path}: {e}")
            return DocumentType.UNKNOWN

    def extract_metadata(self, content: str) -> dict[str, Any] | None:
        """Extract existing metadata from document."""
        try:
            # Look for YAML frontmatter
            frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
            if frontmatter_match:
                yaml_content = frontmatter_match.group(1)
                return yaml.safe_load(yaml_content)

            return None

        except Exception as e:
            self.logger.error(f"Failed to extract metadata: {e}")
            return None

    def parse_document_structure(self, content: str) -> list[ContentSection]:
        """Parse document into structured sections."""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            # Skip YAML frontmatter
            if line.strip() == '---':
                continue

            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content)
                    sections.append(current_section)

                # Start new section
                level = len(header_match.group(1))
                name = header_match.group(2)
                current_section = ContentSection(name=name, content='', level=level)
                current_content = []
            else:
                # Add content to current section
                if current_section:
                    current_content.append(line)

        # Save final section
        if current_section:
            current_section.content = '\n'.join(current_content)
            sections.append(current_section)

        return sections

    def generate_metadata(
        self,
        doc_type: DocumentType,
        existing_metadata: dict | None = None,
        file_path: str | None = None
    ) -> DocumentMetadata:
        """Generate standardized metadata for document."""

        # Extract title from file path if not provided
        title = "Untitled Document"
        if file_path:
            title = Path(file_path).stem.replace('_', ' ').replace('-', ' ').title()

        # Use existing metadata as base
        if existing_metadata:
            metadata = DocumentMetadata(
                title=existing_metadata.get('title', title),
                type=doc_type.value,
                created_at=existing_metadata.get('created_at', datetime.now().strftime('%Y-%m-%d')),
                authors=existing_metadata.get('authors', ['MFC Team']),
                last_modified_at=datetime.now().strftime('%Y-%m-%d'),
                version=existing_metadata.get('version', '1.0'),
                reviewers=existing_metadata.get('reviewers', []),
                tags=existing_metadata.get('tags', self._generate_default_tags(doc_type)),
                status=existing_metadata.get('status', 'draft'),
                related_docs=existing_metadata.get('related_docs', [])
            )
        else:
            # Generate new metadata
            metadata = DocumentMetadata(
                title=title,
                type=doc_type.value,
                created_at=datetime.now().strftime('%Y-%m-%d'),
                authors=['MFC Team'],
                last_modified_at=datetime.now().strftime('%Y-%m-%d'),
                version='1.0',
                reviewers=[],
                tags=self._generate_default_tags(doc_type),
                status='draft',
                related_docs=[]
            )

        return metadata

    def _generate_default_tags(self, doc_type: DocumentType) -> list[str]:
        """Generate default tags for document type."""
        base_tags = ['mfc']

        if doc_type == DocumentType.TECHNICAL_SPEC:
            base_tags.extend(['technical-spec', 'simulation'])
        elif doc_type == DocumentType.API_DOC:
            base_tags.extend(['api', 'interface'])
        elif doc_type == DocumentType.USER_GUIDE:
            base_tags.extend(['user-guide', 'documentation'])
        elif doc_type == DocumentType.ARCHITECTURE:
            base_tags.extend(['architecture', 'system-design'])

        return base_tags

    def apply_template(
        self,
        file_path: str,
        preserve_content: bool = True,
        dry_run: bool = False
    ) -> tuple[bool, str, str]:
        """Apply appropriate template to document.

        Returns:
            Tuple of (success, original_content, standardized_content)
        """
        try:
            # Read original content
            with open(file_path) as f:
                original_content = f.read()

            # Detect document type
            doc_type = self.detect_document_type(file_path)
            if doc_type == DocumentType.UNKNOWN:
                return False, original_content, "Unable to determine document type"

            # Get appropriate template
            template_key = doc_type.value.replace('-', '_')
            if template_key not in self.templates:
                return False, original_content, f"No template found for {doc_type.value}"

            template = self.templates[template_key]

            # Extract existing metadata and content
            existing_metadata = self.extract_metadata(original_content)
            content_sections = self.parse_document_structure(original_content)

            # Generate standardized metadata
            metadata = self.generate_metadata(doc_type, existing_metadata, file_path)

            # Apply template with content migration
            standardized_content = self._migrate_content_to_template(
                template, metadata, content_sections, preserve_content
            )

            # Write standardized content if not dry run
            if not dry_run:
                # Create backup
                backup_path = f"{file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(backup_path, 'w') as f:
                    f.write(original_content)

                # Write standardized version
                with open(file_path, 'w') as f:
                    f.write(standardized_content)

                self.logger.info(f"Standardized {file_path} using {doc_type.value} template")

            return True, original_content, standardized_content

        except Exception as e:
            self.logger.error(f"Failed to apply template to {file_path}: {e}")
            return False, "", str(e)

    def _migrate_content_to_template(
        self,
        template: str,
        metadata: DocumentMetadata,
        sections: list[ContentSection],
        preserve_content: bool
    ) -> str:
        """Migrate existing content to template structure."""

        # Generate metadata YAML
        metadata_yaml = self._generate_metadata_yaml(metadata)

        # Start with template
        result = template

        # Replace metadata placeholders
        result = re.sub(r'^---.*?---', metadata_yaml, result, flags=re.DOTALL)

        if preserve_content and sections:
            # Map existing content to template sections
            result = self._map_content_to_template_sections(result, sections)

        # Clean up template placeholders
        result = self._clean_template_placeholders(result, metadata)

        return result

    def _generate_metadata_yaml(self, metadata: DocumentMetadata) -> str:
        """Generate YAML frontmatter from metadata."""
        yaml_dict = {
            'title': metadata.title,
            'type': metadata.type,
            'created_at': metadata.created_at,
            'authors': metadata.authors
        }

        # Add optional fields if present
        if metadata.last_modified_at:
            yaml_dict['last_modified_at'] = metadata.last_modified_at
        if metadata.version:
            yaml_dict['version'] = metadata.version
        if metadata.reviewers:
            yaml_dict['reviewers'] = metadata.reviewers
        if metadata.tags:
            yaml_dict['tags'] = metadata.tags
        if metadata.status:
            yaml_dict['status'] = metadata.status
        if metadata.related_docs:
            yaml_dict['related_docs'] = metadata.related_docs

        yaml_content = yaml.dump(yaml_dict, default_flow_style=False)
        return f"---\n{yaml_content}---"

    def _map_content_to_template_sections(
        self,
        template: str,
        sections: list[ContentSection]
    ) -> str:
        """Map existing content to template sections."""

        # Create mapping of section names to content
        content_map = {}
        for section in sections:
            # Normalize section names for matching
            normalized_name = section.name.lower().strip()
            content_map[normalized_name] = section.content.strip()

        # Replace template sections with existing content where applicable
        lines = template.split('\n')
        result_lines = []
        skip_template_content = False

        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                _ = len(header_match.group(1))  # Header level not used currently
                section_name = header_match.group(2).strip()
                normalized_section = section_name.lower()

                # Check if we have existing content for this section
                if normalized_section in content_map:
                    result_lines.append(line)  # Keep header
                    result_lines.append('')    # Add blank line
                    result_lines.append(content_map[normalized_section])
                    result_lines.append('')    # Add blank line
                    skip_template_content = True
                    _ = section_name  # Current section tracking not used currently
                else:
                    result_lines.append(line)
                    skip_template_content = False
                    _ = section_name  # Current section tracking not used currently
            else:
                # Check if we should skip template content
                if not skip_template_content:
                    result_lines.append(line)
                elif line.strip() == '' or line.startswith('#') or line.startswith('---'):
                    # Always include blank lines, headers, and metadata separators
                    result_lines.append(line)
                    if line.startswith('#'):
                        skip_template_content = False

        return '\n'.join(result_lines)

    def _clean_template_placeholders(self, content: str, metadata: DocumentMetadata) -> str:
        """Clean up template placeholders with actual values."""

        # Replace common placeholders
        replacements = {
            '[Document Title]': metadata.title,
            '[API Name]': metadata.title,
            '[Component Name]': 'Component',
            '[Group Name]': 'Group',
            'YYYY-MM-DD': datetime.now().strftime('%Y-%m-%d'),
            '[Author Name]': ', '.join(metadata.authors),
            '[Maintainer Name]': metadata.authors[0] if metadata.authors else 'MFC Team'
        }

        result = content
        for placeholder, replacement in replacements.items():
            result = result.replace(placeholder, replacement)

        return result

    def validate_standardized_document(self, file_path: str) -> dict[str, Any]:
        """Validate standardized document against standards."""

        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata_complete': False,
            'structure_valid': False,
            'content_preserved': False
        }

        try:
            with open(file_path) as f:
                content = f.read()

            # Validate metadata
            metadata = self.extract_metadata(content)
            if metadata:
                validation_results['metadata_complete'] = self._validate_metadata(
                    metadata, validation_results
                )
            else:
                validation_results['errors'].append('No metadata found')
                validation_results['valid'] = False

            # Validate structure
            sections = self.parse_document_structure(content)
            validation_results['structure_valid'] = self._validate_structure(
                sections, validation_results
            )

            # Check for content preservation indicators
            validation_results['content_preserved'] = len(sections) > 0

        except Exception as e:
            validation_results['errors'].append(f'Validation failed: {e}')
            validation_results['valid'] = False

        return validation_results

    def _validate_metadata(self, metadata: dict, results: dict) -> bool:
        """Validate metadata completeness."""
        required_fields = self.standards.get('metadata', {}).get('required_fields', [])

        missing_fields = []
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                missing_fields.append(field)

        if missing_fields:
            results['errors'].append(f'Missing required metadata fields: {missing_fields}')
            return False

        return True

    def _validate_structure(self, sections: list[ContentSection], results: dict) -> bool:
        """Validate document structure."""
        if not sections:
            results['errors'].append('No content sections found')
            return False

        # Check for reasonable section count
        if len(sections) < 2:
            results['warnings'].append('Document has very few sections')

        return True

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Documentation Template Engine"
    )
    parser.add_argument("action", choices=["apply", "validate", "detect-type"])
    parser.add_argument("file_path", help="Path to document file")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--preserve-content", action="store_true", default=True,
                        help="Preserve existing content when applying template")

    args = parser.parse_args()

    # Initialize template engine
    engine = DocumentationTemplateEngine()

    if args.action == "apply":
        success, original, standardized = engine.apply_template(
            args.file_path,
            preserve_content=args.preserve_content,
            dry_run=args.dry_run
        )

        if success:
            if args.dry_run:
                print("Template application preview:")
                print("=" * 50)
                print(standardized)
            else:
                print(f"Successfully applied template to {args.file_path}")
        else:
            print(f"Failed to apply template: {standardized}")

    elif args.action == "validate":
        results = engine.validate_standardized_document(args.file_path)

        print(f"Validation results for {args.file_path}:")
        print(f"Valid: {results['valid']}")
        print(f"Metadata complete: {results['metadata_complete']}")
        print(f"Structure valid: {results['structure_valid']}")

        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")

        if results['warnings']:
            print("Warnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")

    elif args.action == "detect-type":
        doc_type = engine.detect_document_type(args.file_path)
        print(f"Detected document type: {doc_type.value}")

if __name__ == "__main__":
    main()
