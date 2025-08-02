"""
Literature Database Query System

This module provides comprehensive literature database functionality
for automated parameter validation in the MFC optimization system.

Created: 2025-08-01
"""
import json
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .pubmed_connector import PubMedArticle, PubMedConnector, PubMedQuery


@dataclass
class ValidationQuery:
    """Container for validation query parameters."""

    parameter_name: str
    parameter_value: float
    units: str
    organism: str = ""
    experimental_conditions: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    max_literature_age_years: int = 20

@dataclass
class ValidationResult:
    """Container for validation results."""

    query: ValidationQuery
    validation_status: str  # "VALIDATED", "ACCEPTABLE", "NEEDS_REVIEW", "NO_DATA"
    confidence_score: float
    literature_matches: List[PubMedArticle] = field(default_factory=list)
    statistical_analysis: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
class LiteratureDatabase:
    """
    Comprehensive literature database for MFC parameter validation.
    
    Provides automated querying, caching, and validation against
    scientific literature through PubMed integration.
    """

    def __init__(self, db_path: str = "data/literature.db", cache_dir: str = "data/pubmed_cache"):
        """
        Initialize literature database.
        
        Args:
            db_path: Path to SQLite database file
            cache_dir: Directory for PubMed cache
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Initialize PubMed connector
        self.pubmed = PubMedConnector(cache_dir=cache_dir)

        # Thread lock for database operations
        self._db_lock = threading.Lock()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize SQLite database schema."""

        with sqlite3.connect(self.db_path) as conn:
            # Parameters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    units TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, category)
                )
            """)

            # Literature values table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS literature_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parameter_id INTEGER,
                    value REAL NOT NULL,
                    pmid TEXT,
                    organism TEXT,
                    experimental_conditions TEXT,
                    extraction_method TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parameter_id) REFERENCES parameters (id)
                )
            """)

            # Validation queries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE,
                    parameter_name TEXT,
                    parameter_value REAL,
                    units TEXT,
                    organism TEXT,
                    query_data TEXT,
                    result_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Article cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS article_metadata (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    journal TEXT,
                    publication_year INTEGER,
                    doi TEXT,
                    abstract TEXT,
                    relevance_score REAL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_param_name ON parameters(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lit_pmid ON literature_values(pmid)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON validation_queries(query_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_article_year ON article_metadata(publication_year)")

    def add_parameter(self, name: str, category: str, units: str, description: str = "") -> int:
        """Add parameter to database."""

        with self._db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO parameters (name, category, units, description)
                    VALUES (?, ?, ?, ?)
                """, (name, category, units, description))

                if cursor.rowcount == 0:
                    # Parameter already exists, get its ID
                    cursor = conn.execute("""
                        SELECT id FROM parameters WHERE name = ? AND category = ?
                    """, (name, category))
                    return cursor.fetchone()[0]
                else:
                    return cursor.lastrowid

    def add_literature_value(self, parameter_name: str, category: str, value: float,
                           pmid: str = "", organism: str = "",
                           experimental_conditions: List[str] = None,
                           extraction_method: str = "", confidence_score: float = 1.0) -> int:
        """Add literature value to database."""

        if experimental_conditions is None:
            experimental_conditions = []

        # Get or create parameter
        param_id = self.add_parameter(parameter_name, category, "", "")

        with self._db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO literature_values 
                    (parameter_id, value, pmid, organism, experimental_conditions, 
                     extraction_method, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (param_id, value, pmid, organism, json.dumps(experimental_conditions),
                      extraction_method, confidence_score))

                return cursor.lastrowid

    def query_literature_values(self, parameter_name: str, category: str = "") -> List[Dict[str, Any]]:
        """Query literature values for a parameter."""

        with sqlite3.connect(self.db_path) as conn:
            if category:
                query = """
                    SELECT lv.*, p.name, p.category, p.units, p.description
                    FROM literature_values lv
                    JOIN parameters p ON lv.parameter_id = p.id
                    WHERE p.name = ? AND p.category = ?
                    ORDER BY lv.confidence_score DESC, lv.created_at DESC
                """
                params = (parameter_name, category)
            else:
                query = """
                    SELECT lv.*, p.name, p.category, p.units, p.description
                    FROM literature_values lv
                    JOIN parameters p ON lv.parameter_id = p.id
                    WHERE p.name = ?
                    ORDER BY lv.confidence_score DESC, lv.created_at DESC
                """
                params = (parameter_name,)

            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]

            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse JSON fields
                result['experimental_conditions'] = json.loads(result['experimental_conditions'] or '[]')
                results.append(result)

            return results

    def search_literature_for_parameter(self, query: ValidationQuery) -> List[PubMedArticle]:
        """Search literature for parameter validation."""

        # Build search terms
        search_terms = [
            "microbial fuel cell",
            query.parameter_name.replace("_", " ")
        ]

        if query.organism:
            search_terms.append(query.organism)

        # Add experimental condition terms
        for condition in query.experimental_conditions:
            if len(condition.split()) <= 3:  # Only add short terms
                search_terms.append(condition)

        # Create PubMed query
        pubmed_query = PubMedQuery(
            search_terms=search_terms,
            mesh_terms=["Bioelectric Energy Sources", "Electrochemistry", "Biofilms"],
            publication_types=["Journal Article", "Research Support"],
            date_range=(f"{datetime.now().year - query.max_literature_age_years}/01/01", ""),
            max_results=50,
            sort_order="relevance"
        )

        # Search and fetch articles
        pmids = self.pubmed.search_literature(pubmed_query)
        articles = self.pubmed.fetch_article_details(pmids)

        # Store article metadata
        self._store_article_metadata(articles)

        return articles

    def _store_article_metadata(self, articles: List[PubMedArticle]):
        """Store article metadata in database."""

        with self._db_lock:
            with sqlite3.connect(self.db_path) as conn:
                for article in articles:
                    # Extract publication year
                    pub_year = None
                    if article.publication_date:
                        try:
                            pub_year = int(article.publication_date.split('-')[0])
                        except (ValueError, IndexError):
                            pass

                    conn.execute("""
                        INSERT OR REPLACE INTO article_metadata
                        (pmid, title, authors, journal, publication_year, doi, abstract, relevance_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        article.pmid,
                        article.title,
                        json.dumps(article.authors),
                        article.journal,
                        pub_year,
                        article.doi,
                        article.abstract,
                        1.0  # Default relevance score
                    ))

    def extract_parameter_values(self, articles: List[PubMedArticle],
                                query: ValidationQuery) -> List[Tuple[float, str, float]]:
        """
        Extract parameter values from article abstracts.
        
        Returns:
            List of (value, pmid, confidence) tuples
        """

        extracted_values = []

        # Simple pattern matching for common parameter formats
        import re

        # Create patterns for the parameter
        param_patterns = [
            rf"{query.parameter_name.replace('_', '\\s*')}.*?([0-9]+\.?[0-9]*)\s*{re.escape(query.units)}",
            rf"([0-9]+\.?[0-9]*)\s*{re.escape(query.units)}.*?{query.parameter_name.replace('_', '\\s*')}",
            rf"{query.parameter_name.replace('_', '\\s*')}.*?([0-9]+\.?[0-9]*)",
        ]

        for article in articles:
            text = f"{article.title} {article.abstract}".lower()

            for pattern in param_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    try:
                        value = float(match.group(1))

                        # Basic sanity check
                        if 0 < value < 1e10:  # Reasonable range
                            confidence = 0.6  # Base confidence for pattern matching

                            # Increase confidence if organism matches
                            if query.organism and query.organism.lower() in text:
                                confidence += 0.2

                            # Increase confidence if experimental conditions match
                            for condition in query.experimental_conditions:
                                if condition.lower() in text:
                                    confidence += 0.1

                            confidence = min(confidence, 1.0)
                            extracted_values.append((value, article.pmid, confidence))

                    except (ValueError, IndexError):
                        continue

        return extracted_values

    def validate_parameter(self, query: ValidationQuery) -> ValidationResult:
        """
        Comprehensive parameter validation against literature.
        
        Args:
            query: Validation query parameters
            
        Returns:
            ValidationResult with comprehensive analysis
        """

        # Check cache first
        cached_result = self._get_cached_validation(query)
        if cached_result:
            return cached_result

        # Search literature
        articles = self.search_literature_for_parameter(query)

        if not articles:
            result = ValidationResult(
                query=query,
                validation_status="NO_DATA",
                confidence_score=0.0,
                recommendations=["No literature data found for this parameter"]
            )
            self._cache_validation_result(query, result)
            return result

        # Extract parameter values from literature
        extracted_values = self.extract_parameter_values(articles, query)

        if not extracted_values:
            result = ValidationResult(
                query=query,
                validation_status="NO_DATA",
                confidence_score=0.0,
                literature_matches=articles,
                recommendations=["Literature found but no parameter values extracted"]
            )
            self._cache_validation_result(query, result)
            return result

        # Statistical analysis
        values = [v[0] for v in extracted_values]
        confidences = [v[2] for v in extracted_values]

        lit_mean = np.mean(values)
        lit_std = np.std(values)
        lit_median = np.median(values)
        lit_min, lit_max = np.min(values), np.max(values)

        # Calculate weighted statistics
        weights = np.array(confidences)
        weighted_mean = np.average(values, weights=weights)

        # Z-score analysis
        if lit_std > 0:
            z_score = (query.parameter_value - lit_mean) / lit_std
            weighted_z_score = (query.parameter_value - weighted_mean) / lit_std
        else:
            z_score = weighted_z_score = 0.0

        # Determine validation status
        within_range = lit_min <= query.parameter_value <= lit_max

        if abs(weighted_z_score) <= 1.0 and within_range:
            status = "VALIDATED"
            confidence = 0.9
        elif abs(weighted_z_score) <= 2.0 and within_range:
            status = "VALIDATED"
            confidence = 0.8
        elif abs(weighted_z_score) <= 2.0 or within_range:
            status = "ACCEPTABLE"
            confidence = 0.6
        elif abs(weighted_z_score) <= 3.0:
            status = "ACCEPTABLE"
            confidence = 0.4
        else:
            status = "NEEDS_REVIEW"
            confidence = 0.2

        # Generate recommendations
        recommendations = []
        if status == "NEEDS_REVIEW":
            if query.parameter_value > lit_max:
                recommendations.append(f"Value {query.parameter_value} exceeds literature maximum {lit_max:.3f}")
            elif query.parameter_value < lit_min:
                recommendations.append(f"Value {query.parameter_value} below literature minimum {lit_min:.3f}")
            recommendations.append(f"Consider using literature mean: {lit_mean:.3f}")

        elif status == "ACCEPTABLE":
            recommendations.append(f"Value within acceptable range. Literature mean: {lit_mean:.3f}")

        else:  # VALIDATED
            recommendations.append("Value well supported by literature")

        # Create result
        result = ValidationResult(
            query=query,
            validation_status=status,
            confidence_score=confidence,
            literature_matches=articles[:10],  # Limit to top 10
            statistical_analysis={
                'literature_mean': lit_mean,
                'literature_std': lit_std,
                'literature_median': lit_median,
                'literature_min': lit_min,
                'literature_max': lit_max,
                'weighted_mean': weighted_mean,
                'z_score': z_score,
                'weighted_z_score': weighted_z_score,
                'sample_size': len(values),
                'confidence_weights_mean': np.mean(confidences)
            },
            recommendations=recommendations
        )

        # Cache result
        self._cache_validation_result(query, result)

        return result

    def _get_query_hash(self, query: ValidationQuery) -> str:
        """Generate hash for query caching."""

        import hashlib
        query_dict = asdict(query)
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()

    def _get_cached_validation(self, query: ValidationQuery) -> Optional[ValidationResult]:
        """Check for cached validation result."""

        query_hash = self._get_query_hash(query)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT result_data FROM validation_queries 
                WHERE query_hash = ? AND created_at > datetime('now', '-7 days')
            """, (query_hash,))

            result = cursor.fetchone()
            if result:
                try:
                    result_dict = json.loads(result[0])
                    # Reconstruct ValidationResult (simplified)
                    return ValidationResult(
                        query=query,
                        validation_status=result_dict['validation_status'],
                        confidence_score=result_dict['confidence_score'],
                        literature_matches=[],  # Not cached for size reasons
                        statistical_analysis=result_dict.get('statistical_analysis', {}),
                        recommendations=result_dict.get('recommendations', []),
                        created_at=result_dict.get('created_at', '')
                    )
                except (json.JSONDecodeError, KeyError):
                    pass

        return None

    def _cache_validation_result(self, query: ValidationQuery, result: ValidationResult):
        """Cache validation result."""

        query_hash = self._get_query_hash(query)

        # Prepare result data (exclude large objects)
        result_data = {
            'validation_status': result.validation_status,
            'confidence_score': result.confidence_score,
            'statistical_analysis': result.statistical_analysis,
            'recommendations': result.recommendations,
            'created_at': result.created_at,
            'literature_count': len(result.literature_matches)
        }

        with self._db_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO validation_queries
                    (query_hash, parameter_name, parameter_value, units, organism, query_data, result_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_hash,
                    query.parameter_name,
                    query.parameter_value,
                    query.units,
                    query.organism,
                    json.dumps(asdict(query)),
                    json.dumps(result_data)
                ))

    def batch_validate_parameters(self, queries: List[ValidationQuery],
                                 max_workers: int = 3) -> List[ValidationResult]:
        """
        Validate multiple parameters in parallel.
        
        Args:
            queries: List of validation queries
            max_workers: Maximum number of concurrent validations
            
        Returns:
            List of validation results
        """

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all validation tasks
            future_to_query = {
                executor.submit(self.validate_parameter, query): query
                for query in queries
            }

            # Collect results
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Validation failed for {query.parameter_name}: {e}")
                    # Create error result
                    error_result = ValidationResult(
                        query=query,
                        validation_status="ERROR",
                        confidence_score=0.0,
                        recommendations=[f"Validation error: {str(e)}"]
                    )
                    results.append(error_result)

        return results

    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""

        summary = {
            'total_parameters': len(results),
            'validated': 0,
            'acceptable': 0,
            'needs_review': 0,
            'no_data': 0,
            'errors': 0,
            'overall_confidence': 0.0,
            'literature_coverage': 0.0
        }

        confidence_scores = []
        has_literature = 0

        for result in results:
            status = result.validation_status
            if status == 'VALIDATED':
                summary['validated'] += 1
            elif status == 'ACCEPTABLE':
                summary['acceptable'] += 1
            elif status == 'NEEDS_REVIEW':
                summary['needs_review'] += 1
            elif status == 'NO_DATA':
                summary['no_data'] += 1
            else:  # ERROR
                summary['errors'] += 1

            if result.confidence_score > 0:
                confidence_scores.append(result.confidence_score)

            if result.literature_matches:
                has_literature += 1

        # Calculate overall metrics
        if confidence_scores:
            summary['overall_confidence'] = np.mean(confidence_scores)

        summary['literature_coverage'] = has_literature / len(results) if results else 0.0

        return {
            'summary': summary,
            'results': [asdict(result) for result in results],
            'generated_at': datetime.now().isoformat(),
            'database_stats': self.get_database_statistics()
        }

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""

        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Count parameters
            cursor = conn.execute("SELECT COUNT(*) FROM parameters")
            stats['total_parameters'] = cursor.fetchone()[0]

            # Count literature values
            cursor = conn.execute("SELECT COUNT(*) FROM literature_values")
            stats['total_literature_values'] = cursor.fetchone()[0]

            # Count cached queries
            cursor = conn.execute("SELECT COUNT(*) FROM validation_queries")
            stats['cached_validations'] = cursor.fetchone()[0]

            # Count articles
            cursor = conn.execute("SELECT COUNT(*) FROM article_metadata")
            stats['cached_articles'] = cursor.fetchone()[0]

            # Get PubMed connector stats
            stats['pubmed_stats'] = self.pubmed.get_cache_statistics()

        return stats

    def export_database(self, filepath: str):
        """Export database to JSON format."""

        export_data = {
            'parameters': [],
            'literature_values': [],
            'articles': [],
            'export_date': datetime.now().isoformat(),
            'statistics': self.get_database_statistics()
        }

        with sqlite3.connect(self.db_path) as conn:
            # Export parameters
            cursor = conn.execute("SELECT * FROM parameters")
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                export_data['parameters'].append(dict(zip(columns, row)))

            # Export literature values
            cursor = conn.execute("""
                SELECT lv.*, p.name as parameter_name, p.category
                FROM literature_values lv
                JOIN parameters p ON lv.parameter_id = p.id
            """)
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                record['experimental_conditions'] = json.loads(record['experimental_conditions'] or '[]')
                export_data['literature_values'].append(record)

            # Export article metadata
            cursor = conn.execute("SELECT * FROM article_metadata LIMIT 1000")  # Limit for size
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                record['authors'] = json.loads(record['authors'] or '[]')
                export_data['articles'].append(record)

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"📊 Database exported to: {filepath}")
if __name__ == "__main__":
    # Example usage
    print("🔬 Literature Database Test")
    print("=" * 40)

    # Initialize database
    lit_db = LiteratureDatabase()

    # Example validation query
    query = ValidationQuery(
        parameter_name="max_current_density",
        parameter_value=2.0,
        units="A/m²",
        organism="Shewanella oneidensis",
        experimental_conditions=["lactate", "anaerobic"],
        confidence_threshold=0.7
    )

    print(f"Validating parameter: {query.parameter_name}")
    print(f"Value: {query.parameter_value} {query.units}")

    # Run validation
    result = lit_db.validate_parameter(query)

    print("\n📊 Validation Result:")
    print(f"  Status: {result.validation_status}")
    print(f"  Confidence: {result.confidence_score:.2f}")
    print(f"  Literature matches: {len(result.literature_matches)}")

    if result.statistical_analysis:
        stats = result.statistical_analysis
        print(f"  Literature mean: {stats.get('literature_mean', 0):.3f}")
        print(f"  Literature range: {stats.get('literature_min', 0):.3f} - {stats.get('literature_max', 0):.3f}")

    print("\n💡 Recommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")

    # Show database statistics
    stats = lit_db.get_database_statistics()
    print("\n📈 Database Statistics:")
    for key, value in stats.items():
        if key != 'pubmed_stats':
            print(f"  {key}: {value}")
