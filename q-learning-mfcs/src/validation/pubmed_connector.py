"""
PubMed API Connector for Literature Validation

This module provides automated literature database queries through the PubMed API
for parameter validation in the MFC optimization system.

Created: 2025-08-01
"""
import hashlib
import json
import sqlite3
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests


@dataclass
class PubMedQuery:
    """Container for PubMed search query parameters."""

    search_terms: list[str]
    mesh_terms: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)
    date_range: tuple[str, str] = ("", "")  # (start_date, end_date) format: YYYY/MM/DD
    max_results: int = 100
    sort_order: str = "relevance"  # "relevance", "pub_date", "first_author"

@dataclass
class PubMedArticle:
    """Container for PubMed article information."""

    pmid: str
    title: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    publication_date: str = ""
    doi: str = ""
    abstract: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

class PubMedConnector:
    """
    PubMed API connector for automated literature searches.

    Provides rate-limited access to PubMed database with caching
    for parameter validation queries.
    """

    def __init__(self, cache_dir: str = "data/pubmed_cache", rate_limit: float = 0.34):
        """
        Initialize PubMed connector.

        Args:
            cache_dir: Directory for caching query results
            rate_limit: Minimum seconds between API calls (max 3 requests/second)
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.rate_limit = rate_limit
        self.last_request_time = 0.0

        # Setup cache directory and database
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db_path = self.cache_dir / "pubmed_cache.db"
        self._init_cache_database()

        # API statistics
        self.api_calls_made = 0
        self.cache_hits = 0
        self.session = requests.Session()

    def _init_cache_database(self):
        """Initialize SQLite cache database."""

        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_terms TEXT,
                    search_date TEXT,
                    result_count INTEGER,
                    pmids TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS article_cache (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    journal TEXT,
                    publication_date TEXT,
                    doi TEXT,
                    abstract TEXT,
                    mesh_terms TEXT,
                    keywords TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_search_date ON search_cache(search_date)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fetched_at ON article_cache(fetched_at)
            """)

    def _enforce_rate_limit(self):
        """Enforce API rate limiting."""

        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _generate_query_hash(self, query: PubMedQuery) -> str:
        """Generate hash for query caching."""

        query_string = f"{query.search_terms}{query.mesh_terms}{query.date_range}{query.max_results}"
        return hashlib.md5(query_string.encode()).hexdigest()

    def _check_search_cache(self, query_hash: str, max_age_days: int = 7) -> list[str] | None:
        """Check if search results are cached and recent."""

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.execute("""
                SELECT pmids FROM search_cache
                WHERE query_hash = ? AND created_at > ?
            """, (query_hash, cutoff_date))

            result = cursor.fetchone()
            if result:
                self.cache_hits += 1
                return result[0].split(',') if result[0] else []

        return None

    def _cache_search_results(self, query_hash: str, query: PubMedQuery, pmids: list[str]):
        """Cache search results."""

        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO search_cache
                (query_hash, query_terms, search_date, result_count, pmids)
                VALUES (?, ?, ?, ?, ?)
            """, (
                query_hash,
                json.dumps(query.search_terms),
                datetime.now().isoformat(),
                len(pmids),
                ','.join(pmids)
            ))

    def _check_article_cache(self, pmid: str, max_age_days: int = 30) -> PubMedArticle | None:
        """Check if article details are cached."""

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM article_cache
                WHERE pmid = ? AND fetched_at > ?
            """, (pmid, cutoff_date))

            result = cursor.fetchone()
            if result:
                self.cache_hits += 1
                return PubMedArticle(
                    pmid=result[0],
                    title=result[1],
                    authors=json.loads(result[2]) if result[2] else [],
                    journal=result[3] or "",
                    publication_date=result[4] or "",
                    doi=result[5] or "",
                    abstract=result[6] or "",
                    mesh_terms=json.loads(result[7]) if result[7] else [],
                    keywords=json.loads(result[8]) if result[8] else []
                )

        return None

    def _cache_article(self, article: PubMedArticle):
        """Cache article details."""

        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO article_cache
                (pmid, title, authors, journal, publication_date, doi, abstract, mesh_terms, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article.pmid,
                article.title,
                json.dumps(article.authors),
                article.journal,
                article.publication_date,
                article.doi,
                article.abstract,
                json.dumps(article.mesh_terms),
                json.dumps(article.keywords)
            ))

    def search_literature(self, query: PubMedQuery) -> list[str]:
        """
        Search PubMed for articles matching query parameters.

        Args:
            query: PubMed query parameters

        Returns:
            List of PMIDs matching the search criteria
        """

        # Check cache first
        query_hash = self._generate_query_hash(query)
        cached_pmids = self._check_search_cache(query_hash)
        if cached_pmids is not None:
            return cached_pmids

        # Build search query
        search_terms = []

        # Add search terms
        for term in query.search_terms:
            search_terms.append(f'"{term}"[Title/Abstract]')

        # Add MeSH terms
        for mesh_term in query.mesh_terms:
            search_terms.append(f'"{mesh_term}"[MeSH Terms]')

        # Add publication types
        for pub_type in query.publication_types:
            search_terms.append(f'"{pub_type}"[Publication Type]')

        # Add date range
        if query.date_range[0] and query.date_range[1]:
            search_terms.append(f'("{query.date_range[0]}"[Date - Publication] : "{query.date_range[1]}"[Date - Publication])')

        search_query = " AND ".join(search_terms)

        # Make API request
        self._enforce_rate_limit()

        params = {
            'db': 'pubmed',
            'term': search_query,
            'retmax': query.max_results,
            'sort': query.sort_order,
            'retmode': 'json'
        }

        try:
            response = self.session.get(f"{self.base_url}/esearch.fcgi", params=params)
            response.raise_for_status()
            self.api_calls_made += 1

            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])

            # Cache results
            self._cache_search_results(query_hash, query, pmids)

            return pmids

        except requests.RequestException as e:
            warnings.warn(f"PubMed search failed: {e}", stacklevel=2)
            return []

    def fetch_article_details(self, pmids: list[str]) -> list[PubMedArticle]:
        """
        Fetch detailed article information for given PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of PubMedArticle objects with detailed information
        """

        articles = []
        uncached_pmids = []

        # Check cache for each PMID
        for pmid in pmids:
            cached_article = self._check_article_cache(pmid)
            if cached_article:
                articles.append(cached_article)
            else:
                uncached_pmids.append(pmid)

        # Fetch uncached articles in batches
        batch_size = 200  # PubMed recommended batch size
        for i in range(0, len(uncached_pmids), batch_size):
            batch_pmids = uncached_pmids[i:i + batch_size]
            batch_articles = self._fetch_article_batch(batch_pmids)
            articles.extend(batch_articles)

        return articles

    def _fetch_article_batch(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch a batch of articles from PubMed."""

        if not pmids:
            return []

        self._enforce_rate_limit()

        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }

        try:
            response = self.session.get(f"{self.base_url}/efetch.fcgi", params=params)
            response.raise_for_status()
            self.api_calls_made += 1

            return self._parse_pubmed_xml(response.text)

        except requests.RequestException as e:
            warnings.warn(f"PubMed fetch failed: {e}", stacklevel=2)
            return []

    def _parse_pubmed_xml(self, xml_text: str) -> list[PubMedArticle]:
        """Parse PubMed XML response into PubMedArticle objects."""

        articles = []

        try:
            root = ET.fromstring(xml_text)

            for article_elem in root.findall('.//PubmedArticle'):
                article = self._parse_single_article(article_elem)
                if article:
                    articles.append(article)
                    self._cache_article(article)

        except ET.ParseError as e:
            warnings.warn(f"XML parsing failed: {e}", stacklevel=2)

        return articles

    def _parse_single_article(self, article_elem) -> PubMedArticle | None:
        """Parse single article XML element."""

        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""

            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""

            # Extract authors
            authors = []
            for author_elem in article_elem.findall('.//Author'):
                lastname = author_elem.find('LastName')
                forename = author_elem.find('ForeName')
                if lastname is not None:
                    author_name = lastname.text
                    if forename is not None:
                        author_name = f"{forename.text} {author_name}"
                    authors.append(author_name)

            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""

            # Extract publication date
            pub_date_elem = article_elem.find('.//PubDate')
            pub_date = self._extract_publication_date(pub_date_elem)

            # Extract DOI
            doi = ""
            for id_elem in article_elem.findall('.//ArticleId'):
                id_type = id_elem.get('IdType')
                if id_type == 'doi':
                    doi = id_elem.text
                    break

            # Extract abstract
            abstract_elem = article_elem.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ""

            # Extract MeSH terms
            mesh_terms = []
            for mesh_elem in article_elem.findall('.//MeshHeading/DescriptorName'):
                mesh_terms.append(mesh_elem.text)

            # Extract keywords
            keywords = []
            for keyword_elem in article_elem.findall('.//Keyword'):
                keywords.append(keyword_elem.text)

            return PubMedArticle(
                pmid=pmid,
                title=title,
                authors=authors,
                journal=journal,
                publication_date=pub_date,
                doi=doi,
                abstract=abstract,
                mesh_terms=mesh_terms,
                keywords=keywords
            )

        except Exception as e:
            warnings.warn(f"Failed to parse article: {e}", stacklevel=2)
            return None

    def _extract_publication_date(self, pub_date_elem) -> str:
        """Extract publication date from PubDate XML element."""

        if pub_date_elem is None:
            return ""

        year_elem = pub_date_elem.find('Year')
        month_elem = pub_date_elem.find('Month')
        day_elem = pub_date_elem.find('Day')

        date_parts = []
        if year_elem is not None:
            date_parts.append(year_elem.text)
        if month_elem is not None:
            date_parts.append(month_elem.text)
        if day_elem is not None:
            date_parts.append(day_elem.text)

        return "-".join(date_parts)

    def search_mfc_parameters(self, parameter_name: str, organism: str = "") -> list[PubMedArticle]:
        """
        Search for literature related to specific MFC parameters.

        Args:
            parameter_name: Name of the parameter to search for
            organism: Optional organism name to include in search

        Returns:
            List of relevant articles
        """

        # Build MFC-specific search terms
        base_terms = [
            "microbial fuel cell",
            parameter_name
        ]

        if organism:
            base_terms.append(organism)

        # Add relevant MeSH terms
        mesh_terms = [
            "Bioelectric Energy Sources",
            "Electrochemistry",
            "Biofilms"
        ]

        query = PubMedQuery(
            search_terms=base_terms,
            mesh_terms=mesh_terms,
            publication_types=["Journal Article", "Research Support"],
            max_results=50,
            sort_order="relevance"
        )

        pmids = self.search_literature(query)
        return self.fetch_article_details(pmids)

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache and API usage statistics."""

        with sqlite3.connect(self.cache_db_path) as conn:
            # Count cached searches
            cursor = conn.execute("SELECT COUNT(*) FROM search_cache")
            cached_searches = cursor.fetchone()[0]

            # Count cached articles
            cursor = conn.execute("SELECT COUNT(*) FROM article_cache")
            cached_articles = cursor.fetchone()[0]

        return {
            'api_calls_made': self.api_calls_made,
            'cache_hits': self.cache_hits,
            'cached_searches': cached_searches,
            'cached_articles': cached_articles,
            'cache_hit_rate': self.cache_hits / max(self.api_calls_made + self.cache_hits, 1)
        }

    def cleanup_old_cache(self, max_age_days: int = 90):
        """Remove old cache entries."""

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        with sqlite3.connect(self.cache_db_path) as conn:
            # Clean search cache
            conn.execute("DELETE FROM search_cache WHERE created_at < ?", (cutoff_date,))

            # Clean article cache
            conn.execute("DELETE FROM article_cache WHERE fetched_at < ?", (cutoff_date,))

            conn.commit()
if __name__ == "__main__":
    # Example usage
    print("🔬 PubMed Connector Test")
    print("=" * 40)

    # Initialize connector
    connector = PubMedConnector()

    # Search for MFC-related articles
    print("Searching for MFC biofilm articles...")
    articles = connector.search_mfc_parameters("biofilm", "Shewanella oneidensis")

    print(f"Found {len(articles)} articles")

    for i, article in enumerate(articles[:3]):  # Show first 3
        print(f"\n{i+1}. {article.title}")
        print(f"   Authors: {', '.join(article.authors[:3])}{'...' if len(article.authors) > 3 else ''}")
        print(f"   Journal: {article.journal} ({article.publication_date})")
        print(f"   DOI: {article.doi}")

    # Show statistics
    stats = connector.get_cache_statistics()
    print("\n📊 Cache Statistics:")
    print(f"  API calls: {stats['api_calls_made']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Hit rate: {stats['cache_hit_rate']:.2%}")
