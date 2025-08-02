#!/usr/bin/env python3
"""
Enhanced Security Git Guardian Integration.

Provides advanced security features to prevent malicious code from entering 
the repository via fragmented commits. Integrates the git-commit-guardian 
with existing chunking logic and adds cross-chunk validation.

Created: 2025-08-01
"""

import hashlib
import json
import logging
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .git_guardian import GitGuardianClient, request_guardian_commit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CommitFragment:
    """Represents a single commit fragment in a series."""
    fragment_id: str
    file_path: str
    content_hash: str
    content_preview: str  # First 200 chars
    commit_message: str
    timestamp: datetime
    change_type: str  # create, edit, delete
    parent_operation_id: str
    security_score: float
    suspicious_patterns: list[str]


@dataclass
class FragmentSeries:
    """Represents a series of related commit fragments."""
    operation_id: str
    file_path: str
    total_fragments: int
    fragments: list[CommitFragment]
    start_time: datetime
    last_update: datetime
    cumulative_security_score: float
    is_complete: bool
    rollback_points: list[str]  # Commit SHAs for rollback


class EnhancedSecurityGuardian:
    """Enhanced security guardian with fragment validation."""

    def __init__(self, db_path: str = ".claude/security/fragment_tracking.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.guardian_client = GitGuardianClient()
        self.init_database()

        # Security thresholds
        self.max_fragment_time_window = timedelta(hours=2)
        self.max_security_score_per_fragment = 0.2
        self.max_cumulative_security_score = 0.4
        self.suspicious_pattern_threshold = 2

        # Malicious pattern detection
        self.malicious_patterns = {
            'obfuscated_code': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__\s*\(',
                r'getattr\s*\([^,]+,\s*["\'][^"\']*["\']',
                r'chr\s*\(\s*\d+\s*\)',
                r'bytes\.fromhex\s*\(',
            ],
            'network_activity': [
                r'urllib\.request',
                r'requests\.(get|post|put|delete)',
                r'socket\.(socket|connect)',
                r'subprocess\.(run|call|Popen)',
                r'os\.system\s*\(',
            ],
            'file_system_access': [
                r'open\s*\([^)]*["\'][\/\\]',  # Absolute paths
                r'os\.(remove|unlink|rmdir|system)',
                r'shutil\.(rmtree|move|copy)',
                r'pathlib\.Path\([^)]*\)\.unlink',
            ],
            'crypto_operations': [
                r'hashlib\.(md5|sha1|sha256)',
                r'base64\.(b64encode|b64decode)',
                r'Crypto\.',
                r'cryptography\.',
            ],
            'data_exfiltration': [
                r'json\.dumps\s*\([^)]*\)',
                r'pickle\.(dumps|loads)',
                r'marshal\.(dumps|loads)',
                r'codecs\.(encode|decode)',
            ]
        }

    def init_database(self) -> None:
        """Initialize SQLite database for fragment tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fragment_series (
                    operation_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    total_fragments INTEGER,
                    start_time TEXT NOT NULL,
                    last_update TEXT NOT NULL,
                    cumulative_security_score REAL DEFAULT 0.0,
                    is_complete BOOLEAN DEFAULT FALSE,
                    rollback_points TEXT DEFAULT '[]'
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS commit_fragments (
                    fragment_id TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    content_preview TEXT,
                    commit_message TEXT,
                    timestamp TEXT NOT NULL,
                    change_type TEXT,
                    security_score REAL DEFAULT 0.0,
                    suspicious_patterns TEXT DEFAULT '[]',
                    FOREIGN KEY (operation_id) REFERENCES fragment_series (operation_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_operation_id 
                ON commit_fragments (operation_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path 
                ON fragment_series (file_path)
            """)

    def generate_operation_id(self, file_path: str, change_type: str) -> str:
        """Generate unique operation ID for a fragment series."""
        timestamp = datetime.now().isoformat()
        data = f"{file_path}:{change_type}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def analyze_security_patterns(self, content: str) -> tuple[float, list[str]]:
        """Analyze content for suspicious security patterns."""
        suspicious_patterns = []
        security_score = 0.0

        for category, patterns in self.malicious_patterns.items():
            category_matches = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    category_matches += len(matches)
                    suspicious_patterns.append(f"{category}: {pattern} ({len(matches)} matches)")

            # Score based on category severity and frequency
            category_weights = {
                'obfuscated_code': 0.4,
                'network_activity': 0.3,
                'file_system_access': 0.25,
                'crypto_operations': 0.15,
                'data_exfiltration': 0.35
            }

            if category_matches > 0:
                weight = category_weights.get(category, 0.1)
                category_score = min(category_matches * weight, 0.5)
                security_score += category_score

        return min(security_score, 1.0), suspicious_patterns

    def start_fragment_series(self, file_path: str, change_type: str,
                            total_fragments: int | None = None) -> str:
        """Start a new fragment series for tracking."""
        operation_id = self.generate_operation_id(file_path, change_type)
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO fragment_series 
                (operation_id, file_path, total_fragments, start_time, last_update)
                VALUES (?, ?, ?, ?, ?)
            """, (operation_id, file_path, total_fragments, now, now))

        logger.info(f"Started fragment series {operation_id} for {file_path}")
        return operation_id

    def add_commit_fragment(self, operation_id: str, file_path: str,
                          content: str, commit_message: str,
                          change_type: str) -> CommitFragment:
        """Add a commit fragment to an existing series."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        content_preview = content[:200] + "..." if len(content) > 200 else content
        fragment_id = f"{operation_id}_{len(self.get_fragments(operation_id))}"

        security_score, suspicious_patterns = self.analyze_security_patterns(content)

        fragment = CommitFragment(
            fragment_id=fragment_id,
            file_path=file_path,
            content_hash=content_hash,
            content_preview=content_preview,
            commit_message=commit_message,
            timestamp=datetime.now(),
            change_type=change_type,
            parent_operation_id=operation_id,
            security_score=security_score,
            suspicious_patterns=suspicious_patterns
        )

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO commit_fragments 
                (fragment_id, operation_id, file_path, content_hash, content_preview,
                 commit_message, timestamp, change_type, security_score, suspicious_patterns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fragment_id, operation_id, file_path, content_hash, content_preview,
                commit_message, fragment.timestamp.isoformat(), change_type,
                security_score, json.dumps(suspicious_patterns)
            ))

            # Update series cumulative score
            conn.execute("""
                UPDATE fragment_series 
                SET cumulative_security_score = cumulative_security_score + ?,
                    last_update = ?
                WHERE operation_id = ?
            """, (security_score, datetime.now().isoformat(), operation_id))

        logger.info(f"Added fragment {fragment_id} with security score {security_score}")
        return fragment

    def get_fragments(self, operation_id: str) -> list[CommitFragment]:
        """Get all fragments for an operation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT fragment_id, file_path, content_hash, content_preview,
                       commit_message, timestamp, change_type, security_score,
                       suspicious_patterns
                FROM commit_fragments 
                WHERE operation_id = ?
                ORDER BY timestamp
            """, (operation_id,))

            fragments = []
            for row in cursor.fetchall():
                fragment = CommitFragment(
                    fragment_id=row[0],
                    file_path=row[1],
                    content_hash=row[2],
                    content_preview=row[3],
                    commit_message=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    change_type=row[6],
                    parent_operation_id=operation_id,
                    security_score=row[7],
                    suspicious_patterns=json.loads(row[8])
                )
                fragments.append(fragment)

            return fragments

    def validate_fragment_series(self, operation_id: str) -> tuple[bool, list[str]]:
        """Validate an entire fragment series for security."""
        fragments = self.get_fragments(operation_id)
        if not fragments:
            return True, []

        issues = []

        # Get series metadata
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT cumulative_security_score, start_time, last_update
                FROM fragment_series WHERE operation_id = ?
            """, (operation_id,))

            row = cursor.fetchone()
            if not row:
                return False, ["Fragment series not found"]

            cumulative_score, start_time, last_update = row
            start_dt = datetime.fromisoformat(start_time)
            last_dt = datetime.fromisoformat(last_update)

        # Check time window
        if last_dt - start_dt > self.max_fragment_time_window:
            issues.append(f"Fragment series spans {last_dt - start_dt}, exceeding maximum window")

        # Check cumulative security score
        if cumulative_score > self.max_cumulative_security_score:
            issues.append(f"Cumulative security score {cumulative_score:.2f} exceeds threshold")

        # Check individual fragment scores
        high_score_fragments = [f for f in fragments
                              if f.security_score > self.max_security_score_per_fragment]
        if high_score_fragments:
            issues.append(f"{len(high_score_fragments)} fragments exceed individual security threshold")

        # Check for pattern concentration
        all_patterns = []
        for fragment in fragments:
            all_patterns.extend(fragment.suspicious_patterns)

        if len(all_patterns) > self.suspicious_pattern_threshold:
            issues.append(f"Total suspicious patterns ({len(all_patterns)}) exceed threshold")

        # Cross-fragment analysis for obfuscation attempts
        content_hashes = [f.content_hash for f in fragments]
        if len(set(content_hashes)) < len(content_hashes) * 0.8:  # Too much similarity
            issues.append("Suspicious content similarity across fragments (possible obfuscation)")

        return len(issues) == 0, issues

    def secure_chunked_commit(self, file_path: str, content_chunks: list[str],
                            commit_message_prefix: str, change_type: str) -> tuple[bool, str]:
        """Execute secure chunked commits with cross-fragment validation."""
        if not content_chunks:
            return True, "No chunks to commit"

        # Start fragment series
        operation_id = self.start_fragment_series(file_path, change_type, len(content_chunks))
        rollback_commits = []

        try:
            # Process each chunk
            for i, chunk in enumerate(content_chunks):
                chunk_msg = f"{commit_message_prefix}chunk {i+1}/{len(content_chunks)}"

                # Add fragment for tracking
                fragment = self.add_commit_fragment(
                    operation_id, file_path, chunk, chunk_msg, change_type
                )

                # Validate individual fragment
                if fragment.security_score > self.max_security_score_per_fragment:
                    raise SecurityError(f"Fragment {i+1} security score too high: {fragment.security_score}")

                # Validate series so far
                is_valid, issues = self.validate_fragment_series(operation_id)
                if not is_valid:
                    raise SecurityError(f"Fragment series validation failed: {'; '.join(issues)}")

                # Create temporary file with chunk content
                with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False) as tmp_file:
                    tmp_file.write(chunk)
                    tmp_file.flush()

                    # Request guardian commit for this chunk
                    success = request_guardian_commit(
                        files=[tmp_file.name],
                        commit_message=chunk_msg,
                        change_type=change_type,
                        auto_generated=True
                    )

                    if not success:
                        raise SecurityError(f"Guardian rejected chunk {i+1}")

                    # Get the commit SHA for rollback tracking
                    try:
                        result = subprocess.run(
                            ['git', 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True
                        )
                        rollback_commits.append(result.stdout.strip())
                    except subprocess.CalledProcessError:
                        logger.warning("Could not get commit SHA for rollback tracking")

            # Final validation of complete series
            is_valid, issues = self.validate_fragment_series(operation_id)
            if not is_valid:
                # Rollback all commits
                self.rollback_fragment_series(operation_id, rollback_commits)
                return False, f"Final validation failed: {'; '.join(issues)}"

            # Mark series as complete
            self.complete_fragment_series(operation_id)
            return True, f"Successfully committed {len(content_chunks)} chunks"

        except SecurityError as e:
            # Rollback any commits made so far
            self.rollback_fragment_series(operation_id, rollback_commits)
            return False, f"Security validation failed: {str(e)}"

        except Exception as e:
            # Rollback on any other error
            self.rollback_fragment_series(operation_id, rollback_commits)
            return False, f"Commit failed: {str(e)}"

    def rollback_fragment_series(self, operation_id: str, commit_shas: list[str]) -> None:
        """Rollback a fragment series by reverting commits."""
        if not commit_shas:
            logger.info(f"No commits to rollback for operation {operation_id}")
            return

        try:
            # Rollback commits in reverse order
            for commit_sha in reversed(commit_shas):
                subprocess.run(
                    ['git', 'revert', '--no-edit', commit_sha],
                    check=True, capture_output=True
                )
                logger.info(f"Reverted commit {commit_sha}")

            # Mark series as rolled back
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE fragment_series 
                    SET is_complete = TRUE, last_update = ?
                    WHERE operation_id = ?
                """, (datetime.now().isoformat(), operation_id))

            logger.info(f"Successfully rolled back fragment series {operation_id}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to rollback operation {operation_id}: {e}")
            # Store rollback points for manual intervention
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE fragment_series 
                    SET rollback_points = ?
                    WHERE operation_id = ?
                """, (json.dumps(commit_shas), operation_id))

    def complete_fragment_series(self, operation_id: str) -> None:
        """Mark a fragment series as complete."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE fragment_series 
                SET is_complete = TRUE, last_update = ?
                WHERE operation_id = ?
            """, (datetime.now().isoformat(), operation_id))

        logger.info(f"Completed fragment series {operation_id}")

    def cleanup_old_series(self, max_age_days: int = 7) -> None:
        """Clean up old fragment series from database."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT operation_id FROM fragment_series 
                WHERE last_update < ? AND is_complete = TRUE
            """, (cutoff_date.isoformat(),))

            old_operations = [row[0] for row in cursor.fetchall()]

            for operation_id in old_operations:
                conn.execute("DELETE FROM commit_fragments WHERE operation_id = ?", (operation_id,))
                conn.execute("DELETE FROM fragment_series WHERE operation_id = ?", (operation_id,))

            logger.info(f"Cleaned up {len(old_operations)} old fragment series")

    def get_security_report(self, days: int = 7) -> dict[str, Any]:
        """Generate security report for recent fragment series."""
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            # Get series statistics
            cursor = conn.execute("""
                SELECT COUNT(*), AVG(cumulative_security_score), 
                       SUM(CASE WHEN is_complete THEN 1 ELSE 0 END)
                FROM fragment_series 
                WHERE start_time > ?
            """, (cutoff_date.isoformat(),))

            total_series, avg_score, completed_series = cursor.fetchone()

            # Get top security issues
            cursor = conn.execute("""
                SELECT f.suspicious_patterns, f.security_score
                FROM commit_fragments f
                JOIN fragment_series s ON f.operation_id = s.operation_id
                WHERE s.start_time > ? AND f.security_score > 0
                ORDER BY f.security_score DESC
                LIMIT 10
            """, (cutoff_date.isoformat(),))

            high_risk_fragments = []
            for patterns_json, score in cursor.fetchall():
                patterns = json.loads(patterns_json)
                high_risk_fragments.append({
                    'security_score': score,
                    'suspicious_patterns': patterns
                })

        return {
            'period_days': days,
            'total_fragment_series': total_series or 0,
            'completed_series': completed_series or 0,
            'average_security_score': avg_score or 0.0,
            'high_risk_fragments': high_risk_fragments,
            'timestamp': datetime.now().isoformat()
        }


class SecurityError(Exception):
    """Exception raised for security validation failures."""
    pass


# Integration functions for existing hook system
def secure_chunked_edit(file_path: str, old_content: str, new_content: str,
                       config: dict[str, Any]) -> bool:
    """Enhanced chunked edit with security validation."""
    guardian = EnhancedSecurityGuardian()

    # Calculate chunks based on config
    max_lines = config.get('max_lines_per_chunk', 50)
    lines = new_content.split('\n')

    chunks = []
    current_chunk = []

    for line in lines:
        current_chunk.append(line)
        if len(current_chunk) >= max_lines:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    # Execute secure chunked commit
    prefix = config.get('commit_message_prefix', 'Auto-commit: ')
    success, message = guardian.secure_chunked_commit(
        file_path, chunks, f"{prefix}edit {file_path} - ", 'edit'
    )

    if success:
        logger.info(f"Secure chunked edit completed: {message}")
    else:
        logger.error(f"Secure chunked edit failed: {message}")

    return success


def secure_chunked_file_creation(file_path: str, content: str,
                                config: dict[str, Any]) -> bool:
    """Enhanced chunked file creation with security validation."""
    guardian = EnhancedSecurityGuardian()

    # Calculate chunks based on config
    max_lines = config.get('max_lines_per_chunk', 100)
    lines = content.split('\n')

    chunks = []
    current_chunk = []

    for line in lines:
        current_chunk.append(line)
        if len(current_chunk) >= max_lines:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    # Execute secure chunked commit
    prefix = config.get('commit_message_prefix', 'Auto-commit: ')
    success, message = guardian.secure_chunked_commit(
        file_path, chunks, f"{prefix}create {file_path} - ", 'create'
    )

    if success:
        logger.info(f"Secure chunked file creation completed: {message}")
    else:
        logger.error(f"Secure chunked file creation failed: {message}")

    return success
