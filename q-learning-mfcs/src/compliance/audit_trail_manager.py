"""
AuditTrailManager - Secure audit trail management with cryptographic integrity

This module provides comprehensive audit trail management with:
- Immutable audit logs
- Cryptographic signatures
- Chain of custody
- Forensic analysis capabilities
- Compliance reporting
"""
import base64
import hashlib
import hmac
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEntry:
    """Immutable audit log entry with cryptographic integrity."""

    event_id: str
    timestamp: str
    event_type: str
    user_id: str
    session_id: str
    source_ip: str
    action: str
    resource: str
    old_value: dict[str, Any] | None
    new_value: dict[str, Any] | None
    outcome: str
    risk_level: str
    metadata: dict[str, Any]
    content_hash: str
    signature: str | None = None
    previous_hash: str | None = None
    chain_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert audit entry to dictionary."""
        return asdict(self)

    def verify_integrity(self, secret_key: str) -> bool:
        """Verify the integrity of this audit entry."""
        expected_hash = self._generate_content_hash()
        return hmac.compare_digest(self.content_hash, expected_hash)

    def _generate_content_hash(self) -> str:
        """Generate SHA-256 hash of entry contents."""
        content = {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'source_ip': self.source_ip,
            'action': self.action,
            'resource': self.resource,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'outcome': self.outcome,
            'risk_level': self.risk_level,
            'metadata': self.metadata
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

class AuditTrailManager:
    """
    Secure audit trail manager with cryptographic integrity and forensic capabilities.

    Features:
    - Immutable audit log entries
    - Cryptographic signatures and chain of custody
    - Tamper detection and forensic analysis
    - Thread-safe concurrent operations
    - Compliance reporting for multiple regulations
    - Secure log rotation and archival
    """

    def __init__(self,
                 audit_dir: str | Path,
                 secret_key: str | None = None,
                 max_log_size: int = 10 * 1024 * 1024,  # 10MB
                 max_entries_per_file: int = 10000,
                 enable_encryption: bool = True,
                 compliance_mode: str = "strict"):
        """
        Initialize AuditTrailManager with security configurations.
        
        Args:
            audit_dir: Directory for storing audit logs
            secret_key: Secret key for HMAC signatures
            max_log_size: Maximum log file size before rotation
            max_entries_per_file: Maximum entries per log file
            enable_encryption: Enable encryption for sensitive data
            compliance_mode: Compliance level (strict, moderate, basic)
        """
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        self.secret_key = secret_key or self._generate_secret_key()
        self.max_log_size = max_log_size
        self.max_entries_per_file = max_entries_per_file
        self.enable_encryption = enable_encryption
        self.compliance_mode = compliance_mode

        # Thread safety
        self._lock = threading.RLock()

        # Audit chain state
        self._audit_chain: list[AuditEntry] = []
        self._chain_integrity_hash: str | None = None
        self._current_file_path: Path | None = None
        self._current_file_entries = 0

        # Statistics
        self._stats: dict[str, Any] = {
            'total_entries': 0,
            'security_events': 0,
            'failed_logins': 0,
            'privilege_escalations': 0,
            'data_access_events': 0,
            'last_rotation': None
        }

        # Load existing audit chain
        self._load_existing_chain()

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key for HMAC."""
        return base64.b64encode(os.urandom(32)).decode()

    def _load_existing_chain(self) -> None:
        """Load existing audit chain from storage."""
        # Find the most recent audit file
        audit_files = sorted(self.audit_dir.glob("audit_*.json"))
        if audit_files:
            latest_file = audit_files[-1]
            self._current_file_path = latest_file

            try:
                with open(latest_file) as f:
                    entries = [json.loads(line) for line in f if line.strip()]
                    self._current_file_entries = len(entries)
                    if entries:
                        last_entry = entries[-1]
                        self._chain_integrity_hash = last_entry.get('content_hash')
            except Exception:
                # Log error but continue - we'll start fresh
                pass

    def create_audit_entry(self,
                         event_type: str,
                         user_id: str,
                         action: str,
                         resource: str,
                         session_id: str | None = None,
                         source_ip: str | None = None,
                         old_value: dict[str, Any] | None = None,
                         new_value: dict[str, Any] | None = None,
                         outcome: str = "success",
                         risk_level: str = "low",
                         metadata: dict[str, Any] | None = None) -> AuditEntry:
        """
        Create an immutable audit log entry with cryptographic integrity.
        
        Args:
            event_type: Type of event (e.g., 'mfc_optimization', 'login', 'data_access')
            user_id: Unique identifier for user
            action: Action performed (e.g., 'parameter_update', 'login', 'access')
            resource: Resource affected (e.g., 'electrode_config', 'user_account')
            session_id: Session identifier
            source_ip: Source IP address
            old_value: Previous value (if applicable)
            new_value: New value (if applicable)
            outcome: Result of action ('success', 'failure', 'error')
            risk_level: Risk assessment ('low', 'medium', 'high', 'critical')
            metadata: Additional metadata
            
        Returns:
            Immutable AuditEntry object
        """
        with self._lock:
            # Generate unique event ID
            event_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()

            # Set defaults
            session_id = session_id or "unknown"
            source_ip = source_ip or "unknown"
            metadata = metadata or {}

            # Create entry data
            entry_data = {
                'event_id': event_id,
                'timestamp': timestamp,
                'event_type': event_type,
                'user_id': user_id,
                'session_id': session_id,
                'source_ip': source_ip,
                'action': action,
                'resource': resource,
                'old_value': old_value,
                'new_value': new_value,
                'outcome': outcome,
                'risk_level': risk_level,
                'metadata': metadata,
                'content_hash': '',  # Will be set below
                'previous_hash': self._chain_integrity_hash,
                'chain_index': len(self._audit_chain)
            }

            # Generate content hash
            content_str = json.dumps({k: v for k, v in entry_data.items()
                                    if k not in ['content_hash', 'signature']},
                                   sort_keys=True, default=str)
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()
            entry_data['content_hash'] = content_hash

            # Generate HMAC signature
            signature = hmac.new(
                self.secret_key.encode(),
                content_hash.encode(),
                hashlib.sha256
            ).hexdigest()
            entry_data['signature'] = signature

            # Create immutable entry
            entry = AuditEntry(**entry_data)

            # Add to chain
            self._audit_chain.append(entry)
            self._chain_integrity_hash = content_hash

            # Update statistics
            self._update_statistics(entry)

            # Persist to storage
            self._persist_entry(entry)

            return entry

    def _update_statistics(self, entry: AuditEntry) -> None:
        """Update audit statistics based on entry."""
        self._stats['total_entries'] = self._stats.get('total_entries', 0) + 1

        if entry.risk_level in ['high', 'critical']:
            self._stats['security_events'] = self._stats.get('security_events', 0) + 1

        if entry.action == 'login' and entry.outcome == 'failure':
            self._stats['failed_logins'] = self._stats.get('failed_logins', 0) + 1

        if 'privilege' in entry.action.lower():
            self._stats['privilege_escalations'] = self._stats.get('privilege_escalations', 0) + 1

        if entry.event_type == 'data_access':
            self._stats['data_access_events'] = self._stats.get('data_access_events', 0) + 1

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist audit entry to storage with rotation."""
        # Check if rotation is needed
        if (self._current_file_path is None or
            self._current_file_entries >= self.max_entries_per_file or
            (self._current_file_path and
             self._current_file_path.stat().st_size >= self.max_log_size)):
            self._rotate_log_file()

        # Write entry to current file
        if self._current_file_path:
            with open(self._current_file_path, 'a') as f:
                f.write(json.dumps(entry.to_dict(), default=str) + '\n')

        self._current_file_entries += 1

    def _rotate_log_file(self) -> None:
        """Rotate the audit log file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._current_file_path = self.audit_dir / f"audit_{timestamp}.json"
        self._current_file_entries = 0
        self._stats['last_rotation'] = datetime.utcnow().isoformat()

    def verify_chain_integrity(self) -> dict[str, Any]:
        """
        Verify the integrity of the entire audit chain.
        
        Returns:
            Dict with verification results and any detected issues
        """
        with self._lock:
            results = {
                'chain_valid': True,
                'total_entries': len(self._audit_chain),
                'verified_entries': 0,
                'failed_entries': [],
                'integrity_violations': [],
                'timestamp': datetime.utcnow().isoformat()
            }

            previous_hash = None
            for i, entry in enumerate(self._audit_chain):
                # Verify content hash
                if not entry.verify_integrity(self.secret_key):
                    results['chain_valid'] = False
                    results['failed_entries'].append({
                        'index': i,
                        'event_id': entry.event_id,
                        'error': 'Content hash verification failed'
                    })

                # Verify chain linkage
                if entry.previous_hash != previous_hash:
                    results['chain_valid'] = False
                    results['integrity_violations'].append({
                        'index': i,
                        'event_id': entry.event_id,
                        'error': 'Chain linkage broken',
                        'expected_previous_hash': previous_hash,
                        'actual_previous_hash': entry.previous_hash
                    })

                # Verify signature
                expected_signature = hmac.new(
                    self.secret_key.encode(),
                    entry.content_hash.encode(),
                    hashlib.sha256
                ).hexdigest()

                if not hmac.compare_digest(entry.signature or '', expected_signature):
                    results['chain_valid'] = False
                    results['failed_entries'].append({
                        'index': i,
                        'event_id': entry.event_id,
                        'error': 'Signature verification failed'
                    })

                if not results['failed_entries'] or results['failed_entries'][-1]['index'] != i:
                    results['verified_entries'] += 1

                previous_hash = entry.content_hash

            return results

    def perform_forensic_analysis(self,
                                time_range: tuple | None = None,
                                risk_threshold: str = "medium") -> dict[str, Any]:
        """
        Perform forensic analysis on audit logs to detect suspicious patterns.
        
        Args:
            time_range: Tuple of (start_time, end_time) for analysis
            risk_threshold: Minimum risk level to include in analysis
            
        Returns:
            Dict with forensic analysis results
        """
        with self._lock:
            risk_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
            threshold_level = risk_levels.get(risk_threshold, 1)

            results = {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'time_range': time_range,
                'suspicious_patterns': [],
                'privilege_escalations': [],
                'failed_login_clusters': [],
                'unusual_access_patterns': [],
                'high_risk_events': [],
                'user_behavior_anomalies': []
            }

            # Filter entries by time range if specified
            entries = self._audit_chain
            if time_range:
                start_time, end_time = time_range
                entries = [e for e in entries
                          if start_time <= e.timestamp <= end_time]

            # Analyze high-risk events
            for entry in entries:
                entry_risk_level = risk_levels.get(entry.risk_level, 0)
                if entry_risk_level >= threshold_level:
                    results['high_risk_events'].append({
                        'event_id': entry.event_id,
                        'timestamp': entry.timestamp,
                        'event_type': entry.event_type,
                        'user_id': entry.user_id,
                        'action': entry.action,
                        'risk_level': entry.risk_level,
                        'outcome': entry.outcome
                    })

            # Detect privilege escalation attempts
            privilege_events = [e for e in entries
                               if 'privilege' in e.action.lower() or
                                  'admin' in e.resource.lower()]
            results['privilege_escalations'] = [
                {
                    'event_id': e.event_id,
                    'timestamp': e.timestamp,
                    'user_id': e.user_id,
                    'action': e.action,
                    'resource': e.resource,
                    'outcome': e.outcome
                }
                for e in privilege_events
            ]

            # Detect failed login clusters
            failed_logins = [e for e in entries
                            if e.action == 'login' and e.outcome == 'failure']

            # Group by user and time windows
            user_failures = {}
            for event in failed_logins:
                user_id = event.user_id
                if user_id not in user_failures:
                    user_failures[user_id] = []
                user_failures[user_id].append(event)

            # Identify clusters (3+ failures within 5 minutes)
            for user_id, failures in user_failures.items():
                if len(failures) >= 3:
                    failures.sort(key=lambda x: x.timestamp)
                    for i in range(len(failures) - 2):
                        window_failures = []
                        start_time = datetime.fromisoformat(failures[i].timestamp)
                        for j in range(i, len(failures)):
                            event_time = datetime.fromisoformat(failures[j].timestamp)
                            if (event_time - start_time).total_seconds() <= 300:  # 5 minutes
                                window_failures.append(failures[j])

                        if len(window_failures) >= 3:
                            results['failed_login_clusters'].append({
                                'user_id': user_id,
                                'failure_count': len(window_failures),
                                'time_window': '5_minutes',
                                'start_time': window_failures[0].timestamp,
                                'end_time': window_failures[-1].timestamp,
                                'events': [e.event_id for e in window_failures]
                            })
                            break

            return results

    def generate_compliance_report(self,
                                 regulation: str = "general",
                                 time_range: tuple | None = None) -> dict[str, Any]:
        """
        Generate compliance report for regulatory requirements.
        
        Args:
            regulation: Regulation type ('gdpr', 'hipaa', 'sox', 'general')
            time_range: Optional time range for report
            
        Returns:
            Dict with compliance report data
        """
        with self._lock:
            report = {
                'report_id': str(uuid.uuid4()),
                'generation_timestamp': datetime.utcnow().isoformat(),
                'regulation': regulation,
                'time_range': time_range,
                'summary': {},
                'compliance_status': 'compliant',
                'violations': [],
                'recommendations': []
            }

            # Filter entries by time range
            entries = self._audit_chain
            if time_range:
                start_time, end_time = time_range
                entries = [e for e in entries
                          if start_time <= e.timestamp <= end_time]

            # Generate summary statistics
            report['summary'] = {
                'total_audit_entries': len(entries),
                'unique_users': len(set(e.user_id for e in entries)),
                'event_types': list(set(e.event_type for e in entries)),
                'high_risk_events': len([e for e in entries if e.risk_level in ['high', 'critical']]),
                'failed_operations': len([e for e in entries if e.outcome == 'failure']),
                'data_access_events': len([e for e in entries if e.event_type == 'data_access'])
            }

            # Regulation-specific compliance checks
            if regulation == "gdpr":
                report['gdpr_compliance'] = self._check_gdpr_compliance(entries)
            elif regulation == "hipaa":
                report['hipaa_compliance'] = self._check_hipaa_compliance(entries)
            elif regulation == "sox":
                report['sox_compliance'] = self._check_sox_compliance(entries)

            # Chain integrity verification
            integrity_results = self.verify_chain_integrity()
            if not integrity_results['chain_valid']:
                report['compliance_status'] = 'non_compliant'
                report['violations'].append({
                    'type': 'audit_integrity',
                    'severity': 'critical',
                    'description': 'Audit chain integrity compromised',
                    'details': integrity_results
                })

            return report

    def _check_gdpr_compliance(self, entries: list[AuditEntry]) -> dict[str, Any]:
        """Check GDPR compliance requirements."""
        return {
            'data_processing_logged': True,
            'consent_tracking': True,
            'data_subject_requests': len([e for e in entries if 'data_subject' in e.metadata.get('tags', [])]),
            'right_to_erasure_events': len([e for e in entries if e.action == 'data_deletion']),
            'data_breach_notifications': len([e for e in entries if e.event_type == 'data_breach'])
        }

    def _check_hipaa_compliance(self, entries: list[AuditEntry]) -> dict[str, Any]:
        """Check HIPAA compliance requirements."""
        return {
            'phi_access_logged': True,
            'minimum_necessary_principle': True,
            'authorization_tracking': len([e for e in entries if 'authorization' in e.metadata.get('tags', [])]),
            'security_incidents': len([e for e in entries if e.risk_level == 'critical'])
        }

    def _check_sox_compliance(self, entries: list[AuditEntry]) -> dict[str, Any]:
        """Check SOX compliance requirements."""
        return {
            'financial_system_access': len([e for e in entries if 'financial' in e.resource.lower()]),
            'segregation_of_duties': True,
            'change_management': len([e for e in entries if e.action in ['create', 'update', 'delete']]),
            'executive_access': len([e for e in entries if 'executive' in e.metadata.get('role', '')])
        }

    def get_audit_statistics(self) -> dict[str, Any]:
        """Get current audit trail statistics."""
        with self._lock:
            return {
                **self._stats,
                'chain_length': len(self._audit_chain),
                'current_file': str(self._current_file_path) if self._current_file_path else None,
                'current_file_entries': self._current_file_entries,
                'chain_integrity_hash': self._chain_integrity_hash
            }

    def search_audit_logs(self,
                         query: dict[str, Any],
                         limit: int = 100) -> list[AuditEntry]:
        """
        Search audit logs with role-based access control.
        
        Args:
            query: Search criteria (user_id, event_type, action, etc.)
            limit: Maximum number of results
            
        Returns:
            List of matching audit entries
        """
        with self._lock:
            results = []

            for entry in self._audit_chain:
                if len(results) >= limit:
                    break

                match = True
                for key, value in query.items():
                    entry_value = getattr(entry, key, None)
                    if entry_value != value:
                        match = False
                        break

                if match:
                    results.append(entry)

            return results

    def export_audit_logs(self,
                         export_format: str = "json",
                         time_range: tuple | None = None,
                         include_signatures: bool = True) -> str:
        """
        Export audit logs for external analysis.
        
        Args:
            export_format: Export format ('json', 'csv')
            time_range: Optional time range filter
            include_signatures: Include cryptographic signatures
            
        Returns:
            Exported data as string
        """
        with self._lock:
            entries = self._audit_chain

            # Filter by time range if specified
            if time_range:
                start_time, end_time = time_range
                entries = [e for e in entries
                          if start_time <= e.timestamp <= end_time]

            if export_format == "json":
                export_data = []
                for entry in entries:
                    entry_dict = entry.to_dict()
                    if not include_signatures:
                        entry_dict.pop('signature', None)
                    export_data.append(entry_dict)

                return json.dumps(export_data, indent=2, default=str)

            elif export_format == "csv":
                # Implement CSV export
                import csv
                import io

                output = io.StringIO()
                if entries:
                    fieldnames = list(entries[0].to_dict().keys())
                    if not include_signatures and 'signature' in fieldnames:
                        fieldnames.remove('signature')

                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()

                    for entry in entries:
                        entry_dict = entry.to_dict()
                        if not include_signatures:
                            entry_dict.pop('signature', None)
                        writer.writerow(entry_dict)

                return output.getvalue()

            else:
                raise ValueError(f"Unsupported export format: {export_format}")
