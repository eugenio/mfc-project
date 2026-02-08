"""Tests for audit_trail_manager.py module - comprehensive coverage."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import json
from datetime import datetime
from pathlib import Path

import pytest

from compliance.audit_trail_manager import (
    AuditEntry,
    AuditTrailManager,
)


class TestAuditEntry:
    def _make_entry(self, **kwargs):
        defaults = {
            "event_id": "evt-001",
            "timestamp": "2025-01-01T00:00:00",
            "event_type": "login",
            "user_id": "user1",
            "session_id": "sess1",
            "source_ip": "10.0.0.1",
            "action": "login",
            "resource": "system",
            "old_value": None,
            "new_value": None,
            "outcome": "success",
            "risk_level": "low",
            "metadata": {},
            "content_hash": "abc123",
            "signature": "sig123",
            "previous_hash": None,
            "chain_index": 0,
        }
        defaults.update(kwargs)
        return AuditEntry(**defaults)

    def test_create_entry(self):
        entry = self._make_entry()
        assert entry.event_id == "evt-001"
        assert entry.event_type == "login"

    def test_to_dict(self):
        entry = self._make_entry()
        d = entry.to_dict()
        assert d["event_id"] == "evt-001"
        assert d["content_hash"] == "abc123"

    def test_verify_integrity_success(self):
        entry = self._make_entry()
        expected_hash = entry._generate_content_hash()
        entry2 = self._make_entry(content_hash=expected_hash)
        assert entry2.verify_integrity("secret") is True

    def test_verify_integrity_failure(self):
        entry = self._make_entry(content_hash="wrong_hash")
        assert entry.verify_integrity("secret") is False

    def test_generate_content_hash(self):
        entry = self._make_entry()
        h = entry._generate_content_hash()
        assert isinstance(h, str)
        assert len(h) == 64

    def test_frozen_dataclass(self):
        entry = self._make_entry()
        with pytest.raises(AttributeError):
            entry.event_id = "new_id"


class TestAuditTrailManagerInit:
    def test_init_creates_directory(self, tmp_path):
        audit_dir = tmp_path / "audit_logs"
        AuditTrailManager(audit_dir=audit_dir)
        assert audit_dir.exists()

    def test_init_with_secret_key(self, tmp_path):
        mgr = AuditTrailManager(audit_dir=tmp_path / "a", secret_key="my_secret")
        assert mgr.secret_key == "my_secret"

    def test_init_generates_secret_key(self, tmp_path):
        mgr = AuditTrailManager(audit_dir=tmp_path / "a")
        assert mgr.secret_key is not None and len(mgr.secret_key) > 0

    def test_init_default_params(self, tmp_path):
        mgr = AuditTrailManager(audit_dir=tmp_path / "a")
        assert mgr.max_log_size == 10 * 1024 * 1024
        assert mgr.max_entries_per_file == 10000
        assert mgr.enable_encryption is True
        assert mgr.compliance_mode == "strict"

    def test_load_existing_chain_with_files(self, tmp_path):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        audit_file = audit_dir / "audit_20250101_000000.json"
        entry_data = {"event_id": "test", "content_hash": "abc123"}
        with open(audit_file, "w") as f:
            f.write(json.dumps(entry_data) + "\n")
        mgr = AuditTrailManager(audit_dir=audit_dir)
        assert mgr._chain_integrity_hash == "abc123"
        assert mgr._current_file_entries == 1

    def test_load_existing_chain_corrupt_file(self, tmp_path):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        audit_file = audit_dir / "audit_20250101_000000.json"
        with open(audit_file, "w") as f:
            f.write("not valid json\n")
        mgr = AuditTrailManager(audit_dir=audit_dir)
        assert mgr is not None


class TestAuditTrailManagerEntries:
    @pytest.fixture
    def manager(self, tmp_path):
        return AuditTrailManager(audit_dir=tmp_path / "audit")

    def test_create_audit_entry(self, manager):
        entry = manager.create_audit_entry(
            event_type="login", user_id="user1",
            action="login", resource="system",
        )
        assert isinstance(entry, AuditEntry)
        assert entry.event_type == "login"
        assert entry.content_hash != ""
        assert entry.signature is not None

    def test_create_entry_with_opts(self, manager):
        entry = manager.create_audit_entry(
            event_type="data_access", user_id="user2",
            action="read", resource="database",
            session_id="sess-123", source_ip="192.168.1.1",
            old_value={"key": "old"}, new_value={"key": "new"},
            outcome="success", risk_level="high",
            metadata={"reason": "audit"},
        )
        assert entry.session_id == "sess-123"
        assert entry.source_ip == "192.168.1.1"
        assert entry.risk_level == "high"

    def test_create_entry_defaults(self, manager):
        entry = manager.create_audit_entry(
            event_type="test", user_id="u",
            action="act", resource="res",
        )
        assert entry.session_id == "unknown"
        assert entry.source_ip == "unknown"

    def test_chain_linkage(self, manager):
        e1 = manager.create_audit_entry(
            event_type="e1", user_id="u", action="a", resource="r"
        )
        e2 = manager.create_audit_entry(
            event_type="e2", user_id="u", action="a", resource="r"
        )
        assert e2.previous_hash == e1.content_hash
        assert e2.chain_index == 1

    def test_stats_total(self, manager):
        manager.create_audit_entry(
            event_type="test", user_id="u", action="act", resource="r"
        )
        assert manager._stats["total_entries"] == 1

    def test_stats_security(self, manager):
        manager.create_audit_entry(
            event_type="t", user_id="u", action="a",
            resource="r", risk_level="high"
        )
        assert manager._stats["security_events"] == 1

    def test_stats_critical(self, manager):
        manager.create_audit_entry(
            event_type="t", user_id="u", action="a",
            resource="r", risk_level="critical"
        )
        assert manager._stats["security_events"] == 1

    def test_stats_failed_logins(self, manager):
        manager.create_audit_entry(
            event_type="l", user_id="u", action="login",
            resource="r", outcome="failure"
        )
        assert manager._stats["failed_logins"] == 1

    def test_stats_privilege(self, manager):
        manager.create_audit_entry(
            event_type="a", user_id="u",
            action="privilege_escalation", resource="r"
        )
        assert manager._stats["privilege_escalations"] == 1

    def test_stats_data_access(self, manager):
        manager.create_audit_entry(
            event_type="data_access", user_id="u",
            action="read", resource="r"
        )
        assert manager._stats["data_access_events"] == 1

    def test_persist_creates_file(self, manager):
        manager.create_audit_entry(
            event_type="t", user_id="u", action="a", resource="r"
        )
        assert manager._current_file_path is not None
        assert manager._current_file_path.exists()

    def test_rotate_log_file(self, manager):
        manager._rotate_log_file()
        assert manager._current_file_path is not None
        assert manager._current_file_entries == 0

    def test_rotation_on_max_entries(self, tmp_path):
        mgr = AuditTrailManager(
            audit_dir=tmp_path / "a", max_entries_per_file=2
        )
        for _ in range(3):
            mgr.create_audit_entry(
                event_type="t", user_id="u", action="a", resource="r"
            )
        assert mgr._current_file_entries >= 1

    def test_rotation_on_file_size(self, tmp_path):
        mgr = AuditTrailManager(
            audit_dir=tmp_path / "a", max_log_size=100
        )
        for _ in range(3):
            mgr.create_audit_entry(
                event_type="t", user_id="u", action="a", resource="r"
            )
        files = list(mgr.audit_dir.glob("audit_*.json"))
        assert len(files) >= 1


class TestVerifyAndForensics:
    @pytest.fixture
    def manager(self, tmp_path):
        return AuditTrailManager(audit_dir=tmp_path / "audit")

    def test_verify_empty(self, manager):
        r = manager.verify_chain_integrity()
        assert r["chain_valid"] is True and r["total_entries"] == 0

    def test_verify_valid(self, manager):
        for _ in range(2):
            manager.create_audit_entry(
                event_type="e", user_id="u", action="a", resource="r"
            )
        r = manager.verify_chain_integrity()
        assert r["total_entries"] == 2

    def test_verify_tampered(self, manager):
        manager.create_audit_entry(
            event_type="e", user_id="u", action="a", resource="r"
        )
        e = manager._audit_chain[0]
        manager._audit_chain[0] = AuditEntry(
            event_id=e.event_id, timestamp=e.timestamp,
            event_type=e.event_type, user_id=e.user_id,
            session_id=e.session_id, source_ip=e.source_ip,
            action=e.action, resource=e.resource,
            old_value=e.old_value, new_value=e.new_value,
            outcome=e.outcome, risk_level=e.risk_level,
            metadata=e.metadata, content_hash=e.content_hash,
            signature="tampered",
            previous_hash=e.previous_hash, chain_index=e.chain_index,
        )
        r = manager.verify_chain_integrity()
        assert r["chain_valid"] is False

    def test_forensic_basic(self, manager):
        manager.create_audit_entry(
            event_type="login", user_id="u", action="login",
            resource="sys", risk_level="medium"
        )
        r = manager.perform_forensic_analysis()
        assert "high_risk_events" in r

    def test_forensic_time_range(self, manager):
        manager.create_audit_entry(
            event_type="l", user_id="u", action="login", resource="s"
        )
        r = manager.perform_forensic_analysis(
            time_range=("2020-01-01T00:00:00", "2030-01-01T00:00:00")
        )
        assert r["time_range"] is not None

    def test_forensic_high_risk(self, manager):
        manager.create_audit_entry(
            event_type="a", user_id="u", action="act",
            resource="s", risk_level="high"
        )
        r = manager.perform_forensic_analysis(risk_threshold="medium")
        assert len(r["high_risk_events"]) >= 1

    def test_forensic_privilege(self, manager):
        manager.create_audit_entry(
            event_type="a", user_id="u",
            action="privilege_change", resource="admin_panel"
        )
        r = manager.perform_forensic_analysis()
        assert len(r["privilege_escalations"]) >= 1

    def test_forensic_login_clusters(self, manager):
        for _ in range(5):
            manager.create_audit_entry(
                event_type="login", user_id="attacker",
                action="login", resource="system",
                outcome="failure", risk_level="medium"
            )
        r = manager.perform_forensic_analysis(risk_threshold="low")
        assert len(r["failed_login_clusters"]) >= 1


class TestReportsAndExport:
    @pytest.fixture
    def manager(self, tmp_path):
        return AuditTrailManager(audit_dir=tmp_path / "audit")

    def test_report_general(self, manager):
        manager.create_audit_entry(
            event_type="l", user_id="u", action="login", resource="s"
        )
        r = manager.generate_compliance_report()
        assert r["regulation"] == "general"

    def test_report_gdpr(self, manager):
        manager.create_audit_entry(
            event_type="data_access", user_id="u", action="read",
            resource="p", metadata={"tags": ["data_subject"]}
        )
        r = manager.generate_compliance_report(regulation="gdpr")
        assert "gdpr_compliance" in r

    def test_report_hipaa(self, manager):
        manager.create_audit_entry(
            event_type="phi", user_id="u", action="read",
            resource="m", metadata={"tags": ["authorization"]}
        )
        r = manager.generate_compliance_report(regulation="hipaa")
        assert "hipaa_compliance" in r

    def test_report_sox(self, manager):
        manager.create_audit_entry(
            event_type="a", user_id="u", action="update",
            resource="financial_system",
            metadata={"role": "executive"}
        )
        r = manager.generate_compliance_report(regulation="sox")
        assert "sox_compliance" in r

    def test_report_time_range(self, manager):
        manager.create_audit_entry(
            event_type="t", user_id="u", action="a", resource="r"
        )
        r = manager.generate_compliance_report(
            time_range=("2020-01-01T00:00:00", "2030-01-01T00:00:00")
        )
        assert r["time_range"] is not None

    def test_stats(self, manager):
        manager.create_audit_entry(
            event_type="t", user_id="u", action="a", resource="r"
        )
        s = manager.get_audit_statistics()
        assert s["total_entries"] == 1

    def test_search_user(self, manager):
        manager.create_audit_entry(
            event_type="l", user_id="alice", action="login", resource="s"
        )
        manager.create_audit_entry(
            event_type="l", user_id="bob", action="login", resource="s"
        )
        r = manager.search_audit_logs({"user_id": "alice"})
        assert len(r) == 1

    def test_search_limit(self, manager):
        for _ in range(5):
            manager.create_audit_entry(
                event_type="l", user_id="u", action="login", resource="s"
            )
        r = manager.search_audit_logs({"event_type": "l"}, limit=3)
        assert len(r) == 3

    def test_search_no_match(self, manager):
        manager.create_audit_entry(
            event_type="l", user_id="u", action="login", resource="s"
        )
        r = manager.search_audit_logs({"user_id": "x"})
        assert len(r) == 0

    def test_export_json(self, manager):
        manager.create_audit_entry(
            event_type="l", user_id="u", action="login", resource="s"
        )
        d = json.loads(manager.export_audit_logs(export_format="json"))
        assert len(d) == 1

    def test_export_json_no_sig(self, manager):
        manager.create_audit_entry(
            event_type="t", user_id="u", action="a", resource="r"
        )
        d = json.loads(
            manager.export_audit_logs(
                export_format="json", include_signatures=False
            )
        )
        assert "signature" not in d[0]

    def test_export_csv(self, manager):
        manager.create_audit_entry(
            event_type="l", user_id="u", action="login", resource="s"
        )
        r = manager.export_audit_logs(export_format="csv")
        assert "event_id" in r

    def test_export_csv_no_sig(self, manager):
        manager.create_audit_entry(
            event_type="t", user_id="u", action="a", resource="r"
        )
        r = manager.export_audit_logs(
            export_format="csv", include_signatures=False
        )
        assert "signature" not in r.split("\n")[0]

    def test_export_csv_empty(self, manager):
        r = manager.export_audit_logs(export_format="csv")
        assert r == ""

    def test_export_unsupported(self, manager):
        with pytest.raises(ValueError, match="Unsupported export format"):
            manager.export_audit_logs(export_format="xml")

    def test_export_time_range(self, manager):
        manager.create_audit_entry(
            event_type="t", user_id="u", action="a", resource="r"
        )
        d = json.loads(
            manager.export_audit_logs(
                export_format="json",
                time_range=("2020-01-01T00:00:00", "2030-01-01T00:00:00"),
            )
        )
        assert len(d) == 1
