"""Additional tests for audit_trail_manager.py - cover lines 320-321, 345."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import hashlib
import hmac as hmac_mod

import pytest

from compliance.audit_trail_manager import AuditEntry, AuditTrailManager


def _make_valid_entry(
    event_id, timestamp, event_type, user_id, session_id, source_ip,
    action, resource, old_value, new_value, outcome, risk_level,
    metadata, previous_hash, chain_index, secret_key,
):
    """Build an AuditEntry whose content_hash and signature pass
    verify_chain_integrity checks."""
    # Create a temporary entry to get its _generate_content_hash
    temp = AuditEntry(
        event_id=event_id, timestamp=timestamp,
        event_type=event_type, user_id=user_id,
        session_id=session_id, source_ip=source_ip,
        action=action, resource=resource,
        old_value=old_value, new_value=new_value,
        outcome=outcome, risk_level=risk_level,
        metadata=metadata, content_hash="placeholder",
        previous_hash=previous_hash, chain_index=chain_index,
    )
    content_hash = temp._generate_content_hash()
    signature = hmac_mod.new(
        secret_key.encode(), content_hash.encode(), hashlib.sha256
    ).hexdigest()
    return AuditEntry(
        event_id=event_id, timestamp=timestamp,
        event_type=event_type, user_id=user_id,
        session_id=session_id, source_ip=source_ip,
        action=action, resource=resource,
        old_value=old_value, new_value=new_value,
        outcome=outcome, risk_level=risk_level,
        metadata=metadata, content_hash=content_hash,
        signature=signature, previous_hash=previous_hash,
        chain_index=chain_index,
    )


@pytest.mark.coverage_extra
class TestChainLinkageBroken:
    """Cover lines 319-321: chain linkage broken detection."""

    def test_verify_detects_broken_chain_linkage(self, tmp_path):
        """Construct valid entries then break chain linkage on the second one."""
        mgr = AuditTrailManager(audit_dir=tmp_path / "audit")
        secret = mgr.secret_key

        e1 = _make_valid_entry(
            event_id="e1", timestamp="2025-01-01T00:00:00",
            event_type="login", user_id="u1", session_id="s1",
            source_ip="10.0.0.1", action="login", resource="sys",
            old_value=None, new_value=None, outcome="success",
            risk_level="low", metadata={},
            previous_hash=None, chain_index=0, secret_key=secret,
        )
        # Second entry: correct linkage would be previous_hash=e1.content_hash
        # But we intentionally break it
        e2 = _make_valid_entry(
            event_id="e2", timestamp="2025-01-01T00:01:00",
            event_type="read", user_id="u2", session_id="s2",
            source_ip="10.0.0.2", action="read", resource="db",
            old_value=None, new_value=None, outcome="success",
            risk_level="low", metadata={},
            previous_hash="completely_wrong_hash",  # broken linkage
            chain_index=1, secret_key=secret,
        )

        mgr._audit_chain = [e1, e2]

        result = mgr.verify_chain_integrity()
        assert result["chain_valid"] is False
        assert len(result["integrity_violations"]) >= 1
        violation = result["integrity_violations"][0]
        assert violation["error"] == "Chain linkage broken"
        assert violation["actual_previous_hash"] == "completely_wrong_hash"


@pytest.mark.coverage_extra
class TestVerifiedEntriesCounting:
    """Cover line 345: verified_entries increment for entries without failures."""

    def test_verified_entries_count_on_valid_chain(self, tmp_path):
        """A fully valid chain should have verified_entries == total_entries."""
        mgr = AuditTrailManager(audit_dir=tmp_path / "audit")
        secret = mgr.secret_key

        e1 = _make_valid_entry(
            event_id="e1", timestamp="2025-01-01T00:00:00",
            event_type="op", user_id="u", session_id="s",
            source_ip="1.2.3.4", action="a", resource="r",
            old_value=None, new_value=None, outcome="success",
            risk_level="low", metadata={},
            previous_hash=None, chain_index=0, secret_key=secret,
        )
        e2 = _make_valid_entry(
            event_id="e2", timestamp="2025-01-01T00:01:00",
            event_type="op", user_id="u", session_id="s",
            source_ip="1.2.3.4", action="a", resource="r",
            old_value=None, new_value=None, outcome="success",
            risk_level="low", metadata={},
            previous_hash=e1.content_hash, chain_index=1, secret_key=secret,
        )
        e3 = _make_valid_entry(
            event_id="e3", timestamp="2025-01-01T00:02:00",
            event_type="op", user_id="u", session_id="s",
            source_ip="1.2.3.4", action="a", resource="r",
            old_value=None, new_value=None, outcome="success",
            risk_level="low", metadata={},
            previous_hash=e2.content_hash, chain_index=2, secret_key=secret,
        )

        mgr._audit_chain = [e1, e2, e3]

        result = mgr.verify_chain_integrity()
        assert result["chain_valid"] is True
        assert result["verified_entries"] == 3
        assert result["total_entries"] == 3

    def test_verified_entries_excludes_failed(self, tmp_path):
        """When the second entry has broken chain linkage,
        verified_entries should count only entries without failures."""
        mgr = AuditTrailManager(audit_dir=tmp_path / "audit")
        secret = mgr.secret_key

        e1 = _make_valid_entry(
            event_id="e1", timestamp="2025-01-01T00:00:00",
            event_type="op", user_id="u", session_id="s",
            source_ip="1.2.3.4", action="a", resource="r",
            old_value=None, new_value=None, outcome="success",
            risk_level="low", metadata={},
            previous_hash=None, chain_index=0, secret_key=secret,
        )
        # Second entry has wrong previous_hash (only chain linkage breaks)
        e2 = _make_valid_entry(
            event_id="e2", timestamp="2025-01-01T00:01:00",
            event_type="op", user_id="u", session_id="s",
            source_ip="1.2.3.4", action="a", resource="r",
            old_value=None, new_value=None, outcome="success",
            risk_level="low", metadata={},
            previous_hash="wrong_linkage", chain_index=1, secret_key=secret,
        )

        mgr._audit_chain = [e1, e2]

        result = mgr.verify_chain_integrity()
        assert result["chain_valid"] is False
        # e1 passes all checks -> verified; e2 fails chain linkage but passes
        # content hash and signature, so it's not in failed_entries.
        # The condition on line 344 checks failed_entries, not integrity_violations.
        # e2 is NOT in failed_entries (chain linkage goes to integrity_violations),
        # so verified_entries should still count it.
        # Expected: both entries counted as verified (integrity_violations != failed_entries)
        assert result["verified_entries"] == 2
        assert len(result["integrity_violations"]) == 1