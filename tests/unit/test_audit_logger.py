# -*- coding: utf-8 -*-
"""
tests/unit/test_audit_logger.py
================================
Unit tests for AuditLogger — the shared logger used by all audit methods.
"""

from pathlib import Path

import pytest

from audit.audit_logger import AuditLogger


class TestAuditLogger:

    def test_creates_log_file(self, tmp_path):
        log = AuditLogger(tmp_path / "audit.log")
        log.close()
        assert (tmp_path / "audit.log").exists()

    def test_log_writes_message(self, tmp_path):
        log = AuditLogger(tmp_path / "audit.log")
        log.log("Method 1 starting")
        log.close()
        content = (tmp_path / "audit.log").read_text()
        assert "Method 1 starting" in content

    def test_sep_light_writes_dashes(self, tmp_path):
        log = AuditLogger(tmp_path / "audit.log")
        log.sep(heavy=False)
        log.close()
        assert "-" * 78 in (tmp_path / "audit.log").read_text()

    def test_sep_heavy_writes_equals(self, tmp_path):
        log = AuditLogger(tmp_path / "audit.log")
        log.sep(heavy=True)
        log.close()
        assert "=" * 78 in (tmp_path / "audit.log").read_text()

    def test_close_writes_footer(self, tmp_path):
        log = AuditLogger(tmp_path / "audit.log")
        log.close()
        content = (tmp_path / "audit.log").read_text()
        assert "finished" in content.lower() or "Audit" in content

    def test_creates_parent_dirs(self, tmp_path):
        log = AuditLogger(tmp_path / "nested" / "deep" / "audit.log")
        log.close()
        assert (tmp_path / "nested" / "deep" / "audit.log").exists()

    def test_repr(self, tmp_path):
        log = AuditLogger(tmp_path / "audit.log")
        log.close()
        assert "AuditLogger" in repr(log)