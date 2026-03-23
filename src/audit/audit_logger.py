# -*- coding: utf-8 -*-
"""
audit/audit_logger.py  (lives at src/audit/audit_logger.py)
============================================================
Shared logger for all data integrity audit methods.

All four audit methods (ExactHashChecker, PHashChecker, EmbeddingChecker,
HardCropProbe) and the AuditRunner import this logger.

Class
-----
AuditLogger
    Writes messages to both stdout and a persistent log file.
    API is intentionally identical to LiveLogger so either can be used
    as a drop-in in tests.

Example
-------
>>> log = AuditLogger(Path("./audit_out/audit_log.txt"))
>>> log.log("Method 1 starting")
>>> log.sep(heavy=True)
>>> log.close()
"""

import time
from pathlib import Path


class AuditLogger:
    """
    Dual-stream logger — stdout + file — for data integrity audits.

    Parameters
    ----------
    path : Path
        Destination log file.  Parent directories are created automatically.

    Example
    -------
    >>> log = AuditLogger(Path("./audit_out/log.txt"))
    >>> log.log("Checking for duplicates...")
    >>> log.sep()
    >>> log.close()
    """

    H: str = "=" * 80
    L: str = "-" * 80

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._f    = open(path, "w", encoding="utf-8")
        self._write(
            f"TRACE Data Audit Log | Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        self._write(self.H + "\n")

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def log(self, msg: str = "") -> None:
        """Write ``msg`` followed by a newline."""
        self._write(msg + "\n")

    def sep(self, heavy: bool = False) -> None:
        """Write a separator line (``=`` heavy, ``-`` light)."""
        self._write((self.H if heavy else self.L) + "\n")

    def close(self) -> None:
        """Write footer and close the file handle."""
        self._write(self.H + "\n")
        self._write(
            f"Audit Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        try:
            self._f.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _write(self, msg: str) -> None:
        print(msg, end="", flush=True)
        try:
            self._f.write(msg)
            self._f.flush()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"AuditLogger({self._path})"
