# This file will handle file and directory operations
# -*- coding: utf-8 -*-
"""
src/utils/io_ops.py
===================
File I/O helpers and the live dual-stream logger used throughout training.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict


# ==============================================================================
# LIVE LOGGER
# ==============================================================================

class LiveLogger:
    """
    Writes log messages to both stdout and a file simultaneously.

    Designed for long-running training jobs where you need to tail
    progress in real-time AND have a persistent log file.

    Parameters
    ----------
    path : Path
        Destination log file.  Parent directories are created automatically.

    Example
    -------
    >>> log = LiveLogger(Path("./runs/train.log"))
    >>> log.log("Epoch 1 complete")
    >>> log.sep()
    >>> log.close()
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._f    = open(path, "w", encoding="utf-8")
        self._write(
            f"TRACE Training Log | Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        self._write("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def log(self, msg: str = "") -> None:
        """Write ``msg`` followed by a newline."""
        self._write(msg + "\n")

    def sep(self, heavy: bool = False) -> None:
        """Write a separator line (``=`` for heavy, ``-`` for light)."""
        self._write(("=" if heavy else "-") * 80 + "\n")

    def close(self) -> None:
        """Flush the footer and close the file handle."""
        self._write("=" * 80 + "\n")
        self._write(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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
        return f"LiveLogger(path={self._path})"


# ==============================================================================
# JSON I/O
# ==============================================================================

class JsonIO:
    """
    Simple JSON read / write helper with atomic-style safety.

    Example
    -------
    >>> JsonIO.save({"epoch": 1}, Path("./history.json"))
    >>> data = JsonIO.load(Path("./history.json"))
    """

    @staticmethod
    def save(data: Any, path: Path, indent: int = 2) -> None:
        """Serialise ``data`` to a JSON file, creating parent dirs if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=indent))

    @staticmethod
    def load(path: Path) -> Any:
        """Load and return parsed JSON from ``path``."""
        return json.loads(path.read_text())


# ==============================================================================
# WORKER COUNT RESOLVER
# ==============================================================================

class WorkerResolver:
    """
    Resolves the optimal DataLoader worker count at runtime.

    Priority:  explicit override > SLURM env var > CPU count heuristic.

    Example
    -------
    >>> n = WorkerResolver.resolve(explicit=None)
    """

    @staticmethod
    def resolve(explicit=None) -> int:
        """Return worker count, respecting SLURM and hardware limits."""
        if explicit is not None:
            return int(explicit)
        if "SLURM_CPUS_PER_TASK" in os.environ:
            return max(2, int(os.environ["SLURM_CPUS_PER_TASK"]) - 2)
        return min(4, os.cpu_count() or 4)