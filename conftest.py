# conftest.py  (project root)
# ============================================================================
# Pytest path configuration.
#
# Adds ``src/`` to sys.path so that audit modules can be imported via their
# natural package name:
#
#   from audit.audit_logger       import AuditLogger
#   from audit.method1_exact_hash import ExactHashChecker
#
# This mirrors how the modules import each other internally.
# ============================================================================

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Reduce OpenMP/MKL thread-pool size BEFORE torch is imported.
# This significantly lowers the virtual-memory footprint of libiomp5md.dll,
# which prevents "WinError 1455 – paging file too small" on Windows.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # suppress duplicate-OMP warnings
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

sys.path.insert(0, str(Path(__file__).parent / "src"))
