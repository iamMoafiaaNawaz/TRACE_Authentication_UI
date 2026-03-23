# -*- coding: utf-8 -*-
"""
src/utils/preflight.py
=======================
Pre-flight library check — verifies all required packages are installed
before any heavy imports are attempted.

Classes
-------
PreflightChecker
    Checks required and optional packages and exits on missing required ones.
"""

import sys
from typing import List, Tuple


class PreflightChecker:
    """
    Verifies required and optional Python packages are installed.

    Parameters
    ----------
    required : list of (import_name, package_name, install_cmd)
    optional : list of (import_name, package_name, description)

    Example
    -------
    >>> PreflightChecker().check()
    """

    REQUIRED: List[Tuple[str, str, str]] = [
        ("torch",       "torch",        "conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia"),
        ("torchvision", "torchvision",  "conda install torchvision -c pytorch"),
        ("sklearn",     "scikit-learn", "conda install scikit-learn -c conda-forge"),
        ("PIL",         "Pillow",       "pip install Pillow"),
        ("tqdm",        "tqdm",         "conda install tqdm -c conda-forge"),
        ("seaborn",     "seaborn",      "conda install seaborn -c conda-forge"),
        ("joblib",      "joblib",       "pip install joblib"),
        ("h5py",        "h5py",         "conda install h5py -c conda-forge"),
    ]

    OPTIONAL: List[Tuple[str, str, str]] = [
        ("imagededup", "imagededup", "pip install imagededup  [audit Method 2 - pHash]"),
        ("umap",       "umap-learn", "pip install umap-learn  [audit Method 3 - UMAP plot]"),
    ]

    def check(self) -> None:
        """Run preflight. Exits with code 1 if any required package is missing."""
        missing = []
        for mod, pkg, install_cmd in self.REQUIRED:
            try:
                __import__(mod)
            except ImportError:
                missing.append((pkg, install_cmd))

        if missing:
            print("\n" + "=" * 70)
            print("PRE-FLIGHT CHECK FAILED — missing required packages:")
            for pkg, cmd in missing:
                print(f"  {pkg:20s}  ->  {cmd}")
            print("=" * 70 + "\n")
            sys.exit(1)

        print("[preflight] All required libraries present.")

        for mod, pkg, desc in self.OPTIONAL:
            try:
                __import__(mod)
            except ImportError:
                print(f"[preflight] Optional not installed: {pkg}  ({desc})")

    def available(self, module_name: str) -> bool:
        """Return True if ``module_name`` can be imported."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def __repr__(self) -> str:
        return "PreflightChecker()"
