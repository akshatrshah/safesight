"""
Pytest configuration file.

WHY THIS FILE EXISTS
----------------------
pytest, by default, only adds a test file's own directory to sys.path
(when there's no __init__.py in tests/, which is our case). That means
`from perception.detection.detector import Detector` fails with
`ModuleNotFoundError: No module named 'perception'`, because the repo
root — where the `perception` package actually lives — was never added
to sys.path.

pytest automatically discovers and loads any conftest.py at the rootdir
before collecting tests, so inserting the repo root into sys.path here
makes `perception` importable everywhere, without needing to manually
set PYTHONPATH or install the package.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
